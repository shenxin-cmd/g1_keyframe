"""
G1 右手多形状连续轨迹生成器（4-DOF IK）
========================================
基于可行域缓存，分层随机放置多个闭合形状，50Hz 拼接为单条 NPZ 轨迹。

用法：
  # 先构建可行域
  python g1_right_hand_workspace.py

  # 预览 + 红色末端轨迹
  python g1_multi_shape_traj_gen.py --preview --save --ws-cache

  # 无头批量
  python g1_multi_shape_traj_gen.py --batch --save --ws-cache
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime
import os
import time

import mujoco
import mujoco.viewer
import numpy as np

from g1_multi_shapes import SHAPE_NAMES, make_shape_local, make_shape_waypoints
from g1_right_hand_workspace import (
    DEFAULT_FRONT_X_MAX,
    DEFAULT_FRONT_X_MIN,
    ROBOT_XML,
    WS_CACHE_PATH,
    ReachableWorkspace,
    _EE_COLLISION_BODY_NAMES,
    _OBSTACLE_BODY_NAMES_MINIMAL,
    _build_addr_table,
    _build_collision_id_sets,
    _has_penetration,
    _reset_to_home,
    build_reachable_workspace,
    load_workspace_cache,
)

os.makedirs("recordings", exist_ok=True)

# ══════════════════════════════════════════════
# 常量
# ══════════════════════════════════════════════

MODEL_PATH = "g1/scene_g1_draggable.xml"
EE_BODY = "right_wrist_yaw_link"

ACTIVE_JOINTS = [
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
]

LOCKED_JOINTS = {
    "right_wrist_roll_joint": 0.0,
    "right_wrist_pitch_joint": 0.0,
    "right_wrist_yaw_joint": 0.0,
}

DEFAULT_FPS = 50.0
DEFAULT_FRAMES_PER_SHAPE = 250
DEFAULT_TRANSITION_FRAMES = 25
DEFAULT_NUM_SHAPES = 8

# 大/小形状占所在八分区可用半径的比例（相对 scale_fit，保证视觉上有明显大小差）
SCALE_LARGE_FRAC = (0.78, 0.95)
SCALE_SMALL_FRAC = (0.22, 0.40)
WS_PROXIMITY = 0.038
IK_ERR_THRESH = WS_PROXIMITY
IK_MIN_OK_RATIO = 0.96

SHARP_SHAPES = frozenset({
    "star", "pentagon", "diamond", "heart", "random_polygon", "random_spline",
})


# ══════════════════════════════════════════════
# 4-DOF IK
# ══════════════════════════════════════════════

class IKSolver4DOF:
    def __init__(self, model, data):
        self.model = model
        self.data = data
        self.ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, EE_BODY)
        self.addr = _build_addr_table(model, ACTIVE_JOINTS)
        self.qpos_addrs = [self.addr[n][0] for n in ACTIVE_JOINTS]
        self.dof_addrs = [self.addr[n][1] for n in ACTIVE_JOINTS]
        self.ranges = [self.addr[n][2] for n in ACTIVE_JOINTS]
        self.lock_addr = _build_addr_table(model, list(LOCKED_JOINTS.keys()))
        self._jacp = np.zeros((3, model.nv))

    def _get_q(self):
        return np.array([self.data.qpos[a] for a in self.qpos_addrs])

    def _set_q(self, q):
        for i, (a, (lo, hi)) in enumerate(zip(self.qpos_addrs, self.ranges)):
            self.data.qpos[a] = np.clip(q[i], lo, hi)

    def seed_joints(self, q4: np.ndarray):
        self._set_q(np.asarray(q4, dtype=np.float64).reshape(4))
        self._lock_wrist()

    def _lock_wrist(self):
        for name, val in LOCKED_JOINTS.items():
            self.data.qpos[self.lock_addr[name][0]] = val
            self.data.qvel[self.lock_addr[name][1]] = 0.0

    def solve(self, target_pos, max_iter=200, tol=5e-4, lam_base=1e-3):
        for _ in range(max_iter):
            self._lock_wrist()
            mujoco.mj_forward(self.model, self.data)
            ee_pos = self.data.xpos[self.ee_id].copy()
            err = target_pos - ee_pos
            e_norm = float(np.linalg.norm(err))
            if e_norm < tol:
                return e_norm, True
            lam = lam_base + 5e-3 * e_norm
            mujoco.mj_jacBody(self.model, self.data, self._jacp, None, self.ee_id)
            J = self._jacp[:, self.dof_addrs]
            dq = J.T @ np.linalg.solve(J @ J.T + lam * np.eye(3), err)
            dq = np.clip(dq, -0.12, 0.12)
            self._set_q(self._get_q() + dq)
        self._lock_wrist()
        mujoco.mj_forward(self.model, self.data)
        final_err = float(np.linalg.norm(target_pos - self.data.xpos[self.ee_id]))
        return final_err, False


# ══════════════════════════════════════════════
# 分层规划
# ══════════════════════════════════════════════

def region_bounds(ws: ReachableWorkspace, left: bool, up: bool, front: bool):
    lo = ws.lo.copy()
    hi = ws.hi.copy()
    if front:
        lo[0] = ws.mid[0]
    else:
        hi[0] = ws.mid[0]
    if left:
        lo[1] = ws.mid[1]
    else:
        hi[1] = ws.mid[1]
    if up:
        lo[2] = ws.mid[2]
    else:
        hi[2] = ws.mid[2]
    return lo, hi


def _binary_flags(n: int, rng: np.random.Generator) -> np.ndarray:
    flags = np.zeros(n, dtype=bool)
    flags[: n // 2] = True
    rng.shuffle(flags)
    return flags


def plan_shapes_stratified(num_shapes: int, rng: np.random.Generator) -> list[dict]:
    left = _binary_flags(num_shapes, rng)
    up = _binary_flags(num_shapes, rng)
    front = _binary_flags(num_shapes, rng)
    large = _binary_flags(num_shapes, rng)
    shapes = list(rng.choice(SHAPE_NAMES, size=num_shapes))
    thetas = rng.uniform(0, 2 * np.pi, size=num_shapes)
    plans = []
    for i in range(num_shapes):
        plans.append({
            "left": bool(left[i]),
            "up": bool(up[i]),
            "front": bool(front[i]),
            "large": bool(large[i]),
            "shape": shapes[i],
            "theta": float(thetas[i]),
        })
    return plans


def _region_mask(ee_cloud, ws, left, up, front):
    lo_r, hi_r = region_bounds(ws, left, up, front)
    eps = 1e-6
    return (
        (ee_cloud[:, 0] >= lo_r[0] - eps) & (ee_cloud[:, 0] <= hi_r[0] + eps) &
        (ee_cloud[:, 1] >= lo_r[1] - eps) & (ee_cloud[:, 1] <= hi_r[1] + eps) &
        (ee_cloud[:, 2] >= lo_r[2] - eps) & (ee_cloud[:, 2] <= hi_r[2] + eps)
    )


def _region_margin(pts: np.ndarray, lo_r: np.ndarray, hi_r: np.ndarray) -> np.ndarray:
    """每个点到八分区边界的最近余量（越大越能画大形状）。"""
    return np.minimum.reduce([
        pts[:, 0] - lo_r[0], hi_r[0] - pts[:, 0],
        pts[:, 1] - lo_r[1], hi_r[1] - pts[:, 1],
        pts[:, 2] - lo_r[2], hi_r[2] - pts[:, 2],
    ])


def pick_center_in_region(ee_cloud, ws, plan, rng):
    mask = _region_mask(ee_cloud, ws, plan["left"], plan["up"], plan["front"])
    pts = ee_cloud[mask]
    lo_r, hi_r = region_bounds(ws, plan["left"], plan["up"], plan["front"])
    if len(pts) < 5:
        target = (lo_r + hi_r) * 0.5
        idx = int(np.argmin(np.linalg.norm(ee_cloud - target, axis=1)))
        return ee_cloud[idx].copy()
    if plan["large"]:
        margins = _region_margin(pts, lo_r, hi_r)
        thresh = float(np.percentile(margins, 65))
        good = pts[margins >= thresh]
        if len(good) < 3:
            good = pts
        return good[rng.integers(0, len(good))].copy()
    margins = _region_margin(pts, lo_r, hi_r)
    thresh = float(np.percentile(margins, 40))
    good = pts[margins <= thresh]
    if len(good) < 3:
        good = pts
    return good[rng.integers(0, len(good))].copy()


def _yz_extents_at_theta(local_yz, theta):
    c, s = np.cos(theta), np.sin(theta)
    yl, zl = local_yz[:, 0], local_yz[:, 1]
    dy = np.abs(yl * c - zl * s)
    dz = np.abs(yl * s + zl * c)
    return float(dy.max()), float(dz.max())


def compute_safe_scale(center, theta, shape_name, plan, ws, rng, margin=0.025):
    """
    在八分区内先算最大可用 scale_fit，再按大/小分层取不同比例。
    避免绝对米数上限被 scale_fit 压扁导致所有形状一样大。
    """
    lo, hi = region_bounds(ws, plan["left"], plan["up"], plan["front"])
    local = make_shape_local(shape_name, 48, rng)
    ext_y, ext_z = _yz_extents_at_theta(local, theta)
    ext_y = max(ext_y, 1e-6)
    ext_z = max(ext_z, 1e-6)
    cy, cz = float(center[1]), float(center[2])
    scale_y = min(cy - lo[1] - margin, hi[1] - cy - margin) / ext_y
    scale_z = min(cz - lo[2] - margin, hi[2] - cz - margin) / ext_z
    scale_fit = min(scale_y, scale_z)
    if scale_fit < 0.025:
        scale_fit = 0.025

    frac_lo, frac_hi = SCALE_LARGE_FRAC if plan["large"] else SCALE_SMALL_FRAC
    frac = float(rng.uniform(frac_lo, frac_hi))
    scale = scale_fit * frac
    return float(max(0.02, min(scale, scale_fit * 0.96))), scale_fit, frac


def _scale_floor_for_plan(plan: dict, scale_fit: float) -> float:
    """拟合可行域时允许缩小的下限，避免大形状被压成与小形状同尺寸。"""
    frac_lo = SCALE_LARGE_FRAC[0] if plan["large"] else SCALE_SMALL_FRAC[0]
    return float(max(0.02, scale_fit * frac_lo * 0.82))


def waypoints_near_workspace(waypoints, ee_cloud, max_dist=WS_PROXIMITY):
    diff = waypoints[:, None, :] - ee_cloud[None, :, :]
    dmin = np.linalg.norm(diff, axis=2).min(axis=1)
    return bool(dmin.max() <= max_dist), float(dmin.max())


def fit_shape_waypoints_to_workspace(shape, center, scale, theta,
                                     frames_per_shape, ee_cloud, rng,
                                     scale_floor: float = 0.02,
                                     is_large: bool = False,
                                     max_attempts: int = 10):
    scale = float(scale)
    initial = scale
    floor = float(scale_floor)
    # 大形状最多缩小 30%，小形状最多缩小 45%，避免全部挤成同一尺寸
    shrink_floor = initial * (0.70 if is_large else 0.55)
    floor = max(floor, shrink_floor)
    prox = WS_PROXIMITY * (1.12 if is_large else 1.0)
    last_dmax = 0.0
    for _ in range(max_attempts):
        shape_wps = make_shape_waypoints(
            shape, center, scale, theta, N=frames_per_shape, rng=rng
        )
        diff = shape_wps[:, None, :] - ee_cloud[None, :, :]
        dmin = np.linalg.norm(diff, axis=2).min(axis=1)
        last_dmax = float(dmin.max())
        if last_dmax <= prox:
            return shape_wps, scale
        if last_dmax > 1e-6:
            scale *= min(0.92, 0.97 * prox / last_dmax)
        else:
            scale *= 0.90
        scale = max(scale, floor)
    raise RuntimeError(
        f"路点超出可行域 (max_dist={last_dmax:.3f}m > {prox:.3f}m)"
    )


def waypoints_in_front_band(waypoints, x_min, x_max, margin=0.01):
    xs = waypoints[:, 0]
    return bool(xs.min() >= x_min - margin and xs.max() <= x_max + margin)


def blend_waypoints_to_workspace(waypoints, ee_cloud, alpha):
    diff = waypoints[:, None, :] - ee_cloud[None, :, :]
    idx = np.argmin(np.linalg.norm(diff, axis=2), axis=1)
    a = float(np.clip(alpha, 0.0, 1.0))
    return ((1.0 - a) * waypoints + a * ee_cloud[idx]).astype(np.float32)


# ══════════════════════════════════════════════
# IK 轨迹求解
# ══════════════════════════════════════════════

def assign_workspace_joint_path(waypoints, ee_cloud, joint_configs, k_neighbors=20):
    n_wp = len(waypoints)
    k_neighbors = min(k_neighbors, len(ee_cloud))
    idx = np.zeros(n_wp, dtype=int)
    idx[0] = int(np.argmin(np.linalg.norm(ee_cloud - waypoints[0], axis=1)))
    for i in range(1, n_wp):
        dist_i = np.linalg.norm(ee_cloud - waypoints[i], axis=1)
        cands = np.argpartition(dist_i, k_neighbors)[:k_neighbors]
        prev_q = joint_configs[idx[i - 1]]
        dj = np.linalg.norm(joint_configs[cands] - prev_q, axis=1)
        idx[i] = int(cands[np.argmin(dj)])
    return idx


def _smooth_joint_path_4d(q4, passes=2):
    q = np.asarray(q4, dtype=np.float64).copy()
    n = len(q)
    if n < 3:
        return q
    for _ in range(passes):
        q_new = q.copy()
        for i in range(n):
            im, ip = (i - 1) % n, (i + 1) % n
            q_new[i] = 0.22 * q[im] + 0.56 * q[i] + 0.22 * q[ip]
        q = q_new
    return q


def _ee_error(model, data, ee_id, target):
    return float(np.linalg.norm(target - data.xpos[ee_id]))


def _qpos_vel_from_sequence(qpos_arr, fps, nv):
    N = len(qpos_arr)
    dt = 1.0 / fps
    qvel_arr = np.zeros((N, nv), dtype=np.float32)
    for i in range(1, N - 1):
        qvel_arr[i, 6:] = (qpos_arr[i + 1, 7:] - qpos_arr[i - 1, 7:]) / (2 * dt)
    qvel_arr[0] = qvel_arr[1]
    qvel_arr[N - 1] = qvel_arr[N - 2]
    timestamps = np.arange(N, dtype=np.float32) * dt
    return qvel_arr, timestamps


def solve_waypoints_ik(
    waypoints, ee_cloud, joint_configs, fps,
    solver, model, data, arm_ids, obstacle_ids,
    max_iter=120, err_thresh=IK_ERR_THRESH,
    show_progress=False, ik_targets=None,
):
    if ik_targets is None:
        ik_targets = waypoints
    path_idx = assign_workspace_joint_path(waypoints, ee_cloud, joint_configs)
    q4_smooth = _smooth_joint_path_4d(joint_configs[path_idx], passes=3)
    ee_id = solver.ee_id
    N = len(waypoints)
    qpos_list, errors, col_flags = [], [], []
    polish_thresh = min(err_thresh, 0.012)

    for i in range(N):
        wp = waypoints[i]
        tgt = ik_targets[i]
        snap = (0.50 * wp + 0.50 * ee_cloud[path_idx[i]]).astype(np.float64)

        solver.seed_joints(q4_smooth[i])
        solver._lock_wrist()
        mujoco.mj_forward(model, data)
        err = _ee_error(model, data, ee_id, wp)

        if err > polish_thresh:
            err, _ = solver.solve(tgt, max_iter=max_iter)
            err = _ee_error(model, data, ee_id, wp)

        if err > err_thresh:
            solver.seed_joints(joint_configs[path_idx[i]])
            solver._lock_wrist()
            mujoco.mj_forward(model, data)
            err, _ = solver.solve(snap, max_iter=max_iter * 2)
            err = _ee_error(model, data, ee_id, wp)

        if err > err_thresh:
            solver.seed_joints(joint_configs[path_idx[i]])
            solver._lock_wrist()
            mujoco.mj_forward(model, data)
            err = _ee_error(model, data, ee_id, wp)

        solver._lock_wrist()
        mujoco.mj_forward(model, data)
        col = _has_penetration(model, data, arm_ids, obstacle_ids)
        qpos_list.append(data.qpos.copy())
        errors.append(err)
        col_flags.append(col)
        if show_progress and (i + 1) % 50 == 0:
            print(f"    IK {i+1}/{N}", end="\r", flush=True)
    if show_progress and N >= 50:
        print(f"    IK {N}/{N} 完成", flush=True)

    qpos_arr = np.array(qpos_list, dtype=np.float32)
    errors = np.array(errors)
    col_ratio = float(np.mean(col_flags))
    qvel_arr, timestamps = _qpos_vel_from_sequence(qpos_arr, fps, model.nv)
    return qpos_arr, qvel_arr, timestamps, errors, col_ratio


def check_feasibility(errors, col_ratio,
                      err_thresh=IK_ERR_THRESH, max_col_ratio=0.05,
                      min_ok_ratio=IK_MIN_OK_RATIO):
    ok_ratio = (errors < err_thresh).mean()
    return ok_ratio >= min_ok_ratio and col_ratio <= max_col_ratio


def make_joint_transition_qpos(q_from, q_to, n_frames, active_qpos_addrs):
    if n_frames <= 0:
        return np.zeros((0, len(q_from)), dtype=np.float32)
    out = []
    for t in np.linspace(0.0, 1.0, n_frames, endpoint=True):
        q = q_from.copy()
        for a in active_qpos_addrs:
            q[a] = (1.0 - t) * q_from[a] + t * q_to[a]
        out.append(q)
    return np.array(out, dtype=np.float32)


def make_transition_waypoints(p0, p1, n_frames):
    t = np.linspace(0, 1, n_frames, endpoint=True)
    pts = p0[None, :] * (1 - t[:, None]) + p1[None, :] * t[:, None]
    return pts.astype(np.float32)


def _plan_label(plan):
    lr = "左" if plan["left"] else "右"
    ud = "上" if plan["up"] else "下"
    fb = "前" if plan["front"] else "后"
    sz = "大" if plan["large"] else "小"
    return f"{lr}/{ud}/{fb}/{sz}"


# ══════════════════════════════════════════════
# 多形状轨迹生成
# ══════════════════════════════════════════════

def generate_multi_shape_trajectory(
    ee_cloud,
    joint_configs,
    num_shapes,
    fps,
    frames_per_shape,
    transition_frames,
    rng,
    front_x_min=DEFAULT_FRONT_X_MIN,
    front_x_max=DEFAULT_FRONT_X_MAX,
):
    model = mujoco.MjModel.from_xml_path(ROBOT_XML)
    data = mujoco.MjData(model)
    addr = _build_addr_table(model, ACTIVE_JOINTS + list(LOCKED_JOINTS.keys()))
    _reset_to_home(model, data, addr)
    solver = IKSolver4DOF(model, data)
    arm_ids, obstacle_ids = _build_collision_id_sets(
        model, _OBSTACLE_BODY_NAMES_MINIMAL, ee_body_names=_EE_COLLISION_BODY_NAMES
    )

    ee_cloud, joint_configs, ws = build_reachable_workspace(
        ee_cloud, joint_configs, front_x_min, front_x_max
    )
    active_qpos_addrs = [addr[n][0] for n in ACTIVE_JOINTS]
    print(f"  正前方可达域: {ws.n_samples} 点 | "
          f"盒 x[{ws.lo[0]:.2f},{ws.hi[0]:.2f}] "
          f"y[{ws.lo[1]:.2f},{ws.hi[1]:.2f}] z[{ws.lo[2]:.2f},{ws.hi[2]:.2f}]")

    plans = plan_shapes_stratified(num_shapes, rng)
    print("  形状规划（各轴约 N/2 分层）:")
    for i, pl in enumerate(plans):
        print(f"    [{i+1}] {pl['shape']:14s} {_plan_label(pl)}")

    all_qpos, all_qvel, all_ts = [], [], []
    shape_names, segment_starts = [], [0]
    all_waypoints_vis, shape_centers = [], []
    prev_last_wp = None
    qpos_seed = None
    t_accum = 0.0
    dt = 1.0 / fps

    for s_idx, plan in enumerate(plans):
        shape = plan["shape"]
        theta = plan["theta"]
        print(f"  形状 {s_idx+1}/{num_shapes}: {_plan_label(plan)} ...", flush=True)

        if qpos_seed is not None:
            data.qpos[:] = qpos_seed
            data.qvel[:] = 0.0
            mujoco.mj_forward(model, data)
        else:
            _reset_to_home(model, data, addr)

        center = pick_center_in_region(ee_cloud, ws, plan, rng)
        scale, scale_fit, scale_frac = compute_safe_scale(
            center, theta, shape, plan, ws, rng
        )
        scale_floor = _scale_floor_for_plan(plan, scale_fit)
        try:
            shape_wps, scale = fit_shape_waypoints_to_workspace(
                shape, center, scale, theta, frames_per_shape, ee_cloud, rng,
                scale_floor=scale_floor, is_large=plan["large"],
            )
        except RuntimeError as exc:
            raise RuntimeError(f"形状 {s_idx+1}: {exc}") from exc

        if not waypoints_in_front_band(shape_wps, front_x_min, front_x_max):
            raise RuntimeError(f"形状 {s_idx+1} 超出正前方 x 带")

        ik_targets = None
        if shape in SHARP_SHAPES:
            ik_targets = blend_waypoints_to_workspace(shape_wps, ee_cloud, 0.30)

        placed = False
        ok_ratio = 0.0
        col_r = 1.0
        for ik_try in range(6):
            extra = ""
            max_iter = 120
            if ik_try == 1:
                extra = ", 更多 IK 迭代"
                max_iter = 200
            elif ik_try == 2:
                extra = ", 换中心"
                center = pick_center_in_region(ee_cloud, ws, plan, rng)
                scale, scale_fit, scale_frac = compute_safe_scale(
                    center, theta, shape, plan, ws, rng
                )
                scale_floor = _scale_floor_for_plan(plan, scale_fit)
                try:
                    shape_wps, scale = fit_shape_waypoints_to_workspace(
                        shape, center, scale, theta, frames_per_shape, ee_cloud, rng,
                        scale_floor=scale_floor, is_large=plan["large"],
                    )
                except RuntimeError:
                    continue
                ik_targets = (
                    blend_waypoints_to_workspace(shape_wps, ee_cloud, 0.30)
                    if shape in SHARP_SHAPES else None
                )
            elif ik_try == 3:
                extra = ", snap α=0.40"
                ik_targets = blend_waypoints_to_workspace(shape_wps, ee_cloud, 0.40)
                max_iter = 180
            elif ik_try == 4:
                extra = ", snap α=0.55"
                ik_targets = blend_waypoints_to_workspace(shape_wps, ee_cloud, 0.55)
                max_iter = 200
            elif ik_try >= 5:
                scale = max(scale * 0.90, scale_floor)
                extra = f", scale={scale:.3f}"
                ik_targets = blend_waypoints_to_workspace(shape_wps, ee_cloud, 0.45)
                try:
                    shape_wps, scale = fit_shape_waypoints_to_workspace(
                        shape, center, scale, theta, frames_per_shape, ee_cloud, rng,
                        scale_floor=scale_floor, is_large=plan["large"],
                    )
                except RuntimeError:
                    continue

            print(f"  形状 {s_idx+1}/{num_shapes}: IK ({len(shape_wps)} 点{extra}) ...",
                  flush=True)
            qp_shape, qv_shape, ts_shape, errs, col_r = solve_waypoints_ik(
                shape_wps, ee_cloud, joint_configs, fps,
                solver, model, data, arm_ids, obstacle_ids,
                max_iter=max_iter, show_progress=(ik_try == 0),
                ik_targets=ik_targets,
            )
            ok_ratio = float(np.mean(errs < IK_ERR_THRESH))
            if check_feasibility(errs, col_r):
                placed = True
                break

        if not placed:
            raise RuntimeError(
                f"形状 {s_idx+1} IK 不可行 "
                f"(ok_ratio={ok_ratio:.2f}, col={col_r:.2f}, scale={scale:.3f})"
            )

        segment_wps = shape_wps
        qp_seg = qp_shape
        qv_seg = qv_shape
        ts_seg = ts_shape

        if qpos_seed is not None and transition_frames > 0:
            qp_trans = make_joint_transition_qpos(
                qpos_seed, qp_shape[0], transition_frames, active_qpos_addrs
            )
            qv_trans, ts_trans = _qpos_vel_from_sequence(qp_trans, fps, model.nv)
            qp_seg = np.vstack([qp_trans, qp_shape])
            qv_seg = np.vstack([qv_trans, qv_shape])
            ts_seg = np.concatenate([
                ts_trans,
                ts_shape + (ts_trans[-1] + 1.0 / fps),
            ])
            trans_vis = make_transition_waypoints(
                prev_last_wp, shape_wps[0], transition_frames
            )
            segment_wps = np.vstack([trans_vis, shape_wps])

        if len(all_ts) > 0:
            ts_seg = ts_seg + t_accum

        all_qpos.append(qp_seg)
        all_qvel.append(qv_seg)
        all_ts.append(ts_seg)
        all_waypoints_vis.append(segment_wps)
        shape_names.append(shape)
        shape_centers.append(center.copy())
        segment_starts.append(segment_starts[-1] + len(qp_seg))
        t_accum = all_ts[-1][-1] + dt
        prev_last_wp = shape_wps[-1].copy()
        qpos_seed = qp_seg[-1].copy()
        sz = "大" if plan["large"] else "小"
        print(f"  形状 {s_idx+1}/{num_shapes}: OK {shape} [{sz}] "
              f"center=({center[0]:.2f},{center[1]:.2f},{center[2]:.2f}) "
              f"scale={scale:.3f} (fit={scale_fit:.3f} frac={scale_frac:.2f}) "
              f"ok_ratio={ok_ratio:.2f}", flush=True)

    return {
        "qpos": np.concatenate(all_qpos, axis=0),
        "qvel": np.concatenate(all_qvel, axis=0),
        "timestamps": np.concatenate(all_ts, axis=0),
        "shape_names": np.array(shape_names, dtype=object),
        "segment_starts": np.array(segment_starts, dtype=np.int32),
        "waypoints": np.concatenate(all_waypoints_vis, axis=0),
        "shape_centers": np.array(shape_centers, dtype=np.float32),
    }


# ══════════════════════════════════════════════
# 保存 / 预览
# ══════════════════════════════════════════════

def save_trajectory_npz(traj, fps, out_path):
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    np.savez_compressed(
        out_path,
        qpos=traj["qpos"],
        qvel=traj["qvel"],
        timestamps=traj["timestamps"],
        record_dt=np.float32(1.0 / fps),
        shape_names=traj["shape_names"],
        segment_starts=traj["segment_starts"],
        shape_centers=traj["shape_centers"],
    )
    n = len(traj["qpos"])
    print(f"\n已保存 -> {out_path}")
    print(f"  帧数={n} fps={fps:.0f} 时长={n/fps:.1f}s 形状数={len(traj['shape_names'])}")


def _add_red_markers(scene, positions, size=0.007, rgba=None):
    if rgba is None:
        rgba = np.array([1.0, 0.15, 0.15, 0.75], dtype=np.float32)
    for pos in positions:
        if scene.ngeom >= scene.maxgeom:
            break
        g = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(
            g, mujoco.mjtGeom.mjGEOM_SPHERE,
            np.array([size, 0.0, 0.0]),
            np.zeros(3), np.eye(3).flatten(), rgba,
        )
        g.pos[:] = pos
        scene.ngeom += 1


def preview_trajectory(traj, fps, out_path=None, auto_save=False):
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)
    qpos_arr = traj["qpos"]
    qvel_arr = traj["qvel"]
    waypoints = traj["waypoints"]
    N = len(qpos_arr)
    frame_dt = 1.0 / fps
    ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, EE_BODY)

    vis_step = max(1, len(waypoints) // 400)
    vis_wps = waypoints[::vis_step]
    save_flag = [False]

    def key_cb(keycode):
        if keycode in (ord("p"), ord("P")):
            save_flag[0] = True

    print(f"\n[预览] {N} 帧 @ {fps:.0f} Hz | P=保存 | ESC=退出")
    trail_max = 800
    trail = []

    with mujoco.viewer.launch_passive(model, data, key_callback=key_cb) as viewer:
        loop = 0
        while viewer.is_running():
            loop += 1
            for i in range(N):
                if not viewer.is_running():
                    break
                t0 = time.perf_counter()
                data.qpos[:] = qpos_arr[i]
                data.qvel[:] = qvel_arr[i]
                mujoco.mj_forward(model, data)
                trail.append(data.xpos[ee_id].copy())
                if len(trail) > trail_max:
                    trail.pop(0)
                try:
                    scn = viewer.user_scn
                    scn.ngeom = 0
                    _add_red_markers(
                        scn, vis_wps, size=0.006,
                        rgba=np.array([1, 0.2, 0.2, 0.5], np.float32),
                    )
                    step = max(1, len(trail) // 200)
                    _add_red_markers(
                        scn, trail[::step], size=0.008,
                        rgba=np.array([1, 0.1, 0.1, 0.9], np.float32),
                    )
                except Exception:
                    pass
                viewer.sync()
                sleep_t = max(0.0, frame_dt - (time.perf_counter() - t0))
                if sleep_t > 0:
                    time.sleep(sleep_t)
            if (save_flag[0] or auto_save) and out_path:
                save_trajectory_npz(traj, fps, out_path)
                break
            if loop >= 2:
                if auto_save and out_path:
                    save_trajectory_npz(traj, fps, out_path)
                while viewer.is_running():
                    viewer.sync()
                    time.sleep(0.02)
                break


# ══════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(description="G1 右手多形状连续轨迹生成")
    p.add_argument("--preview", action="store_true", help="MuJoCo 预览 + 红色轨迹")
    p.add_argument("--batch", action="store_true", help="无头生成")
    p.add_argument("--num-shapes", type=int, default=DEFAULT_NUM_SHAPES)
    p.add_argument("--fps", type=float, default=DEFAULT_FPS)
    p.add_argument("--frames-per-shape", type=int, default=DEFAULT_FRAMES_PER_SHAPE)
    p.add_argument("--transition-frames", type=int, default=DEFAULT_TRANSITION_FRAMES)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--save", action="store_true", help="保存 NPZ")
    p.add_argument("--ws-cache", action="store_true", help="加载可行域缓存")
    p.add_argument("--front-x-min", type=float, default=DEFAULT_FRONT_X_MIN)
    p.add_argument("--front-x-max", type=float, default=DEFAULT_FRONT_X_MAX)
    args = p.parse_args()

    if not args.preview and not args.batch:
        print("请使用 --preview 和/或 --batch")
        print("  例: python g1_multi_shape_traj_gen.py --preview --save --ws-cache")
        return

    rng = np.random.default_rng(args.seed)

    if args.ws_cache:
        ee_cloud, joint_configs = load_workspace_cache(WS_CACHE_PATH)
        print(f"已加载可行域缓存: {len(ee_cloud)} 点")
    else:
        from g1_right_hand_workspace import build_workspace_cache
        print("无 --ws-cache，正在快速采样可行域...")
        ee_cloud, joint_configs = build_workspace_cache(
            n_per_joint=10, n_random=5000, show_plot=False,
            front_x_min=args.front_x_min, front_x_max=args.front_x_max,
        )

    est = args.num_shapes * (args.frames_per_shape + args.transition_frames)
    print(f"\n生成 {args.num_shapes} 个分层形状 @ {args.fps:.0f} Hz, "
          f"{args.frames_per_shape} 帧/形状 (~{est} IK 帧)...")

    traj = generate_multi_shape_trajectory(
        ee_cloud, joint_configs,
        num_shapes=args.num_shapes,
        fps=args.fps,
        frames_per_shape=args.frames_per_shape,
        transition_frames=args.transition_frames,
        rng=rng,
        front_x_min=args.front_x_min,
        front_x_max=args.front_x_max,
    )

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_npz = os.path.join(
        "recordings",
        f"g1_multi_{int(args.fps)}hz_S{args.num_shapes}_{ts}.npz",
    )

    if args.batch:
        if args.save:
            save_trajectory_npz(traj, args.fps, out_npz)
        else:
            print("批量完成（加 --save 写 NPZ）")
    elif args.preview:
        preview_trajectory(
            traj, args.fps,
            out_path=out_npz if args.save else None,
            auto_save=args.save,
        )

    if args.save:
        print(f"  回放: python g1_player.py --traj {out_npz} --fps {int(args.fps)}")


if __name__ == "__main__":
    main()
