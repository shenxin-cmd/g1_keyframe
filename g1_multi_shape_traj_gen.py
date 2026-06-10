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

from g1_arm_posture import (
    ELBOW_BRANCH_AUTO,
    ELBOW_BRANCH_INWARD,
    ELBOW_BRANCH_OUTWARD,
    NATURAL_Q4_REF,
    is_allowed_q4,
    pick_locked_branch_from_candidates,
    q4_branch_stable_with_anchor,
    resolve_branch_mode,
)
from g1_multi_shapes import SHAPE_NAMES, make_shape_local, make_shape_waypoints
from g1_right_hand_workspace import (
    CENTER_SAMPLE_INSET,
    TRAJ_BOUNDARY_MARGIN,
    DEFAULT_FRONT_X_MAX,
    DEFAULT_FRONT_X_MIN,
    ROBOT_XML,
    WS_CACHE_PATH,
    ReachableWorkspace,
    _RIGHT_ARM_COLLISION_BODY_NAMES,
    _OBSTACLE_BODY_NAMES_MINIMAL,
    _build_addr_table,
    _build_collision_id_sets,
    _has_penetration,
    _reset_to_home,
    build_reachable_workspace,
    envelope_margins_at,
    inset_bounds,
    load_workspace_cache,
    mask_points_in_box,
    sample_center_from_cloud,
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

# 大/小形状在「中心处最大可用 scale」上的采样比例区间
SCALE_LARGE_FRAC_OF_MAX = (0.62, 0.92)
SCALE_SMALL_FRAC_OF_MAX = (0.22, 0.48)
SCALE_SHRINK_STEP = 0.01
SCALE_MIN = 0.025
WS_PROXIMITY = 0.038
IK_ERR_THRESH = 0.045
IK_POLISH_TOL = 0.014
IK_MIN_OK_RATIO = 0.96
MAX_COL_RATIO = 0.0
# 抛光相对锚点的最大关节偏移（严格，防止肘部翻支）
IK_MAX_ANCHOR_DELTA = 0.10
IK_MAX_SEED_DELTA = 0.12
# 肩 roll 是解支/深度(X) 敏感轴：路径与抛光均强约束
IK_MAX_ROLL_STEP = 0.06
IK_MAX_ROLL_ANCHOR_SLACK = 0.05
IK_MAX_DEPTH_X_SLACK = 0.012
Q4_ROLL_IDX = 1
JOINT_PATH_SMOOTH_PASSES = 0

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

    def solve_constrained(
        self,
        target_pos: np.ndarray,
        q4_anchor: np.ndarray,
        q4_seed: np.ndarray | None = None,
        max_iter: int = 100,
        tol: float = IK_POLISH_TOL,
        max_delta: float = IK_MAX_ANCHOR_DELTA,
        lam_base: float = 1e-3,
        hold_depth_x: bool = False,
    ) -> tuple[float, np.ndarray]:
        """DLS 抛光：以锚点解支为基准，上一帧为初值，小步逼近目标。"""
        anchor = np.asarray(q4_anchor, dtype=np.float64).reshape(4)
        if q4_seed is not None:
            self.seed_joints(q4_seed)
        else:
            self.seed_joints(anchor)
        best_q = self._get_q().copy()
        best_err = float("inf")
        target_pos = np.asarray(target_pos, dtype=np.float64).reshape(3)
        for _ in range(max_iter):
            self._lock_wrist()
            mujoco.mj_forward(self.model, self.data)
            ee_pos = self.data.xpos[self.ee_id].copy()
            err_vec = target_pos - ee_pos
            if hold_depth_x:
                err_vec[0] = 0.0
            e_norm = float(np.linalg.norm(err_vec))
            q_now = self._get_q()
            if e_norm < best_err:
                best_err = e_norm
                best_q = q_now.copy()
            if e_norm < tol:
                return e_norm, q_now
            lam = lam_base + 5e-3 * e_norm
            mujoco.mj_jacBody(self.model, self.data, self._jacp, None, self.ee_id)
            J = self._jacp[:, self.dof_addrs]
            dq = J.T @ np.linalg.solve(J @ J.T + lam * np.eye(3), err_vec)
            dq = np.clip(dq, -0.05, 0.05)
            dq[Q4_ROLL_IDX] = np.clip(dq[Q4_ROLL_IDX], -0.035, 0.035)
            q_new = q_now + dq
            q_new = anchor + np.clip(q_new - anchor, -max_delta, max_delta)
            q_new[Q4_ROLL_IDX] = anchor[Q4_ROLL_IDX] + np.clip(
                q_new[Q4_ROLL_IDX] - anchor[Q4_ROLL_IDX],
                -IK_MAX_ROLL_STEP, IK_MAX_ROLL_STEP,
            )
            self._set_q(q_new)
        self.seed_joints(best_q)
        self._lock_wrist()
        mujoco.mj_forward(self.model, self.data)
        return best_err, best_q


# ══════════════════════════════════════════════
# 分层规划
# ══════════════════════════════════════════════

def _binary_flags(n: int, rng: np.random.Generator) -> np.ndarray:
    flags = np.zeros(n, dtype=bool)
    flags[: n // 2] = True
    rng.shuffle(flags)
    return flags


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


def _region_mask(ee_cloud, ws, left, up, front):
    lo_r, hi_r = region_bounds(ws, left, up, front)
    eps = 1e-6
    return (
        (ee_cloud[:, 0] >= lo_r[0] - eps) & (ee_cloud[:, 0] <= hi_r[0] + eps) &
        (ee_cloud[:, 1] >= lo_r[1] - eps) & (ee_cloud[:, 1] <= hi_r[1] + eps) &
        (ee_cloud[:, 2] >= lo_r[2] - eps) & (ee_cloud[:, 2] <= hi_r[2] + eps)
    )


def plan_shapes_stratified(num_shapes: int, rng: np.random.Generator,
                           ws: ReachableWorkspace) -> list[dict]:
    """左/右、上/下、前/后、大/小各约 N/2 分层。"""
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


def pick_center_in_region(ee_cloud, ws, plan, rng,
                          inset: float = CENTER_SAMPLE_INSET):
    lo_r, hi_r = region_bounds(ws, plan["left"], plan["up"], plan["front"])
    lo_r, hi_r = inset_bounds(lo_r, hi_r, inset)
    mask = mask_points_in_box(ee_cloud, lo_r, hi_r)
    pool = ee_cloud[mask]
    if len(pool) < 8:
        return sample_center_from_cloud(ee_cloud, ws.lo, ws.hi, rng, inset)
    target = (lo_r + hi_r) * 0.5
    target += rng.uniform(-0.03, 0.03, size=3)
    d = np.linalg.norm(pool - target, axis=1)
    k = min(48, len(pool))
    cands = np.argpartition(d, k - 1)[:k]
    best = int(cands[np.argmin(d[cands])])
    return pool[best].copy()


def _yz_extents_at_theta(local_yz, theta):
    c, s = np.cos(theta), np.sin(theta)
    yl, zl = local_yz[:, 0], local_yz[:, 1]
    dy = np.abs(yl * c - zl * s)
    dz = np.abs(yl * s + zl * c)
    return float(dy.max()), float(dz.max())


def compute_safe_scale(center, theta, shape_name, plan, ws, rng,
                       margin: float = TRAJ_BOUNDARY_MARGIN):
    """
    用 x 切片 yz 边界估算中心处最大 scale，再在大/小比例区间内随机采样。
    """
    local = make_shape_local(shape_name, 48, rng)
    ext_y, ext_z = _yz_extents_at_theta(local, theta)
    ext_y = max(ext_y, 1e-6)
    ext_z = max(ext_z, 1e-6)
    if ws.envelope is not None:
        my, mz = envelope_margins_at(ws.envelope, center, margin=margin)
    else:
        cy, cz = float(center[1]), float(center[2])
        my = min(cy - ws.lo[1], ws.hi[1] - cy) - margin
        mz = min(cz - ws.lo[2], ws.hi[2] - cz) - margin
    scale_max = min(my / ext_y, mz / ext_z)
    scale_max = float(max(SCALE_MIN, scale_max))
    frac_lo, frac_hi = (SCALE_LARGE_FRAC_OF_MAX if plan["large"]
                        else SCALE_SMALL_FRAC_OF_MAX)
    frac = float(rng.uniform(frac_lo, frac_hi))
    scale = float(max(SCALE_MIN, min(frac * scale_max, scale_max * 0.98)))
    return scale, scale_max, frac


def waypoints_in_reachable_box(waypoints, ws: ReachableWorkspace,
                               margin: float = TRAJ_BOUNDARY_MARGIN):
    lo, hi = inset_bounds(ws.lo, ws.hi, margin)
    return bool((waypoints >= lo).all() and (waypoints <= hi).all())


def fit_shape_waypoints_to_workspace(shape, center, scale, theta,
                                     frames_per_shape, ee_cloud, rng,
                                     ws: ReachableWorkspace,
                                     scale_max: float | None = None,
                                     min_scale: float = SCALE_MIN):
    """边界采样为主；若仍略超可行域则每次减 1cm。"""
    scale = float(scale)
    floor = float(min_scale)
    if scale_max is not None:
        floor = max(floor, scale_max * SCALE_SMALL_FRAC_OF_MAX[0] * 0.85)
    prox = WS_PROXIMITY * 1.06
    last_dmax = 0.0
    while scale >= floor - 1e-9:
        shape_wps = make_shape_waypoints(
            shape, center, scale, theta, N=frames_per_shape, rng=rng
        )
        diff = shape_wps[:, None, :] - ee_cloud[None, :, :]
        dmin = np.linalg.norm(diff, axis=2).min(axis=1)
        last_dmax = float(dmin.max())
        if last_dmax <= prox and waypoints_in_reachable_box(shape_wps, ws):
            return shape_wps, scale
        scale -= SCALE_SHRINK_STEP
    raise RuntimeError(
        f"路点超出可行域 (max_dist={last_dmax:.3f}m > {prox:.3f}m, "
        f"scale<{floor:.3f}m)"
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

def _q4_has_arm_collision(solver, model, data, q4, arm_ids, obstacle_ids):
    solver.seed_joints(np.asarray(q4, dtype=np.float64))
    solver._lock_wrist()
    mujoco.mj_forward(model, data)
    return _has_penetration(model, data, arm_ids, obstacle_ids)


def assign_workspace_joint_path(
    waypoints, ee_cloud, joint_configs, k_neighbors=20,
    solver=None, model=None, data=None, arm_ids=None, obstacle_ids=None,
    prox=WS_PROXIMITY, elbow_branch_mode=ELBOW_BRANCH_AUTO,
    locked_branch: str | None = None,
):
    n_wp = len(waypoints)
    k_neighbors = min(k_neighbors, len(ee_cloud))
    idx = np.zeros(n_wp, dtype=int)
    use_col = all(x is not None for x in (
        solver, model, data, arm_ids, obstacle_ids,
    ))
    addr = None
    if model is not None:
        addr = _build_addr_table(model, ACTIVE_JOINTS + list(LOCKED_JOINTS.keys()))

    def _cand_order_for(wp, prev_idx=None):
        dist_i = np.linalg.norm(ee_cloud - wp, axis=1)
        near = np.where(dist_i <= prox * 1.15)[0]
        if len(near) < 3:
            order = np.argsort(dist_i)
            near = order[:max(k_neighbors, 48)]
        if prev_idx is None:
            return near[np.argsort(dist_i[near])]
        prev_q = joint_configs[prev_idx]
        dj = np.linalg.norm(joint_configs[near] - prev_q, axis=1)
        d_roll = np.abs(joint_configs[near, Q4_ROLL_IDX] - prev_q[Q4_ROLL_IDX])
        d_x = np.abs(ee_cloud[near, 0] - wp[0])
        # 深度(X) 一致 > roll 连续 > 关节距离 > 末端距离
        return near[np.lexsort((dist_i[near], dj, d_roll, d_x))]

    def _pick_for_waypoint(wp, prev_idx=None, branch=None, roll_hint: float | None = None):
        dist_i = np.linalg.norm(ee_cloud - wp, axis=1)
        cand_order = _cand_order_for(wp, prev_idx)

        def _pick_best(candidates, max_dx: float | None = None):
            valid = []
            for c in candidates:
                q4 = joint_configs[c]
                if not is_allowed_q4(q4, branch, model, data, addr):
                    continue
                if max_dx is not None and abs(float(ee_cloud[c, 0] - wp[0])) > max_dx:
                    continue
                if use_col and _q4_has_arm_collision(
                    solver, model, data, q4, arm_ids, obstacle_ids,
                ):
                    continue
                valid.append(int(c))
            if not valid:
                return None
            if len(valid) == 1:
                return valid[0]
            prev_q = (
                joint_configs[prev_idx] if prev_idx is not None else None
            )
            scored = []
            for c in valid[:24]:
                d_j = (
                    float(np.linalg.norm(joint_configs[c] - prev_q))
                    if prev_q is not None else 0.0
                )
                d_roll = 0.0
                if prev_q is not None:
                    d_roll = abs(float(joint_configs[c, Q4_ROLL_IDX] - prev_q[Q4_ROLL_IDX]))
                elif roll_hint is not None:
                    d_roll = abs(float(joint_configs[c, Q4_ROLL_IDX] - roll_hint))
                d_nat = abs(float(joint_configs[c, Q4_ROLL_IDX] - NATURAL_Q4_REF[1]))
                roll_pen = 1 if d_roll > IK_MAX_ROLL_STEP else 0
                d_x = abs(float(ee_cloud[c, 0] - wp[0]))
                scored.append((roll_pen, d_roll, d_x, d_j, d_nat, float(dist_i[c]), c))
            scored.sort(key=lambda x: (x[0], x[1], x[2], x[3], x[4], x[5]))
            return scored[0][6]

        dx_limits = [IK_MAX_DEPTH_X_SLACK, IK_MAX_DEPTH_X_SLACK * 2.5, None]
        for dx_lim in dx_limits:
            picked = _pick_best(cand_order, max_dx=dx_lim)
            if picked is not None:
                return picked
            picked = _pick_best(np.argsort(dist_i), max_dx=dx_lim)
            if picked is not None:
                return picked
        raise RuntimeError("无满足姿态/碰撞约束的关节配置")

    if locked_branch is None:
        c0 = _cand_order_for(waypoints[0])
        locked_branch = pick_locked_branch_from_candidates(
            joint_configs, c0, elbow_branch_mode, model, data, addr,
        )
        if locked_branch is None:
            raise RuntimeError("首点无法确定肘部折向（可能均为向上翻折）")
    idx[0] = _pick_for_waypoint(waypoints[0], branch=locked_branch)
    roll_hint = float(joint_configs[idx[0], Q4_ROLL_IDX])
    for i in range(1, n_wp):
        idx[i] = _pick_for_waypoint(
            waypoints[i], idx[i - 1], locked_branch, roll_hint=roll_hint,
        )
    return idx, locked_branch


def _smooth_joint_path_4d(q4, passes=2, closed: bool = True):
    q = np.asarray(q4, dtype=np.float64).copy()
    n = len(q)
    if n < 3:
        return q
    for _ in range(passes):
        q_new = q.copy()
        for i in range(n):
            if closed:
                im, ip = (i - 1) % n, (i + 1) % n
            else:
                im, ip = max(0, i - 1), min(n - 1, i + 1)
            q_new[i] = 0.22 * q[im] + 0.56 * q[i] + 0.22 * q[ip]
            q_new[i, Q4_ROLL_IDX] = q[i, Q4_ROLL_IDX]
        q = q_new
    return q


def _qpos_from_q4_seed(solver, model, data, q4: np.ndarray) -> np.ndarray:
    solver.seed_joints(np.asarray(q4, dtype=np.float64))
    solver._lock_wrist()
    mujoco.mj_forward(model, data)
    return data.qpos.copy()


def _clamp_q4_to_anchor(q4, anchor, max_delta: float) -> np.ndarray:
    q4 = np.asarray(q4, dtype=np.float64).reshape(4)
    anchor = np.asarray(anchor, dtype=np.float64).reshape(4)
    return anchor + np.clip(q4 - anchor, -max_delta, max_delta)


def _ee_error(model, data, ee_id, target):
    return float(np.linalg.norm(target - data.xpos[ee_id]))


def _q4_from_qpos(data, addr) -> np.ndarray:
    return np.array(
        [float(data.qpos[addr[n][0]]) for n in ACTIVE_JOINTS],
        dtype=np.float64,
    )


def _enforce_posture_qpos(qpos, q4_fallback, locked_branch,
                          solver, model, data, addr, ee_id, wp):
    """IK 抛光后若折向/伸直约束被破坏，回退到可行域关节角。"""
    data.qpos[:] = qpos
    mujoco.mj_forward(model, data)
    if locked_branch is None:
        return qpos, _ee_error(model, data, ee_id, wp)
    q4 = _q4_from_qpos(data, addr)
    if is_allowed_q4(q4, locked_branch, model, data, addr):
        return qpos, _ee_error(model, data, ee_id, wp)
    solver.seed_joints(np.asarray(q4_fallback, dtype=np.float64))
    solver._lock_wrist()
    mujoco.mj_forward(model, data)
    return data.qpos.copy(), _ee_error(model, data, ee_id, wp)


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


def _solve_frame_with_collision_fallback(
    wp, tgt, snap, q4_ws_i, q4_smooth_i,
    solver, model, data, ee_id, arm_ids, obstacle_ids,
    max_iter, err_thresh, polish_thresh, locked_branch=None,
):
    """优先用可行域关节角（已验证无穿透），仅在 IK 后仍无碰撞时才采纳。"""
    addr = _build_addr_table(model, ACTIVE_JOINTS + list(LOCKED_JOINTS.keys()))
    if not is_allowed_q4(q4_ws_i, locked_branch, model, data, addr):
        raise RuntimeError("路点关节解违反肘部姿态约束")
    seeds = [q4_ws_i]
    if (not _q4_has_arm_collision(
        solver, model, data, q4_smooth_i, arm_ids, obstacle_ids,
    ) and is_allowed_q4(q4_smooth_i, locked_branch, model, data, addr)):
        seeds.insert(0, q4_smooth_i)
    best_qpos = None
    best_err = float("inf")
    best_col = True
    for si, q4_seed in enumerate(seeds):
        solver.seed_joints(q4_seed)
        solver._lock_wrist()
        mujoco.mj_forward(model, data)
        err = _ee_error(model, data, ee_id, wp)
        col = _has_penetration(model, data, arm_ids, obstacle_ids)
        if not col and err <= err_thresh:
            qpos, err = _enforce_posture_qpos(
                data.qpos.copy(), q4_ws_i, locked_branch,
                solver, model, data, addr, ee_id, wp,
            )
            return qpos, err, False
        if err > polish_thresh * 1.8 and not col:
            solve_tgt = snap
            iters = max(30, max_iter // 3)
            err_try, _ = solver.solve(solve_tgt, max_iter=iters)
            err_try = _ee_error(model, data, ee_id, wp)
            solver._lock_wrist()
            mujoco.mj_forward(model, data)
            if not _has_penetration(model, data, arm_ids, obstacle_ids):
                err = err_try
            else:
                solver.seed_joints(q4_seed)
                solver._lock_wrist()
                mujoco.mj_forward(model, data)
                err = _ee_error(model, data, ee_id, wp)
        solver._lock_wrist()
        mujoco.mj_forward(model, data)
        col = _has_penetration(model, data, arm_ids, obstacle_ids)
        if not col and (best_col or err < best_err):
            best_qpos = data.qpos.copy()
            best_err = err
            best_col = False
        if not col and err <= err_thresh:
            qpos, err = _enforce_posture_qpos(
                data.qpos.copy(), q4_ws_i, locked_branch,
                solver, model, data, addr, ee_id, wp,
            )
            return qpos, err, False
    if best_qpos is not None:
        qpos, err = _enforce_posture_qpos(
            best_qpos, q4_ws_i, locked_branch,
            solver, model, data, addr, ee_id, wp,
        )
        return qpos, err, best_col
    qpos, err = _enforce_posture_qpos(
        data.qpos.copy(), q4_ws_i, locked_branch,
        solver, model, data, addr, ee_id, wp,
    )
    return qpos, err, col


def solve_waypoints_ik(
    waypoints, ee_cloud, joint_configs, fps,
    solver, model, data, arm_ids, obstacle_ids,
    max_iter=120, err_thresh=IK_ERR_THRESH,
    show_progress=False, ik_targets=None,
    elbow_branch_mode=ELBOW_BRANCH_AUTO,
    locked_branch: str | None = None,
    closed_path: bool = True,
    smooth_passes: int = JOINT_PATH_SMOOTH_PASSES,
):
    """
    可行域选解（roll 连续优先）→ 轻量关节平滑 → 锚点约束 DLS 抛光。
    抛光时限制肩 roll 步长，避免同支解内 X 锯齿。
    """
    targets = waypoints if ik_targets is None else ik_targets
    path_idx, locked_branch = assign_workspace_joint_path(
        waypoints, ee_cloud, joint_configs,
        solver=solver, model=model, data=data,
        arm_ids=arm_ids, obstacle_ids=obstacle_ids,
        elbow_branch_mode=elbow_branch_mode,
        locked_branch=locked_branch,
    )
    q4_anchors = joint_configs[path_idx]
    q4_path = _smooth_joint_path_4d(
        q4_anchors, passes=smooth_passes, closed=closed_path,
    )
    hold_depth_x = float(np.ptp(waypoints[:, 0])) < 0.006
    ee_id = solver.ee_id
    addr = _build_addr_table(model, ACTIVE_JOINTS + list(LOCKED_JOINTS.keys()))
    N = len(waypoints)
    qpos_list, errors, col_flags = [], [], []
    q4_prev = None

    for i in range(N):
        q4_anchor = q4_anchors[i]
        q4_out = q4_anchor
        wp = waypoints[i]
        tgt = targets[i]

        if not hold_depth_x:
            q4_base = q4_path[i]
            if q4_prev is not None:
                q4_seed = _clamp_q4_to_anchor(q4_prev, q4_anchor, IK_MAX_SEED_DELTA)
            else:
                q4_seed = q4_base
            polish_anchor = (
                q4_prev.copy() if q4_prev is not None else q4_anchor.copy()
            )
            solver.seed_joints(q4_base)
            solver._lock_wrist()
            mujoco.mj_forward(model, data)
            err_base = _ee_error(model, data, ee_id, wp)
            q4_out = q4_base
            if (
                err_base > IK_POLISH_TOL
                and q4_branch_stable_with_anchor(
                    q4_base, polish_anchor, locked_branch, model, data, addr,
                )
            ):
                _, q4_pol = solver.solve_constrained(
                    tgt, polish_anchor, q4_seed=q4_seed,
                    max_iter=min(50, max_iter),
                    tol=IK_POLISH_TOL,
                    max_delta=IK_MAX_ANCHOR_DELTA,
                )
                solver.seed_joints(q4_pol)
                solver._lock_wrist()
                mujoco.mj_forward(model, data)
                err_pol = _ee_error(model, data, ee_id, wp)
                col_pol = _has_penetration(model, data, arm_ids, obstacle_ids)
                roll_jump = (
                    q4_prev is not None
                    and abs(q4_pol[Q4_ROLL_IDX] - q4_prev[Q4_ROLL_IDX]) > IK_MAX_ROLL_STEP
                )
                if (
                    not col_pol
                    and not roll_jump
                    and err_pol < err_base
                    and q4_branch_stable_with_anchor(
                        q4_pol, polish_anchor, locked_branch, model, data, addr,
                    )
                ):
                    q4_out = q4_pol
            if not q4_branch_stable_with_anchor(
                q4_out, polish_anchor, locked_branch, model, data, addr,
            ):
                q4_out = q4_anchor
            if (
                q4_prev is not None
                and abs(q4_out[Q4_ROLL_IDX] - q4_prev[Q4_ROLL_IDX]) > IK_MAX_ROLL_STEP
            ):
                q4_out = q4_anchor

        qpos_i = _qpos_from_q4_seed(solver, model, data, q4_out)
        err = _ee_error(model, data, ee_id, wp)
        col = _has_penetration(model, data, arm_ids, obstacle_ids)
        qpos_i, err = _enforce_posture_qpos(
            qpos_i, q4_anchor, locked_branch,
            solver, model, data, addr, ee_id, wp,
        )
        col = _has_penetration(model, data, arm_ids, obstacle_ids) or col

        qpos_list.append(qpos_i)
        errors.append(err)
        col_flags.append(col)
        q4_prev = _q4_from_qpos(data, addr)
        if show_progress and (i + 1) % 50 == 0:
            print(f"    IK {i+1}/{N}", end="\r", flush=True)
    if show_progress and N >= 50:
        print(f"    IK {N}/{N} 完成", flush=True)

    qpos_arr = np.array(qpos_list, dtype=np.float32)
    errors = np.array(errors)
    col_ratio = float(np.mean(col_flags))
    qvel_arr, timestamps = _qpos_vel_from_sequence(qpos_arr, fps, model.nv)
    return qpos_arr, qvel_arr, timestamps, errors, col_ratio, locked_branch


def check_feasibility(errors, col_ratio,
                      err_thresh=IK_ERR_THRESH, max_col_ratio=MAX_COL_RATIO,
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


def solve_transition_segment(
    p_from: np.ndarray,
    p_to: np.ndarray,
    n_frames: int,
    ee_cloud: np.ndarray,
    joint_configs: np.ndarray,
    fps: float,
    solver,
    model,
    data,
    arm_ids,
    obstacle_ids,
    locked_branch: str | None,
    elbow_branch_mode: str = ELBOW_BRANCH_AUTO,
):
    """层间/形状间过渡：对末端路点做 IK，避免关节线性插值导致肘部翻折。"""
    if n_frames <= 0:
        return (
            np.zeros((0, model.nq), dtype=np.float32),
            np.zeros((0, model.nv), dtype=np.float32),
            np.zeros(0, dtype=np.float32),
        )
    wps = make_transition_waypoints(p_from, p_to, n_frames)
    qp, qv, ts, _, _, _ = solve_waypoints_ik(
        wps, ee_cloud, joint_configs, fps,
        solver, model, data, arm_ids, obstacle_ids,
        show_progress=False,
        elbow_branch_mode=elbow_branch_mode,
        locked_branch=locked_branch,
        closed_path=False,
        smooth_passes=2,
    )
    return qp, qv, ts


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
    elbow_branch_mode=ELBOW_BRANCH_AUTO,
):
    model = mujoco.MjModel.from_xml_path(ROBOT_XML)
    data = mujoco.MjData(model)
    addr = _build_addr_table(model, ACTIVE_JOINTS + list(LOCKED_JOINTS.keys()))
    _reset_to_home(model, data, addr)
    solver = IKSolver4DOF(model, data)
    arm_ids, obstacle_ids = _build_collision_id_sets(
        model, _OBSTACLE_BODY_NAMES_MINIMAL,
        ee_body_names=_RIGHT_ARM_COLLISION_BODY_NAMES,
    )

    ee_cloud, joint_configs, ws = build_reachable_workspace(
        ee_cloud, joint_configs, front_x_min, front_x_max
    )
    active_qpos_addrs = [addr[n][0] for n in ACTIVE_JOINTS]
    print(f"  正前方可达域: {ws.n_samples} 点 | "
          f"盒 x[{ws.lo[0]:.2f},{ws.hi[0]:.2f}] "
          f"y[{ws.lo[1]:.2f},{ws.hi[1]:.2f}] z[{ws.lo[2]:.2f},{ws.hi[2]:.2f}]")

    plans = plan_shapes_stratified(num_shapes, rng, ws)
    print("  形状规划（左/右/上/下/前/后/大/小约 N/2，尺度由边界余量采样）:")
    for i, pl in enumerate(plans):
        print(f"    [{i+1}] {pl['shape']:14s} {_plan_label(pl)}")

    all_qpos, all_qvel, all_ts = [], [], []
    shape_names, segment_starts = [], [0]
    all_waypoints_vis, shape_centers = [], []
    prev_last_wp = None
    qpos_seed = None
    t_accum = 0.0
    dt = 1.0 / fps
    traj_locked_branch = None

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
        scale, scale_max, scale_frac = compute_safe_scale(
            center, theta, shape, plan, ws, rng
        )
        try:
            shape_wps, scale = fit_shape_waypoints_to_workspace(
                shape, center, scale, theta, frames_per_shape, ee_cloud, rng,
                ws=ws, scale_max=scale_max,
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
                scale, scale_max, scale_frac = compute_safe_scale(
                    center, theta, shape, plan, ws, rng
                )
                try:
                    shape_wps, scale = fit_shape_waypoints_to_workspace(
                        shape, center, scale, theta, frames_per_shape, ee_cloud, rng,
                        ws=ws, scale_max=scale_max,
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
                scale = max(scale - SCALE_SHRINK_STEP, SCALE_MIN)
                extra = f", scale={scale:.3f}"
                ik_targets = blend_waypoints_to_workspace(shape_wps, ee_cloud, 0.45)
                try:
                    shape_wps, scale = fit_shape_waypoints_to_workspace(
                        shape, center, scale, theta, frames_per_shape, ee_cloud, rng,
                        ws=ws, scale_max=scale_max, min_scale=scale,
                    )
                except RuntimeError:
                    continue

            print(f"  形状 {s_idx+1}/{num_shapes}: IK ({len(shape_wps)} 点{extra}) ...",
                  flush=True)
            qp_shape, qv_shape, ts_shape, errs, col_r, traj_locked_branch = (
                solve_waypoints_ik(
                    shape_wps, ee_cloud, joint_configs, fps,
                    solver, model, data, arm_ids, obstacle_ids,
                    max_iter=max_iter, show_progress=(ik_try == 0),
                    ik_targets=ik_targets,
                    elbow_branch_mode=elbow_branch_mode,
                    locked_branch=traj_locked_branch,
                )
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
            qp_trans, qv_trans, ts_trans = solve_transition_segment(
                prev_last_wp, shape_wps[0], transition_frames,
                ee_cloud, joint_configs, fps,
                solver, model, data, arm_ids, obstacle_ids,
                locked_branch=traj_locked_branch,
                elbow_branch_mode=elbow_branch_mode,
            )
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
              f"scale={scale:.3f} (max={scale_max:.3f} frac={scale_frac:.2f}) "
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
    payload = dict(
        qpos=traj["qpos"],
        qvel=traj["qvel"],
        timestamps=traj["timestamps"],
        record_dt=np.float32(1.0 / fps),
        shape_names=traj["shape_names"],
        segment_starts=traj["segment_starts"],
        shape_centers=traj["shape_centers"],
    )
    if "waypoints" in traj:
        payload["waypoints"] = traj["waypoints"]
    if "shape_name" in traj:
        payload["shape_name"] = traj["shape_name"]
    np.savez_compressed(out_path, **payload)
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
                        rgba=np.array([0.95, 0.85, 0.1, 0.55], np.float32),
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
    p.add_argument("--elbow-branch", type=str, default=ELBOW_BRANCH_AUTO,
                   help="肘部折向: outward(外翻)/inward(内翻)/auto(默认,参考图四自然姿态)")
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

    elbow_mode = resolve_branch_mode(args.elbow_branch)
    print(f"肘部折向模式: {elbow_mode}")

    traj = generate_multi_shape_trajectory(
        ee_cloud, joint_configs,
        num_shapes=args.num_shapes,
        fps=args.fps,
        frames_per_shape=args.frames_per_shape,
        transition_frames=args.transition_frames,
        rng=rng,
        front_x_min=args.front_x_min,
        front_x_max=args.front_x_max,
        elbow_branch_mode=elbow_mode,
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
