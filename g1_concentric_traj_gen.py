"""
G1 右手同心递减轨迹生成器
========================
从可达空间采样中心点，测量该处最大可画尺度，再生成同形状、同心、逐渐减小的多层闭合轨迹。

用法：
  # 单条：圆，6 层同心递减
  python g1_concentric_traj_gen.py --shape circle --layers 6 --preview --ws-cache --save

  # 批量：五角星，10 个不同中心，各 5 层
  python g1_concentric_traj_gen.py --shape star --layers 5 --num-trajs 10 --batch --ws-cache --save

  # 中文形状名
  python g1_concentric_traj_gen.py --shape 椭圆 --layers 4 --preview --ws-cache

支持形状：circle, ellipse, triangle, triangle_down, square, rectangle,
         star, pentagon, random_polygon, random_spline
         （及中文别名：圆形/椭圆/三角形/倒三角/正方形/长方形/五角星/五边形/随机图形）
"""

from __future__ import annotations

import argparse
import datetime
import os
import time

import mujoco
import numpy as np

from g1_multi_shapes import CONCENTRIC_SHAPE_NAMES, make_shape_local, make_shape_waypoints
from g1_arm_posture import ELBOW_BRANCH_AUTO, resolve_branch_mode
from g1_multi_shape_traj_gen import (
    ACTIVE_JOINTS,
    IK_ERR_THRESH,
    LOCKED_JOINTS,
    SCALE_MIN,
    SCALE_SHRINK_STEP,
    WS_PROXIMITY,
    _qpos_vel_from_sequence,
    check_feasibility,
    make_transition_waypoints,
    solve_transition_segment,
    preview_trajectory,
    save_trajectory_npz,
    solve_waypoints_ik,
    waypoints_in_front_band,
)
from g1_multi_shape_traj_gen import IKSolver4DOF as _IKSolver4DOF
from g1_right_hand_workspace import (
    CENTER_SAMPLE_INSET,
    TRAJ_BOUNDARY_MARGIN,
    DEFAULT_FRONT_X_MAX,
    DEFAULT_FRONT_X_MIN,
    ROBOT_XML,
    WS_CACHE_PATH,
    ReachableWorkspace,
    _OBSTACLE_BODY_NAMES_MINIMAL,
    _RIGHT_ARM_COLLISION_BODY_NAMES,
    _build_addr_table,
    _build_collision_id_sets,
    _reset_to_home,
    build_reachable_workspace,
    envelope_margins_at,
    inset_bounds,
    load_workspace_cache,
    sample_center_from_cloud,
)

os.makedirs("recordings", exist_ok=True)

DEFAULT_FPS = 30.0
DEFAULT_FRAMES_PER_RING = 200
DEFAULT_TRANSITION_FRAMES = 15
DEFAULT_LAYERS = 5
DEFAULT_MIN_FRAC = 0.18
# 向内递减下限：当前圈画完后，下一圈边长 < 5cm 则不再向内生成
MIN_INNER_SCALE = 0.05

SHAPE_ALIASES: dict[str, str] = {
    "circle": "circle", "圆形": "circle", "圆": "circle",
    "ellipse": "ellipse", "椭圆": "ellipse",
    "triangle": "triangle", "三角形": "triangle", "三角": "triangle",
    "triangle_down": "triangle_down", "inverted_triangle": "triangle_down",
    "倒三角": "triangle_down", "倒三角形": "triangle_down",
    "square": "square", "正方形": "square", "方形": "square",
    "rectangle": "rectangle", "长方形": "rectangle", "矩形": "rectangle",
    "star": "star", "五角星": "star", "星形": "star",
    "pentagon": "pentagon", "五边形": "pentagon",
    "heart": "heart", "爱心": "heart",
    "random_polygon": "random_polygon", "随机": "random_polygon",
    "随机图形": "random_polygon", "random_spline": "random_spline",
}


def resolve_shape_name(name: str) -> str:
    key = name.strip().lower()
    if key in SHAPE_ALIASES:
        return SHAPE_ALIASES[key]
    if key in CONCENTRIC_SHAPE_NAMES:
        return key
    opts = ", ".join(CONCENTRIC_SHAPE_NAMES)
    raise ValueError(f"未知形状: {name}，可选: {opts}")


def _yz_extents_at_theta(local_yz, theta: float):
    c, s = np.cos(theta), np.sin(theta)
    yl, zl = local_yz[:, 0], local_yz[:, 1]
    return float(np.abs(yl * c - zl * s).max()), float(np.abs(yl * s + zl * c).max())


def sample_center(
    ee_cloud: np.ndarray,
    ws: ReachableWorkspace,
    rng: np.random.Generator,
    inset: float = CENTER_SAMPLE_INSET,
) -> np.ndarray:
    return sample_center_from_cloud(ee_cloud, ws.lo, ws.hi, rng, inset)


def _waypoints_fit_workspace(shape_wps, ee_cloud, ws: ReachableWorkspace,
                             prox: float = WS_PROXIMITY * 1.06,
                             boundary_margin: float = TRAJ_BOUNDARY_MARGIN) -> bool:
    diff = shape_wps[:, None, :] - ee_cloud[None, :, :]
    dmax = float(np.linalg.norm(diff, axis=2).min(axis=1).max())
    if dmax > prox:
        return False
    lo, hi = inset_bounds(ws.lo, ws.hi, boundary_margin)
    return bool((shape_wps >= lo).all() and (shape_wps <= hi).all())


def find_max_scale(
    shape: str,
    center: np.ndarray,
    theta: float,
    ws: ReachableWorkspace,
    ee_cloud: np.ndarray,
    rng: np.random.Generator,
    probe_frames: int = 64,
    margin: float = TRAJ_BOUNDARY_MARGIN,
) -> tuple[float, float]:
    """
    先由 x 切片 yz 边界估算上界，再以 1cm 步长向下搜索最大可画 scale。
    返回 (scale_envelope, scale_fit)。
    """
    local = make_shape_local(shape, probe_frames, rng)
    ext_y, ext_z = _yz_extents_at_theta(local, theta)
    ext_y = max(ext_y, 1e-6)
    ext_z = max(ext_z, 1e-6)
    if ws.envelope is not None:
        my, mz = envelope_margins_at(ws.envelope, center, margin=margin)
    else:
        cy, cz = float(center[1]), float(center[2])
        my = min(cy - ws.lo[1], ws.hi[1] - cy) - margin
        mz = min(cz - ws.lo[2], ws.hi[2] - cz) - margin
    scale_env = float(max(SCALE_MIN, min(my / ext_y, mz / ext_z) * 0.98))
    scale = scale_env
    scale_fit = SCALE_MIN
    while scale >= SCALE_MIN - 1e-9:
        wps = make_shape_waypoints(
            shape, center, scale, theta, N=probe_frames, rng=rng,
        )
        if _waypoints_fit_workspace(wps, ee_cloud, ws):
            scale_fit = float(scale)
            break
        scale -= SCALE_SHRINK_STEP
    if scale_fit < SCALE_MIN:
        raise RuntimeError(
            f"中心 ({center[0]:.2f},{center[1]:.2f},{center[2]:.2f}) "
            f"处无法放置形状 {shape}（scale<{SCALE_MIN:.3f}m）"
        )
    return scale_env, scale_fit


def build_concentric_scales(
    scale_max: float,
    n_layers: int,
    min_frac: float = DEFAULT_MIN_FRAC,
    min_inner: float = MIN_INNER_SCALE,
) -> np.ndarray:
    """
    从最大 scale 向内递减；每圈画完后，若下一圈边长 < min_inner 则停止向内。
    """
    scales = [float(scale_max)]
    if n_layers <= 1:
        return np.array(scales, dtype=np.float64)

    lo_target = max(min_inner, scale_max * float(min_frac))
    need_span = SCALE_SHRINK_STEP * (n_layers - 1)
    if scale_max - lo_target >= need_span:
        planned = np.linspace(scale_max, lo_target, n_layers, dtype=np.float64)
    else:
        planned = [scale_max]
        s = scale_max
        for _ in range(n_layers - 1):
            s -= SCALE_SHRINK_STEP
            planned.append(max(SCALE_MIN, s))
        planned = np.array(planned, dtype=np.float64)

    for sc in planned[1:]:
        if float(sc) < min_inner - 1e-9:
            break
        scales.append(float(sc))
    return np.array(scales, dtype=np.float64)


def build_concentric_waypoints(
    shape: str,
    center: np.ndarray,
    scales: np.ndarray,
    theta: float,
    frames_per_ring: int,
    rng: np.random.Generator,
) -> list[np.ndarray]:
    rings = []
    for sc in scales:
        rings.append(make_shape_waypoints(
            shape, center, float(sc), theta, N=frames_per_ring, rng=rng,
        ))
    return rings


def generate_concentric_trajectory(
    ee_cloud: np.ndarray,
    joint_configs: np.ndarray,
    shape: str,
    center: np.ndarray,
    theta: float,
    n_layers: int,
    fps: float,
    frames_per_ring: int,
    transition_frames: int,
    min_frac: float,
    front_x_min: float,
    front_x_max: float,
    rng: np.random.Generator,
    ws: ReachableWorkspace,
    show_progress: bool = True,
    elbow_branch_mode: str = ELBOW_BRANCH_AUTO,
) -> dict:
    model = mujoco.MjModel.from_xml_path(ROBOT_XML)
    data = mujoco.MjData(model)
    addr = _build_addr_table(model, ACTIVE_JOINTS + list(LOCKED_JOINTS.keys()))
    _reset_to_home(model, data, addr)
    solver = _IKSolver4DOF(model, data)
    arm_ids, obstacle_ids = _build_collision_id_sets(
        model, _OBSTACLE_BODY_NAMES_MINIMAL,
        ee_body_names=_RIGHT_ARM_COLLISION_BODY_NAMES,
    )
    scale_env, scale_max = find_max_scale(
        shape, center, theta, ws, ee_cloud, rng,
    )
    scales = build_concentric_scales(scale_max, n_layers, min_frac=min_frac)
    n_actual = len(scales)
    rings = build_concentric_waypoints(
        shape, center, scales, theta, frames_per_ring, rng,
    )

    print(f"  中心=({center[0]:.2f},{center[1]:.2f},{center[2]:.2f}) "
          f"shape={shape} theta={theta:.2f}")
    print(f"  边界估算 scale_env={scale_env:.3f}m | 实测最大 scale_max={scale_max:.3f}m")
    inner_note = ""
    if n_actual < n_layers:
        inner_note = f"（向内止于 {MIN_INNER_SCALE:.2f}m，原计划 {n_layers} 层）"
    print(f"  实际层数={n_actual} scales={[f'{s:.3f}' for s in scales]}{inner_note}")

    all_qpos, all_qvel, all_ts = [], [], []
    all_wps, layer_scales = [], []
    segment_starts = [0]
    qpos_seed = None
    prev_last_wp = None
    t_accum = 0.0
    dt = 1.0 / fps
    traj_locked_branch = None

    for li, (scale, ring_wps) in enumerate(zip(scales, rings)):
        if not waypoints_in_front_band(ring_wps, front_x_min, front_x_max):
            raise RuntimeError(f"第 {li+1} 层超出正前方 x 带")

        if qpos_seed is not None:
            data.qpos[:] = qpos_seed
            data.qvel[:] = 0.0
            mujoco.mj_forward(model, data)
        else:
            _reset_to_home(model, data, addr)

        print(f"  层 {li+1}/{n_actual}: scale={scale:.3f}m IK ...", flush=True)
        qp_ring, qv_ring, ts_ring, errs, col_r, traj_locked_branch = (
            solve_waypoints_ik(
                ring_wps, ee_cloud, joint_configs, fps,
                solver, model, data, arm_ids, obstacle_ids,
                show_progress=show_progress,
                elbow_branch_mode=elbow_branch_mode,
                locked_branch=traj_locked_branch,
            )
        )
        if li == 0:
            print(f"  肘部折向锁定: {traj_locked_branch}")
        ok_ratio = float(np.mean(errs < IK_ERR_THRESH))
        if not check_feasibility(errs, col_r):
            raise RuntimeError(
                f"第 {li+1} 层 IK 不可行 (ok_ratio={ok_ratio:.2f}, col={col_r:.2f})"
            )

        seg_wps = ring_wps
        qp_seg = qp_ring
        qv_seg = qv_ring
        ts_seg = ts_ring

        if qpos_seed is not None and transition_frames > 0:
            qp_trans, qv_trans, ts_trans = solve_transition_segment(
                prev_last_wp, ring_wps[0], transition_frames,
                ee_cloud, joint_configs, fps,
                solver, model, data, arm_ids, obstacle_ids,
                locked_branch=traj_locked_branch,
                elbow_branch_mode=elbow_branch_mode,
            )
            qp_seg = np.vstack([qp_trans, qp_ring])
            qv_seg = np.vstack([qv_trans, qv_ring])
            ts_seg = np.concatenate([
                ts_trans,
                ts_ring + (ts_trans[-1] + 1.0 / fps),
            ])
            trans_vis = make_transition_waypoints(
                prev_last_wp, ring_wps[0], transition_frames,
            )
            seg_wps = np.vstack([trans_vis, ring_wps])

        if len(all_ts) > 0:
            ts_seg = ts_seg + t_accum

        all_qpos.append(qp_seg)
        all_qvel.append(qv_seg)
        all_ts.append(ts_seg)
        all_wps.append(seg_wps)
        layer_scales.append(float(scale))
        segment_starts.append(segment_starts[-1] + len(qp_seg))
        t_accum = all_ts[-1][-1] + dt
        prev_last_wp = ring_wps[-1].copy()
        qpos_seed = qp_seg[-1].copy()
        print(f"  层 {li+1}/{n_actual}: OK ok_ratio={ok_ratio:.2f}", flush=True)

    return {
        "qpos": np.concatenate(all_qpos, axis=0),
        "qvel": np.concatenate(all_qvel, axis=0),
        "timestamps": np.concatenate(all_ts, axis=0),
        "waypoints": np.concatenate(all_wps, axis=0),
        "shape_name": shape,
        "center": center.astype(np.float32),
        "theta": float(theta),
        "scale_max": float(scale_max),
        "layer_scales": np.array(layer_scales, dtype=np.float32),
        "segment_starts": np.array(segment_starts, dtype=np.int32),
        "n_layers": int(n_actual),
        "n_layers_planned": int(n_layers),
    }


def try_generate_one(
    ee_cloud, joint_configs, shape, rng,
    center: np.ndarray | None = None,
    theta: float | None = None,
    **kwargs,
) -> dict | None:
    max_tries = int(kwargs.pop("max_tries", 12))
    for attempt in range(max_tries):
        ws = kwargs.get("ws")
        if center is not None and attempt == 0:
            c = center.copy()
        elif ws is not None:
            c = sample_center(ee_cloud, ws, rng)
        else:
            idx = int(rng.integers(0, len(ee_cloud)))
            c = ee_cloud[idx].astype(np.float64).copy()
        th = theta if (theta is not None and attempt == 0) else float(rng.uniform(0, 2 * np.pi))
        try:
            return generate_concentric_trajectory(
                ee_cloud, joint_configs, shape, c, th, rng=rng, **kwargs,
            )
        except RuntimeError:
            if center is not None:
                break
            continue
    return None


def main():
    p = argparse.ArgumentParser(description="G1 右手同心递减轨迹生成")
    p.add_argument("--shape", type=str, required=True,
                   help="形状名，如 circle / 五角星 / random_polygon")
    p.add_argument("--layers", type=int, default=DEFAULT_LAYERS,
                   help="同心层数（从大到小）")
    p.add_argument("--num-trajs", type=int, default=1,
                   help="生成多少条轨迹（每次随机采样不同中心）")
    p.add_argument("--min-frac", type=float, default=DEFAULT_MIN_FRAC,
                   help="最小层相对最大层的比例")
    p.add_argument("--fps", type=float, default=DEFAULT_FPS)
    p.add_argument("--frames-per-ring", type=int, default=DEFAULT_FRAMES_PER_RING)
    p.add_argument("--transition-frames", type=int, default=DEFAULT_TRANSITION_FRAMES)
    p.add_argument("--center", type=float, nargs=3, metavar=("X", "Y", "Z"),
                   help="手动指定中心点（不指定则随机采样）")
    p.add_argument("--theta", type=float, default=None, help="形状旋转角（弧度）")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--preview", action="store_true")
    p.add_argument("--batch", action="store_true")
    p.add_argument("--save", action="store_true")
    p.add_argument("--ws-cache", action="store_true")
    p.add_argument("--front-x-min", type=float, default=DEFAULT_FRONT_X_MIN)
    p.add_argument("--front-x-max", type=float, default=DEFAULT_FRONT_X_MAX)
    p.add_argument("--elbow-branch", type=str, default=ELBOW_BRANCH_AUTO,
                   help="肘部折向: outward/inward/auto")
    args = p.parse_args()

    if not args.preview and not args.batch:
        print("请使用 --preview 和/或 --batch")
        print("  例: python g1_concentric_traj_gen.py --shape circle --layers 6 --preview --ws-cache")
        return

    shape = resolve_shape_name(args.shape)
    rng = np.random.default_rng(args.seed)

    if args.ws_cache:
        ee_cloud, joint_configs = load_workspace_cache(WS_CACHE_PATH)
        print(f"已加载可行域缓存: {len(ee_cloud)} 点")
    else:
        from g1_right_hand_workspace import build_workspace_cache
        ee_cloud, joint_configs = build_workspace_cache(
            n_per_joint=10, n_random=5000, show_plot=False,
            front_x_min=args.front_x_min, front_x_max=args.front_x_max,
        )

    ee_cloud, joint_configs, ws = build_reachable_workspace(
        ee_cloud, joint_configs, args.front_x_min, args.front_x_max,
    )
    print(f"正前方可达域: {ws.n_samples} 点 | "
          f"盒 x[{ws.lo[0]:.2f},{ws.hi[0]:.2f}] "
          f"y[{ws.lo[1]:.2f},{ws.hi[1]:.2f}] z[{ws.lo[2]:.2f},{ws.hi[2]:.2f}]")

    elbow_mode = resolve_branch_mode(args.elbow_branch)
    print(f"肘部折向模式: {elbow_mode}")

    gen_kw = dict(
        shape=shape,
        n_layers=args.layers,
        fps=args.fps,
        frames_per_ring=args.frames_per_ring,
        transition_frames=args.transition_frames,
        min_frac=args.min_frac,
        front_x_min=args.front_x_min,
        front_x_max=args.front_x_max,
        ws=ws,
        show_progress=args.batch,
        elbow_branch_mode=elbow_mode,
    )
    user_center = np.array(args.center, dtype=np.float64) if args.center else None

    saved_paths = []
    for ti in range(args.num_trajs):
        print(f"\n=== 轨迹 {ti+1}/{args.num_trajs} | {shape} x{args.layers} 层 ===")
        traj = try_generate_one(
            ee_cloud, joint_configs, rng=rng,
            center=user_center if ti == 0 else None,
            theta=args.theta if ti == 0 else None,
            **gen_kw,
        )
        if traj is None:
            print(f"  轨迹 {ti+1} 失败（中心不可行）")
            continue

        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        c = traj["center"]
        out_npz = os.path.join(
            "recordings",
            f"g1_concentric_{shape}_L{args.layers}_"
            f"c{c[0]:.2f}_{c[1]:.2f}_{c[2]:.2f}_{ts}.npz",
        )

        if args.save:
            os.makedirs("recordings", exist_ok=True)
            np.savez_compressed(
                out_npz,
                qpos=traj["qpos"],
                qvel=traj["qvel"],
                timestamps=traj["timestamps"],
                fps=np.float32(args.fps),
                shape_name=shape,
                center=traj["center"],
                theta=np.float32(traj["theta"]),
                scale_max=np.float32(traj["scale_max"]),
                layer_scales=traj["layer_scales"],
                segment_starts=traj["segment_starts"],
                waypoints=traj["waypoints"],
                mode="concentric",
            )
            print(f"  已保存 -> {out_npz}")
            saved_paths.append(out_npz)

        if args.preview and ti == 0:
            preview_trajectory(
                {
                    "qpos": traj["qpos"],
                    "qvel": traj["qvel"],
                    "timestamps": traj["timestamps"],
                    "waypoints": traj["waypoints"],
                    "shape_names": np.array([shape] * traj["n_layers"], dtype=object),
                    "segment_starts": traj["segment_starts"],
                    "shape_centers": np.tile(traj["center"], (traj["n_layers"], 1)),
                },
                args.fps,
                out_path=out_npz if args.save else None,
                auto_save=args.save,
            )

    if args.save and saved_paths:
        print("\n回放示例:")
        print(f"  python g1_player.py --traj {saved_paths[-1]} --fps {int(args.fps)}")


if __name__ == "__main__":
    main()
