"""
G1 批量轨迹采样 MuJoCo 可视化
================================
在 MuJoCo 中用红点叠加显示 batch 采集的所有轨迹路点，黄点标圆心，
用于检查随机采样（圆心/形状/尺度）的空间分布与覆盖效果。

用法：
  python g1_batch_sampler_viewer.py
  python g1_batch_sampler_viewer.py --batch-root batch_data --shapes circle
  python g1_batch_sampler_viewer.py --source raw --show-workspace
  python g1_batch_sampler_viewer.py --animate --fps 50   # 逐条回放，红点保持
"""

from __future__ import annotations

import argparse
import glob
import os
import time

import mujoco
import mujoco.viewer
import numpy as np

from g1_player import extract_ee_positions
from g1_right_hand_workspace import (
    ACTIVE_JOINTS,
    LOCKED_JOINTS,
    ROBOT_XML,
    WS_CACHE_PATH,
    _build_addr_table,
    _reset_to_home,
    load_workspace_cache,
)
from g1_trajectory_postprocess import DEFAULT_BATCH_ROOT

RED = np.array([1.0, 0.15, 0.15, 0.72], dtype=np.float32)
YELLOW = np.array([1.0, 0.85, 0.1, 0.95], dtype=np.float32)
GREEN = np.array([0.15, 0.85, 0.25, 0.65], dtype=np.float32)
GREY = np.array([0.45, 0.48, 0.52, 0.35], dtype=np.float32)

SHAPE_COLORS = {
    "circle": np.array([1.0, 0.2, 0.2, 0.75], dtype=np.float32),
    "ellipse": np.array([1.0, 0.5, 0.1, 0.75], dtype=np.float32),
    "triangle": np.array([0.2, 0.8, 0.2, 0.75], dtype=np.float32),
    "triangle_down": np.array([0.2, 0.6, 0.9, 0.75], dtype=np.float32),
    "square": np.array([0.9, 0.2, 0.9, 0.75], dtype=np.float32),
    "rectangle": np.array([0.6, 0.2, 0.9, 0.75], dtype=np.float32),
    "star": np.array([1.0, 0.85, 0.1, 0.75], dtype=np.float32),
    "pentagon": np.array([0.2, 0.85, 0.85, 0.75], dtype=np.float32),
    "heart": np.array([1.0, 0.3, 0.5, 0.75], dtype=np.float32),
    "random_polygon": np.array([0.55, 0.55, 0.55, 0.75], dtype=np.float32),
}


def _add_spheres(scene, positions, size: float, rgba: np.ndarray) -> int:
    added = 0
    for pos in positions:
        if scene.ngeom >= scene.maxgeom:
            break
        g = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(
            g,
            mujoco.mjtGeom.mjGEOM_SPHERE,
            np.array([size, 0.0, 0.0], dtype=np.float64),
            np.zeros(3, dtype=np.float64),
            np.eye(3, dtype=np.float64).flatten(),
            rgba,
        )
        g.pos[:] = np.asarray(pos, dtype=np.float64).reshape(3)
        scene.ngeom += 1
        added += 1
    return added


def _subsample(points: np.ndarray, max_n: int) -> np.ndarray:
    if len(points) <= max_n:
        return points
    idx = np.linspace(0, len(points) - 1, max_n, dtype=int)
    return points[idx]


def _discover_npz(batch_root: str, source: str) -> list[str]:
    if source == "raw":
        pattern = os.path.join(batch_root, "raw", "**", "*.npz")
    else:
        pattern = os.path.join(batch_root, "clips_obs", "**", "*_obs.npz")
    return sorted(glob.glob(pattern, recursive=True))


def load_batch_samples(
    batch_root: str,
    source: str = "raw",
    shapes: list[str] | None = None,
    max_trajs: int | None = None,
) -> list[dict]:
    files = _discover_npz(batch_root, source)
    samples: list[dict] = []
    for path in files:
        if shapes:
            shape_dir = os.path.basename(os.path.dirname(path))
            if shape_dir not in shapes:
                continue
        try:
            data = np.load(path, allow_pickle=False)
        except Exception as e:
            print(f"跳过损坏文件 {path}: {e}")
            continue
        if "waypoints" not in data:
            print(f"跳过无 waypoints: {path}")
            continue
        shape = (
            str(data["shape_name"])
            if "shape_name" in data
            else os.path.basename(os.path.dirname(path))
        )
        center = (
            data["center"].astype(np.float64)
            if "center" in data
            else np.zeros(3, dtype=np.float64)
        )
        samples.append({
            "path": path,
            "shape": shape,
            "seed": int(data["seed"]) if "seed" in data else -1,
            "center": center,
            "waypoints": data["waypoints"].astype(np.float64),
            "qpos": data["qpos"].astype(np.float64) if "qpos" in data else None,
            "qvel": data["qvel"].astype(np.float64) if "qvel" in data else None,
            "fps": float(data["fps"]) if "fps" in data else 50.0,
        })
        if max_trajs is not None and len(samples) >= max_trajs:
            break
    return samples


def _budget_waypoints(samples: list[dict], wp_budget: int, wp_step: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """返回 [(points, rgba), ...]，按轨迹分配路点显示配额。"""
    if not samples:
        return []
    if wp_step > 1:
        chunks = [s["waypoints"][::wp_step] for s in samples]
    else:
        total = sum(len(s["waypoints"]) for s in samples)
        per_traj = max(8, wp_budget // max(1, len(samples)))
        chunks = [_subsample(s["waypoints"], per_traj) for s in samples]

    out: list[tuple[np.ndarray, np.ndarray]] = []
    for s, pts in zip(samples, chunks):
        if len(pts) == 0:
            continue
        rgba = SHAPE_COLORS.get(s["shape"], RED)
        out.append((pts, rgba))
    return out


def _draw_scene(
    scene,
    samples: list[dict],
    *,
    wp_budget: int,
    wp_step: int,
    show_centers: bool,
    show_waypoints: bool,
    show_ee: bool,
    color_by_shape: bool,
    workspace_ee: np.ndarray | None,
    ee_model: mujoco.MjModel | None,
) -> None:
    scene.ngeom = 0
    if workspace_ee is not None and len(workspace_ee) > 0:
        step = max(1, len(workspace_ee) // 2000)
        _add_spheres(scene, workspace_ee[::step], 0.004, GREY)

    if show_centers:
        centers = np.array([s["center"] for s in samples], dtype=np.float64)
        _add_spheres(scene, centers, 0.012, YELLOW)

    if show_waypoints:
        if color_by_shape:
            for pts, rgba in _budget_waypoints(samples, wp_budget, wp_step):
                _add_spheres(scene, pts, 0.006, rgba)
        else:
            all_pts = []
            for s in samples:
                pts = s["waypoints"]
                if wp_step > 1:
                    pts = pts[::wp_step]
                all_pts.append(pts)
            if all_pts:
                merged = np.concatenate(all_pts, axis=0)
                merged = _subsample(merged, wp_budget)
                _add_spheres(scene, merged, 0.006, RED)

    if show_ee and ee_model is not None:
        ee_pts = []
        for s in samples:
            if s["qpos"] is None:
                continue
            ee = extract_ee_positions(ee_model, s["qpos"])
            ee_pts.append(_subsample(ee, max(12, wp_budget // max(1, len(samples)))))
        if ee_pts:
            merged = np.concatenate(ee_pts, axis=0)
            _add_spheres(scene, merged, 0.005, GREEN)


def run_viewer(args: argparse.Namespace) -> None:
    samples = load_batch_samples(
        args.batch_root,
        source=args.source,
        shapes=args.shapes,
        max_trajs=args.max_trajs,
    )
    if not samples:
        print(f"未找到轨迹文件。请检查 {args.batch_root}/{args.source}")
        return

    shapes_seen = sorted({s["shape"] for s in samples})
    n_wp = sum(len(s["waypoints"]) for s in samples)
    print(f"加载 {len(samples)} 条轨迹 | 形状: {', '.join(shapes_seen)}")
    print(f"路点总数: {n_wp} | 黄点=圆心 | 红点=规划路点", end="")
    if args.show_ee:
        print(" | 绿点=实际末端 FK", end="")
    print()

    workspace_ee = None
    if args.show_workspace and os.path.isfile(WS_CACHE_PATH):
        workspace_ee, _ = load_workspace_cache(WS_CACHE_PATH)
        print(f"叠加可行域点云: {WS_CACHE_PATH} ({len(workspace_ee)} pts)")

    # 与可行域 / IK / 路点生成使用同一模型与 home 姿态（非 scene_g1_draggable keyframe）
    model = mujoco.MjModel.from_xml_path(ROBOT_XML)
    data = mujoco.MjData(model)
    addr = _build_addr_table(model, ACTIVE_JOINTS + list(LOCKED_JOINTS.keys()))
    _reset_to_home(model, data, addr)
    print(f"机器人参考姿态: {ROBOT_XML} + _reset_to_home（与批量采集坐标系一致）")

    ee_model = model if args.show_ee else None
    traj_idx = [0]

    def key_cb(keycode: int) -> None:
        if keycode in (ord("n"), ord("N")):
            traj_idx[0] = (traj_idx[0] + 1) % len(samples)
            s = samples[traj_idx[0]]
            print(f"切换轨迹 [{traj_idx[0]+1}/{len(samples)}] {s['shape']} seed={s['seed']}")
        elif keycode in (ord("p"), ord("P")):
            traj_idx[0] = (traj_idx[0] - 1) % len(samples)
            s = samples[traj_idx[0]]
            print(f"切换轨迹 [{traj_idx[0]+1}/{len(samples)}] {s['shape']} seed={s['seed']}")

    print("\n[MuJoCo] ESC 退出 | N/P 切换高亮轨迹（配合 --animate）")
    with mujoco.viewer.launch_passive(model, data, key_callback=key_cb) as viewer:
        if not args.animate:
            while viewer.is_running():
                try:
                    scn = viewer.user_scn
                    _draw_scene(
                        scn,
                        samples,
                        wp_budget=args.wp_budget,
                        wp_step=args.wp_step,
                        show_centers=args.show_centers,
                        show_waypoints=args.show_waypoints,
                        show_ee=args.show_ee,
                        color_by_shape=args.color_by_shape,
                        workspace_ee=workspace_ee,
                        ee_model=ee_model,
                    )
                except Exception:
                    pass
                viewer.sync()
                time.sleep(0.02)
            return

        frame_dt = 1.0 / args.fps
        while viewer.is_running():
            s = samples[traj_idx[0]]
            qpos = s["qpos"]
            if qpos is None:
                print(f"无 qpos，无法动画: {s['path']}")
                traj_idx[0] = (traj_idx[0] + 1) % len(samples)
                continue
            for i in range(len(qpos)):
                if not viewer.is_running():
                    break
                t0 = time.perf_counter()
                data.qpos[:] = qpos[i]
                qvel = s.get("qvel")
                data.qvel[:] = qvel[i] if qvel is not None else 0.0
                mujoco.mj_forward(model, data)
                try:
                    scn = viewer.user_scn
                    _draw_scene(
                        scn,
                        samples,
                        wp_budget=args.wp_budget,
                        wp_step=args.wp_step,
                        show_centers=args.show_centers,
                        show_waypoints=args.show_waypoints,
                        show_ee=args.show_ee,
                        color_by_shape=args.color_by_shape,
                        workspace_ee=workspace_ee,
                        ee_model=ee_model,
                    )
                except Exception:
                    pass
                viewer.sync()
                sleep_t = max(0.0, frame_dt - (time.perf_counter() - t0))
                if sleep_t > 0:
                    time.sleep(sleep_t)
            traj_idx[0] = (traj_idx[0] + 1) % len(samples)


def main():
    ap = argparse.ArgumentParser(description="G1 批量轨迹采样 MuJoCo 可视化")
    ap.add_argument("--batch-root", default=DEFAULT_BATCH_ROOT)
    ap.add_argument("--source", choices=("raw", "obs"), default="raw",
                    help="raw=IK 原始轨迹；obs=clips_obs")
    ap.add_argument("--shapes", nargs="*", default=None, help="只显示指定形状子目录")
    ap.add_argument("--max-trajs", type=int, default=None, help="最多加载条数")
    ap.add_argument("--wp-budget", type=int, default=8000,
                    help="路点红点总数上限（受 MuJoCo maxgeom 限制）")
    ap.add_argument("--wp-step", type=int, default=1,
                    help="每条轨迹路点步长降采样，>1 时优先于 wp-budget 均分")
    ap.add_argument("--no-centers", action="store_true", help="不显示黄点圆心")
    ap.add_argument("--no-waypoints", action="store_true", help="不显示红点的路点")
    ap.add_argument("--show-ee", action="store_true",
                    help="叠加绿色实际末端 FK（较慢）")
    ap.add_argument("--show-workspace", action="store_true",
                    help="叠加灰色可行域点云")
    ap.add_argument("--color-by-shape", action="store_true",
                    help="按形状上色路点（默认统一红色）")
    ap.add_argument("--animate", action="store_true",
                    help="逐条回放机器人，路点保持显示")
    ap.add_argument("--fps", type=float, default=50.0)
    args = ap.parse_args()
    args.show_centers = not args.no_centers
    args.show_waypoints = not args.no_waypoints
    run_viewer(args)


if __name__ == "__main__":
    main()
