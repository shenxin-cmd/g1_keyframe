"""
G1 300 帧轨迹片段可视化（无 MuJoCo 窗口）
"""

from __future__ import annotations

import argparse
import glob
import os

import numpy as np

from g1_player import extract_ee_positions, plot_ee_trajectory, ROBOT_XML
from g1_trajectory_postprocess import DEFAULT_BATCH_ROOT, batch_dirs

import mujoco


def inspect_clip_file(
    clip_path: str,
    batch_root: str = DEFAULT_BATCH_ROOT,
    out_path: str | None = None,
) -> str:
    """为单个 clips_raw 或 clips_obs NPZ 生成 PNG 预览。"""
    data = np.load(clip_path, allow_pickle=False)
    qpos = data["qpos"]
    waypoints = data["waypoints"].astype(np.float64) if "waypoints" in data else None

    shape = str(data["shape_name"]) if "shape_name" in data else "unknown"
    scale = float(data["scale"]) if "scale" in data else 0.0
    seed = int(data["seed"]) if "seed" in data else -1
    center = data["center"] if "center" in data else np.zeros(3)

    ee_model = mujoco.MjModel.from_xml_path(ROBOT_XML)
    ee_traj = extract_ee_positions(ee_model, qpos)

    dirs = batch_dirs(batch_root)
    if out_path is None:
        base = os.path.splitext(os.path.basename(clip_path))[0]
        if base.endswith("_obs"):
            base = base[:-4]
        elif base.endswith("_raw"):
            base = base[:-4]
        out_dir = os.path.join(dirs["previews"], shape)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{base}_preview.png")

    c = np.asarray(center, dtype=np.float64).reshape(3)
    title = (
        f"{shape} | center=({c[0]:.2f},{c[1]:.2f},{c[2]:.2f}) "
        f"| scale={scale:.3f} seed={seed} | {len(qpos)}f"
    )
    plot_ee_trajectory(
        ee_traj,
        waypoints=waypoints,
        segment_starts=None,
        title=title,
        out_path=out_path,
    )
    return out_path


def inspect_batch(
    batch_dir: str,
    batch_root: str = DEFAULT_BATCH_ROOT,
    only_new: bool = False,
) -> int:
    patterns = [
        os.path.join(batch_dir, "**", "*_obs.npz"),
        os.path.join(batch_dir, "**", "*_raw.npz"),
    ]
    files = sorted({f for pat in patterns for f in glob.glob(pat, recursive=True)})
    n = 0
    for f in files:
        base = os.path.splitext(os.path.basename(f))[0]
        if base.endswith("_obs"):
            base = base[:-4]
        elif base.endswith("_raw"):
            base = base[:-4]
        shape = os.path.basename(os.path.dirname(f))
        preview = os.path.join(batch_root, "previews", shape, f"{base}_preview.png")
        if only_new and os.path.isfile(preview):
            continue
        inspect_clip_file(f, batch_root, preview)
        print(f"预览: {preview}")
        n += 1
    return n


def main():
    ap = argparse.ArgumentParser(description="G1 轨迹片段可视化")
    ap.add_argument("--clip", default=None, help="单个 clip NPZ")
    ap.add_argument("--batch-dir", default=None, help="批量扫描目录")
    ap.add_argument("--batch-root", default=DEFAULT_BATCH_ROOT)
    ap.add_argument("--only-new", action="store_true")
    args = ap.parse_args()

    if args.clip:
        out = inspect_clip_file(args.clip, args.batch_root)
        print(f"已保存: {out}")
    elif args.batch_dir:
        n = inspect_batch(args.batch_dir, args.batch_root, args.only_new)
        print(f"共生成 {n} 张预览图")
    else:
        obs_root = os.path.join(args.batch_root, "clips_obs")
        n = inspect_batch(obs_root, args.batch_root, args.only_new)
        print(f"共生成 {n} 张预览图")


if __name__ == "__main__":
    main()
