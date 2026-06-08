"""
G1 批量轨迹后处理：单圈 300 帧轨迹 → 质量筛选 → obs NPZ（无需裁切）
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, asdict
from typing import Any

import mujoco
import numpy as np

from g1_player import (
    MODEL_PATH,
    PRIV_STATE_DT,
    PrivilegedStateComputer,
    compute_state,
)
from g1_trajectory_ik import G1RightArmModel, ROBOT_XML, evaluate_trajectory

DEFAULT_BATCH_ROOT = os.environ.get("G1_BATCH_ROOT", "batch_data")


@dataclass
class FilterThresholds:
    err_mean_mm: float = 15.0
    ok_ratio: float = 0.85
    branch_viol: int = 0
    max_droll: float = 0.12
    max_dpitch: float = 0.12
    max_dyaw: float = 0.06
    xflip: int = 40


def batch_dirs(root: str = DEFAULT_BATCH_ROOT) -> dict[str, str]:
    sub = ("raw", "clips_raw", "clips_obs", "previews", "rejected", "logs", "state")
    out = {k: os.path.join(root, k) for k in sub}
    for p in out.values():
        os.makedirs(p, exist_ok=True)
    return out


def format_center_tag(center: np.ndarray) -> str:
    c = np.asarray(center, dtype=np.float64).reshape(3)
    return f"c{c[0]:.2f}_{c[1]:.2f}_{c[2]:.2f}"


def traj_basename(
    shape: str,
    center: np.ndarray,
    scale: float,
    seed: int,
    frames: int = 300,
) -> str:
    """单圈轨迹命名：形状 + 圆心 + 半径 + seed + 帧数"""
    return (
        f"{shape}_{format_center_tag(center)}_s{scale:.3f}"
        f"_seed{seed:03d}_F{frames}"
    )


def _right_arm_joint_steps(qpos_seg: np.ndarray, arm: G1RightArmModel) -> dict[str, float]:
    q4 = np.array(
        [[qpos_seg[i, a] for a in arm.qpos_addrs] for i in range(len(qpos_seg))],
        dtype=np.float64,
    )
    if len(q4) < 2:
        return {"max_dpitch": 0.0, "max_droll": 0.0, "max_dyaw": 0.0}
    dq = np.abs(np.diff(q4, axis=0))
    return {
        "max_dpitch": float(dq[:, 0].max()),
        "max_droll": float(dq[:, 1].max()),
        "max_dyaw": float(dq[:, 2].max()),
    }


def assess_clip(
    qpos_seg: np.ndarray,
    wps_seg: np.ndarray,
    locked_branch: str | None,
    thresholds: FilterThresholds,
) -> tuple[bool, dict[str, Any], list[str]]:
    metrics = evaluate_trajectory(qpos_seg, wps_seg, locked_branch)
    model = mujoco.MjModel.from_xml_path(ROBOT_XML)
    arm = G1RightArmModel(model)
    metrics.update(_right_arm_joint_steps(qpos_seg, arm))

    reasons: list[str] = []
    if metrics["err_mean_mm"] > thresholds.err_mean_mm:
        reasons.append(f"err_mean_mm={metrics['err_mean_mm']:.1f}>{thresholds.err_mean_mm}")
    if metrics["ok_ratio"] < thresholds.ok_ratio:
        reasons.append(f"ok_ratio={metrics['ok_ratio']:.3f}<{thresholds.ok_ratio}")
    if metrics["branch_viol"] > thresholds.branch_viol:
        reasons.append(f"branch_viol={metrics['branch_viol']}>{thresholds.branch_viol}")
    if metrics["max_droll"] > thresholds.max_droll:
        reasons.append(f"max_droll={metrics['max_droll']:.3f}>{thresholds.max_droll}")
    if metrics["max_dpitch"] > thresholds.max_dpitch:
        reasons.append(f"max_dpitch={metrics['max_dpitch']:.3f}>{thresholds.max_dpitch}")
    if metrics["max_dyaw"] > thresholds.max_dyaw:
        reasons.append(f"max_dyaw={metrics['max_dyaw']:.3f}>{thresholds.max_dyaw}")
    if metrics["xflip"] > thresholds.xflip:
        reasons.append(f"xflip={metrics['xflip']}>{thresholds.xflip}")

    return len(reasons) == 0, metrics, reasons


def qpos_to_obs(qpos_arr: np.ndarray, qvel_arr: np.ndarray, fps: float) -> dict[str, np.ndarray]:
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)
    priv = PrivilegedStateComputer(model, dt=PRIV_STATE_DT)
    priv.reset()

    n = len(qpos_arr)
    out_state = np.zeros((n, 64), dtype=np.float32)
    out_action = np.zeros((n, 29), dtype=np.float32)
    sample_priv = None
    out_priv = None

    for i in range(n):
        data.qpos[:] = qpos_arr[i]
        data.qvel[:] = qvel_arr[i]
        mujoco.mj_forward(model, data)
        out_state[i] = compute_state(data)
        p = priv.compute(data)
        if sample_priv is None:
            sample_priv = p
            out_priv = np.zeros((n, len(p)), dtype=np.float32)
        out_priv[i] = p

    ts = np.arange(n, dtype=np.float32) / float(fps)
    return {
        "state": out_state,
        "last_action": out_action,
        "privileged_state": out_priv,
        "qpos": qpos_arr.astype(np.float32),
        "qvel": qvel_arr.astype(np.float32),
        "timestamps": ts,
        "fps": np.float32(fps),
    }


def _atomic_save_npz(path: str, **arrays) -> None:
    tmp = path[:-4] + ".part" if path.endswith(".npz") else path + ".part"
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    np.savez_compressed(tmp, **arrays)
    final = tmp + ".npz" if not tmp.endswith(".npz") else tmp
    if not os.path.isfile(final):
        raise FileNotFoundError(f"savez 未生成预期文件: {final}")
    os.replace(final, path)


def process_raw_trajectory(
    raw_path: str,
    batch_root: str = DEFAULT_BATCH_ROOT,
    thresholds: FilterThresholds | None = None,
    save_clips_raw: bool = False,
    event_cb=None,
) -> dict[str, Any]:
    """
    处理单个 raw NPZ（单圈 300 帧，整段筛选转 obs，无需裁切）。
    event_cb(event_type: str, payload: dict) 可选，用于写 events.jsonl。
    """
    thresholds = thresholds or FilterThresholds()
    dirs = batch_dirs(batch_root)
    data = np.load(raw_path, allow_pickle=False)

    qpos = data["qpos"]
    qvel = data["qvel"]
    waypoints = data["waypoints"]
    shape = str(data["shape_name"])
    center = data["center"].astype(np.float64)
    locked_branch = str(data["locked_branch"]) if "locked_branch" in data else None
    if locked_branch == "None":
        locked_branch = None

    seed = int(data["seed"]) if "seed" in data else -1
    fps = float(data["fps"]) if "fps" in data else 50.0
    frames_per_ring = int(data["frames_per_ring"]) if "frames_per_ring" in data else 300
    if "layer_scales" in data and len(data["layer_scales"]) > 0:
        scale = float(data["layer_scales"][0])
    else:
        scale = float(data["scale_max"]) if "scale_max" in data else 0.0

    n_frames = len(qpos)
    base = traj_basename(shape, center, scale, seed, frames_per_ring)
    payload_base = {
        "shape": shape,
        "seed": seed,
        "scale": scale,
        "center": center.tolist(),
        "raw_path": raw_path,
        "traj_name": base,
    }
    clip_records: list[dict] = []

    if n_frames != frames_per_ring:
        reason = f"bad_frame_count={n_frames}!={frames_per_ring}"
        rej_path = os.path.join(dirs["rejected"], f"{base}.reject.json")
        with open(rej_path, "w", encoding="utf-8") as f:
            json.dump({**payload_base, "reasons": [reason]}, f, indent=2)
        if event_cb:
            event_cb("traj_rejected", {**payload_base, "reasons": [reason]})
        clip_records.append({**payload_base, "status": "rejected", "reasons": [reason]})
        return {
            "raw_path": raw_path,
            "shape": shape,
            "seed": seed,
            "n_clips": 1,
            "n_pass": 0,
            "n_reject": 1,
            "clips": clip_records,
        }

    ok, metrics, reasons = assess_clip(qpos, waypoints, locked_branch, thresholds)
    if not ok:
        rej_path = os.path.join(dirs["rejected"], f"{base}.reject.json")
        with open(rej_path, "w", encoding="utf-8") as f:
            json.dump({
                **payload_base,
                "metrics": metrics,
                "thresholds": asdict(thresholds),
                "reasons": reasons,
            }, f, indent=2)
        clip_records.append({**payload_base, "status": "rejected", "reasons": reasons, "metrics": metrics})
        if event_cb:
            event_cb("traj_rejected", {**payload_base, "reasons": reasons, "metrics": metrics})
        return {
            "raw_path": raw_path,
            "shape": shape,
            "seed": seed,
            "n_clips": 1,
            "n_pass": 0,
            "n_reject": 1,
            "clips": clip_records,
        }

    shape_dir_obs = os.path.join(dirs["clips_obs"], shape)
    os.makedirs(shape_dir_obs, exist_ok=True)
    obs_path = os.path.join(shape_dir_obs, f"{base}_obs.npz")

    if save_clips_raw:
        shape_dir_raw = os.path.join(dirs["clips_raw"], shape)
        os.makedirs(shape_dir_raw, exist_ok=True)
        raw_out = os.path.join(shape_dir_raw, f"{base}_raw.npz")
        _atomic_save_npz(
            raw_out,
            qpos=qpos,
            qvel=qvel,
            waypoints=waypoints.astype(np.float32),
            shape_name=shape,
            center=center.astype(np.float32),
            scale=np.float32(scale),
            seed=np.int32(seed),
            frames_per_ring=np.int32(frames_per_ring),
            fps=np.float32(fps),
            source_raw=raw_path,
        )

    obs = qpos_to_obs(qpos, qvel, fps)
    obs.update(
        shape_name=shape,
        center=center.astype(np.float32),
        scale=np.float32(scale),
        seed=np.int32(seed),
        waypoints=waypoints.astype(np.float32),
        traj_name=base,
        source_raw=raw_path,
    )
    _atomic_save_npz(obs_path, **obs)

    rec = {
        **payload_base,
        "status": "passed",
        "metrics": metrics,
        "obs_clip_path": obs_path,
    }
    clip_records.append(rec)
    if event_cb:
        event_cb("traj_passed", rec)
        event_cb("obs_saved", {"traj_name": base, "obs_clip_path": obs_path})

    return {
        "raw_path": raw_path,
        "shape": shape,
        "seed": seed,
        "n_clips": 1,
        "n_pass": 1,
        "n_reject": 0,
        "clips": clip_records,
    }


def main():
    ap = argparse.ArgumentParser(description="G1 raw 轨迹后处理")
    ap.add_argument("--raw", required=True, help="raw NPZ 路径")
    ap.add_argument("--batch-root", default=DEFAULT_BATCH_ROOT)
    ap.add_argument("--err-mean-mm", type=float, default=15.0)
    ap.add_argument("--ok-ratio", type=float, default=0.85)
    ap.add_argument("--max-droll", type=float, default=0.12)
    args = ap.parse_args()

    th = FilterThresholds(err_mean_mm=args.err_mean_mm, ok_ratio=args.ok_ratio, max_droll=args.max_droll)
    result = process_raw_trajectory(args.raw, args.batch_root, th)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
