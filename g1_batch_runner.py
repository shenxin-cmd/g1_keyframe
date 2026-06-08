"""
G1 批量轨迹采集主控：多形状 × 多 seed IK → 后处理 → 预览
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import sys
import time
from datetime import datetime, timezone
from typing import Any

import numpy as np

from g1_arm_posture import ELBOW_BRANCH_OUTWARD
from g1_concentric_traj_gen import resolve_shape_name
from g1_right_hand_workspace import WS_CACHE_PATH
from g1_trajectory_ik import (
    _load_reachable_workspace,
    _try_generate_concentric_demo,
    save_trajectory_npz,
)
from g1_trajectory_postprocess import (
    DEFAULT_BATCH_ROOT,
    batch_dirs,
    format_center_tag,
    process_raw_trajectory,
    FilterThresholds,
)

BATCH_SHAPES = [
    "circle",
    "ellipse",
    "triangle",
    "triangle_down",
    "square",
    "rectangle",
    "star",
    "pentagon",
    "heart",
    "random_polygon",
]

DEFAULT_CENTER = np.array([0.27, -0.25, 1.01], dtype=np.float64)


class BatchRunner:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.root = args.batch_root
        self.dirs = batch_dirs(self.root)
        self.logger = self._setup_logging()
        self.manifest_path = os.path.join(self.dirs["logs"], "manifest.jsonl")
        self.events_path = os.path.join(self.dirs["logs"], "events.jsonl")
        self.checkpoint_path = os.path.join(self.dirs["state"], "checkpoint.json")
        self.lock_path = os.path.join(self.dirs["state"], "runner.lock")
        self._manifest: dict[str, dict] = {}
        self._current_job: str | None = None
        self._load_manifest()
        self._install_signals()

    def _setup_logging(self) -> logging.Logger:
        log_path = os.path.join(self.root, "logs", "run.log")
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        logger = logging.getLogger("g1_batch_runner")
        logger.setLevel(logging.INFO)
        logger.handlers.clear()
        fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        fh = logging.FileHandler(log_path, encoding="utf-8")
        fh.setFormatter(fmt)
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(fmt)
        logger.addHandler(fh)
        logger.addHandler(sh)
        return logger

    def _utcnow(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def _load_manifest(self) -> None:
        if not os.path.isfile(self.manifest_path):
            return
        with open(self.manifest_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                self._manifest[rec["job_id"]] = rec

    def _append_manifest(self, rec: dict) -> None:
        self._manifest[rec["job_id"]] = rec
        with open(self.manifest_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def _update_manifest(self, job_id: str, **fields) -> None:
        rec = self._manifest.get(job_id, {"job_id": job_id})
        rec.update(fields)
        rec["updated_at"] = self._utcnow()
        self._manifest[job_id] = rec
        self._append_manifest(rec)

    def _emit_event(self, event_type: str, payload: dict) -> None:
        evt = {"ts": self._utcnow(), "type": event_type, **payload}
        with open(self.events_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(evt, ensure_ascii=False, default=str) + "\n")

    def _write_checkpoint(self, job_id: str) -> None:
        with open(self.checkpoint_path, "w", encoding="utf-8") as f:
            json.dump({"job_id": job_id, "ts": self._utcnow()}, f, indent=2)

    def _install_signals(self) -> None:
        def _handler(signum, _frame):
            self.logger.warning("收到信号 %s，正在安全退出...", signum)
            if self._current_job:
                self._update_manifest(
                    self._current_job,
                    status="interrupted",
                    error=f"signal_{signum}",
                )
            sys.exit(128 + signum)

        signal.signal(signal.SIGINT, _handler)
        signal.signal(signal.SIGTERM, _handler)

    def _acquire_lock(self) -> None:
        if os.path.isfile(self.lock_path):
            with open(self.lock_path, encoding="utf-8") as f:
                info = f.read().strip()
            if not self.args.force:
                raise RuntimeError(f"已有 runner 锁: {self.lock_path} ({info})，使用 --force 覆盖")
        with open(self.lock_path, "w", encoding="utf-8") as f:
            f.write(f"pid={os.getpid()} started={self._utcnow()}")

    def _release_lock(self) -> None:
        if os.path.isfile(self.lock_path):
            os.remove(self.lock_path)

    def _job_id(self, shape: str, seed: int) -> str:
        tag = format_center_tag(self.args.center)
        return f"{shape}_{tag}_seed{seed:03d}"

    def _raw_path(self, shape: str, seed: int) -> str:
        tag = format_center_tag(self.args.center)
        fname = f"{tag}_seed{seed:03d}_L{self.args.layers}_F{self.args.frames_per_ring}.npz"
        return os.path.join(self.dirs["raw"], shape, fname)

    def _job_status(self, job_id: str) -> str | None:
        rec = self._manifest.get(job_id)
        return rec.get("status") if rec else None

    def _should_skip(self, job_id: str) -> bool:
        if not self.args.resume:
            return False
        return self._job_status(job_id) == "post_done"

    def _build_jobs(self) -> list[tuple[str, int]]:
        shapes = [resolve_shape_name(s) for s in (self.args.shapes or BATCH_SHAPES)]
        jobs = [(shape, seed) for shape in shapes for seed in range(
            self.args.seed_start, self.args.seed_start + self.args.seed_count,
        )]
        if self.args.max_jobs is not None:
            jobs = jobs[: self.args.max_jobs]
        return jobs

    def run(self) -> None:
        if not os.path.isfile(WS_CACHE_PATH):
            raise RuntimeError(f"缺少可行域缓存 {WS_CACHE_PATH}，请先运行 g1_right_hand_workspace.py")

        self._acquire_lock()
        jobs = self._build_jobs()
        self.logger.info(
            "开始批量任务: %d jobs | center=%s layers=%d F=%d fps=%.0f",
            len(jobs), self.args.center.tolist(), self.args.layers,
            self.args.frames_per_ring, self.args.fps,
        )

        ee, joints, ws = _load_reachable_workspace(True)
        thresholds = FilterThresholds(
            err_mean_mm=self.args.err_mean_mm,
            ok_ratio=self.args.ok_ratio,
            max_droll=self.args.max_droll,
            max_dpitch=self.args.max_dpitch,
            max_dyaw=self.args.max_dyaw,
            xflip=self.args.max_xflip,
        )

        stats = {"done": 0, "failed": 0, "skipped": 0, "clips_pass": 0, "clips_reject": 0}

        try:
            for ji, (shape, seed) in enumerate(jobs):
                job_id = self._job_id(shape, seed)
                raw_path = self._raw_path(shape, seed)
                if self._should_skip(job_id):
                    stats["skipped"] += 1
                    self.logger.info("[%d/%d] 跳过已完成 %s", ji + 1, len(jobs), job_id)
                    continue

                self._current_job = job_id
                self._update_manifest(
                    job_id,
                    shape=shape,
                    seed=seed,
                    status="running",
                    started_at=self._utcnow(),
                )
                t0 = time.perf_counter()
                resume_post_only = (
                    self.args.resume
                    and self._job_status(job_id) == "ik_done"
                    and os.path.isfile(raw_path)
                )

                try:
                    if resume_post_only:
                        self.logger.info("[%d/%d] %s 续跑后处理（跳过 IK）", ji + 1, len(jobs), job_id)
                        ik_sec = 0.0
                    else:
                        self._emit_event("ik_start", {"job_id": job_id, "shape": shape, "seed": seed})
                        rng = np.random.default_rng(seed)
                        traj = _try_generate_concentric_demo(
                            shape,
                            self.args.layers,
                            self.args.fps,
                            self.args.frames_per_ring,
                            ELBOW_BRANCH_OUTWARD,
                            ee, joints, ws,
                            rng,
                            refine=not self.args.no_refine,
                            center=self.args.center.copy(),
                            theta=None,
                        )
                        if traj is None:
                            raise RuntimeError("IK 失败：中心不可行")

                        save_trajectory_npz(raw_path, traj, seed=seed)
                        ik_sec = time.perf_counter() - t0
                        self._emit_event("ik_done", {"job_id": job_id, "raw_path": raw_path, "sec": ik_sec})
                        self._update_manifest(job_id, status="ik_done", raw_path=raw_path, ik_sec=ik_sec)

                    def event_cb(evt_type: str, payload: dict):
                        self._emit_event(evt_type, {"job_id": job_id, **payload})

                    post = process_raw_trajectory(
                        raw_path,
                        batch_root=self.root,
                        thresholds=thresholds,
                        event_cb=event_cb,
                    )
                    stats["clips_pass"] += post["n_pass"]
                    stats["clips_reject"] += post["n_reject"]

                    if self.args.preview:
                        from g1_clip_inspector import inspect_clip_file
                        for clip in post["clips"]:
                            obs_path = clip.get("obs_clip_path")
                            if clip.get("status") != "passed" or not obs_path:
                                continue
                            try:
                                inspect_clip_file(obs_path, self.root)
                            except Exception as e:
                                self.logger.warning("预览失败 %s: %s", obs_path, e)

                    self._update_manifest(
                        job_id,
                        status="post_done",
                        n_clips=post["n_clips"],
                        n_pass=post["n_pass"],
                        n_reject=post["n_reject"],
                        finished_at=self._utcnow(),
                    )
                    self._write_checkpoint(job_id)
                    stats["done"] += 1
                    self.logger.info(
                        "[%d/%d] %s IK %.1fs | %d pass / %d reject (of %d) | "
                        "累计 pass=%d reject=%d failed=%d",
                        ji + 1, len(jobs), job_id, ik_sec,
                        post["n_pass"], post["n_reject"], post["n_clips"],
                        stats["clips_pass"], stats["clips_reject"], stats["failed"],
                    )

                except Exception as e:
                    stats["failed"] += 1
                    self._update_manifest(
                        job_id,
                        status="failed",
                        error=str(e),
                        finished_at=self._utcnow(),
                    )
                    self._emit_event("ik_failed", {"job_id": job_id, "error": str(e)})
                    self.logger.exception("[%d/%d] %s 失败: %s", ji + 1, len(jobs), job_id, e)

                self._current_job = None

        finally:
            self._release_lock()

        self.logger.info(
            "批量结束: done=%d skipped=%d failed=%d | clips pass=%d reject=%d",
            stats["done"], stats["skipped"], stats["failed"],
            stats["clips_pass"], stats["clips_reject"],
        )


def main():
    ap = argparse.ArgumentParser(description="G1 批量轨迹采集")
    ap.add_argument("--batch-root", default=DEFAULT_BATCH_ROOT)
    ap.add_argument("--shapes", nargs="*", default=None, help="默认 10 种形状全集")
    ap.add_argument("--center", nargs=3, type=float, default=DEFAULT_CENTER.tolist())
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--frames-per-ring", type=int, default=300)
    ap.add_argument("--fps", type=float, default=50.0)
    ap.add_argument("--seed-start", type=int, default=0)
    ap.add_argument("--seed-count", type=int, default=50)
    ap.add_argument("--max-jobs", type=int, default=None, help="调试用，限制 job 数")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--force", action="store_true", help="覆盖 runner.lock")
    ap.add_argument("--no-refine", action="store_true")
    ap.add_argument("--preview", action="store_true", help="每个 job 通过后生成 PNG 预览")
    ap.add_argument("--err-mean-mm", type=float, default=15.0)
    ap.add_argument("--ok-ratio", type=float, default=0.85)
    ap.add_argument("--max-droll", type=float, default=0.12)
    ap.add_argument("--max-dpitch", type=float, default=0.12)
    ap.add_argument("--max-dyaw", type=float, default=0.06)
    ap.add_argument("--max-xflip", type=int, default=40)
    args = ap.parse_args()
    args.center = np.array(args.center, dtype=np.float64)

    BatchRunner(args).run()


if __name__ == "__main__":
    main()
