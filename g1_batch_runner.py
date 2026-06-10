"""
G1 批量轨迹采集主控：多形状 × 随机采样 → 单圈 300 帧 IK → 筛选 → obs NPZ
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import multiprocessing as mp
import os
import signal
import sys
import time
from datetime import datetime, timezone
from typing import Any

import numpy as np

from g1_arm_posture import ELBOW_BRANCH_OUTWARD
from g1_concentric_traj_gen import resolve_shape_name
from g1_multi_shapes import DEFAULT_SHAPE_PLANE, resolve_shape_plane
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

# center=None 时从可达域随机采样圆心

_WORKER_STATE: dict[str, Any] = {}


@dataclasses.dataclass
class WorkerConfig:
    batch_root: str
    layers: int
    frames_per_ring: int
    fps: float
    no_refine: bool
    no_filter: bool
    fast_ik: bool
    shape_plane: str
    center: list[float] | None
    err_mean_mm: float
    ok_ratio: float
    max_droll: float
    max_dpitch: float
    max_dyaw: float
    xflip: int


def _worker_init(cfg_dict: dict) -> None:
    """子进程初始化：各 worker 独立加载可行域与 MuJoCo 上下文。"""
    global _WORKER_STATE
    cfg = WorkerConfig(**cfg_dict)
    ee, joints, ws = _load_reachable_workspace(True)
    _WORKER_STATE = {
        "cfg": cfg,
        "ee": ee,
        "joints": joints,
        "ws": ws,
        "thresholds": FilterThresholds(
            err_mean_mm=cfg.err_mean_mm,
            ok_ratio=cfg.ok_ratio,
            max_droll=cfg.max_droll,
            max_dpitch=cfg.max_dpitch,
            max_dyaw=cfg.max_dyaw,
            xflip=cfg.xflip,
        ),
    }


def _worker_run_job(task: dict) -> dict:
    """在子进程中执行单个 job（IK + 后处理）。"""
    cfg: WorkerConfig = _WORKER_STATE["cfg"]
    ee = _WORKER_STATE["ee"]
    joints = _WORKER_STATE["joints"]
    ws = _WORKER_STATE["ws"]
    thresholds: FilterThresholds = _WORKER_STATE["thresholds"]

    job_id = task["job_id"]
    shape = task["shape"]
    seed = task["seed"]
    raw_path = task["raw_path"]
    resume_post_only = task["resume_post_only"]
    t0 = time.perf_counter()
    ik_sec = 0.0

    try:
        if not resume_post_only:
            rng = np.random.default_rng(seed)
            center_arg = (
                np.array(cfg.center, dtype=np.float64)
                if cfg.center is not None else None
            )
            traj = _try_generate_concentric_demo(
                shape,
                cfg.layers,
                cfg.fps,
                cfg.frames_per_ring,
                ELBOW_BRANCH_OUTWARD,
                ee, joints, ws,
                rng,
                refine=not cfg.no_refine,
                center=center_arg,
                theta=None,
                fast_ik=cfg.fast_ik,
                shape_plane=cfg.shape_plane,
            )
            if traj is None:
                raise RuntimeError("IK 失败：中心不可行")
            os.makedirs(os.path.dirname(raw_path) or ".", exist_ok=True)
            save_trajectory_npz(raw_path, traj, seed=seed)
            ik_sec = time.perf_counter() - t0

        post = process_raw_trajectory(
            raw_path,
            batch_root=cfg.batch_root,
            thresholds=thresholds,
            no_filter=cfg.no_filter,
            event_cb=None,
        )
        return {
            "job_id": job_id,
            "shape": shape,
            "seed": seed,
            "status": "ok",
            "ik_sec": ik_sec,
            "resume_post_only": resume_post_only,
            "post": post,
            "raw_path": raw_path,
        }
    except Exception as exc:
        return {
            "job_id": job_id,
            "shape": shape,
            "seed": seed,
            "status": "failed",
            "ik_sec": time.perf_counter() - t0,
            "error": str(exc),
            "raw_path": raw_path,
        }


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
        plane = self.args.shape_plane
        if self.args.center is not None:
            return f"{shape}_{plane}_{format_center_tag(self.args.center)}_seed{seed:03d}"
        return f"{shape}_{plane}_seed{seed:03d}"

    def _raw_path(self, shape: str, seed: int) -> str:
        fname = f"seed{seed:03d}_L{self.args.layers}_F{self.args.frames_per_ring}.npz"
        return os.path.join(
            self.dirs["raw"], shape, self.args.shape_plane, fname,
        )

    def _job_status(self, job_id: str) -> str | None:
        rec = self._manifest.get(job_id)
        return rec.get("status") if rec else None

    def _should_skip(self, job_id: str) -> bool:
        if not self.args.resume:
            return False
        if self._job_status(job_id) != "post_done":
            return False
        if self.args.no_filter:
            rec = self._manifest.get(job_id, {})
            return int(rec.get("n_pass", 0)) > 0
        return True

    def _build_jobs(self) -> list[tuple[str, int]]:
        shapes = [resolve_shape_name(s) for s in (self.args.shapes or BATCH_SHAPES)]
        jobs = [(shape, seed) for shape in shapes for seed in range(
            self.args.seed_start, self.args.seed_start + self.args.seed_count,
        )]
        if self.args.max_jobs is not None:
            jobs = jobs[: self.args.max_jobs]
        return jobs

    def _worker_config_dict(self) -> dict:
        center = None
        if self.args.center is not None:
            center = self.args.center.tolist()
        return dataclasses.asdict(WorkerConfig(
            batch_root=self.root,
            layers=self.args.layers,
            frames_per_ring=self.args.frames_per_ring,
            fps=self.args.fps,
            no_refine=self.args.no_refine,
            no_filter=self.args.no_filter,
            fast_ik=self.args.fast_ik,
            shape_plane=self.args.shape_plane,
            center=center,
            err_mean_mm=self.args.err_mean_mm,
            ok_ratio=self.args.ok_ratio,
            max_droll=self.args.max_droll,
            max_dpitch=self.args.max_dpitch,
            max_dyaw=self.args.max_dyaw,
            xflip=self.args.max_xflip,
        ))

    def _resume_post_only(self, job_id: str, raw_path: str) -> bool:
        rec = self._manifest.get(job_id, {})
        return (
            self.args.resume
            and os.path.isfile(raw_path)
            and (
                self._job_status(job_id) == "ik_done"
                or (
                    self._job_status(job_id) == "post_done"
                    and self.args.no_filter
                    and int(rec.get("n_pass", 0)) == 0
                )
            )
        )

    def _process_job_result(
        self,
        ji: int,
        total: int,
        result: dict,
        stats: dict[str, int],
    ) -> None:
        job_id = result["job_id"]
        if result["status"] == "failed":
            stats["failed"] += 1
            self._update_manifest(
                job_id,
                status="failed",
                error=result.get("error", "unknown"),
                finished_at=self._utcnow(),
            )
            self._emit_event("ik_failed", {
                "job_id": job_id,
                "error": result.get("error", "unknown"),
            })
            self.logger.error(
                "[%d/%d] %s 失败: %s",
                ji + 1, total, job_id, result.get("error", "unknown"),
            )
            return

        post = result["post"]
        ik_sec = float(result.get("ik_sec", 0.0))
        raw_path = result["raw_path"]
        if not result.get("resume_post_only"):
            self._emit_event("ik_done", {
                "job_id": job_id,
                "raw_path": raw_path,
                "sec": ik_sec,
            })
            self._update_manifest(
                job_id,
                status="ik_done",
                raw_path=raw_path,
                ik_sec=ik_sec,
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
        status = "PASS" if post["n_pass"] else "REJECT"
        self.logger.info(
            "[%d/%d] %s IK %.1fs | %s | 累计 pass=%d reject=%d failed=%d",
            ji + 1, total, job_id, ik_sec, status,
            stats["clips_pass"], stats["clips_reject"], stats["failed"],
        )

    def _run_serial(self, jobs: list[tuple[str, int]], thresholds: FilterThresholds,
                    ee, joints, ws) -> dict[str, int]:
        stats = {"done": 0, "failed": 0, "skipped": 0, "clips_pass": 0, "clips_reject": 0}
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
            resume_post_only = self._resume_post_only(job_id, raw_path)

            try:
                if resume_post_only:
                    self.logger.info(
                        "[%d/%d] %s 续跑后处理（跳过 IK）", ji + 1, len(jobs), job_id,
                    )
                    ik_sec = 0.0
                    result = {
                        "job_id": job_id,
                        "shape": shape,
                        "seed": seed,
                        "status": "ok",
                        "ik_sec": ik_sec,
                        "resume_post_only": True,
                        "raw_path": raw_path,
                        "post": process_raw_trajectory(
                            raw_path,
                            batch_root=self.root,
                            thresholds=thresholds,
                            no_filter=self.args.no_filter,
                            event_cb=lambda evt, payload, jid=job_id: self._emit_event(
                                evt, {"job_id": jid, **payload},
                            ),
                        ),
                    }
                else:
                    self._emit_event("ik_start", {
                        "job_id": job_id, "shape": shape, "seed": seed,
                    })
                    rng = np.random.default_rng(seed)
                    center_arg = (
                        self.args.center.copy()
                        if self.args.center is not None else None
                    )
                    traj = _try_generate_concentric_demo(
                        shape,
                        self.args.layers,
                        self.args.fps,
                        self.args.frames_per_ring,
                        ELBOW_BRANCH_OUTWARD,
                        ee, joints, ws,
                        rng,
                        refine=not self.args.no_refine,
                        center=center_arg,
                        theta=None,
                        fast_ik=self.args.fast_ik,
                        shape_plane=self.args.shape_plane,
                    )
                    if traj is None:
                        raise RuntimeError("IK 失败：中心不可行")
                    os.makedirs(os.path.dirname(raw_path) or ".", exist_ok=True)
                    save_trajectory_npz(raw_path, traj, seed=seed)
                    ik_sec = time.perf_counter() - t0
                    self._emit_event("ik_done", {
                        "job_id": job_id, "raw_path": raw_path, "sec": ik_sec,
                    })
                    self._update_manifest(
                        job_id, status="ik_done", raw_path=raw_path, ik_sec=ik_sec,
                    )
                    result = {
                        "job_id": job_id,
                        "shape": shape,
                        "seed": seed,
                        "status": "ok",
                        "ik_sec": ik_sec,
                        "resume_post_only": False,
                        "raw_path": raw_path,
                        "post": process_raw_trajectory(
                            raw_path,
                            batch_root=self.root,
                            thresholds=thresholds,
                            no_filter=self.args.no_filter,
                            event_cb=lambda evt, payload, jid=job_id: self._emit_event(
                                evt, {"job_id": jid, **payload},
                            ),
                        ),
                    }
                self._process_job_result(ji, len(jobs), result, stats)
            except Exception as e:
                stats["failed"] += 1
                self._update_manifest(
                    job_id,
                    status="failed",
                    error=str(e),
                    finished_at=self._utcnow(),
                )
                self._emit_event("ik_failed", {"job_id": job_id, "error": str(e)})
                self.logger.exception(
                    "[%d/%d] %s 失败: %s", ji + 1, len(jobs), job_id, e,
                )
            self._current_job = None
        return stats

    def _run_parallel(self, jobs: list[tuple[str, int]]) -> dict[str, int]:
        stats = {"done": 0, "failed": 0, "skipped": 0, "clips_pass": 0, "clips_reject": 0}
        pending: list[tuple[int, dict]] = []
        for ji, (shape, seed) in enumerate(jobs):
            job_id = self._job_id(shape, seed)
            raw_path = self._raw_path(shape, seed)
            if self._should_skip(job_id):
                stats["skipped"] += 1
                self.logger.info("[%d/%d] 跳过已完成 %s", ji + 1, len(jobs), job_id)
                continue
            self._update_manifest(
                job_id,
                shape=shape,
                seed=seed,
                status="queued",
                started_at=self._utcnow(),
            )
            pending.append((ji, {
                "job_index": ji,
                "job_id": job_id,
                "shape": shape,
                "seed": seed,
                "raw_path": raw_path,
                "resume_post_only": self._resume_post_only(job_id, raw_path),
            }))

        if not pending:
            return stats

        workers = min(self.args.workers, len(pending))
        self.logger.info("并行执行 %d 个待处理 job，workers=%d", len(pending), workers)
        ctx = mp.get_context("spawn")
        with ctx.Pool(
            processes=workers,
            initializer=_worker_init,
            initargs=(self._worker_config_dict(),),
        ) as pool:
            tasks = [task for _, task in pending]
            index_by_job = {task["job_id"]: ji for ji, task in pending}
            for result in pool.imap_unordered(_worker_run_job, tasks):
                job_id = result["job_id"]
                ji = index_by_job[job_id]
                self._current_job = job_id
                if result["status"] == "ok" and not result.get("resume_post_only"):
                    self._emit_event("ik_start", {
                        "job_id": job_id,
                        "shape": result["shape"],
                        "seed": result["seed"],
                    })
                self._process_job_result(ji, len(jobs), result, stats)
                self._current_job = None
        return stats

    def run(self) -> None:
        if not os.path.isfile(WS_CACHE_PATH):
            raise RuntimeError(f"缺少可行域缓存 {WS_CACHE_PATH}，请先运行 g1_right_hand_workspace.py")

        self._acquire_lock()
        jobs = self._build_jobs()
        center_desc = (
            self.args.center.tolist() if self.args.center is not None else "随机采样"
        )
        fast_note = " | fast-ik" if self.args.fast_ik else ""
        worker_note = (
            f" | workers={self.args.workers}"
            if self.args.workers > 1 else " | workers=1(串行)"
        )
        self.logger.info(
            "开始批量任务: %d jobs | plane=%s | center=%s layers=%d F=%d fps=%.0f%s%s%s",
            len(jobs), self.args.shape_plane, center_desc, self.args.layers,
            self.args.frames_per_ring, self.args.fps,
            " | 筛选已关闭(--no-filter)" if self.args.no_filter else "",
            worker_note,
            fast_note,
        )

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
            if self.args.workers > 1:
                stats = self._run_parallel(jobs)
            else:
                ee, joints, ws = _load_reachable_workspace(True)
                stats = self._run_serial(jobs, thresholds, ee, joints, ws)

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
    ap.add_argument("--center", nargs=3, type=float, default=None,
                    help="固定圆心；不指定则从可达域随机采样")
    ap.add_argument("--layers", type=int, default=1,
                    help="每条轨迹圈数（默认 1 圈 = 300 帧）")
    ap.add_argument("--frames-per-ring", type=int, default=300)
    ap.add_argument("--fps", type=float, default=30.0,
                    help="轨迹采样/回放帧率（帧数仍由 --frames-per-ring 决定）")
    ap.add_argument("--seed-start", type=int, default=0)
    ap.add_argument("--seed-count", type=int, default=100,
                    help="每种形状随机采样次数")
    ap.add_argument("--max-jobs", type=int, default=None, help="调试用，限制 job 数")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--force", action="store_true", help="覆盖 runner.lock")
    ap.add_argument("--no-refine", action="store_true")
    ap.add_argument(
        "--workers", type=int, default=1,
        help="并行 worker 数（MuJoCo IK 为 CPU 任务，建议设为 CPU 核数-1）",
    )
    ap.add_argument(
        "--fast-ik", action="store_true",
        help="批量快速 IK：减少 SQP 迭代与种子数（略降精度，显著提速）",
    )
    ap.add_argument(
        "--shape-plane", type=str, default=DEFAULT_SHAPE_PLANE,
        help="形状所在平面: yz(默认,垂直于x) | xy(水平) | xz",
    )
    ap.add_argument("--preview", action="store_true", help="每个 job 通过后生成 PNG 预览")
    ap.add_argument("--no-filter", action="store_true",
                    help="跳过后处理质量筛选，全部生成 obs NPZ（仍会记录 metrics）")
    ap.add_argument("--err-mean-mm", type=float, default=15.0)
    ap.add_argument("--ok-ratio", type=float, default=0.85)
    ap.add_argument("--max-droll", type=float, default=0.12)
    ap.add_argument("--max-dpitch", type=float, default=0.12)
    ap.add_argument("--max-dyaw", type=float, default=0.06)
    ap.add_argument("--max-xflip", type=int, default=40)
    args = ap.parse_args()
    args.center = (
        np.array(args.center, dtype=np.float64) if args.center is not None else None
    )
    args.shape_plane = resolve_shape_plane(args.shape_plane)

    BatchRunner(args).run()


if __name__ == "__main__":
    main()
