"""
G1 右手轨迹逆运动学（成熟方法实现，独立于 g1_multi_shape_traj_gen）
====================================================================

问题
----
4-DOF 臂跟踪 3D 末端路点，需同时满足：
  1) 解支一致：整条轨迹肘部保持同一折向（内翻/外翻），不跳支；
  2) 高精度：末端尽量贴合规划路点。

方法选型（文献/工业实践）
------------------------
不采用「可行域离散跳解 + 事后抛光」（易在冗余维上跳支）。

采用两阶段成熟管线：

**阶段 A — TRAC-IK 风格多种子 SQP（逐路点）**
  - 对每个路点从多个初值（q_{i-1}、可行域近邻、分支参考姿态）启动 L-BFGS-B
  - 目标：min ||FK(q)-p*||²，约束关节限位 + 肘部折向
  - 在收敛解中选 ||q-q_{i-1}|| 最小者（TRAC-IK SolveType=Distance）
  - 避免纯 Jacobian 积分在闭合轨迹上的漂移

**阶段 B — 任务优先级 DLS 精化（可选，Gauss-Seidel 扫描）**
  - Buss (2009) DLS + 零空间姿态维持（Pink / Stack-of-Tasks）
  - 在阶段 A 解的基础上做 2~3 轮序列精化

**人类化运动约束**
  - 优先使用肘关节，其次肩 roll/pitch，最后肩 yaw（加权 Δq 代价）
  - 肩 roll/pitch/yaw 每帧严格限速，禁止突变

参考文献
--------
- Buss, "Introduction to Inverse Kinematics" (DLS / 零空间)
- Chiaverini et al., "Singularity-robust task-priority redundancy resolution"
- Caron, Pink IK (Stack-of-Tasks)
- TRAC-IK: seed 连续性 + 距离最优
- arXiv:2604.02021 TP-DLS 轨迹跟踪层

用法
----
  python g1_trajectory_ik.py --shape circle --center 0.27 -0.25 1.01 --preview --save
  python g1_trajectory_ik.py --shape circle --layers 4 --ws-cache --elbow-branch outward --save
  python g1_trajectory_ik.py --shape circle --layers 6 --num-trajs 5 --seed 42 --ws-cache --save

在其它脚本中：
  from g1_trajectory_ik import TrajectoryIKSolver, solve_trajectory_ik
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime
import os
import time
from typing import Sequence

import mujoco
import numpy as np
from scipy.optimize import least_squares, minimize

from g1_arm_posture import (
    ELBOW_BRANCH_AUTO,
    ELBOW_BRANCH_INWARD,
    ELBOW_BRANCH_OUTWARD,
    NATURAL_Q4_REF,
    elbow_branch_of_q4,
    is_allowed_q4,
    pick_locked_branch_from_candidates,
    resolve_branch_mode,
)
from g1_multi_shapes import make_shape_waypoints
from g1_right_hand_workspace import (
    ACTIVE_JOINTS,
    LOCKED_JOINTS,
    ROBOT_XML,
    WS_CACHE_PATH,
    _OBSTACLE_BODY_NAMES_MINIMAL,
    _RIGHT_ARM_COLLISION_BODY_NAMES,
    _build_addr_table,
    _build_collision_id_sets,
    _has_penetration,
    _reset_to_home,
    build_reachable_workspace,
    load_workspace_cache,
)

os.makedirs("recordings", exist_ok=True)

EE_BODY = "right_wrist_yaw_link"
MODEL_PATH = "g1/scene_g1_draggable.xml"

# ── 默认超参 ──
POS_TOL = 0.0015          # 1.5 mm
MAX_DLS_ITERS = 100
MAX_DQ = 0.08
LAMBDA0 = 1e-3
LAMBDA_MANIP = 5e-2       # Chiaverini 自适应阻尼
W_NULL_POSTURE = 2.5      # 零空间姿态权重
W_NULL_CONT = 1.2         # 零空间连续性权重
REFINE_SMOOTH = 8.0
REFINE_POSTURE = 1.5
REFINE_BRANCH_PENALTY = 50.0
REFINE_COL_PENALTY = 200.0
ERR_THRESH = 0.008        # 8 mm 可行性阈值
WS_NEIGHBOR_K = 6
BRANCH_COST = 80.0
DIST_BLEND = 0.55         # TRAC-IK Distance：连续性权重（防解支跳变）
ROLL_BLEND = 0.35
MAX_ROLL_STEP = 0.10
SQP_MAXITER = 45

# q4 顺序：pitch(0), roll(1), yaw(2), elbow(3)
IDX_PITCH, IDX_ROLL, IDX_YAW, IDX_ELBOW = 0, 1, 2, 3
SHOULDER_IDXS = (IDX_PITCH, IDX_ROLL, IDX_YAW)

# 加权 Δq：值越大越不愿动该关节（人类：肘优先，yaw 最后）
W_DQ_ELBOW = 0.35
W_DQ_ROLL = 2.8
W_DQ_PITCH = 2.8
W_DQ_YAW = 6.5
HUMAN_MOTION_BLEND = 0.0   # 人类化由逐步硬限位实现，不用软代价
ENABLE_GS_REFINE = False   # GS 精化会破坏肩逐步限速

# 每帧最大关节步长 (rad)，肩三轴严格限速
MAX_STEP_ELBOW = 0.13
MAX_STEP_SHOULDER = 0.038   # pitch / roll 每帧上限 (rad)
MAX_STEP_YAW = 0.018        # yaw 最慢
SHOULDER_DLS_SCALE = 0.30  # DLS 主任务中肩关节增量缩放


@dataclasses.dataclass
class IkResult:
    qpos: np.ndarray
    qvel: np.ndarray
    timestamps: np.ndarray
    errors: np.ndarray
    col_ratio: float
    locked_branch: str | None
    branch_violations: int


class G1RightArmModel:
    """MuJoCo 正运动学 + 位置雅可比。"""

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData | None = None):
        self.model = model
        self.data = data if data is not None else mujoco.MjData(model)
        self.ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, EE_BODY)
        self.addr = _build_addr_table(model, ACTIVE_JOINTS + list(LOCKED_JOINTS.keys()))
        self.qpos_addrs = [self.addr[n][0] for n in ACTIVE_JOINTS]
        self.dof_addrs = [self.addr[n][1] for n in ACTIVE_JOINTS]
        self.ranges = np.array([self.addr[n][2] for n in ACTIVE_JOINTS], dtype=np.float64)
        self.lock_addr = _build_addr_table(model, list(LOCKED_JOINTS.keys()))
        self._jacp = np.zeros((3, model.nv))

    def clip_q4(self, q4: np.ndarray) -> np.ndarray:
        q4 = np.asarray(q4, dtype=np.float64).reshape(4)
        lo = self.ranges[:, 0]
        hi = self.ranges[:, 1]
        return np.clip(q4, lo, hi)

    def set_q4(self, q4: np.ndarray) -> None:
        q4 = self.clip_q4(q4)
        for i, a in enumerate(self.qpos_addrs):
            self.data.qpos[a] = float(q4[i])
        for name, val in LOCKED_JOINTS.items():
            self.data.qpos[self.lock_addr[name][0]] = val
            self.data.qvel[self.lock_addr[name][1]] = 0.0

    def forward(self) -> np.ndarray:
        mujoco.mj_forward(self.model, self.data)
        return self.data.xpos[self.ee_id].copy()

    def position_jacobian(self) -> np.ndarray:
        mujoco.mj_jacBody(self.model, self.data, self._jacp, None, self.ee_id)
        return self._jacp[:, self.dof_addrs].copy()

    def qpos_full(self) -> np.ndarray:
        return self.data.qpos.copy()

    def pos_error(self, target: np.ndarray) -> tuple[float, np.ndarray]:
        target = np.asarray(target, dtype=np.float64).reshape(3)
        ee = self.forward()
        err_vec = target - ee
        return float(np.linalg.norm(err_vec)), err_vec


def _adaptive_lambda(J: np.ndarray, lam0: float = LAMBDA0, lam1: float = LAMBDA_MANIP) -> float:
    """Chiaverini 可操作度自适应阻尼。"""
    JJt = J @ J.T
    w = float(np.sqrt(max(0.0, np.linalg.det(JJt))))
    return lam0 + lam1 / (w + 1e-4)


def _dls_step(J: np.ndarray, err_vec: np.ndarray, lam: float) -> np.ndarray:
    """阻尼最小二乘关节增量。"""
    JJt = J @ J.T + (lam ** 2) * np.eye(3)
    dq = J.T @ np.linalg.solve(JJt, err_vec)
    return np.clip(dq, -MAX_DQ, MAX_DQ)


def _null_projector(J: np.ndarray, lam: float) -> np.ndarray:
    """阻尼伪逆对应的零空间投影 N = I - J⁺J。"""
    n = J.shape[1]
    JJt = J @ J.T + (lam ** 2) * np.eye(3)
    J_pinv = J.T @ np.linalg.solve(JJt, np.eye(3))
    return np.eye(n) - J_pinv @ J


def _weighted_dq_cost(dq: np.ndarray) -> float:
    """人类化加权关节位移代价（越小越像人）。"""
    dq = np.asarray(dq, dtype=np.float64).reshape(4)
    return float(
        W_DQ_PITCH * dq[IDX_PITCH] ** 2
        + W_DQ_ROLL * dq[IDX_ROLL] ** 2
        + W_DQ_YAW * dq[IDX_YAW] ** 2
        + W_DQ_ELBOW * dq[IDX_ELBOW] ** 2
    )


def _max_shoulder_step(q_new: np.ndarray, q_prev: np.ndarray) -> float:
    dq = np.abs(np.asarray(q_new, dtype=np.float64) - np.asarray(q_prev, dtype=np.float64))
    return float(np.max(dq[list(SHOULDER_IDXS)]))


def _human_step_bounds(
    q_prev: np.ndarray | None,
    lo: np.ndarray,
    hi: np.ndarray,
) -> list[tuple[float, float]]:
    """逐帧动态限位：优化在「人类可接受的小步」框内求末端最优。"""
    if q_prev is None:
        return list(zip(lo.tolist(), hi.tolist()))
    q_prev = np.asarray(q_prev, dtype=np.float64).reshape(4)
    d_el = MAX_STEP_ELBOW
    d_sh = MAX_STEP_SHOULDER
    d_yaw = MAX_STEP_YAW
    lo_d = lo.copy()
    hi_d = hi.copy()
    lo_d[IDX_ELBOW] = max(lo[IDX_ELBOW], q_prev[IDX_ELBOW] - d_el)
    hi_d[IDX_ELBOW] = min(hi[IDX_ELBOW], q_prev[IDX_ELBOW] + d_el)
    lo_d[IDX_PITCH] = max(lo[IDX_PITCH], q_prev[IDX_PITCH] - d_sh)
    hi_d[IDX_PITCH] = min(hi[IDX_PITCH], q_prev[IDX_PITCH] + d_sh)
    lo_d[IDX_ROLL] = max(lo[IDX_ROLL], q_prev[IDX_ROLL] - d_sh)
    hi_d[IDX_ROLL] = min(hi[IDX_ROLL], q_prev[IDX_ROLL] + d_sh)
    lo_d[IDX_YAW] = max(lo[IDX_YAW], q_prev[IDX_YAW] - d_yaw)
    hi_d[IDX_YAW] = min(hi[IDX_YAW], q_prev[IDX_YAW] + d_yaw)
    return list(zip(lo_d.tolist(), hi_d.tolist()))


def _project_human_step(q_opt: np.ndarray, q_prev: np.ndarray, clip_fn) -> np.ndarray:
    """肩三轴硬限速投影；肘保留优化结果并限幅。"""
    q_opt = np.asarray(q_opt, dtype=np.float64).reshape(4)
    q_prev = np.asarray(q_prev, dtype=np.float64).reshape(4)
    q_out = q_prev.copy()
    q_out[IDX_PITCH] = np.clip(
        q_opt[IDX_PITCH], q_prev[IDX_PITCH] - MAX_STEP_SHOULDER, q_prev[IDX_PITCH] + MAX_STEP_SHOULDER,
    )
    q_out[IDX_ROLL] = np.clip(
        q_opt[IDX_ROLL], q_prev[IDX_ROLL] - MAX_STEP_SHOULDER, q_prev[IDX_ROLL] + MAX_STEP_SHOULDER,
    )
    q_out[IDX_YAW] = np.clip(
        q_opt[IDX_YAW], q_prev[IDX_YAW] - MAX_STEP_YAW, q_prev[IDX_YAW] + MAX_STEP_YAW,
    )
    q_out[IDX_ELBOW] = np.clip(
        q_opt[IDX_ELBOW], q_prev[IDX_ELBOW] - MAX_STEP_ELBOW, q_prev[IDX_ELBOW] + MAX_STEP_ELBOW,
    )
    return clip_fn(q_out)


def _apply_dq_human_limits(dq: np.ndarray) -> np.ndarray:
    """DLS 增量按关节优先级限幅。"""
    dq = np.asarray(dq, dtype=np.float64).reshape(4).copy()
    step = {IDX_PITCH: MAX_STEP_SHOULDER, IDX_ROLL: MAX_STEP_SHOULDER, IDX_YAW: MAX_STEP_YAW}
    for j in SHOULDER_IDXS:
        dq[j] *= SHOULDER_DLS_SCALE
        dq[j] = np.clip(dq[j], -step[j], step[j])
    dq[IDX_ELBOW] = np.clip(dq[IDX_ELBOW], -MAX_STEP_ELBOW, MAX_STEP_ELBOW)
    return dq


def _elbow_compensate(
    arm: G1RightArmModel,
    target: np.ndarray,
    q_start: np.ndarray,
    q_prev: np.ndarray,
    locked_branch: str | None,
    arm_ids,
    obstacle_ids,
    tol: float = POS_TOL,
    max_iters: int = 40,
) -> np.ndarray:
    """
    肩三轴已限速后，仅用肘关节微调末端（人类：先动肘）。
    肩关节相对 q_prev 保持硬约束。
    """
    q = arm.clip_q4(q_start)
    shoulder_fix = q[list(SHOULDER_IDXS)].copy()
    target = np.asarray(target, dtype=np.float64).reshape(3)
    for _ in range(max_iters):
        arm.set_q4(q)
        err_norm, err_vec = arm.pos_error(target)
        if err_norm <= tol:
            break
        J = arm.position_jacobian()
        col = J[:, IDX_ELBOW]
        denom = float(col @ col) + LAMBDA0 ** 2
        dq_el = float(col @ err_vec / denom)
        q_try = q.copy()
        q_try[IDX_ELBOW] = np.clip(
            q[IDX_ELBOW] + np.clip(dq_el, -MAX_STEP_ELBOW, MAX_STEP_ELBOW),
            q_prev[IDX_ELBOW] - MAX_STEP_ELBOW,
            q_prev[IDX_ELBOW] + MAX_STEP_ELBOW,
        )
        q_try[list(SHOULDER_IDXS)] = shoulder_fix
        q_try = arm.clip_q4(q_try)
        arm.set_q4(q_try)
        if locked_branch and not is_allowed_q4(
            q_try, locked_branch, arm.model, arm.data, arm.addr,
        ):
            break
        if _has_penetration(arm.model, arm.data, arm_ids, obstacle_ids):
            break
        err_try, _ = arm.pos_error(target)
        if err_try >= err_norm:
            break
        q = q_try
    return q


def _branch_reference_q4(branch: str | None) -> np.ndarray:
    """分支参考姿态（零空间吸引目标）。"""
    q = NATURAL_Q4_REF.copy()
    if branch == ELBOW_BRANCH_OUTWARD:
        q[1] = min(q[1], -0.55)
        q[3] = np.clip(q[3], -0.35, 0.85)
    elif branch == ELBOW_BRANCH_INWARD:
        q[1] = max(q[1], -0.25)
        q[3] = np.clip(q[3], -0.35, 1.0)
    return q


class TracIkStyleSolver:
    """
    TRAC-IK 思路：多初值非线性优化 + 在可行解中选取最接近上一帧的配置。
    参考 TRACLabs TRAC-IK (KDL Newton + SQP 并行) 与 MoveIt kinematics。
    """

    def __init__(self, arm: G1RightArmModel, arm_ids, obstacle_ids):
        self.arm = arm
        self.arm_ids = arm_ids
        self.obstacle_ids = obstacle_ids
        self._lo = arm.ranges[:, 0].copy()
        self._hi = arm.ranges[:, 1].copy()

    def _optimize_from_seed(
        self,
        target: np.ndarray,
        q_seed: np.ndarray,
        locked_branch: str | None,
        q_prev: np.ndarray | None = None,
    ) -> tuple[np.ndarray, float]:
        target = np.asarray(target, dtype=np.float64).reshape(3)
        q_seed = self.arm.clip_q4(q_seed)

        def objective(qv: np.ndarray) -> float:
            qv = self.arm.clip_q4(qv)
            self.arm.set_q4(qv)
            pos_cost = float(np.sum((self.arm.forward() - target) ** 2))
            if q_prev is not None:
                pos_cost += HUMAN_MOTION_BLEND * _weighted_dq_cost(qv - q_prev)
            if locked_branch is not None and not is_allowed_q4(
                qv, locked_branch, self.arm.model, self.arm.data, self.arm.addr,
            ):
                pos_cost += BRANCH_COST
            if _has_penetration(
                self.arm.model, self.arm.data, self.arm_ids, self.obstacle_ids,
            ):
                pos_cost += REFINE_COL_PENALTY
            return pos_cost

        if q_prev is not None:
            bounds = _human_step_bounds(q_prev, self._lo, self._hi)
        else:
            bounds = list(zip(self._lo.tolist(), self._hi.tolist()))
        res = minimize(
            objective, q_seed, method="L-BFGS-B",
            bounds=bounds, options={"maxiter": SQP_MAXITER, "ftol": 1e-9},
        )
        q_best = self.arm.clip_q4(res.x)
        self.arm.set_q4(q_best)
        err = float(np.linalg.norm(self.arm.forward() - target))
        return q_best, err

    def _workspace_seeds(
        self,
        target: np.ndarray,
        ee_cloud: np.ndarray,
        joint_configs: np.ndarray,
        locked_branch: str | None,
        k: int = WS_NEIGHBOR_K,
    ) -> list[np.ndarray]:
        dist = np.linalg.norm(ee_cloud - target, axis=1)
        order = np.argsort(dist)[:max(k, 24)]
        seeds = []
        for c in order:
            q4 = joint_configs[c]
            if locked_branch and not is_allowed_q4(
                q4, locked_branch, self.arm.model, self.arm.data, self.arm.addr,
            ):
                continue
            seeds.append(q4.copy())
            if len(seeds) >= k:
                break
        return seeds

    def solve_one(
        self,
        target: np.ndarray,
        q_prev: np.ndarray | None,
        q_ref: np.ndarray,
        ee_cloud: np.ndarray,
        joint_configs: np.ndarray,
        locked_branch: str | None,
        tol: float = POS_TOL,
    ) -> tuple[np.ndarray, float, bool]:
        seeds: list[np.ndarray] = []
        if q_prev is not None:
            seeds.append(self.arm.clip_q4(q_prev))
            # 人类化逐步限位下，只在 q_{i-1} 邻域内搜索；多种子易选到不可达支路
        else:
            seeds.extend(self._workspace_seeds(
                target, ee_cloud, joint_configs, locked_branch, k=WS_NEIGHBOR_K,
            ))
            if not seeds:
                seeds.append(self.arm.clip_q4(q_ref))
        uniq: list[np.ndarray] = []
        for s in seeds:
            if not any(np.allclose(s, u, atol=1e-4) for u in uniq):
                uniq.append(s)

        candidates: list[tuple[float, float, float, float, np.ndarray]] = []
        for s in uniq:
            q_opt, err = self._optimize_from_seed(
                target, s, locked_branch, q_prev=q_prev,
            )
            if q_prev is not None:
                dq = q_opt - q_prev
                d_w = float(np.sqrt(_weighted_dq_cost(dq)))
                d_roll = abs(float(dq[IDX_ROLL]))
                d_sh = _max_shoulder_step(q_opt, q_prev)
            else:
                d_w, d_roll, d_sh = 0.0, 0.0, 0.0
            candidates.append((err, d_w, d_roll, d_sh, q_opt))

        # 逐步限位不可达时，回退到全空间 IK，但仍优先选人类化候选
        if q_prev is not None and candidates:
            best_bounded = min(candidates, key=lambda x: x[0])
            if best_bounded[0] > POS_TOL * 4:
                fallback_seeds = [self.arm.clip_q4(q_prev)]
                fallback_seeds.extend(self._workspace_seeds(
                    target, ee_cloud, joint_configs, locked_branch, k=WS_NEIGHBOR_K,
                ))
                fb_uniq: list[np.ndarray] = []
                for s in fallback_seeds:
                    if not any(np.allclose(s, u, atol=1e-4) for u in fb_uniq):
                        fb_uniq.append(s)
                for s in fb_uniq:
                    q_opt, err = self._optimize_from_seed(
                        target, s, locked_branch, q_prev=None,
                    )
                    dq = q_opt - q_prev
                    d_w = float(np.sqrt(_weighted_dq_cost(dq)))
                    d_roll = abs(float(dq[IDX_ROLL]))
                    d_sh = _max_shoulder_step(q_opt, q_prev)
                    candidates.append((err, d_w, d_roll, d_sh, q_opt))

        if not candidates:
            raise RuntimeError("无可用 IK 初值")

        good = [c for c in candidates if c[0] <= tol]
        pool = good if good else candidates
        if q_prev is not None:
            tight = [c for c in pool if c[3] <= MAX_STEP_SHOULDER + 1e-5]
            if tight:
                pool = tight
            tight_roll = [c for c in pool if c[2] <= MAX_ROLL_STEP]
            if tight_roll:
                pool = tight_roll
        pool.sort(key=lambda x: (
            x[0] + DIST_BLEND * x[1] + ROLL_BLEND * x[2], x[1], x[2],
        ))
        err, _, _, _, q_best = pool[0]
        self.arm.set_q4(q_best)
        err = float(np.linalg.norm(self.arm.forward() - target))
        return q_best, err, err <= tol


class TaskPriorityDlsIk:
    """
    单点 TP-DLS：主任务位置 + 零空间姿态/连续性。
    等价于工业界「seed 连续 + 冗余解析」的标准做法。
    """

    def __init__(self, arm: G1RightArmModel):
        self.arm = arm

    def solve_one(
        self,
        target: np.ndarray,
        q_seed: np.ndarray,
        q_ref: np.ndarray,
        q_prev: np.ndarray | None = None,
        locked_branch: str | None = None,
        max_iter: int = MAX_DLS_ITERS,
        tol: float = POS_TOL,
    ) -> tuple[np.ndarray, float, bool]:
        q = self.arm.clip_q4(q_seed)
        best_q = q.copy()
        best_err = float("inf")

        for _ in range(max_iter):
            self.arm.set_q4(q)
            err_norm, err_vec = self.arm.pos_error(target)
            if err_norm < best_err:
                best_err = err_norm
                best_q = q.copy()
            if err_norm < tol:
                return best_q, err_norm, True

            J = self.arm.position_jacobian()
            lam = _adaptive_lambda(J)
            dq_primary = _dls_step(J, err_vec, lam)
            N = _null_projector(J, lam)

            e_post = q_ref - q
            dq_null = W_NULL_POSTURE * e_post
            if q_prev is not None:
                dq_null += W_NULL_CONT * (q_prev - q)
            dq = dq_primary + N @ dq_null
            dq = _apply_dq_human_limits(dq)
            if q_prev is not None:
                step_bounds = _human_step_bounds(
                    q_prev, self.arm.ranges[:, 0], self.arm.ranges[:, 1],
                )
                q_tgt = self.arm.clip_q4(q + dq)
                for j in range(4):
                    q_tgt[j] = float(np.clip(q_tgt[j], step_bounds[j][0], step_bounds[j][1]))
                q = q_tgt
            else:
                q = self.arm.clip_q4(q + dq)

            if locked_branch is not None and not is_allowed_q4(
                q, locked_branch, self.arm.model, self.arm.data, self.arm.addr,
            ):
                q = best_q.copy()

        self.arm.set_q4(best_q)
        return best_q, best_err, best_err < tol * 3.0


class TrajectoryGaussSeidelRefiner:
    """
    序列 Gauss-Seidel 精化（Kenwright 2012 / 实时角色 IK 常用）。
    多轮扫描路点，每点 TP-DLS warm-start，兼顾精度与速度。
    """

    def __init__(
        self,
        tp_ik: TaskPriorityDlsIk,
        locked_branch: str | None,
        q_ref: np.ndarray,
    ):
        self.tp_ik = tp_ik
        self.locked_branch = locked_branch
        self.q_ref = q_ref

    def refine(
        self,
        waypoints: np.ndarray,
        q_init: np.ndarray,
        closed: bool,
        passes: int = 4,
    ) -> np.ndarray:
        n = len(waypoints)
        q_path = np.asarray(q_init, dtype=np.float64).reshape(n, 4).copy()
        indices = list(range(n))
        if closed and n > 2:
            scan = indices + indices[::-1][1:-1]
        else:
            scan = indices

        for _ in range(passes):
            for i in scan:
                q_prev = q_path[i - 1] if i > 0 else (q_path[-1] if closed else q_path[i])
                q_i, _, _ = self.tp_ik.solve_one(
                    waypoints[i], q_path[i], self.q_ref,
                    q_prev=q_prev,
                    locked_branch=self.locked_branch,
                    max_iter=60,
                    tol=POS_TOL,
                )
                q_path[i] = q_i
        return q_path


class TrajectoryGaussNewtonRefiner:
    """短轨迹（≤80 路点）的全局非线性最小二乘精化。"""

    def __init__(
        self,
        arm: G1RightArmModel,
        arm_ids,
        obstacle_ids,
        locked_branch: str | None,
        q_ref: np.ndarray,
    ):
        self.arm = arm
        self.arm_ids = arm_ids
        self.obstacle_ids = obstacle_ids
        self.locked_branch = locked_branch
        self.q_ref = q_ref

    def refine(
        self,
        waypoints: np.ndarray,
        q_init: np.ndarray,
        closed: bool,
        max_nfev: int = 800,
    ) -> np.ndarray:
        n = len(waypoints)
        lo = self.arm.ranges[:, 0]
        hi = self.arm.ranges[:, 1]
        q_init = np.asarray(q_init, dtype=np.float64).reshape(n, 4)
        x0 = np.stack([self.arm.clip_q4(q_init[i]) for i in range(n)], axis=0).reshape(-1)

        def residuals(x: np.ndarray) -> np.ndarray:
            q_path = x.reshape(n, 4)
            parts = []
            for i in range(n):
                self.arm.set_q4(q_path[i])
                parts.append(self.arm.forward() - waypoints[i])
            for i in range(n - 1):
                parts.append(np.sqrt(REFINE_SMOOTH) * (q_path[i + 1] - q_path[i]))
            if closed and n > 2:
                parts.append(np.sqrt(REFINE_SMOOTH) * (q_path[0] - q_path[-1]))
            parts.append(np.sqrt(REFINE_POSTURE) * (q_path - self.q_ref).reshape(-1))
            return np.concatenate(parts)

        res = least_squares(
            residuals, x0, bounds=(np.tile(lo, n), np.tile(hi, n)),
            method="trf", max_nfev=max_nfev, ftol=1e-8, xtol=1e-8,
        )
        return res.x.reshape(n, 4)


class TrajectoryIKSolver:
    """
    轨迹 IK 求解器：TP-DLS 初解 + 可选全轨迹 Gauss-Newton 精化。
    """

    def __init__(
        self,
        model: mujoco.MjModel | None = None,
        data: mujoco.MjData | None = None,
    ):
        self.model = model or mujoco.MjModel.from_xml_path(ROBOT_XML)
        self.data = data or mujoco.MjData(self.model)
        self.arm = G1RightArmModel(self.model, self.data)
        self.arm_ids, self.obstacle_ids = _build_collision_id_sets(
            self.model, _OBSTACLE_BODY_NAMES_MINIMAL,
            ee_body_names=_RIGHT_ARM_COLLISION_BODY_NAMES,
        )
        self.trac_ik = TracIkStyleSolver(self.arm, self.arm_ids, self.obstacle_ids)
        self.tp_ik = TaskPriorityDlsIk(self.arm)
        self.addr = self.arm.addr

    def _pick_initial_q4(
        self,
        waypoint0: np.ndarray,
        ee_cloud: np.ndarray,
        joint_configs: np.ndarray,
        locked_branch: str | None,
    ) -> np.ndarray:
        dist = np.linalg.norm(ee_cloud - waypoint0, axis=1)
        order = np.argsort(dist)[:64]
        for c in order:
            q4 = joint_configs[c]
            if locked_branch and not is_allowed_q4(
                q4, locked_branch, self.model, self.data, self.addr,
            ):
                continue
            self.arm.set_q4(q4)
            if not _has_penetration(self.model, self.data, self.arm_ids, self.obstacle_ids):
                return q4.copy()
        q4, _, _ = self.trac_ik.solve_one(
            waypoint0, None, _branch_reference_q4(locked_branch),
            ee_cloud, joint_configs, locked_branch, tol=POS_TOL * 4,
        )
        return q4

    def _resolve_locked_branch(
        self,
        waypoint0: np.ndarray,
        ee_cloud: np.ndarray,
        joint_configs: np.ndarray,
        branch_mode: str,
    ) -> str | None:
        if branch_mode in (ELBOW_BRANCH_INWARD, ELBOW_BRANCH_OUTWARD):
            return branch_mode
        dist = np.linalg.norm(ee_cloud - waypoint0, axis=1)
        c0 = np.argsort(dist)[:48]
        return pick_locked_branch_from_candidates(
            joint_configs, c0, branch_mode,
            self.model, self.data, self.addr,
        )

    def solve(
        self,
        waypoints: np.ndarray,
        ee_cloud: np.ndarray | None = None,
        joint_configs: np.ndarray | None = None,
        fps: float = 50.0,
        elbow_branch_mode: str = ELBOW_BRANCH_AUTO,
        locked_branch: str | None = None,
        closed_path: bool = True,
        refine: bool = True,
        show_progress: bool = False,
    ) -> IkResult:
        waypoints = np.asarray(waypoints, dtype=np.float64)
        n = len(waypoints)
        if ee_cloud is None or joint_configs is None:
            raise ValueError("需提供 ee_cloud 与 joint_configs（TRAC-IK 多种子）")
        if locked_branch is None:
            locked_branch = self._resolve_locked_branch(
                waypoints[0], ee_cloud, joint_configs, elbow_branch_mode,
            )
            if locked_branch is None:
                raise RuntimeError("无法确定肘部折向")

        q_ref = _branch_reference_q4(locked_branch)
        if ee_cloud is not None and joint_configs is not None:
            q_prev = self._pick_initial_q4(
                waypoints[0], ee_cloud, joint_configs, locked_branch,
            )
        else:
            q_prev = _branch_reference_q4(locked_branch)

        q_path = np.zeros((n, 4), dtype=np.float64)
        errors = np.zeros(n, dtype=np.float64)
        col_flags = np.zeros(n, dtype=bool)
        branch_viol = 0

        for i in range(n):
            q_i, err_i, _ = self.trac_ik.solve_one(
                waypoints[i],
                q_prev if i > 0 else None,
                q_ref, ee_cloud, joint_configs, locked_branch,
            )
            self.arm.set_q4(q_i)
            col_flags[i] = _has_penetration(
                self.model, self.data, self.arm_ids, self.obstacle_ids,
            )
            if locked_branch and elbow_branch_of_q4(
                q_i, self.model, self.data, self.addr,
            ) != locked_branch:
                branch_viol += 1
            q_path[i] = q_i
            errors[i] = err_i
            q_prev = q_i
            if show_progress and (i + 1) % 50 == 0:
                print(f"    TRAC-IK {i+1}/{n}", flush=True)

        if refine and ENABLE_GS_REFINE and n <= 120:
            if show_progress:
                print("    Gauss-Seidel DLS 精化 ...", flush=True)
            gs = TrajectoryGaussSeidelRefiner(self.tp_ik, locked_branch, q_ref)
            q_path = gs.refine(waypoints, q_path, closed=closed_path, passes=2)
            for i in range(n):
                self.arm.set_q4(q_path[i])
                errors[i], _ = self.arm.pos_error(waypoints[i])

        qpos_list = []
        for i in range(n):
            self.arm.set_q4(q_path[i])
            qpos_list.append(self.arm.qpos_full())

        qpos_arr = np.array(qpos_list, dtype=np.float32)
        qvel_arr, timestamps = _qpos_vel_from_sequence(qpos_arr, fps, self.model.nv)
        return IkResult(
            qpos=qpos_arr,
            qvel=qvel_arr,
            timestamps=timestamps,
            errors=errors.astype(np.float32),
            col_ratio=float(np.mean(col_flags)),
            locked_branch=locked_branch,
            branch_violations=branch_viol,
        )


def _qpos_vel_from_sequence(qpos_arr: np.ndarray, fps: float, nv: int):
    n = len(qpos_arr)
    dt = 1.0 / fps
    qvel = np.zeros((n, nv), dtype=np.float32)
    for i in range(1, n - 1):
        qvel[i, 6:] = (qpos_arr[i + 1, 7:] - qpos_arr[i - 1, 7:]) / (2 * dt)
    if n > 1:
        qvel[0] = qvel[1]
        qvel[n - 1] = qvel[n - 2]
    ts = np.arange(n, dtype=np.float32) * dt
    return qvel, ts


def solve_trajectory_ik(
    waypoints: np.ndarray,
    ee_cloud: np.ndarray,
    joint_configs: np.ndarray,
    fps: float = 50.0,
    elbow_branch_mode: str = ELBOW_BRANCH_AUTO,
    locked_branch: str | None = None,
    closed_path: bool = True,
    refine: bool = True,
    show_progress: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, str | None]:
    """
    与旧 solve_waypoints_ik 兼容的返回签名。
    """
    solver = TrajectoryIKSolver()
    res = solver.solve(
        waypoints, ee_cloud, joint_configs, fps=fps,
        elbow_branch_mode=elbow_branch_mode,
        locked_branch=locked_branch,
        closed_path=closed_path,
        refine=refine,
        show_progress=show_progress,
    )
    return (
        res.qpos, res.qvel, res.timestamps, res.errors,
        res.col_ratio, res.locked_branch,
    )


def evaluate_trajectory(
    qpos: np.ndarray,
    waypoints: np.ndarray,
    locked_branch: str | None = None,
) -> dict:
    """评估末端误差、折向违规、X 抖动。"""
    model = mujoco.MjModel.from_xml_path(ROBOT_XML)
    data = mujoco.MjData(model)
    arm = G1RightArmModel(model, data)
    ee_id = arm.ee_id
    n = min(len(qpos), len(waypoints))
    ee_pts = np.zeros((n, 3))
    rolls = []
    viol = 0
    for i in range(n):
        data.qpos[:] = qpos[i]
        mujoco.mj_forward(model, data)
        ee_pts[i] = data.xpos[ee_id].copy()
        q4 = np.array([data.qpos[arm.qpos_addrs[j]] for j in range(4)])
        rolls.append(q4[1])
        if locked_branch and elbow_branch_of_q4(q4, model, data, arm.addr) != locked_branch:
            viol += 1
    errs = np.linalg.norm(ee_pts - waypoints[:n], axis=1)
    dx = np.diff(ee_pts[:, 0])
    return {
        "err_mean_mm": float(errs.mean() * 1000),
        "err_max_mm": float(errs.max() * 1000),
        "ok_ratio": float(np.mean(errs < ERR_THRESH)),
        "branch_viol": viol,
        "xflip": int(np.sum(dx[1:] * dx[:-1] < 0)),
        "xspan_mm": float((ee_pts[:, 0].max() - ee_pts[:, 0].min()) * 1000),
        "max_droll": float(np.abs(np.diff(rolls)).max()) if n > 1 else 0.0,
    }


# ══════════════════════════════════════════════
# 同心圆 CLI（演示新 IK，不修改旧 traj_gen）
# ══════════════════════════════════════════════

def _load_reachable_workspace(ws_cache: bool):
    if ws_cache and os.path.isfile(WS_CACHE_PATH):
        ee, joints = load_workspace_cache()
    else:
        raise RuntimeError("请先运行 python g1_right_hand_workspace.py 生成可行域")
    return build_reachable_workspace(ee, joints)


def _generate_concentric_demo(
    shape: str,
    center: np.ndarray,
    n_layers: int,
    fps: float,
    frames_per_ring: int,
    elbow_branch_mode: str,
    rng: np.random.Generator,
    theta: float,
    ee,
    joints,
    ws,
    refine: bool,
) -> dict:
    from g1_concentric_traj_gen import (
        build_concentric_scales,
        build_concentric_waypoints,
        find_max_scale,
    )

    scale_env, scale_max = find_max_scale(shape, center, theta, ws, ee, rng)
    scales = build_concentric_scales(scale_max, n_layers)
    rings = build_concentric_waypoints(shape, center, scales, theta, frames_per_ring, rng)

    solver = TrajectoryIKSolver()
    all_qpos, all_qvel, all_ts, all_wps = [], [], [], []
    segment_starts = [0]
    layer_metrics: list[dict] = []
    locked = None
    t_accum = 0.0
    dt = 1.0 / fps

    print(f"中心=({center[0]:.2f},{center[1]:.2f},{center[2]:.2f}) "
          f"theta={theta:.2f} scale_max={scale_max:.3f} 层数={len(scales)}")

    for li, ring_wps in enumerate(rings):
        print(f"  层 {li+1}/{len(scales)} IK (TrajectoryIKSolver) ...", flush=True)
        res = solver.solve(
            ring_wps, ee, joints, fps=fps,
            elbow_branch_mode=elbow_branch_mode,
            locked_branch=locked,
            closed_path=True,
            refine=refine,
        )
        locked = res.locked_branch
        if li == 0:
            print(f"  肘部折向锁定: {locked}")
        m = evaluate_trajectory(res.qpos, ring_wps, locked)
        layer_metrics.append({
            "layer_idx": li + 1,
            "scale": float(scales[li]),
            **m,
        })
        print(f"    err={m['err_mean_mm']:.1f}mm max={m['err_max_mm']:.1f}mm "
              f"ok={m['ok_ratio']:.2f} xflip={m['xflip']} "
              f"xspan={m['xspan_mm']:.1f}mm branch_viol={m['branch_viol']}")

        ts = res.timestamps.copy()
        if t_accum > 0:
            ts += t_accum
        all_qpos.append(res.qpos)
        all_qvel.append(res.qvel)
        all_ts.append(ts)
        all_wps.append(ring_wps)
        segment_starts.append(segment_starts[-1] + len(res.qpos))
        t_accum = all_ts[-1][-1] + dt

    return {
        "qpos": np.concatenate(all_qpos),
        "qvel": np.concatenate(all_qvel),
        "timestamps": np.concatenate(all_ts),
        "waypoints": np.concatenate(all_wps),
        "segment_starts": np.array(segment_starts, dtype=np.int32),
        "shape_name": shape,
        "center": center.astype(np.float32),
        "scale_max": float(scale_max),
        "layer_scales": scales.astype(np.float32),
        "layer_metrics": layer_metrics,
        "theta": float(theta),
        "fps": float(fps),
        "frames_per_ring": int(frames_per_ring),
        "n_layers": int(len(scales)),
        "locked_branch": locked,
    }


def save_trajectory_npz(path: str, traj: dict, seed: int | None = None) -> None:
    """原子写入轨迹 NPZ（供批量流水线复用）。"""
    # np.savez_compressed 会自动追加 .npz，故临时文件不能再用 *.npz.tmp
    tmp = path[:-4] + ".part" if path.endswith(".npz") else path + ".part"
    payload = dict(
        qpos=traj["qpos"],
        qvel=traj["qvel"],
        timestamps=traj["timestamps"],
        waypoints=traj["waypoints"],
        segment_starts=traj["segment_starts"],
        shape_name=traj["shape_name"],
        center=traj["center"],
        scale_max=np.float32(traj["scale_max"]),
        layer_scales=traj["layer_scales"],
        theta=np.float32(traj["theta"]),
        fps=np.float32(traj["fps"]),
        frames_per_ring=np.int32(traj["frames_per_ring"]),
        n_layers=np.int32(traj["n_layers"]),
        locked_branch=traj["locked_branch"],
        ik_method="trac_ik_human",
    )
    if seed is not None:
        payload["seed"] = np.int32(seed)
    metrics = traj.get("layer_metrics")
    if metrics:
        for k in ("err_mean_mm", "err_max_mm", "ok_ratio", "branch_viol",
                  "xflip", "xspan_mm", "max_droll"):
            payload[f"layer_{k}"] = np.array(
                [m[k] for m in metrics], dtype=np.float32,
            )
        payload["layer_scale_per_metric"] = np.array(
            [m["scale"] for m in metrics], dtype=np.float32,
        )
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    np.savez_compressed(tmp, **payload)
    final = tmp + ".npz"
    if not os.path.isfile(final):
        raise FileNotFoundError(f"savez 未生成预期文件: {final}")
    os.replace(final, path)


def _try_generate_concentric_demo(
    shape: str,
    n_layers: int,
    fps: float,
    frames_per_ring: int,
    elbow_branch_mode: str,
    ee,
    joints,
    ws,
    rng: np.random.Generator,
    refine: bool,
    center: np.ndarray | None = None,
    theta: float | None = None,
    max_tries: int = 12,
) -> dict | None:
    from g1_concentric_traj_gen import sample_center

    for attempt in range(max_tries):
        if center is not None and attempt == 0:
            c = center.copy()
        else:
            c = sample_center(ee, ws, rng)
        th = theta if (theta is not None and attempt == 0) else float(rng.uniform(0, 2 * np.pi))
        try:
            return _generate_concentric_demo(
                shape, c, n_layers, fps, frames_per_ring,
                elbow_branch_mode, rng, th, ee, joints, ws, refine,
            )
        except RuntimeError:
            if center is not None:
                break
            continue
    return None


def main():
    ap = argparse.ArgumentParser(description="G1 轨迹 IK（TP-DLS + Gauss-Newton）")
    ap.add_argument("--shape", default="circle")
    ap.add_argument("--center", nargs=3, type=float, default=None,
                    help="手动指定圆心；不指定则从可达域随机采样")
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--num-trajs", type=int, default=1,
                    help="生成多少条轨迹（每次随机采样不同中心/朝向）")
    ap.add_argument("--theta", type=float, default=None, help="形状旋转角（弧度）；不指定则随机")
    ap.add_argument("--seed", type=int, default=None, help="随机种子（可复现）")
    ap.add_argument("--fps", type=float, default=50.0)
    ap.add_argument("--frames-per-ring", type=int, default=200)
    ap.add_argument("--elbow-branch", default="outward")
    ap.add_argument("--ws-cache", action="store_true")
    ap.add_argument("--no-refine", action="store_true", help="跳过全轨迹 Gauss-Newton")
    ap.add_argument("--preview", action="store_true")
    ap.add_argument("--save", action="store_true")
    args = ap.parse_args()

    if not args.ws_cache:
        print("随机采样需要可行域缓存，请加上 --ws-cache")
        return

    from g1_concentric_traj_gen import resolve_shape_name

    branch = resolve_branch_mode(args.elbow_branch)
    shape = resolve_shape_name(args.shape)
    rng = np.random.default_rng(args.seed)
    ee, joints, ws = _load_reachable_workspace(args.ws_cache)
    user_center = np.array(args.center, dtype=np.float64) if args.center else None
    refine = not args.no_refine
    last_traj = None

    for ti in range(args.num_trajs):
        print(f"\n=== 轨迹 {ti + 1}/{args.num_trajs} | {shape} x{args.layers} 层 ===")
        traj = _try_generate_concentric_demo(
            shape, args.layers, args.fps, args.frames_per_ring,
            branch, ee, joints, ws, rng, refine,
            center=user_center if ti == 0 else None,
            theta=args.theta if ti == 0 else None,
        )
        if traj is None:
            print(f"  轨迹 {ti + 1} 失败（中心不可行）")
            continue

        m = evaluate_trajectory(traj["qpos"], traj["waypoints"], traj["locked_branch"])
        print(f"\n整体: err={m['err_mean_mm']:.1f}mm ok={m['ok_ratio']:.2f} "
              f"xflip={m['xflip']} branch_viol={m['branch_viol']}")
        last_traj = traj

        if args.save:
            tag = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            c = traj["center"]
            out = (
                f"recordings/g1_tpik_{shape}_L{args.layers}_"
                f"c{c[0]:.2f}_{c[1]:.2f}_{c[2]:.2f}_{tag}.npz"
            )
            save_trajectory_npz(out, traj, seed=args.seed)
            print(f"已保存: {out}")

    if args.preview and last_traj is not None:
        from g1_multi_shape_traj_gen import preview_trajectory
        preview_trajectory(last_traj["qpos"], last_traj["waypoints"], MODEL_PATH)


if __name__ == "__main__":
    main()
