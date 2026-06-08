"""
G1 右手 4-DOF 姿态分类与自然臂约束。
- 禁止肘部向上翻折（肘高于肩且为手臂最高点）
- 单次轨迹内肘部折向一致：inward / outward
"""

from __future__ import annotations

import dataclasses

import mujoco
import numpy as np

from g1_right_hand_workspace import (
    ACTIVE_JOINTS,
    LOCKED_JOINTS,
    ROBOT_XML,
    _build_addr_table,
    _reset_to_home,
)

# 图四自然参考姿态（MuJoCo 面板读数）
NATURAL_Q4_REF = np.array([0.0786, -0.792, 0.0, 0.0], dtype=np.float64)

ELBOW_BRANCH_OUTWARD = "outward"
ELBOW_BRANCH_INWARD = "inward"
ELBOW_BRANCH_AUTO = "auto"
ELBOW_BRANCHES = (ELBOW_BRANCH_OUTWARD, ELBOW_BRANCH_INWARD, ELBOW_BRANCH_AUTO)

_SHOULDER_BODY = "right_shoulder_yaw_link"
_ELBOW_BODY = "right_elbow_link"
_WRIST_BODY = "right_wrist_yaw_link"

# 肘相对肩的 y 偏移：+Y 为机器人左侧，向内折肘靠躯干
INWARD_EL_Y_MIN = 0.01

# 肩 roll：图四外翻 roll≈-0.79；内翻/突变时 roll 接近 0 或为正
OUTWARD_ROLL_MAX = -0.10
INWARD_ROLL_MIN = -0.06

# 外翻禁止极端弯曲（视觉上像内翻）；伸直由 STRAIGHT_ELBOW_MAX 约束
OUTWARD_ELBOW_HI = 1.05
INWARD_ELBOW_HI = 1.10

# 向上翻折：肘 z 高于肩，且不低于腕（肘为拱顶）
UP_EL_Z_MIN = 0.02
UP_WR_Z_MARGIN = -0.01

# 肘关节接近伸直（range 下限约 -1.05 rad）
STRAIGHT_ELBOW_MAX = -0.70


@dataclasses.dataclass
class PostureFeatures:
    pitch: float
    roll: float
    yaw: float
    elbow: float
    el_z_rel: float
    wr_z_rel: float
    el_y_rel: float
    upward_flip: bool

    @property
    def branch(self) -> str:
        if self._is_outward_features():
            return ELBOW_BRANCH_OUTWARD
        if self._is_inward_features():
            return ELBOW_BRANCH_INWARD
        if self.roll <= (OUTWARD_ROLL_MAX + INWARD_ROLL_MIN) * 0.5:
            return ELBOW_BRANCH_OUTWARD
        return ELBOW_BRANCH_INWARD

    def _is_outward_features(self) -> bool:
        if self.roll > OUTWARD_ROLL_MAX:
            return False
        if self.elbow > OUTWARD_ELBOW_HI:
            return False
        if self.el_y_rel >= INWARD_EL_Y_MIN:
            return False
        return True

    def _is_inward_features(self) -> bool:
        return (
            self.roll >= INWARD_ROLL_MIN
            or self.el_y_rel >= INWARD_EL_Y_MIN
            or self.elbow >= INWARD_ELBOW_HI
        )


def _ensure_model_data(model=None, data=None, addr=None):
    if model is None:
        model = mujoco.MjModel.from_xml_path(ROBOT_XML)
    if data is None:
        data = mujoco.MjData(model)
    if addr is None:
        addr = _build_addr_table(model, ACTIVE_JOINTS + list(LOCKED_JOINTS.keys()))
    return model, data, addr


def set_q4(model, data, addr, q4: np.ndarray):
    _reset_to_home(model, data, addr)
    q4 = np.asarray(q4, dtype=np.float64).reshape(4)
    for i, n in enumerate(ACTIVE_JOINTS):
        data.qpos[addr[n][0]] = float(q4[i])
    for name, val in LOCKED_JOINTS.items():
        data.qpos[addr[name][0]] = val
    mujoco.mj_forward(model, data)


def compute_posture_features(model, data) -> PostureFeatures:
    ids = {
        "sp": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, _SHOULDER_BODY),
        "el": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, _ELBOW_BODY),
        "wr": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, _WRIST_BODY),
    }
    sp = data.xpos[ids["sp"]]
    el = data.xpos[ids["el"]]
    wr = data.xpos[ids["wr"]]
    addr = _build_addr_table(model, ACTIVE_JOINTS)
    q4 = [float(data.qpos[addr[n][0]]) for n in ACTIVE_JOINTS]
    el_z_rel = float(el[2] - sp[2])
    wr_z_rel = float(wr[2] - sp[2])
    el_y_rel = float(el[1] - sp[1])
    upward = (el_z_rel > UP_EL_Z_MIN) and (el_z_rel >= wr_z_rel + UP_WR_Z_MARGIN)
    return PostureFeatures(
        pitch=q4[0], roll=q4[1], yaw=q4[2], elbow=q4[3],
        el_z_rel=el_z_rel, wr_z_rel=wr_z_rel, el_y_rel=el_y_rel,
        upward_flip=upward,
    )


def posture_from_q4(q4: np.ndarray, model=None, data=None, addr=None) -> PostureFeatures:
    model, data, addr = _ensure_model_data(model, data, addr)
    set_q4(model, data, addr, q4)
    return compute_posture_features(model, data)


def is_upward_flip_q4(q4: np.ndarray, model=None, data=None, addr=None) -> bool:
    return posture_from_q4(q4, model, data, addr).upward_flip


def elbow_branch_of_q4(q4: np.ndarray, model=None, data=None, addr=None) -> str:
    return posture_from_q4(q4, model, data, addr).branch


def q4_branch_stable_with_anchor(
    q4: np.ndarray,
    q4_anchor: np.ndarray,
    branch: str | None,
    model=None,
    data=None,
    addr=None,
    roll_slack: float = 0.05,
) -> bool:
    """抛光结果须与锚点同属 locked_branch，且肩 roll 不能跨支突变。"""
    if branch not in (ELBOW_BRANCH_INWARD, ELBOW_BRANCH_OUTWARD):
        return True
    if not is_allowed_q4(q4, branch, model, data, addr):
        return False
    f = posture_from_q4(q4, model, data, addr)
    fa = posture_from_q4(q4_anchor, model, data, addr)
    if abs(f.roll - fa.roll) > roll_slack:
        return False
    if branch == ELBOW_BRANCH_OUTWARD and f.roll > fa.roll + roll_slack:
        return False
    if branch == ELBOW_BRANCH_INWARD and f.roll < fa.roll - roll_slack:
        return False
    return True


def q4_matches_branch(q4: np.ndarray, branch: str,
                      model=None, data=None, addr=None) -> bool:
    if branch not in (ELBOW_BRANCH_INWARD, ELBOW_BRANCH_OUTWARD):
        return True
    feat = posture_from_q4(q4, model, data, addr)
    if feat.upward_flip:
        return False
    return feat.branch == branch


def is_near_straight_arm_q4(q4: np.ndarray, model=None, data=None, addr=None) -> bool:
    feat = posture_from_q4(q4, model, data, addr)
    return feat.elbow < STRAIGHT_ELBOW_MAX


def is_allowed_q4(q4: np.ndarray, branch: str | None = None,
                  model=None, data=None, addr=None) -> bool:
    feat = posture_from_q4(q4, model, data, addr)
    if feat.upward_flip:
        return False
    if feat.elbow < STRAIGHT_ELBOW_MAX:
        return False
    if branch == ELBOW_BRANCH_OUTWARD:
        return feat._is_outward_features()
    if branch == ELBOW_BRANCH_INWARD:
        return feat._is_inward_features()
    return True


def naturalness_score(
    q4: np.ndarray,
    branch: str | None = None,
    model=None,
    data=None,
    addr=None,
) -> float:
    q4 = np.asarray(q4, dtype=np.float64)
    d = float(np.linalg.norm(q4 - NATURAL_Q4_REF))
    score = 1.0 / (1.0 + 2.5 * d)
    if branch in (ELBOW_BRANCH_INWARD, ELBOW_BRANCH_OUTWARD):
        model, data, addr = _ensure_model_data(model, data, addr)
        feat = posture_from_q4(q4, model, data, addr)
        if feat.branch == branch:
            score += 0.35
        else:
            score -= 1.0
        if feat.upward_flip:
            score -= 2.0
    return score


def resolve_branch_mode(mode: str | None) -> str:
    if mode is None:
        return ELBOW_BRANCH_AUTO
    m = mode.strip().lower()
    aliases = {
        "out": ELBOW_BRANCH_OUTWARD, "外": ELBOW_BRANCH_OUTWARD,
        "向外": ELBOW_BRANCH_OUTWARD, "外翻": ELBOW_BRANCH_OUTWARD,
        "in": ELBOW_BRANCH_INWARD, "内": ELBOW_BRANCH_INWARD,
        "向内": ELBOW_BRANCH_INWARD, "内翻": ELBOW_BRANCH_INWARD,
        "auto": ELBOW_BRANCH_AUTO, "自动": ELBOW_BRANCH_AUTO,
    }
    if m in aliases:
        return aliases[m]
    if m in ELBOW_BRANCHES:
        return m
    raise ValueError(
        f"未知肘部折向模式: {mode}，可选 outward/inward/auto"
    )


def pick_locked_branch_from_candidates(
    joint_configs: np.ndarray,
    cand_indices: np.ndarray,
    branch_mode: str,
    model=None,
    data=None,
    addr=None,
) -> str | None:
    model, data, addr = _ensure_model_data(model, data, addr)
    if branch_mode in (ELBOW_BRANCH_INWARD, ELBOW_BRANCH_OUTWARD):
        return branch_mode
    best_score = -1e9
    best_branch = None
    for c in cand_indices:
        q4 = joint_configs[c]
        feat = posture_from_q4(q4, model, data, addr)
        if feat.upward_flip:
            continue
        sc = naturalness_score(q4)
        if sc > best_score:
            best_score = sc
            best_branch = feat.branch
    return best_branch
