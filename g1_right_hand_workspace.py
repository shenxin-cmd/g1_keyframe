"""
G1 右手 4-DOF 可行域分析
========================
在关节限位内采样右臂 4 自由度（肩 pitch/roll/yaw + 肘），腕三关节锁定为 0，
过滤穿透碰撞后得到右手末端可达点云，并限制在机器人正前方。

用法：
  python g1_right_hand_workspace.py
  python g1_right_hand_workspace.py --mujoco-viewer
  python g1_right_hand_workspace.py --show-plot
  python g1_right_hand_workspace.py --probe-q4 -0.28 0.03 -0.40 -0.32
  python g1_right_hand_workspace.py --probe-sphere 0.28 -0.30 0.98
  # --ws-cache 与 g1_multi_shape_traj_gen.py 一致，可写可不写（probe 默认读缓存）

输出：
  recordings/g1_ws_4dof_front.npz
  recordings/g1_ws_4dof_front_3d.png
  recordings/g1_ws_4dof_front_slices.png
"""

from __future__ import annotations

import argparse
import dataclasses
import os
import time

import mujoco
import mujoco.viewer
import numpy as np

os.makedirs("recordings", exist_ok=True)

# ══════════════════════════════════════════════
# 常量
# ══════════════════════════════════════════════

# 与 g1_recorder / scene_g1_draggable 使用同一机器人模型
ROBOT_XML = "g1/g1_for_reward_inference.xml"
DRAGGABLE_XML = "g1/scene_g1_draggable.xml"
EE_BODY = "right_wrist_yaw_link"
HAND_BODY = "right_rubber_hand"
WS_CACHE_PATH = "recordings/g1_ws_4dof_front.npz"
OUT_3D = "recordings/g1_ws_4dof_front_3d.png"
OUT_SLICES = "recordings/g1_ws_4dof_front_slices.png"

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

# 右臂碰撞检测：肩→肘→腕→手 全部参与，防止前臂/肘穿躯干。
_RIGHT_ARM_COLLISION_BODY_NAMES = [
    "right_shoulder_pitch_link",
    "right_shoulder_roll_link",
    "right_shoulder_yaw_link",
    "right_elbow_link",
    "right_wrist_roll_link",
    "right_wrist_pitch_link",
    "right_wrist_yaw_link",
    "right_rubber_hand",
]
# 兼容旧名
_EE_COLLISION_BODY_NAMES = _RIGHT_ARM_COLLISION_BODY_NAMES

_OBSTACLE_BODY_NAMES_MINIMAL = ["pelvis", "torso_link", "head_link"]

HOME_PELVIS_POS = np.array([0.0, 0.0, 0.793])
HOME_PELVIS_QUAT = np.array([1.0, 0.0, 0.0, 0.0])

# 骨盆前方 x>0 即算身前；0.15 过严会裁掉贴近胸口的拖拽姿态
DEFAULT_FRONT_X_MIN = 0.05
DEFAULT_FRONT_X_MAX = 0.55
# 轨迹中心随机采样时，相对可达域边界 xyz 各向内缩进（避免伸直难动区域）
CENTER_SAMPLE_INSET = 0.20
# 形状轨迹路点相对可达域边界（x 切片 yz 包络或轴对齐盒）至少保留的安全距离
TRAJ_BOUNDARY_MARGIN = 0.10
PENETRATION_THRESH = -0.0005
PROBE_OK_DIST = 0.045


# ══════════════════════════════════════════════
# 工具函数
# ══════════════════════════════════════════════

def _int(x):
    return int(np.asarray(x).reshape(-1)[0])


def _build_addr_table(model, names):
    tbl = {}
    for name in names:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        tbl[name] = (
            _int(model.jnt_qposadr[jid]),
            _int(model.jnt_dofadr[jid]),
            tuple(model.jnt_range[jid].tolist()),
        )
    return tbl


def _build_collision_id_sets(model, obstacle_names,
                             ee_body_names=None):
    if ee_body_names is None:
        ee_body_names = _RIGHT_ARM_COLLISION_BODY_NAMES
    def _ids(names):
        s = set()
        for n in names:
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n)
            if bid >= 0:
                s.add(bid)
        return s
    return _ids(ee_body_names), _ids(obstacle_names)


def _has_penetration(model, data, arm_ids, obstacle_ids):
    """检测右臂与躯干/头部的真实穿透（须先 mj_forward + mj_collision）。"""
    mujoco.mj_forward(model, data)
    try:
        mujoco.mj_collision(model, data)
    except Exception:
        return True
    for i in range(data.ncon):
        c = data.contact[i]
        if c.dist > PENETRATION_THRESH:
            continue
        b1 = model.geom_bodyid[c.geom1]
        b2 = model.geom_bodyid[c.geom2]
        if ((b1 in arm_ids) or (b2 in arm_ids)) and \
           ((b1 in obstacle_ids) or (b2 in obstacle_ids)):
            return True
    return False


def _reset_to_home(model, data, addr):
    data.qpos[:3] = HOME_PELVIS_POS
    data.qpos[3:7] = HOME_PELVIS_QUAT
    data.qpos[7:] = 0.0
    data.qvel[:] = 0.0
    for name, val in LOCKED_JOINTS.items():
        if name in addr:
            data.qpos[addr[name][0]] = val
    mujoco.mj_forward(model, data)


@dataclasses.dataclass
class WorkspaceEnvelope:
    """沿 x 轴分片的 yz 平面可达边界（用于估算中心点处最大形状尺度）。"""
    x_edges: np.ndarray
    y_lo: np.ndarray
    y_hi: np.ndarray
    z_lo: np.ndarray
    z_hi: np.ndarray

    @property
    def n_slices(self) -> int:
        return len(self.y_lo)


@dataclasses.dataclass
class ReachableWorkspace:
    """正前方过滤后的右手可达域盒。"""
    lo: np.ndarray
    hi: np.ndarray
    mid: np.ndarray
    n_samples: int
    x_band: tuple[float, float]
    envelope: WorkspaceEnvelope | None = None


def build_workspace_envelope(ee_cloud: np.ndarray,
                           n_x_bins: int = 48) -> WorkspaceEnvelope:
    """按 x 分箱，记录每个切片在 yz 平面上的点云边界。"""
    ee = np.asarray(ee_cloud, dtype=np.float64)
    x_edges = np.linspace(ee[:, 0].min(), ee[:, 0].max(), n_x_bins + 1)
    y_lo = np.zeros(n_x_bins, dtype=np.float64)
    y_hi = np.zeros(n_x_bins, dtype=np.float64)
    z_lo = np.zeros(n_x_bins, dtype=np.float64)
    z_hi = np.zeros(n_x_bins, dtype=np.float64)
    global_y = (ee[:, 1].min(), ee[:, 1].max())
    global_z = (ee[:, 2].min(), ee[:, 2].max())
    for i in range(n_x_bins):
        lo_x, hi_x = x_edges[i], x_edges[i + 1]
        if i < n_x_bins - 1:
            m = (ee[:, 0] >= lo_x) & (ee[:, 0] < hi_x)
        else:
            m = (ee[:, 0] >= lo_x) & (ee[:, 0] <= hi_x)
        sl = ee[m]
        if len(sl) < 8:
            y_lo[i], y_hi[i] = global_y
            z_lo[i], z_hi[i] = global_z
        else:
            y_lo[i] = sl[:, 1].min()
            y_hi[i] = sl[:, 1].max()
            z_lo[i] = sl[:, 2].min()
            z_hi[i] = sl[:, 2].max()
    return WorkspaceEnvelope(
        x_edges=x_edges, y_lo=y_lo, y_hi=y_hi, z_lo=z_lo, z_hi=z_hi,
    )


def _envelope_slice_weights(envelope: WorkspaceEnvelope, x: float):
    edges = envelope.x_edges
    n = envelope.n_slices
    if x <= edges[0]:
        return 0, 1.0, 0.0
    if x >= edges[-1]:
        return n - 1, 1.0, 0.0
    i = int(np.searchsorted(edges, x, side="right") - 1)
    i = min(max(i, 0), n - 2)
    w = (x - edges[i]) / max(edges[i + 1] - edges[i], 1e-9)
    return i, 1.0 - w, w


def envelope_bounds_at(envelope: WorkspaceEnvelope, x: float):
    i, w0, w1 = _envelope_slice_weights(envelope, x)
    y_lo = w0 * envelope.y_lo[i] + w1 * envelope.y_lo[i + 1]
    y_hi = w0 * envelope.y_hi[i] + w1 * envelope.y_hi[i + 1]
    z_lo = w0 * envelope.z_lo[i] + w1 * envelope.z_lo[i + 1]
    z_hi = w0 * envelope.z_hi[i] + w1 * envelope.z_hi[i + 1]
    return y_lo, y_hi, z_lo, z_hi


def envelope_margins_at(envelope: WorkspaceEnvelope, center: np.ndarray,
                        margin: float = TRAJ_BOUNDARY_MARGIN):
    """中心点到当前 x 切片 yz 边界的最近余量 (my, mz)，已扣除 margin。"""
    y_lo, y_hi, z_lo, z_hi = envelope_bounds_at(envelope, float(center[0]))
    cy, cz = float(center[1]), float(center[2])
    my = min(cy - y_lo, y_hi - cy) - margin
    mz = min(cz - z_lo, z_hi - cz) - margin
    return float(my), float(mz)


# ══════════════════════════════════════════════
# 采样
# ══════════════════════════════════════════════

def _sample_workspace_grid(n_per_joint: int) -> tuple[np.ndarray, np.ndarray, int, int]:
    model = mujoco.MjModel.from_xml_path(ROBOT_XML)
    data = mujoco.MjData(model)
    all_names = ACTIVE_JOINTS + list(LOCKED_JOINTS.keys())
    addr = _build_addr_table(model, all_names)
    _reset_to_home(model, data, addr)
    ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, EE_BODY)
    arm_ids, obstacle_ids = _build_collision_id_sets(model, _OBSTACLE_BODY_NAMES_MINIMAL)

    grids = [np.linspace(lo, hi, n_per_joint) for lo, hi in
             [addr[n][2] for n in ACTIVE_JOINTS]]
    ee_pts, joint_cfgs = [], []
    n_col = 0

    def _recurse(dim: int, qvals: list[float]):
        nonlocal n_col
        if dim == 4:
            for name, val in LOCKED_JOINTS.items():
                data.qpos[addr[name][0]] = val
            mujoco.mj_forward(model, data)
            if _has_penetration(model, data, arm_ids, obstacle_ids):
                n_col += 1
                return
            from g1_arm_posture import is_near_straight_arm_q4, is_upward_flip_q4
            if is_upward_flip_q4(qvals, model, data, addr):
                n_col += 1
                return
            if is_near_straight_arm_q4(qvals, model, data, addr):
                n_col += 1
                return
            ee_pts.append(data.xpos[ee_id].copy())
            joint_cfgs.append(qvals.copy())
            return
        for q in grids[dim]:
            data.qpos[addr[ACTIVE_JOINTS[dim]][0]] = q
            qvals.append(float(q))
            _recurse(dim + 1, qvals)
            qvals.pop()

    _recurse(0, [])
    return (
        np.array(ee_pts, dtype=np.float32),
        np.array(joint_cfgs, dtype=np.float32),
        n_per_joint ** 4,
        n_col,
    )


def _sample_workspace_random(n_random: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    model = mujoco.MjModel.from_xml_path(ROBOT_XML)
    data = mujoco.MjData(model)
    addr = _build_addr_table(model, ACTIVE_JOINTS + list(LOCKED_JOINTS.keys()))
    _reset_to_home(model, data, addr)
    ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, EE_BODY)
    arm_ids, obstacle_ids = _build_collision_id_sets(model, _OBSTACLE_BODY_NAMES_MINIMAL)
    rng = np.random.default_rng(seed)
    ranges = [addr[n][2] for n in ACTIVE_JOINTS]
    ee_pts, joint_cfgs = [], []
    for _ in range(n_random):
        qvals = [float(rng.uniform(lo, hi)) for lo, hi in ranges]
        for i, n in enumerate(ACTIVE_JOINTS):
            data.qpos[addr[n][0]] = qvals[i]
        for name, val in LOCKED_JOINTS.items():
            data.qpos[addr[name][0]] = val
        mujoco.mj_forward(model, data)
        if _has_penetration(model, data, arm_ids, obstacle_ids):
            continue
        from g1_arm_posture import is_allowed_q4
        if not is_allowed_q4(qvals, None, model, data, addr):
            continue
        ee_pts.append(data.xpos[ee_id].copy())
        joint_cfgs.append(qvals)
    return np.array(ee_pts, dtype=np.float32), np.array(joint_cfgs, dtype=np.float32)


def _sample_kinematic_envelope(n_per_joint: int = 12,
                               front_x_min: float = DEFAULT_FRONT_X_MIN,
                               front_x_max: float = DEFAULT_FRONT_X_MAX) -> np.ndarray:
    """4-DOF 运动学包络（无碰撞检查），作灰色参考。"""
    model = mujoco.MjModel.from_xml_path(ROBOT_XML)
    data = mujoco.MjData(model)
    addr = _build_addr_table(model, ACTIVE_JOINTS + list(LOCKED_JOINTS.keys()))
    _reset_to_home(model, data, addr)
    ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, EE_BODY)
    grids = [np.linspace(lo, hi, n_per_joint) for lo, hi in
             [addr[n][2] for n in ACTIVE_JOINTS]]
    ee_pts = []

    def _recurse(dim: int):
        if dim == 4:
            for name, val in LOCKED_JOINTS.items():
                data.qpos[addr[name][0]] = val
            mujoco.mj_forward(model, data)
            ee_pts.append(data.xpos[ee_id].copy())
            return
        for q in grids[dim]:
            data.qpos[addr[ACTIVE_JOINTS[dim]][0]] = q
            _recurse(dim + 1)

    _recurse(0)
    pts = np.array(ee_pts, dtype=np.float32)
    m = (pts[:, 0] >= front_x_min) & (pts[:, 0] <= front_x_max)
    return pts[m]


def _merge_samples(ee_a, joints_a, ee_b, joints_b, dedup_tol=0.004):
    if len(ee_a) == 0:
        return ee_b, joints_b
    if len(ee_b) == 0:
        return ee_a, joints_a
    ee = np.vstack([ee_a, ee_b])
    joints = np.vstack([joints_a, joints_b])
    keep = np.ones(len(ee), dtype=bool)
    for i in range(len(ee)):
        if not keep[i]:
            continue
        d = np.linalg.norm(ee[i + 1:] - ee[i], axis=1)
        keep[i + 1:][d < dedup_tol] = False
    return ee[keep], joints[keep]


def filter_workspace_front(ee_cloud, joint_configs,
                           x_min=DEFAULT_FRONT_X_MIN,
                           x_max=DEFAULT_FRONT_X_MAX):
    m = (ee_cloud[:, 0] >= x_min) & (ee_cloud[:, 0] <= x_max)
    filtered = ee_cloud[m]
    if len(filtered) < 50:
        raise RuntimeError(
            f"正前方带 x=[{x_min},{x_max}] 内有效点过少: {len(filtered)}"
        )
    return filtered, joint_configs[m]


def filter_collision_free_samples(ee_cloud: np.ndarray,
                                  joint_configs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """用全臂穿透检测过滤旧缓存中碰撞样本。"""
    model = mujoco.MjModel.from_xml_path(ROBOT_XML)
    data = mujoco.MjData(model)
    addr = _build_addr_table(model, ACTIVE_JOINTS + list(LOCKED_JOINTS.keys()))
    arm_ids, obstacle_ids = _build_collision_id_sets(model, _OBSTACLE_BODY_NAMES_MINIMAL)
    keep = []
    for i, q4 in enumerate(joint_configs):
        _reset_to_home(model, data, addr)
        for j, n in enumerate(ACTIVE_JOINTS):
            data.qpos[addr[n][0]] = float(q4[j])
        if not _has_penetration(model, data, arm_ids, obstacle_ids):
            from g1_arm_posture import is_allowed_q4
            if is_allowed_q4(q4, None, model, data, addr):
                keep.append(i)
    if len(keep) < 50:
        raise RuntimeError(
            f"全臂碰撞过滤后有效点过少: {len(keep)} / {len(ee_cloud)}，"
            "请重建可行域: python g1_right_hand_workspace.py"
        )
    idx = np.asarray(keep, dtype=int)
    return ee_cloud[idx], joint_configs[idx]


def build_reachable_workspace(ee_cloud, joint_configs,
                            x_min=DEFAULT_FRONT_X_MIN,
                            x_max=DEFAULT_FRONT_X_MAX):
    ee, joints = filter_workspace_front(ee_cloud, joint_configs, x_min, x_max)
    ee, joints = filter_collision_free_samples(ee, joints)
    lo = ee.min(axis=0)
    hi = ee.max(axis=0)
    ws = ReachableWorkspace(
        lo=lo, hi=hi, mid=(lo + hi) * 0.5,
        n_samples=len(ee), x_band=(x_min, x_max),
        envelope=build_workspace_envelope(ee),
    )
    return ee, joints, ws


def inset_bounds(lo: np.ndarray, hi: np.ndarray,
                 inset: float = CENTER_SAMPLE_INSET) -> tuple[np.ndarray, np.ndarray]:
    """将轴对齐包围盒各面向内缩 inset（米）。"""
    lo = np.asarray(lo, dtype=np.float64).copy()
    hi = np.asarray(hi, dtype=np.float64).copy()
    lo2 = lo + inset
    hi2 = hi - inset
    for i in range(3):
        if lo2[i] > hi2[i]:
            mid = 0.5 * (lo[i] + hi[i])
            lo2[i] = hi2[i] = mid
    return lo2, hi2


def mask_points_in_box(points: np.ndarray, lo: np.ndarray, hi: np.ndarray,
                       eps: float = 1e-6) -> np.ndarray:
    return (
        (points[:, 0] >= lo[0] - eps) & (points[:, 0] <= hi[0] + eps) &
        (points[:, 1] >= lo[1] - eps) & (points[:, 1] <= hi[1] + eps) &
        (points[:, 2] >= lo[2] - eps) & (points[:, 2] <= hi[2] + eps)
    )


def sample_center_from_cloud(
    ee_cloud: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    rng: np.random.Generator,
    inset: float = CENTER_SAMPLE_INSET,
) -> np.ndarray:
    """在缩进后的包围盒内随机选轨迹中心，避开边缘伸直区域。"""
    lo_i, hi_i = inset_bounds(lo, hi, inset)
    mask = mask_points_in_box(ee_cloud, lo_i, hi_i)
    pool = ee_cloud[mask]
    if len(pool) < 8:
        pool = ee_cloud
    idx = int(rng.integers(0, len(pool)))
    return pool[idx].astype(np.float64).copy()


def fk_ee_from_4dof(q4: np.ndarray) -> np.ndarray:
    model = mujoco.MjModel.from_xml_path(ROBOT_XML)
    data = mujoco.MjData(model)
    addr = _build_addr_table(model, ACTIVE_JOINTS + list(LOCKED_JOINTS.keys()))
    _reset_to_home(model, data, addr)
    for i, n in enumerate(ACTIVE_JOINTS):
        data.qpos[addr[n][0]] = float(q4[i])
    for name, val in LOCKED_JOINTS.items():
        data.qpos[addr[name][0]] = val
    mujoco.mj_forward(model, data)
    ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, EE_BODY)
    return data.xpos[ee_id].copy()


# ══════════════════════════════════════════════
# 可视化
# ══════════════════════════════════════════════

def _draw_box_wireframe(ax, lo, hi, color="0.4", linestyle="-", linewidth=1.0, alpha=0.9):
    x0, y0, z0 = lo
    x1, y1, z1 = hi
    corners = np.array([
        [x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
        [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1],
    ])
    edges = (
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    )
    for i, j in edges:
        ax.plot(
            [corners[i, 0], corners[j, 0]],
            [corners[i, 1], corners[j, 1]],
            [corners[i, 2], corners[j, 2]],
            color=color, linestyle=linestyle, linewidth=linewidth, alpha=alpha,
        )


def _print_workspace(ws: ReachableWorkspace, x_min: float, x_max: float):
    print(f"  正前方过滤 x∈[{x_min:.2f}, {x_max:.2f}] m")
    print(f"  碰撞-free 样本数: {ws.n_samples}")
    print(f"  可达域盒 x[{ws.lo[0]:.2f},{ws.hi[0]:.2f}] "
          f"y[{ws.lo[1]:.2f},{ws.hi[1]:.2f}] z[{ws.lo[2]:.2f},{ws.hi[2]:.2f}]")
    print(f"  八分区分割 @ mid x={ws.mid[0]:.2f} y={ws.mid[1]:.2f} z={ws.mid[2]:.2f}")
    print("  (左/右/上/下/前/后 均相对该可达域盒，非全身身后空间)")


def visualize_workspace(
    ee_cloud: np.ndarray,
    joint_configs: np.ndarray,
    front_x_min: float = DEFAULT_FRONT_X_MIN,
    front_x_max: float = DEFAULT_FRONT_X_MAX,
    out_3d: str = OUT_3D,
    out_slices: str = OUT_SLICES,
    show_plot: bool = False,
    show_kinematic: bool = True,
):
    try:
        import matplotlib
        if not show_plot:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    except ImportError:
        print("需要 matplotlib: pip install matplotlib")
        return None

    ee, _, ws = build_reachable_workspace(
        ee_cloud, joint_configs, front_x_min, front_x_max
    )
    _print_workspace(ws, front_x_min, front_x_max)

    kin4 = None
    if show_kinematic:
        print("  采样 4-DOF 运动学包络（无碰撞，灰色参考）...")
        kin4 = _sample_kinematic_envelope(12, front_x_min, front_x_max)
        print(f"    运动学前方点: {len(kin4)}")

    fig = plt.figure(figsize=(11, 9))
    ax = fig.add_subplot(111, projection="3d")
    if kin4 is not None and len(kin4) > 0:
        s4 = max(1, len(kin4) // 5000)
        ax.scatter(
            kin4[::s4, 0], kin4[::s4, 1], kin4[::s4, 2],
            c="0.55", s=2, alpha=0.22, depthshade=True,
            label=f"运动学包络 ({len(kin4)})",
        )
    ax.scatter(
        ee[:, 0], ee[:, 1], ee[:, 2],
        c="red", s=4, alpha=0.65, depthshade=True,
        label=f"碰撞-free ({len(ee)})",
    )
    _draw_box_wireframe(ax, ws.lo, ws.hi, color="royalblue", linewidth=1.4)
    _draw_box_wireframe(
        ax,
        np.array([ws.mid[0], ws.lo[1], ws.lo[2]]),
        np.array([ws.hi[0], ws.mid[1], ws.hi[2]]),
        color="orange", linestyle="--", linewidth=1.0, alpha=0.85,
    )
    ax.set_xlabel("X (前方, m)")
    ax.set_ylabel("Y (左+, m)")
    ax.set_zlabel("Z (上, m)")
    ax.set_title(
        f"G1 右手 4-DOF 可行域 (正前方 x∈[{front_x_min:.2f},{front_x_max:.2f}])\n"
        f"红=碰撞-free | 灰=运动学 | 蓝=可达盒 | 橙虚线=八分区分割",
        fontsize=10,
    )
    ax.legend(loc="upper left", fontsize=9)
    ax.view_init(elev=22, azim=-58)
    plt.tight_layout()
    plt.savefig(out_3d, dpi=160, bbox_inches="tight")
    print(f"  3D 图 -> {out_3d}")
    if show_plot:
        plt.show()
    else:
        plt.close(fig)

    fig2, axes = plt.subplots(1, 3, figsize=(12, 4))
    fig2.suptitle("正前方可达域 — YZ / XZ / XY 切片 @ mid", fontsize=10)
    axes[0].scatter(ee[:, 1], ee[:, 2], c="red", s=2, alpha=0.4)
    axes[0].axvline(ws.mid[1], color="orange", ls="--", lw=0.8)
    axes[0].axhline(ws.mid[2], color="orange", ls="--", lw=0.8)
    axes[0].set_xlabel("Y"); axes[0].set_ylabel("Z")
    axes[0].set_title(f"X≈{ws.mid[0]:.2f}")
    axes[0].set_aspect("equal"); axes[0].grid(True, alpha=0.3)

    axes[1].scatter(ee[:, 0], ee[:, 2], c="red", s=2, alpha=0.4)
    axes[1].axvline(ws.mid[0], color="orange", ls="--", lw=0.8)
    axes[1].axhline(ws.mid[2], color="orange", ls="--", lw=0.8)
    axes[1].set_xlabel("X"); axes[1].set_ylabel("Z")
    axes[1].set_title(f"Y≈{ws.mid[1]:.2f}")
    axes[1].set_aspect("equal"); axes[1].grid(True, alpha=0.3)

    axes[2].scatter(ee[:, 0], ee[:, 1], c="red", s=2, alpha=0.4)
    axes[2].axvline(ws.mid[0], color="orange", ls="--", lw=0.8)
    axes[2].axhline(ws.mid[1], color="orange", ls="--", lw=0.8)
    axes[2].set_xlabel("X"); axes[2].set_ylabel("Y")
    axes[2].set_title(f"Z≈{ws.mid[2]:.2f}")
    axes[2].set_aspect("equal"); axes[2].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_slices, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"  切片图 -> {out_slices}")

    return ee, ws, kin4


def _add_markers(scene, positions, size=0.007, rgba=None):
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


def _lock_recorder_upper(model, data):
    """与 g1_recorder 一致：拖手时腕+腰锁定为 0。"""
    for n in [
        "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
        "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
        "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    ]:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)
        if jid < 0:
            continue
        data.qpos[int(model.jnt_qposadr[jid])] = 0.0
        data.qvel[int(model.jnt_dofadr[jid])] = 0.0


def simulate_sphere_drag(sphere_xyz: np.ndarray, n_steps: int = 400):
    """
    在 scene_g1_draggable 中模拟拖黄色球到目标位置（与 g1_recorder 约束一致）。
    返回 (wrist_pos, hand_pos, q4)。
    """
    model = mujoco.MjModel.from_xml_path(DRAGGABLE_XML)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    sp_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "sphere_right_hand")
    ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, EE_BODY)
    hand_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, HAND_BODY)
    model.body_pos[sp_id] = np.asarray(sphere_xyz, dtype=np.float64)
    for _ in range(n_steps):
        mujoco.mj_step(model, data)
        _lock_recorder_upper(model, data)
    wrist = data.xpos[ee_id].copy()
    hand = data.xpos[hand_id].copy() if hand_id >= 0 else wrist.copy()
    q4 = []
    for n in ACTIVE_JOINTS:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)
        q4.append(float(data.qpos[int(model.jnt_qposadr[jid])]))
    return wrist, hand, np.array(q4, dtype=np.float64)


def probe_sphere_drag(sphere_xyz: np.ndarray, ee_cloud: np.ndarray,
                      joint_configs: np.ndarray,
                      front_x_min: float, front_x_max: float):
    """验证拖拽球目标位置对应的腕部姿态是否在可行域内。"""
    wrist, hand, q4 = simulate_sphere_drag(sphere_xyz)
    diff = np.linalg.norm(ee_cloud - wrist, axis=1)
    j = int(np.argmin(diff))
    dmin = float(diff[j])
    in_front = front_x_min <= wrist[0] <= front_x_max
    ok = dmin <= PROBE_OK_DIST
    print(f"  拖拽球目标: {sphere_xyz}")
    print(f"  仿真后腕部: x={wrist[0]:.3f} y={wrist[1]:.3f} z={wrist[2]:.3f}")
    print(f"  仿真后手掌: x={hand[0]:.3f} y={hand[1]:.3f} z={hand[2]:.3f}")
    print(f"  4 关节角: {[round(x, 3) for x in q4]}")
    print(f"  最近缓存腕点距离: {dmin:.4f} m (阈值 {PROBE_OK_DIST} m)")
    print(f"  在正前方带 x∈[{front_x_min},{front_x_max}]: {in_front}")
    print(f"  判定: {'在可行域内' if ok and in_front else '不在可行域内'}")


def mujoco_viewer_workspace(ee: np.ndarray, ws: ReachableWorkspace,
                            kin4: np.ndarray | None = None):
    model = mujoco.MjModel.from_xml_path(DRAGGABLE_XML)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    for _ in range(300):
        mujoco.mj_step(model, data)
        _lock_recorder_upper(model, data)

    print(f"\n[MuJoCo] scene_g1_draggable（与 recorder 相同）+ 红=腕部可行域 | ESC 退出")
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            try:
                scn = viewer.user_scn
                scn.ngeom = 0
                if kin4 is not None and len(kin4) > 0:
                    s4 = max(1, len(kin4) // 1200)
                    _add_markers(
                        scn, kin4[::s4], size=0.005,
                        rgba=np.array([0.45, 0.48, 0.52, 0.45], dtype=np.float32),
                    )
                s = max(1, len(ee) // 2500)
                _add_markers(
                    scn, ee[::s], size=0.006,
                    rgba=np.array([1.0, 0.12, 0.12, 0.75], dtype=np.float32),
                )
                corners = np.array([
                    ws.lo,
                    [ws.hi[0], ws.lo[1], ws.lo[2]],
                    [ws.hi[0], ws.hi[1], ws.lo[2]],
                    [ws.lo[0], ws.hi[1], ws.lo[2]],
                    [ws.lo[0], ws.lo[1], ws.hi[2]],
                    [ws.hi[0], ws.lo[1], ws.hi[2]],
                    ws.hi,
                    [ws.lo[0], ws.hi[1], ws.hi[2]],
                ])
                _add_markers(
                    scn, corners, size=0.012,
                    rgba=np.array([0.2, 0.4, 1.0, 0.85], dtype=np.float32),
                )
            except Exception:
                pass
            viewer.sync()
            time.sleep(0.02)


def probe_pose(q4: np.ndarray, ee_cloud: np.ndarray, joint_configs: np.ndarray,
               front_x_min: float, front_x_max: float):
    ee = fk_ee_from_4dof(q4)
    diff = np.linalg.norm(ee_cloud - ee, axis=1)
    j = int(np.argmin(diff))
    dmin = float(diff[j])
    in_front = front_x_min <= ee[0] <= front_x_max
    ok = dmin <= PROBE_OK_DIST
    print(f"  FK 末端: x={ee[0]:.3f} y={ee[1]:.3f} z={ee[2]:.3f}")
    print(f"  最近缓存点距离: {dmin:.4f} m (阈值 {PROBE_OK_DIST} m)")
    print(f"  最近关节角: {joint_configs[j]}")
    print(f"  在正前方带 x∈[{front_x_min},{front_x_max}]: {in_front}")
    print(f"  判定: {'在可行域内' if ok and in_front else '不在可行域内'}")


# ══════════════════════════════════════════════
# 主流程
# ══════════════════════════════════════════════

def build_workspace_cache(
    n_per_joint: int = 14,
    n_random: int = 30000,
    save_path: str = WS_CACHE_PATH,
    front_x_min: float = DEFAULT_FRONT_X_MIN,
    front_x_max: float = DEFAULT_FRONT_X_MAX,
    show_plot: bool = False,
    mujoco_viewer: bool = False,
    show_kinematic: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    print(f"\n[4-DOF 可行域] 模型={ROBOT_XML}")
    print(f"  网格 {n_per_joint}^4 + 随机 {n_random}")
    print(f"  障碍: pelvis/torso/head | 碰撞: 腕+手穿透（上臂轻触不算）")
    t0 = time.perf_counter()
    pts_g, joints_g, total, n_col = _sample_workspace_grid(n_per_joint)
    pts_r, joints_r = _sample_workspace_random(n_random)
    pts, joints = _merge_samples(pts_g, joints_g, pts_r, joints_r)
    elapsed = time.perf_counter() - t0
    print(f"  完成 {elapsed:.1f}s | 网格有效 {len(pts_g)} (剔除 {n_col}/{total}) "
          f"| 随机有效 {len(pts_r)} | 合并 {len(pts)}")
    if len(pts) == 0:
        raise RuntimeError("无有效可行域点")

    np.savez_compressed(
        save_path,
        ee_positions=pts,
        joint_configs=joints,
        obstacle_preset="minimal",
        collision_mode="full_arm_penetration",
        robot_xml=ROBOT_XML,
        front_x_min=np.float32(front_x_min),
        front_x_max=np.float32(front_x_max),
    )
    print(f"  缓存 -> {save_path}")

    result = visualize_workspace(
        pts, joints, front_x_min, front_x_max,
        show_plot=show_plot, show_kinematic=show_kinematic,
    )
    if result is not None and mujoco_viewer:
        ee, ws, kin4 = result
        mujoco_viewer_workspace(ee, ws, kin4)

    return pts, joints


def load_workspace_cache(path: str = WS_CACHE_PATH):
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"{path} 不存在，请先运行: python g1_right_hand_workspace.py"
        )
    d = np.load(path)
    return d["ee_positions"].astype(np.float32), d["joint_configs"].astype(np.float32)


def main():
    p = argparse.ArgumentParser(description="G1 右手 4-DOF 可行域分析")
    p.add_argument("--grid", type=int, default=14, help="每关节网格采样数")
    p.add_argument("--random", type=int, default=30000, help="随机补充采样数")
    p.add_argument("--front-x-min", type=float, default=DEFAULT_FRONT_X_MIN)
    p.add_argument("--front-x-max", type=float, default=DEFAULT_FRONT_X_MAX)
    p.add_argument("--show-plot", action="store_true", help="显示 matplotlib 窗口")
    p.add_argument("--mujoco-viewer", action="store_true", help="MuJoCo 叠加红点")
    p.add_argument("--no-kinematic", action="store_true", help="不绘制灰色运动学包络")
    p.add_argument("--probe-q4", type=float, nargs=4, metavar=("SP", "SR", "SY", "EL"),
                   help="验证一组 4 关节角是否在可行域内")
    p.add_argument("--probe-sphere", type=float, nargs=3, metavar=("X", "Y", "Z"),
                   help="模拟拖拽黄球到世界坐标 (x,y,z) 米，验证腕部是否在可行域")
    p.add_argument("--cache", type=str, default=WS_CACHE_PATH, help="NPZ 缓存路径")
    p.add_argument("--ws-cache", action="store_true",
                   help="使用默认缓存 recordings/g1_ws_4dof_front.npz（与 multi_shape 脚本同名参数）")
    args = p.parse_args()

    if args.ws_cache:
        args.cache = WS_CACHE_PATH

    if args.probe_q4 or args.probe_sphere:
        ee_cloud, joint_configs = load_workspace_cache(args.cache)
        if args.probe_q4:
            print("探测 4 关节角 vs 可行域缓存:")
            probe_pose(
                np.array(args.probe_q4, dtype=np.float64),
                ee_cloud, joint_configs,
                args.front_x_min, args.front_x_max,
            )
        if args.probe_sphere:
            print("模拟拖拽球 vs 可行域缓存:")
            probe_sphere_drag(
                np.array(args.probe_sphere, dtype=np.float64),
                ee_cloud, joint_configs,
                args.front_x_min, args.front_x_max,
            )
        return

    build_workspace_cache(
        n_per_joint=args.grid,
        n_random=args.random,
        save_path=args.cache,
        front_x_min=args.front_x_min,
        front_x_max=args.front_x_max,
        show_plot=args.show_plot,
        mujoco_viewer=args.mujoco_viewer,
        show_kinematic=not args.no_kinematic,
    )


if __name__ == "__main__":
    main()
