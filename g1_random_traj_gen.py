"""
G1 random multi-shape right-hand trajectory generator (4-DOF IK).

Does NOT modify g1_traj_gen.py. New workflow:
  1) workspace: 4-DOF joint sampling + collision filter -> cache EE cloud
  2) generate: stratified shape placement (L/R, U/D, F/B, large/small) then IK @ 50Hz
  3) --preview or --batch

Examples:
  python g1_random_traj_gen.py --mode workspace
  python g1_random_traj_gen.py --mode visualize_ws --ws-cache
  python g1_random_traj_gen.py --preview --num-shapes 8 --save
  python g1_random_traj_gen.py --batch --num-shapes 8 --save --ws-cache
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
from scipy.spatial.transform import Rotation

from g1_shape_library import SHAPE_NAMES, make_shape_waypoints

try:
    import imageio.v2 as imageio
    HAS_IMAGEIO = True
except ImportError:
    imageio = None
    HAS_IMAGEIO = False

os.makedirs("recordings", exist_ok=True)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_PATH = "g1/scene_g1_draggable.xml"
PLAIN_XML = "g1/g1_29dof.xml"
EE_BODY = "right_wrist_yaw_link"
WS_CACHE_PATH = "recordings/workspace_4dof_ee.npz"

# 与 g1_recorder 拖右手一致：肩 pitch/roll/yaw + 肘；腕三关节锁定为 0
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

_RIGHT_ARM_BODY_NAMES = [
    "right_shoulder_pitch_link", "right_shoulder_roll_link",
    "right_shoulder_yaw_link", "right_elbow_link",
    "right_wrist_roll_link", "right_wrist_pitch_link",
    "right_wrist_yaw_link",
]

_OBSTACLE_BODY_NAMES = [
    "pelvis", "torso_link", "head_link",
    "waist_yaw_link", "waist_roll_link",
    "left_shoulder_pitch_link", "left_shoulder_roll_link",
    "left_shoulder_yaw_link", "left_elbow_link",
    "left_wrist_roll_link", "left_wrist_pitch_link",
    "left_wrist_yaw_link",
]

# 仅躯干/头/骨盆 — 用于 workspace 采样，避免左臂/腰把可达域裁得过小
_OBSTACLE_BODY_NAMES_MINIMAL = ["pelvis", "torso_link", "head_link"]

HOME_PELVIS_POS = np.array([0.0, 0.0, 0.793])
HOME_PELVIS_QUAT = np.array([1.0, 0.0, 0.0, 0.0])

DEFAULT_FPS = 50.0
DEFAULT_FRAMES_PER_SHAPE = 250
DEFAULT_TRANSITION_FRAMES = 25
DEFAULT_NUM_SHAPES = 8

# Robot faces +X; trajectories only in front of pelvis (world frame)
DEFAULT_FRONT_X_MIN = 0.15
DEFAULT_FRONT_X_MAX = 0.50

# Scale caps after margin fit (meters, local shape radius ~1)
SCALE_LARGE = (0.065, 0.10)
SCALE_SMALL = (0.035, 0.065)
WS_PROXIMITY = 0.038  # max dist to nearest EE sample (~14^4 grid spacing)
# Feasibility tied to discrete workspace resolution (not sub-mm IK tolerance)
IK_ERR_THRESH = WS_PROXIMITY
IK_MIN_OK_RATIO = 0.96

# Shapes with sharp corners: pre-blend Cartesian targets toward workspace
SHARP_SHAPES = frozenset({
    "star", "pentagon", "diamond", "heart", "random_polygon", "random_spline",
})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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


def _build_collision_id_sets(model, obstacle_names=None):
    if obstacle_names is None:
        obstacle_names = _OBSTACLE_BODY_NAMES

    def _ids(names):
        s = set()
        for n in names:
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n)
            if bid >= 0:
                s.add(bid)
        return s
    return _ids(_RIGHT_ARM_BODY_NAMES), _ids(obstacle_names)


def _has_self_collision(
    model, data, arm_ids, obstacle_ids,
    mode: str = "penetration",
):
    """
    penetration: 仅真实穿透算碰撞（与 MuJoCo 拖拽时轻触仍可用一致）
    touch: dist < 2mm 也算碰撞（旧逻辑，会删掉大量拖拽可达姿态）
    """
    if mode == "none":
        return False
    thresh = -0.0005 if mode == "penetration" else 0.002
    for i in range(data.ncon):
        c = data.contact[i]
        if c.dist > thresh:
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


# ---------------------------------------------------------------------------
# 4-DOF IK
# ---------------------------------------------------------------------------

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
        """Initialize from a collision-free workspace sample (4 active joints)."""
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


# ---------------------------------------------------------------------------
# Workspace
# ---------------------------------------------------------------------------

def _sample_workspace_grid(
    n_per_joint: int,
    active_joint_names: list[str],
    locked: dict[str, float],
    check_collision: bool = True,
    obstacle_names: list[str] | None = None,
    collision_mode: str = "penetration",
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Grid-sample EE positions; returns (ee_pts, joint_cfgs, n_total, n_filtered)."""
    model = mujoco.MjModel.from_xml_path(PLAIN_XML)
    data = mujoco.MjData(model)
    all_joint_names = active_joint_names + list(locked.keys())
    addr = _build_addr_table(model, all_joint_names)
    _reset_to_home(model, data, addr)
    ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, EE_BODY)
    arm_ids, obstacle_ids = _build_collision_id_sets(model, obstacle_names)

    ranges = [addr[n][2] for n in active_joint_names]
    grids = [np.linspace(lo, hi, n_per_joint) for lo, hi in ranges]
    n_active = len(active_joint_names)
    total = n_per_joint ** n_active

    ee_pts, joint_cfgs = [], []
    n_col = 0

    def _recurse(dim: int, qvals: list[float]):
        nonlocal n_col
        if dim == n_active:
            for name, val in locked.items():
                data.qpos[addr[name][0]] = val
            mujoco.mj_forward(model, data)
            if check_collision and _has_self_collision(
                model, data, arm_ids, obstacle_ids, mode=collision_mode
            ):
                n_col += 1
                return
            ee_pts.append(data.xpos[ee_id].copy())
            joint_cfgs.append(qvals.copy())
            return
        for q in grids[dim]:
            data.qpos[addr[active_joint_names[dim]][0]] = q
            qvals.append(float(q))
            _recurse(dim + 1, qvals)
            qvals.pop()

    _recurse(0, [])
    pts = np.array(ee_pts, dtype=np.float32)
    joints = np.array(joint_cfgs, dtype=np.float32)
    return pts, joints, total, n_col


def _sample_workspace_random_4dof(
    n_random: int,
    obstacle_names: list[str],
    collision_mode: str = "penetration",
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Random uniform 4-DOF samples (fills gaps between grid nodes)."""
    model = mujoco.MjModel.from_xml_path(PLAIN_XML)
    data = mujoco.MjData(model)
    addr = _build_addr_table(model, ACTIVE_JOINTS + list(LOCKED_JOINTS.keys()))
    _reset_to_home(model, data, addr)
    ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, EE_BODY)
    arm_ids, obstacle_ids = _build_collision_id_sets(model, obstacle_names)
    rng = np.random.default_rng(seed)
    ee_pts, joint_cfgs = [], []
    ranges = [addr[n][2] for n in ACTIVE_JOINTS]
    for _ in range(n_random):
        qvals = [float(rng.uniform(lo, hi)) for lo, hi in ranges]
        for i, n in enumerate(ACTIVE_JOINTS):
            data.qpos[addr[n][0]] = qvals[i]
        for name, val in LOCKED_JOINTS.items():
            data.qpos[addr[name][0]] = val
        mujoco.mj_forward(model, data)
        if _has_self_collision(model, data, arm_ids, obstacle_ids, mode=collision_mode):
            continue
        ee_pts.append(data.xpos[ee_id].copy())
        joint_cfgs.append(qvals)
    return np.array(ee_pts, dtype=np.float32), np.array(joint_cfgs, dtype=np.float32)


def _merge_workspace_samples(
    ee_a: np.ndarray, joints_a: np.ndarray,
    ee_b: np.ndarray, joints_b: np.ndarray,
    dedup_tol: float = 0.004,
) -> tuple[np.ndarray, np.ndarray]:
    """Concatenate and drop near-duplicate EE points."""
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


def analyze_workspace_4dof(
    n_per_joint: int = 14,
    n_random: int = 30000,
    save_path: str = WS_CACHE_PATH,
    show_plot: bool = False,
    obstacle_preset: str = "minimal",
    collision_mode: str = "penetration",
):
    try:
        import matplotlib
        if not show_plot:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib required: pip install matplotlib")
        return None

    obstacles = (
        _OBSTACLE_BODY_NAMES_MINIMAL if obstacle_preset == "minimal"
        else _OBSTACLE_BODY_NAMES
    )
    total = n_per_joint ** 4
    print(f"\n[Workspace 4DOF] grid {n_per_joint}^4 + random {n_random} "
          f"| obstacles={obstacle_preset} | collision={collision_mode}")
    t0 = time.perf_counter()
    pts_g, joints_g, total, n_col = _sample_workspace_grid(
        n_per_joint,
        ACTIVE_JOINTS,
        LOCKED_JOINTS,
        check_collision=True,
        obstacle_names=obstacles,
        collision_mode=collision_mode,
    )
    pts_r, joints_r = _sample_workspace_random_4dof(
        n_random, obstacles, collision_mode=collision_mode
    )
    pts, joints = _merge_workspace_samples(pts_g, joints_g, pts_r, joints_r)
    elapsed = time.perf_counter() - t0
    print(f"  Done {elapsed:.1f}s | grid valid {len(pts_g)} (filtered {n_col}) "
          f"| random valid {len(pts_r)} | merged {len(pts)}")
    if len(pts) == 0:
        print("  No valid points!")
        return None
    print(f"  EE X: [{pts[:,0].min():.3f}, {pts[:,0].max():.3f}]")
    print(f"  EE Y: [{pts[:,1].min():.3f}, {pts[:,1].max():.3f}]")
    print(f"  EE Z: [{pts[:,2].min():.3f}, {pts[:,2].max():.3f}]")
    print("  Same 4 DOF as g1_recorder (wrist locked); collision=penetration only.")

    np.savez_compressed(
        save_path,
        ee_positions=pts,
        joint_configs=joints,
        obstacle_preset=obstacle_preset,
        collision_mode=collision_mode,
    )
    print(f"  Cache saved -> {save_path}")

    x_planes = np.linspace(
        max(pts[:, 0].min() + 0.02, 0.2),
        min(pts[:, 0].max() - 0.02, 0.55),
        4,
    )
    fig, axes = plt.subplots(1, len(x_planes), figsize=(4 * len(x_planes), 5))
    fig.suptitle("4DOF workspace (collision-free)", fontsize=11)
    for ax, xp in zip(axes, x_planes):
        mask = np.abs(pts[:, 0] - xp) < 0.025
        sl = pts[mask]
        ax.scatter(sl[:, 1], sl[:, 2], s=1, alpha=0.25)
        ax.set_title(f"x={xp:.2f} ({len(sl)} pts)")
        ax.set_xlabel("Y"); ax.set_ylabel("Z")
        ax.set_aspect("equal"); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("workspace_4dof.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Plot -> workspace_4dof.png")
    if show_plot:
        plt.show()
    return pts


def load_workspace_cache(
    path: str = WS_CACHE_PATH,
    with_joints: bool = True,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"{path} not found. Run: python g1_random_traj_gen.py --mode workspace"
        )
    d = np.load(path)
    ee = d["ee_positions"].astype(np.float32)
    if with_joints and "joint_configs" in d:
        return ee, d["joint_configs"].astype(np.float32)
    return ee


def filter_workspace_front(
    ee_cloud: np.ndarray,
    joint_configs: np.ndarray,
    x_min: float = DEFAULT_FRONT_X_MIN,
    x_max: float = DEFAULT_FRONT_X_MAX,
) -> tuple[np.ndarray, np.ndarray]:
    """Filter EE cloud and matching 4-DOF joint rows to the front band."""
    m = (ee_cloud[:, 0] >= x_min) & (ee_cloud[:, 0] <= x_max)
    filtered_ee = ee_cloud[m]
    if len(filtered_ee) < 50:
        raise RuntimeError(
            f"Too few workspace points in front band x=[{x_min},{x_max}]: "
            f"{len(filtered_ee)} (of {len(ee_cloud)}). "
            "Re-run --mode workspace or widen --front-x-min/--front-x-max."
        )
    return filtered_ee, joint_configs[m]


@dataclasses.dataclass
class ReachableWorkspace:
    """
    Right-hand reachable region AFTER front constraint (x band filter).
    All 左/右/上/下/前/后 labels are relative to this box only, not the full robot.
    """
    lo: np.ndarray   # min corner (x,y,z)
    hi: np.ndarray   # max corner
    mid: np.ndarray  # center = (lo+hi)/2, splits each axis into two halves
    n_samples: int
    x_band: tuple[float, float]


def build_reachable_workspace(
    ee_cloud: np.ndarray,
    joint_configs: np.ndarray,
    x_min: float = DEFAULT_FRONT_X_MIN,
    x_max: float = DEFAULT_FRONT_X_MAX,
) -> tuple[np.ndarray, np.ndarray, ReachableWorkspace]:
    """Filter to front EE cloud, then define stratification box from its bounds."""
    ee, joints = filter_workspace_front(ee_cloud, joint_configs, x_min, x_max)
    lo = ee.min(axis=0)
    hi = ee.max(axis=0)
    ws = ReachableWorkspace(
        lo=lo,
        hi=hi,
        mid=(lo + hi) * 0.5,
        n_samples=len(ee),
        x_band=(x_min, x_max),
    )
    return ee, joints, ws


def _draw_box_wireframe(ax, lo, hi, color="0.4", linestyle="-", linewidth=1.0, alpha=0.9):
    """Draw 3D box edges on matplotlib Axes3D."""
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


def sample_kinematic_envelope_4dof(
    n_per_joint: int = 10,
    front_x_min: float = DEFAULT_FRONT_X_MIN,
    front_x_max: float = DEFAULT_FRONT_X_MAX,
) -> np.ndarray:
    """4-DOF grid without collision check (motion limit only)."""
    pts, _, _, _ = _sample_workspace_grid(
        n_per_joint, ACTIVE_JOINTS, LOCKED_JOINTS, check_collision=False
    )
    m = (pts[:, 0] >= front_x_min) & (pts[:, 0] <= front_x_max)
    return pts[m]


def fk_ee_from_4dof(q4: np.ndarray) -> np.ndarray:
    """Forward kinematics on g1_29dof.xml (same model as workspace cache)."""
    model = mujoco.MjModel.from_xml_path(PLAIN_XML)
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


def probe_pose_in_workspace(
    q4: np.ndarray,
    ee_cloud: np.ndarray,
    joint_configs: np.ndarray,
) -> None:
    """Print whether a 4-DOF pose (e.g. copied from recorder UI) lies near the cache."""
    ee = fk_ee_from_4dof(q4)
    diff = np.linalg.norm(ee_cloud - ee, axis=1)
    j = int(np.argmin(diff))
    print(f"  FK EE: x={ee[0]:.3f} y={ee[1]:.3f} z={ee[2]:.3f}")
    print(f"  Nearest cache EE dist={diff[j]:.4f} m")
    print(f"  Nearest joints: {joint_configs[j]}")
    in_front = DEFAULT_FRONT_X_MIN <= ee[0] <= DEFAULT_FRONT_X_MAX
    print(f"  In front band x∈[{DEFAULT_FRONT_X_MIN},{DEFAULT_FRONT_X_MAX}]: {in_front}")


def visualize_front_reachable_workspace(
    ee_cloud: np.ndarray,
    joint_configs: np.ndarray,
    front_x_min: float = DEFAULT_FRONT_X_MIN,
    front_x_max: float = DEFAULT_FRONT_X_MAX,
    out_path: str = "workspace_front_reachable.png",
    show_plot: bool = False,
    mujoco_viewer: bool = False,
    show_kinematic: bool = True,
):
    """
    Plot front-filtered workspace:
      gray = 4-DOF kinematic envelope (no collision, wrist locked)
      red  = collision-free samples used for trajectory generation
    """
    try:
        import matplotlib
        if not show_plot and not mujoco_viewer:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    except ImportError:
        print("matplotlib required: pip install matplotlib")
        return None

    ee, _, ws = build_reachable_workspace(
        ee_cloud, joint_configs, front_x_min, front_x_max
    )
    _print_reachable_workspace(ws, front_x_min, front_x_max)

    kin4 = None
    if show_kinematic:
        print("  Sampling 4-DOF kinematic envelope (no collision, wrist locked) ...")
        kin4 = sample_kinematic_envelope_4dof(12, front_x_min, front_x_max)
        print(f"    kinematic front: {len(kin4)} pts | "
              f"Z [{kin4[:,2].min():.2f}, {kin4[:,2].max():.2f}]")
        print(f"  Collision-free red Z [{ee[:,2].min():.2f}, {ee[:,2].max():.2f}]")

    fig = plt.figure(figsize=(11, 9))
    ax = fig.add_subplot(111, projection="3d")
    if kin4 is not None and len(kin4) > 0:
        step4 = max(1, len(kin4) // 5000)
        ax.scatter(
            kin4[::step4, 0], kin4[::step4, 1], kin4[::step4, 2],
            c="0.55", s=2, alpha=0.22, depthshade=True,
            label=f"4-DOF kinematic ({len(kin4)})",
        )
    ax.scatter(
        ee[:, 0], ee[:, 1], ee[:, 2],
        c="red", s=4, alpha=0.65, depthshade=True,
        label=f"4-DOF collision-free ({len(ee)})",
    )
    _draw_box_wireframe(ax, ws.lo, ws.hi, color="royalblue", linewidth=1.4)
    _draw_box_wireframe(
        ax,
        np.array([ws.mid[0], ws.lo[1], ws.lo[2]]),
        np.array([ws.hi[0], ws.mid[1], ws.hi[2]]),
        color="orange", linestyle="--", linewidth=1.0, alpha=0.85,
    )
    ax.plot(
        [ws.mid[0], ws.mid[0]], [ws.lo[1], ws.hi[1]], [ws.lo[2], ws.lo[2]],
        color="orange", linestyle="--", linewidth=0.9,
    )
    ax.plot(
        [ws.mid[0], ws.mid[0]], [ws.lo[1], ws.hi[1]], [ws.hi[2], ws.hi[2]],
        color="orange", linestyle="--", linewidth=0.9,
    )
    ax.plot(
        [ws.lo[0], ws.hi[0]], [ws.mid[1], ws.mid[1]], [ws.mid[2], ws.mid[2]],
        color="orange", linestyle="--", linewidth=0.9,
    )

    ax.set_xlabel("X (forward, m)")
    ax.set_ylabel("Y (left+, m)")
    ax.set_zlabel("Z (up, m)")
    ax.set_title(
        f"G1 right wrist 4-DOF workspace (front x∈[{front_x_min:.2f},{front_x_max:.2f}] m)\n"
        f"red=collision-free | gray=kinematic (no collision) | blue=split box",
        fontsize=10,
    )
    ax.legend(loc="upper left", fontsize=9)
    ax.view_init(elev=22, azim=-58)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    print(f"  Saved plot -> {out_path}")
    if show_plot:
        plt.show()
    else:
        plt.close(fig)

    # 2D slices for quick reference
    slice_path = out_path.replace(".png", "_slices.png")
    fig2, axes = plt.subplots(1, 3, figsize=(12, 4))
    fig2.suptitle("Front reachable workspace — YZ / XZ / XY slices @ mid", fontsize=10)
    axes[0].scatter(ee[:, 1], ee[:, 2], c="red", s=2, alpha=0.4)
    axes[0].axvline(ws.mid[1], color="orange", ls="--", lw=0.8)
    axes[0].axhline(ws.mid[2], color="orange", ls="--", lw=0.8)
    axes[0].set_xlabel("Y"); axes[0].set_ylabel("Z")
    axes[0].set_title(f"view X≈{ws.mid[0]:.2f}")
    axes[0].set_aspect("equal"); axes[0].grid(True, alpha=0.3)

    axes[1].scatter(ee[:, 0], ee[:, 2], c="red", s=2, alpha=0.4)
    axes[1].axvline(ws.mid[0], color="orange", ls="--", lw=0.8)
    axes[1].axhline(ws.mid[2], color="orange", ls="--", lw=0.8)
    axes[1].set_xlabel("X"); axes[1].set_ylabel("Z")
    axes[1].set_title(f"view Y≈{ws.mid[1]:.2f}")
    axes[1].set_aspect("equal"); axes[1].grid(True, alpha=0.3)

    axes[2].scatter(ee[:, 0], ee[:, 1], c="red", s=2, alpha=0.4)
    axes[2].axvline(ws.mid[0], color="orange", ls="--", lw=0.8)
    axes[2].axhline(ws.mid[1], color="orange", ls="--", lw=0.8)
    axes[2].set_xlabel("X"); axes[2].set_ylabel("Y")
    axes[2].set_title(f"view Z≈{ws.mid[2]:.2f}")
    axes[2].set_aspect("equal"); axes[2].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(slice_path, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"  Saved slices -> {slice_path}")

    if mujoco_viewer:
        _visualize_front_reachable_mujoco(ee, ws, kin4=kin4)

    return ee, ws


def _visualize_front_reachable_mujoco(
    ee: np.ndarray,
    ws: ReachableWorkspace,
    kin4: np.ndarray | None = None,
):
    """MuJoCo: same model as workspace (g1_29dof) + red/gray EE points."""
    model = mujoco.MjModel.from_xml_path(PLAIN_XML)
    data = mujoco.MjData(model)
    addr = _build_addr_table(model, ACTIVE_JOINTS + list(LOCKED_JOINTS.keys()))
    _reset_to_home(model, data, addr)

    print(f"\n[MuJoCo] model={PLAIN_XML} | red={len(ee)} feasible | "
          f"gray=kinematic | ESC quit")
    print("  (Use this view to compare poses; scene_g1_draggable may differ slightly.)")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            try:
                scn = viewer.user_scn
                scn.ngeom = 0
                if kin4 is not None and len(kin4) > 0:
                    s4 = max(1, len(kin4) // 1200)
                    _add_red_markers(
                        scn, kin4[::s4], size=0.005,
                        rgba=np.array([0.45, 0.48, 0.52, 0.45], dtype=np.float32),
                    )
                s = max(1, len(ee) // 2500)
                _add_red_markers(
                    scn, ee[::s], size=0.006,
                    rgba=np.array([1.0, 0.12, 0.12, 0.75], dtype=np.float32),
                )
                # box corners (blue-ish)
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
                _add_red_markers(
                    scn, corners, size=0.012,
                    rgba=np.array([0.2, 0.4, 1.0, 0.85], dtype=np.float32),
                )
            except Exception:
                pass
            viewer.sync()
            time.sleep(0.02)


def region_bounds(
    ws: ReachableWorkspace,
    left: bool,
    up: bool,
    front: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Octant inside the reachable box [ws.lo, ws.hi]:
      前/后 = upper/lower x half (farther/closer within reachable x extent)
      左/右 = upper/lower y half (robot +Y = left)
      上/下 = upper/lower z half
    """
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


def waypoints_in_front_band(
    waypoints: np.ndarray,
    x_min: float,
    x_max: float,
    margin: float = 0.01,
) -> bool:
    """All trajectory points must lie in the forward x band."""
    xs = waypoints[:, 0]
    return bool(
        xs.min() >= x_min - margin and xs.max() <= x_max + margin
    )


def _binary_flags(n: int, rng: np.random.Generator) -> np.ndarray:
    """Exactly n//2 True and n - n//2 False, shuffled."""
    flags = np.zeros(n, dtype=bool)
    flags[: n // 2] = True
    rng.shuffle(flags)
    return flags


def _shape_local_template(shape_name: str, rng: np.random.Generator) -> np.ndarray:
    """Unit local polyline (N=48) for extent / margin math."""
    from g1_shape_library import (
        make_circle_local, make_square_local, make_rectangle_local,
        make_triangle_local, make_pentagon_local, make_star_local,
        make_heart_local, make_wave_local, make_diamond_local,
        make_random_polygon_local, make_random_spline_local,
    )
    name = shape_name.lower()
    n = 48
    makers = {
        "circle": lambda: make_circle_local(n),
        "square": lambda: make_square_local(n),
        "rectangle": lambda: make_rectangle_local(n),
        "triangle": lambda: make_triangle_local(n),
        "pentagon": lambda: make_pentagon_local(n),
        "star": lambda: make_star_local(n),
        "heart": lambda: make_heart_local(n),
        "wave": lambda: make_wave_local(n),
        "diamond": lambda: make_diamond_local(n),
        "random_polygon": lambda: make_random_polygon_local(n, rng),
        "random_spline": lambda: make_random_spline_local(n, rng),
    }
    return makers[name]()


def _yz_extents_at_theta(local_yz: np.ndarray, theta: float) -> tuple[float, float]:
    """Max |delta y| and |delta z| per unit scale at rotation theta."""
    c, s = np.cos(theta), np.sin(theta)
    yl, zl = local_yz[:, 0], local_yz[:, 1]
    dy = np.abs(yl * c - zl * s)
    dz = np.abs(yl * s + zl * c)
    return float(dy.max()), float(dz.max())


def plan_shapes_stratified(
    num_shapes: int,
    rng: np.random.Generator,
) -> list[dict]:
    """
    ~N/2 per side within the reachable workspace box (L/R, U/D, F/B) + large/small.
    Each shape also gets a random type and rotation.
    """
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


def _region_mask(
    ee_cloud: np.ndarray,
    ws: ReachableWorkspace,
    left: bool,
    up: bool,
    front: bool,
) -> np.ndarray:
    lo_r, hi_r = region_bounds(ws, left, up, front)
    eps = 1e-6
    return (
        (ee_cloud[:, 0] >= lo_r[0] - eps) & (ee_cloud[:, 0] <= hi_r[0] + eps) &
        (ee_cloud[:, 1] >= lo_r[1] - eps) & (ee_cloud[:, 1] <= hi_r[1] + eps) &
        (ee_cloud[:, 2] >= lo_r[2] - eps) & (ee_cloud[:, 2] <= hi_r[2] + eps)
    )


def pick_center_in_region(
    ee_cloud: np.ndarray,
    ws: ReachableWorkspace,
    plan: dict,
    rng: np.random.Generator,
) -> np.ndarray:
    """Center = random EE sample inside the reachable octant for this plan."""
    mask = _region_mask(ee_cloud, ws, plan["left"], plan["up"], plan["front"])
    pts = ee_cloud[mask]
    if len(pts) < 5:
        lo_r, hi_r = region_bounds(ws, plan["left"], plan["up"], plan["front"])
        target = (lo_r + hi_r) * 0.5
        idx = int(np.argmin(np.linalg.norm(ee_cloud - target, axis=1)))
        return ee_cloud[idx].copy()
    return pts[rng.integers(0, len(pts))].copy()


def compute_safe_scale(
    center: np.ndarray,
    theta: float,
    shape_name: str,
    plan: dict,
    ws: ReachableWorkspace,
    rng: np.random.Generator,
    margin: float = 0.025,
) -> float:
    """Fit scale inside the reachable octant box (same frame as L/R/U/D/F/B)."""
    lo, hi = region_bounds(ws, plan["left"], plan["up"], plan["front"])

    local = _shape_local_template(shape_name, rng)
    ext_y, ext_z = _yz_extents_at_theta(local, theta)
    ext_y = max(ext_y, 1e-6)
    ext_z = max(ext_z, 1e-6)

    cy, cz = float(center[1]), float(center[2])
    scale_y = min(cy - lo[1] - margin, hi[1] - cy - margin) / ext_y
    scale_z = min(cz - lo[2] - margin, hi[2] - cz - margin) / ext_z
    scale_fit = min(scale_y, scale_z)

    lo_r, hi_r = SCALE_LARGE if plan["large"] else SCALE_SMALL
    scale_target = float(rng.uniform(lo_r, hi_r))
    return float(max(0.03, min(scale_target, scale_fit * 0.92)))


def waypoints_near_workspace(
    waypoints: np.ndarray,
    ee_cloud: np.ndarray,
    max_dist: float = WS_PROXIMITY,
) -> tuple[bool, float]:
    """Every waypoint must lie near a sampled reachable EE pose."""
    diff = waypoints[:, None, :] - ee_cloud[None, :, :]
    dmin = np.linalg.norm(diff, axis=2).min(axis=1)
    return bool(dmin.max() <= max_dist), float(dmin.max())


def fit_shape_waypoints_to_workspace(
    shape: str,
    center: np.ndarray,
    scale: float,
    theta: float,
    frames_per_shape: int,
    ee_cloud: np.ndarray,
    rng: np.random.Generator,
    max_attempts: int = 8,
) -> tuple[np.ndarray, float]:
    """Shrink scale until all waypoints lie within WS_PROXIMITY of the EE cloud."""
    scale = float(scale)
    last_dmax = 0.0
    for _ in range(max_attempts):
        shape_wps = make_shape_waypoints(
            shape, center, scale, theta, N=frames_per_shape, rng=rng
        )
        ok_ws, last_dmax = waypoints_near_workspace(shape_wps, ee_cloud)
        if ok_ws:
            return shape_wps, scale
        if last_dmax > 1e-6:
            scale *= min(0.92, 0.97 * WS_PROXIMITY / last_dmax)
        else:
            scale *= 0.90
        scale = max(scale, 0.02)
    raise RuntimeError(
        f"Waypoints exceed workspace after {max_attempts} scale reductions "
        f"(max_dist={last_dmax:.3f}m > {WS_PROXIMITY}m)"
    )


# ---------------------------------------------------------------------------
# Trajectory IK + feasibility
# ---------------------------------------------------------------------------

def _nearest_workspace_indices(
    waypoints: np.ndarray,
    ee_cloud: np.ndarray,
) -> np.ndarray:
    diff = waypoints[:, None, :] - ee_cloud[None, :, :]
    return np.argmin(np.linalg.norm(diff, axis=2), axis=1)


def assign_workspace_joint_path(
    waypoints: np.ndarray,
    ee_cloud: np.ndarray,
    joint_configs: np.ndarray,
    k_neighbors: int = 20,
) -> np.ndarray:
    """
    Ordered joint indices along the closed shape: each frame picks a nearby
    workspace sample that is close to the previous frame in joint space
    (avoids star/corner index jumps).
    """
    n_wp = len(waypoints)
    k_neighbors = min(k_neighbors, len(ee_cloud))
    idx = np.zeros(n_wp, dtype=int)
    idx[0] = int(np.argmin(np.linalg.norm(ee_cloud - waypoints[0], axis=1)))

    for i in range(1, n_wp):
        dist_i = np.linalg.norm(ee_cloud - waypoints[i], axis=1)
        cands = np.argpartition(dist_i, k_neighbors)[:k_neighbors]
        prev_q = joint_configs[idx[i - 1]]
        dj = np.linalg.norm(joint_configs[cands] - prev_q, axis=1)
        idx[i] = int(cands[np.argmin(dj)])
    return idx


def blend_waypoints_to_workspace(
    waypoints: np.ndarray,
    ee_cloud: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """Pull Cartesian targets toward nearest reachable EE samples."""
    idx = _nearest_workspace_indices(waypoints, ee_cloud)
    a = float(np.clip(alpha, 0.0, 1.0))
    return ((1.0 - a) * waypoints + a * ee_cloud[idx]).astype(np.float32)


def _smooth_joint_path_4d(q4: np.ndarray, passes: int = 2) -> np.ndarray:
    """Circular moving average on workspace joint lookups (closed shape)."""
    q = np.asarray(q4, dtype=np.float64).copy()
    n = len(q)
    if n < 3:
        return q
    for _ in range(passes):
        q_new = q.copy()
        for i in range(n):
            im, ip = (i - 1) % n, (i + 1) % n
            q_new[i] = 0.22 * q[im] + 0.56 * q[i] + 0.22 * q[ip]
        q = q_new
    return q


def _ee_error(model, data, ee_id: int, target: np.ndarray) -> float:
    return float(np.linalg.norm(target - data.xpos[ee_id]))


def _qpos_vel_from_sequence(qpos_arr: np.ndarray, fps: float, nv: int):
    N = len(qpos_arr)
    dt = 1.0 / fps
    qvel_arr = np.zeros((N, nv), dtype=np.float32)
    for i in range(1, N - 1):
        qvel_arr[i, 6:] = (qpos_arr[i + 1, 7:] - qpos_arr[i - 1, 7:]) / (2 * dt)
    qvel_arr[0] = qvel_arr[1]
    qvel_arr[N - 1] = qvel_arr[N - 2]
    timestamps = np.arange(N, dtype=np.float32) * dt
    return qvel_arr, timestamps


def solve_waypoints_ik_ws_seed(
    waypoints: np.ndarray,
    ee_cloud: np.ndarray,
    joint_configs: np.ndarray,
    fps: float,
    solver: IKSolver4DOF,
    model,
    data,
    arm_ids,
    obstacle_ids,
    max_iter: int = 120,
    err_thresh: float = IK_ERR_THRESH,
    show_progress: bool = False,
    ik_targets: np.ndarray | None = None,
):
    """
    Joint-tracked IK: smooth workspace joint lookups, polish toward targets.
    `waypoints` = shape geometry for error metric; `ik_targets` may be workspace-blended.
    """
    if ik_targets is None:
        ik_targets = waypoints
    path_idx = assign_workspace_joint_path(waypoints, ee_cloud, joint_configs)
    q4_smooth = _smooth_joint_path_4d(joint_configs[path_idx], passes=3)
    ee_id = solver.ee_id
    N = len(waypoints)
    qpos_list = []
    errors = []
    col_flags = []
    polish_thresh = min(err_thresh, 0.012)

    for i in range(N):
        wp = waypoints[i]
        tgt = ik_targets[i]
        snap = (0.50 * wp + 0.50 * ee_cloud[path_idx[i]]).astype(np.float64)

        solver.seed_joints(q4_smooth[i])
        solver._lock_wrist()
        mujoco.mj_forward(model, data)
        err = _ee_error(model, data, ee_id, wp)

        if err > polish_thresh:
            err, _ = solver.solve(tgt, max_iter=max_iter)
            err = _ee_error(model, data, ee_id, wp)

        if err > err_thresh:
            solver.seed_joints(joint_configs[path_idx[i]])
            solver._lock_wrist()
            mujoco.mj_forward(model, data)
            err, _ = solver.solve(snap, max_iter=max_iter * 2)
            err = _ee_error(model, data, ee_id, wp)

        if err > err_thresh:
            solver.seed_joints(joint_configs[path_idx[i]])
            solver._lock_wrist()
            mujoco.mj_forward(model, data)
            err = _ee_error(model, data, ee_id, wp)

        solver._lock_wrist()
        mujoco.mj_forward(model, data)
        col = _has_self_collision(model, data, arm_ids, obstacle_ids)
        qpos_list.append(data.qpos.copy())
        errors.append(err)
        col_flags.append(col)
        if show_progress and (i + 1) % 50 == 0:
            print(f"    IK {i+1}/{N}", end="\r", flush=True)
    if show_progress and N >= 50:
        print(f"    IK {N}/{N} done", flush=True)

    qpos_arr = np.array(qpos_list, dtype=np.float32)
    errors = np.array(errors)
    col_ratio = float(np.mean(col_flags))
    qvel_arr, timestamps = _qpos_vel_from_sequence(qpos_arr, fps, model.nv)
    return qpos_arr, qvel_arr, timestamps, errors, col_ratio


def make_joint_transition_qpos(
    q_from: np.ndarray,
    q_to: np.ndarray,
    n_frames: int,
    active_qpos_addrs: list[int],
) -> np.ndarray:
    """Interpolate only the 4 active arm joints (avoids Cartesian transition collisions)."""
    if n_frames <= 0:
        return np.zeros((0, len(q_from)), dtype=np.float32)
    out = []
    for t in np.linspace(0.0, 1.0, n_frames, endpoint=True):
        q = q_from.copy()
        for a in active_qpos_addrs:
            q[a] = (1.0 - t) * q_from[a] + t * q_to[a]
        out.append(q)
    return np.array(out, dtype=np.float32)


def precheck_waypoints_ik(waypoints, solver, model, data, arm_ids, obstacle_ids,
                          n_samples: int = 24, max_iter: int = 60):
    """Fast reject: IK only on sparse samples before full 250-point solve."""
    n = len(waypoints)
    if n <= n_samples:
        idx = np.arange(n)
    else:
        idx = np.unique(np.linspace(0, n - 1, n_samples, dtype=int))
    errs, cols = [], []
    for i in idx:
        err, _ = solver.solve(waypoints[i], max_iter=max_iter)
        solver._lock_wrist()
        mujoco.mj_forward(model, data)
        errs.append(err)
        cols.append(_has_self_collision(model, data, arm_ids, obstacle_ids))
    return np.array(errs), float(np.mean(cols))


def check_feasibility(
    errors,
    col_ratio,
    err_thresh: float = IK_ERR_THRESH,
    max_col_ratio: float = 0.05,
    min_ok_ratio: float = IK_MIN_OK_RATIO,
):
    ok = errors < err_thresh
    ok_ratio = ok.mean()
    return ok_ratio >= min_ok_ratio and col_ratio <= max_col_ratio


def make_transition_waypoints(p0, p1, n_frames):
    t = np.linspace(0, 1, n_frames, endpoint=True)
    pts = p0[None, :] * (1 - t[:, None]) + p1[None, :] * t[:, None]
    return pts.astype(np.float32)


def _plan_label(plan: dict) -> str:
    lr = "左" if plan["left"] else "右"
    ud = "上" if plan["up"] else "下"
    fb = "前" if plan["front"] else "后"
    sz = "大" if plan["large"] else "小"
    return f"{lr}/{ud}/{fb}/{sz}"


def _print_reachable_workspace(ws: ReachableWorkspace, front_x_min: float, front_x_max: float):
    print(f"  Front filter (world x): [{front_x_min:.2f}, {front_x_max:.2f}] m")
    print(f"  Right-hand reachable domain: {ws.n_samples} EE samples")
    print(f"    box x[{ws.lo[0]:.2f},{ws.hi[0]:.2f}] "
          f"y[{ws.lo[1]:.2f},{ws.hi[1]:.2f}] z[{ws.lo[2]:.2f},{ws.hi[2]:.2f}]")
    print(f"    split @ mid x={ws.mid[0]:.2f} y={ws.mid[1]:.2f} z={ws.mid[2]:.2f}")
    print("    (前/后/左/右/上/下 均相对该可达域盒，非全身或身后空间)")


def generate_multi_shape_trajectory(
    ee_cloud: np.ndarray,
    joint_configs: np.ndarray,
    num_shapes: int,
    fps: float,
    frames_per_shape: int,
    transition_frames: int,
    rng: np.random.Generator,
    front_x_min: float = DEFAULT_FRONT_X_MIN,
    front_x_max: float = DEFAULT_FRONT_X_MAX,
):
    """
    1) Filter front workspace from cache
    2) Stratify N shapes (L/R, U/D, F/B, large/small)
    3) Per shape: center + scale from region geometry (no random retry loop)
    4) Stitch transitions, single IK pass per shape
    """
    model = mujoco.MjModel.from_xml_path(PLAIN_XML)
    data = mujoco.MjData(model)
    addr = _build_addr_table(model, ACTIVE_JOINTS + list(LOCKED_JOINTS.keys()))
    _reset_to_home(model, data, addr)
    solver = IKSolver4DOF(model, data)
    arm_ids, obstacle_ids = _build_collision_id_sets(model)

    ee_cloud, joint_configs, ws = build_reachable_workspace(
        ee_cloud, joint_configs, front_x_min, front_x_max
    )
    active_qpos_addrs = [addr[n][0] for n in ACTIVE_JOINTS]
    _print_reachable_workspace(ws, front_x_min, front_x_max)

    plans = plan_shapes_stratified(num_shapes, rng)
    print("  Shape plan (each axis ~N/2 within reachable domain):")
    for i, pl in enumerate(plans):
        print(f"    [{i+1}] {pl['shape']:14s} {_plan_label(pl)}")

    all_qpos, all_qvel, all_ts = [], [], []
    shape_names = []
    segment_starts = [0]
    all_waypoints_vis = []
    shape_centers = []

    prev_last_wp = None
    qpos_seed = None
    t_accum = 0.0
    dt = 1.0 / fps

    for s_idx, plan in enumerate(plans):
        shape = plan["shape"]
        theta = plan["theta"]
        print(f"  Shape {s_idx+1}/{num_shapes}: plan {_plan_label(plan)} ...", flush=True)

        if qpos_seed is not None:
            data.qpos[:] = qpos_seed
            data.qvel[:] = 0.0
            mujoco.mj_forward(model, data)
        else:
            _reset_to_home(model, data, addr)

        center = pick_center_in_region(ee_cloud, ws, plan, rng)
        scale = compute_safe_scale(center, theta, shape, plan, ws, rng)
        try:
            shape_wps, scale = fit_shape_waypoints_to_workspace(
                shape, center, scale, theta, frames_per_shape, ee_cloud, rng
            )
        except RuntimeError as exc:
            raise RuntimeError(f"Shape {s_idx+1}: {exc}") from exc

        if not waypoints_in_front_band(shape_wps, front_x_min, front_x_max):
            raise RuntimeError(f"Shape {s_idx+1} leaves front x band")

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
                extra = ", more IK iters"
                max_iter = 200
            elif ik_try == 2:
                extra = ", new center"
                center = pick_center_in_region(ee_cloud, ws, plan, rng)
                scale = compute_safe_scale(center, theta, shape, plan, ws, rng)
                try:
                    shape_wps, scale = fit_shape_waypoints_to_workspace(
                        shape, center, scale, theta, frames_per_shape, ee_cloud, rng
                    )
                except RuntimeError:
                    continue
                ik_targets = (
                    blend_waypoints_to_workspace(shape_wps, ee_cloud, 0.30)
                    if shape in SHARP_SHAPES else None
                )
            elif ik_try == 3:
                extra = ", snap targets α=0.40"
                ik_targets = blend_waypoints_to_workspace(shape_wps, ee_cloud, 0.40)
                max_iter = 180
            elif ik_try == 4:
                extra = ", snap targets α=0.55"
                ik_targets = blend_waypoints_to_workspace(shape_wps, ee_cloud, 0.55)
                max_iter = 200
            elif ik_try >= 5:
                scale *= 0.90
                extra = f", scale={scale:.3f}"
                ik_targets = blend_waypoints_to_workspace(shape_wps, ee_cloud, 0.45)

            if ik_try >= 5:
                try:
                    shape_wps, scale = fit_shape_waypoints_to_workspace(
                        shape, center, scale, theta, frames_per_shape, ee_cloud, rng
                    )
                except RuntimeError:
                    continue
                ik_targets = blend_waypoints_to_workspace(shape_wps, ee_cloud, 0.45)

            print(f"  Shape {s_idx+1}/{num_shapes}: IK ({len(shape_wps)} pts{extra}) ...",
                  flush=True)
            qp_shape, qv_shape, ts_shape, errs, col_r = solve_waypoints_ik_ws_seed(
                shape_wps, ee_cloud, joint_configs, fps,
                solver, model, data, arm_ids, obstacle_ids,
                max_iter=max_iter, show_progress=(ik_try == 0),
                ik_targets=ik_targets,
            )
            ok_ratio = float(np.mean(errs < IK_ERR_THRESH))
            if check_feasibility(errs, col_r):
                placed = True
                break

        if not placed:
            raise RuntimeError(
                f"Shape {s_idx+1} IK infeasible "
                f"(ok_ratio={ok_ratio:.2f}, col={col_r:.2f}, scale={scale:.3f})"
            )

        segment_wps = shape_wps
        qp_seg = qp_shape
        qv_seg = qv_shape
        ts_seg = ts_shape

        if qpos_seed is not None and transition_frames > 0:
            qp_trans = make_joint_transition_qpos(
                qpos_seed, qp_shape[0], transition_frames, active_qpos_addrs
            )
            qv_trans, ts_trans = _qpos_vel_from_sequence(qp_trans, fps, model.nv)
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
        print(f"  Shape {s_idx+1}/{num_shapes}: OK {shape} center="
              f"({center[0]:.2f},{center[1]:.2f},{center[2]:.2f}) "
              f"scale={scale:.3f} ok_ratio={ok_ratio:.2f}", flush=True)

    qpos_out = np.concatenate(all_qpos, axis=0)
    qvel_out = np.concatenate(all_qvel, axis=0)
    ts_out = np.concatenate(all_ts, axis=0)
    waypoints_out = np.concatenate(all_waypoints_vis, axis=0)
    return {
        "qpos": qpos_out,
        "qvel": qvel_out,
        "timestamps": ts_out,
        "shape_names": np.array(shape_names, dtype=object),
        "segment_starts": np.array(segment_starts, dtype=np.int32),
        "waypoints": waypoints_out,
        "shape_centers": np.array(shape_centers, dtype=np.float32),
    }


# ---------------------------------------------------------------------------
# Save / preview / video
# ---------------------------------------------------------------------------

def save_trajectory_npz(traj: dict, fps: float, out_path: str):
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    np.savez_compressed(
        out_path,
        qpos=traj["qpos"],
        qvel=traj["qvel"],
        timestamps=traj["timestamps"],
        record_dt=np.float32(1.0 / fps),
        shape_names=traj["shape_names"],
        segment_starts=traj["segment_starts"],
        shape_centers=traj["shape_centers"],
    )
    n = len(traj["qpos"])
    print(f"\nSaved -> {out_path}")
    print(f"  frames={n} fps={fps:.0f} duration={n/fps:.1f}s shapes={len(traj['shape_names'])}")


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


def preview_trajectory(traj: dict, fps: float, out_path: str | None = None,
                       auto_save: bool = False):
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)
    qpos_arr = traj["qpos"]
    qvel_arr = traj["qvel"]
    waypoints = traj["waypoints"]
    N = len(qpos_arr)
    frame_dt = 1.0 / fps
    ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, EE_BODY)

    # Downsample waypoint cloud for display
    vis_step = max(1, len(waypoints) // 400)
    vis_wps = waypoints[::vis_step]

    save_flag = [False]

    def key_cb(keycode):
        if keycode in (ord("p"), ord("P")):
            save_flag[0] = True

    print(f"\n[Preview] {N} frames @ {fps:.0f} Hz | P=save | ESC=quit")
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
                    _add_red_markers(scn, vis_wps, size=0.006,
                                     rgba=np.array([1, 0.2, 0.2, 0.5], np.float32))
                    step = max(1, len(trail) // 200)
                    _add_red_markers(scn, trail[::step], size=0.008,
                                     rgba=np.array([1, 0.1, 0.1, 0.9], np.float32))
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


def render_video(traj: dict, video_path: str, fps: float = 50.0,
                 width=1280, height=720):
    if not HAS_IMAGEIO:
        raise RuntimeError("pip install imageio imageio-ffmpeg")

    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    if width > model.vis.global_.offwidth:
        model.vis.global_.offwidth = width
    if height > model.vis.global_.offheight:
        model.vis.global_.offheight = height
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=height, width=width)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    camera.trackbodyid = model.body("pelvis").id
    camera.distance = 2.5
    camera.elevation = -20.0
    camera.azimuth = -140.0

    qpos_arr = traj["qpos"]
    waypoints = traj["waypoints"]
    vis_step = max(1, len(waypoints) // 400)
    vis_wps = waypoints[::vis_step]
    ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, EE_BODY)

    frames = []
    trail = []
    try:
        for i in range(len(qpos_arr)):
            data.qpos[:] = qpos_arr[i]
            data.qvel[:] = traj["qvel"][i]
            mujoco.mj_forward(model, data)
            trail.append(data.xpos[ee_id].copy())
            if len(trail) > 500:
                trail.pop(0)
            renderer.update_scene(data, camera=camera)
            scn = renderer.scene
            base = scn.ngeom
            _add_red_markers(scn, vis_wps, size=0.006)
            step = max(1, len(trail) // 150)
            _add_red_markers(scn, trail[::step], size=0.009)
            frames.append(renderer.render())
            if (i + 1) % 100 == 0:
                print(f"  video {i+1}/{len(qpos_arr)}", end="\r")
        print()
    finally:
        renderer.close()

    os.makedirs(os.path.dirname(video_path) or ".", exist_ok=True)
    try:
        imageio.mimsave(video_path, frames, fps=float(fps))
    except Exception as exc:
        video_path = os.path.splitext(video_path)[0] + ".gif"
        print(f"  MP4 failed ({exc}), saved GIF -> {video_path}")
        imageio.mimsave(video_path, frames, fps=float(fps))
    print(f"  Video -> {video_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="G1 random multi-shape trajectory generator")
    p.add_argument("--mode", choices=["workspace", "visualize_ws", "probe_pose", "generate"],
                   default="generate",
                   help="workspace / visualize_ws / probe_pose / generate")
    p.add_argument("--probe-q4", type=float, nargs=4, metavar=("SP", "SR", "SY", "EL"),
                   help="With probe_pose: 4 right-arm joint angles (rad) from recorder UI")
    p.add_argument("--preview", action="store_true", help="MuJoCo preview with red trail")
    p.add_argument("--batch", action="store_true", help="Headless generate (no window)")
    p.add_argument("--num-shapes", type=int, default=DEFAULT_NUM_SHAPES)
    p.add_argument("--fps", type=float, default=DEFAULT_FPS)
    p.add_argument("--frames-per-shape", type=int, default=DEFAULT_FRAMES_PER_SHAPE)
    p.add_argument("--transition-frames", type=int, default=DEFAULT_TRANSITION_FRAMES)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--save", action="store_true", help="Save NPZ (auto in --batch)")
    p.add_argument("--ws-cache", action="store_true",
                   help="Load workspace_4dof_ee.npz instead of re-sampling")
    p.add_argument("--ws-samples", type=int, default=12,
                   help="Grid points per joint for workspace mode")
    p.add_argument("--ws-obstacles", choices=["full", "minimal"], default="minimal",
                   help="Obstacles for workspace (default minimal=torso+head+pelvis)")
    p.add_argument("--ws-random", type=int, default=30000,
                   help="Extra random 4-DOF samples when building workspace cache")
    p.add_argument("--ws-collision", choices=["penetration", "touch", "none"],
                   default="penetration",
                   help="Collision rule for workspace (default penetration, matches drag)")
    p.add_argument("--no-kinematic-overlay", action="store_true",
                   help="visualize_ws: skip gray 4/7-DOF kinematic envelope")
    p.add_argument("--video", nargs="?", const="", default=None, metavar="PATH")
    p.add_argument("--show-plot", action="store_true",
                   help="Show matplotlib window (workspace / visualize_ws)")
    p.add_argument("--mujoco-viewer", action="store_true",
                   help="With visualize_ws: also open MuJoCo window with red EE points")
    p.add_argument("--ws-plot", type=str, default="workspace_front_reachable.png",
                   help="Output PNG for visualize_ws mode")
    p.add_argument("--front-x-min", type=float, default=DEFAULT_FRONT_X_MIN,
                   help="Min x (m) in front of robot for trajectory centers (default 0.15)")
    p.add_argument("--front-x-max", type=float, default=DEFAULT_FRONT_X_MAX,
                   help="Max x (m) in front of robot for trajectory centers (default 0.50)")
    args = p.parse_args()

    if args.mode == "workspace":
        analyze_workspace_4dof(
            n_per_joint=args.ws_samples,
            n_random=args.ws_random,
            show_plot=args.show_plot,
            obstacle_preset=args.ws_obstacles,
            collision_mode=args.ws_collision,
        )
        return

    if args.mode == "probe_pose":
        if not args.probe_q4:
            print("Use --probe-q4 SP SR SY EL  (joint angles in rad from recorder sliders)")
            return
        ee_cloud, joint_configs = load_workspace_cache(with_joints=True)
        print("Probe 4-DOF pose against workspace cache:")
        probe_pose_in_workspace(np.array(args.probe_q4, dtype=np.float64),
                                ee_cloud, joint_configs)
        return

    if args.mode == "visualize_ws":
        ee_cloud, joint_configs = load_workspace_cache(with_joints=True)
        print(f"Loaded workspace cache: {len(ee_cloud)} EE points")
        visualize_front_reachable_workspace(
            ee_cloud,
            joint_configs,
            front_x_min=args.front_x_min,
            front_x_max=args.front_x_max,
            out_path=args.ws_plot,
            show_plot=args.show_plot,
            mujoco_viewer=args.mujoco_viewer,
            show_kinematic=not args.no_kinematic_overlay,
        )
        return

    if not args.preview and not args.batch:
        print("Use --preview and/or --batch (e.g. --preview --save or --batch --save)")
        return

    rng = np.random.default_rng(args.seed)

    if args.ws_cache:
        ee_cloud, joint_configs = load_workspace_cache(with_joints=True)
        print(f"Loaded workspace cache: {len(ee_cloud)} EE points + joint configs")
    else:
        print("No --ws-cache: running quick workspace sample (may take ~1-2 min)...")
        ee_cloud = analyze_workspace_4dof(n_per_joint=min(args.ws_samples, 10))
        if ee_cloud is None:
            return
        joint_configs = np.load(WS_CACHE_PATH)["joint_configs"].astype(np.float32)

    print(f"Restricting trajectories to robot front (+X): "
          f"x in [{args.front_x_min:.2f}, {args.front_x_max:.2f}] m")

    est_frames = args.num_shapes * (args.frames_per_shape + args.transition_frames)
    print(f"\nGenerating {args.num_shapes} stratified shapes @ {args.fps:.0f} Hz, "
          f"{args.frames_per_shape} frames/shape (~{est_frames} IK frames total) ...")
    print("  Plan: N/2 per side within reachable box (左/右, 上/下, 前/后, 大/小) -> IK.")
    traj = generate_multi_shape_trajectory(
        ee_cloud,
        joint_configs,
        num_shapes=args.num_shapes,
        fps=args.fps,
        frames_per_shape=args.frames_per_shape,
        transition_frames=args.transition_frames,
        rng=rng,
        front_x_min=args.front_x_min,
        front_x_max=args.front_x_max,
    )

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_npz = os.path.join(
        "recordings",
        f"g1_random_{int(args.fps)}hz_S{args.num_shapes}_{ts}.npz",
    )

    if args.video is not None:
        vpath = args.video if args.video else out_npz.replace(".npz", ".mp4")
        render_video(traj, vpath, fps=args.fps)

    if args.batch:
        if args.save:
            save_trajectory_npz(traj, args.fps, out_npz)
        else:
            print("Batch done (use --save to write NPZ)")
    elif args.preview:
        preview_trajectory(
            traj, args.fps,
            out_path=out_npz if args.save else None,
            auto_save=args.save,
        )

    if args.save:
        print(f"  Replay: python g1_player.py --traj {out_npz} --fps {int(args.fps)}")


if __name__ == "__main__":
    main()
