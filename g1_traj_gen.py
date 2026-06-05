"""
G1 右手轨迹生成器
=================
用途：为 G1 右手生成精确的圆形 / 正方形轨迹，通过逆运动学求关节角，
      在 MuJoCo 窗口预览后保存为 NPZ，可直接用 g1_player.py 回放。

运行模式：
  # 第一步：工作空间分析，输出截面图并打印建议参数
  python g1_traj_gen.py --mode workspace

  # 第二步：生成圆形并预览（用 workspace 建议的参数）
  python g1_traj_gen.py --mode circle --x 0.40 --cy -0.15 --cz 0.90 --radius 0.12

  # 第三步：生成正方形并预览
  python g1_traj_gen.py --mode square --x 0.40 --cy -0.15 --cz 0.90 --half 0.10

  # 预览窗口内按 P 键保存 NPZ；或加 --save 参数自动保存

  # 导出 MP4（离屏渲染，无需录屏）
  python g1_traj_gen.py --mode circle --x 0.30 --cy -0.15 --cz 0.90 --radius 0.07 --video
  python g1_traj_gen.py --mode circle ... --video recordings/demo.mp4 --save --no-preview

约束说明（与 g1_recorder.py 的腕锁一致）：
  活动关节（3 DOF）：right_shoulder_pitch, right_shoulder_roll, right_elbow
  锁定关节（保持0）：right_shoulder_yaw, right_wrist_roll/pitch/yaw
  其余关节：全部固定在 T 形站立零姿态
"""

import argparse
import datetime
import os
import time

import mujoco
import mujoco.viewer
import numpy as np

try:
    import imageio.v2 as imageio
    HAS_IMAGEIO = True
except ImportError:
    imageio = None
    HAS_IMAGEIO = False

# matplotlib 仅工作空间分析时使用，懒导入
os.makedirs("recordings", exist_ok=True)

# ══════════════════════════════════════════════
# 常量
# ══════════════════════════════════════════════

MODEL_PATH  = "g1/scene_g1_draggable.xml"
PLAIN_XML   = "g1/g1_29dof.xml"       # 无 sphere 约束，用于纯运动学 IK
EE_BODY     = "right_wrist_yaw_link"  # 末端执行器 body

# IK 中唯一自由的三个关节
ACTIVE_JOINTS = [
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_elbow_joint",
]

# IK 中强制归零的四个关节
LOCKED_JOINTS = {
    "right_shoulder_yaw_joint": 0.0,
    "right_wrist_roll_joint":   0.0,
    "right_wrist_pitch_joint":  0.0,
    "right_wrist_yaw_joint":    0.0,
}

# 碰撞检测：右臂 body（正在运动的部分）
_RIGHT_ARM_BODY_NAMES = [
    "right_shoulder_pitch_link", "right_shoulder_roll_link",
    "right_shoulder_yaw_link",   "right_elbow_link",
    "right_wrist_roll_link",     "right_wrist_pitch_link",
    "right_wrist_yaw_link",
]

# 碰撞检测：障碍物 body（右臂不应穿入的部分）
_OBSTACLE_BODY_NAMES = [
    "pelvis", "torso_link", "head_link",
    "waist_yaw_link", "waist_roll_link",
    "left_shoulder_pitch_link", "left_shoulder_roll_link",
    "left_shoulder_yaw_link",   "left_elbow_link",
    "left_wrist_roll_link",     "left_wrist_pitch_link",
    "left_wrist_yaw_link",
]

# 机器人基座初始位姿（T 形站立）
HOME_PELVIS_POS  = np.array([0.0, 0.0, 0.793])
HOME_PELVIS_QUAT = np.array([1.0, 0.0, 0.0, 0.0])  # w x y z


# ══════════════════════════════════════════════
# 工具函数
# ══════════════════════════════════════════════

def _int(x):
    """把 MuJoCo 3.x 返回的 array([n]) 安全转为 int。"""
    return int(np.asarray(x).reshape(-1)[0])


def _build_addr_table(model, names):
    """为每个关节名建立 {name: (qpos_addr, dof_addr, (lo, hi))} 映射。"""
    tbl = {}
    for name in names:
        jid  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        tbl[name] = (
            _int(model.jnt_qposadr[jid]),
            _int(model.jnt_dofadr[jid]),
            tuple(model.jnt_range[jid].tolist()),
        )
    return tbl


def _build_collision_id_sets(model):
    """
    Build (arm_ids, obstacle_ids) – sets of body IDs for self-collision check.
    mj_forward() calls mj_collision() internally, so data.contact is populated
    after every mj_forward call.  We only need to filter relevant contacts here.
    """
    def _ids(names):
        s = set()
        for n in names:
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n)
            if bid >= 0:
                s.add(bid)
        return s
    return _ids(_RIGHT_ARM_BODY_NAMES), _ids(_OBSTACLE_BODY_NAMES)


def _has_self_collision(model, data, arm_ids, obstacle_ids, tol=0.002):
    """
    Returns True if any right-arm geom physically penetrates an obstacle geom.
    contact.dist < 0 means penetration; we allow a small tolerance (tol metres)
    to avoid false positives from geoms that merely touch at the surface.
    Adjacent arm-chain links are NOT in obstacle_ids, so they never trigger.
    """
    for i in range(data.ncon):
        c  = data.contact[i]
        if c.dist > tol:        # positive dist = separation, not a penetration
            continue
        b1 = model.geom_bodyid[c.geom1]
        b2 = model.geom_bodyid[c.geom2]
        if ((b1 in arm_ids) or (b2 in arm_ids)) and \
           ((b1 in obstacle_ids) or (b2 in obstacle_ids)):
            return True
    return False


def _reset_to_home(model, data, addr_tbl):
    """重置为 T 形站立零姿态（仅铰链关节归零，基座归位）。"""
    data.qpos[:3]   = HOME_PELVIS_POS
    data.qpos[3:7]  = HOME_PELVIS_QUAT
    data.qpos[7:]   = 0.0
    data.qvel[:]    = 0.0
    # 确认锁定关节是 0
    for name, val in LOCKED_JOINTS.items():
        if name in addr_tbl:
            data.qpos[addr_tbl[name][0]] = val
    mujoco.mj_forward(model, data)


# ══════════════════════════════════════════════
# IK 求解器（阻尼最小二乘 Jacobian 法）
# ══════════════════════════════════════════════

class IKSolver:
    """
    3-DOF 位置 IK，仅操作 ACTIVE_JOINTS，其余关节固定。

    算法：阻尼最小二乘（DLS）
        dq = Jᵀ (J Jᵀ + λI)⁻¹ · e
    其中 e = target_pos - ee_pos，λ 随误差自适应调整（近奇异时增大）。
    """

    def __init__(self, model, data):
        self.model = model
        self.data  = data
        self.ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, EE_BODY)
        self.addr  = _build_addr_table(model, ACTIVE_JOINTS)
        self.qpos_addrs = [self.addr[n][0] for n in ACTIVE_JOINTS]
        self.dof_addrs  = [self.addr[n][1] for n in ACTIVE_JOINTS]
        self.ranges     = [self.addr[n][2] for n in ACTIVE_JOINTS]
        self._jacp = np.zeros((3, model.nv))

    def _get_q(self):
        return np.array([self.data.qpos[a] for a in self.qpos_addrs])

    def _set_q(self, q):
        for i, (a, (lo, hi)) in enumerate(zip(self.qpos_addrs, self.ranges)):
            self.data.qpos[a] = np.clip(q[i], lo, hi)

    def _lock(self):
        for name, val in LOCKED_JOINTS.items():
            if name in self.addr:
                self.data.qpos[self.addr[name][0]] = val
                self.data.qvel[self.addr[name][1]] = 0.0

    def solve(self, target_pos, max_iter=150, tol=5e-4, lam_base=1e-3):
        """
        求解 IK，使末端到达 target_pos。
        返回 (final_error, converged)。
        """
        for _ in range(max_iter):
            self._lock()
            mujoco.mj_forward(self.model, self.data)

            ee_pos = self.data.xpos[self.ee_id].copy()
            err    = target_pos - ee_pos
            e_norm = float(np.linalg.norm(err))

            if e_norm < tol:
                return e_norm, True

            # 自适应阻尼：误差大时加大阻尼防跳变，接近目标时减小
            lam = lam_base + 5e-3 * e_norm

            mujoco.mj_jacBody(self.model, self.data, self._jacp, None, self.ee_id)
            J = self._jacp[:, self.dof_addrs]   # (3, 3)

            # 阻尼最小二乘
            dq = J.T @ np.linalg.solve(J @ J.T + lam * np.eye(3), err)

            # 限制每步步长，防止奇异附近跳变
            dq_clip = np.clip(dq, -0.15, 0.15)
            self._set_q(self._get_q() + dq_clip)

        self._lock()
        mujoco.mj_forward(self.model, self.data)
        final_err = float(np.linalg.norm(
            target_pos - self.data.xpos[self.ee_id]
        ))
        return final_err, False


# ══════════════════════════════════════════════
# 工作空间分析
# ══════════════════════════════════════════════

def analyze_workspace(N_per_joint=22):
    """
    在关节空间均匀采样 N³ 个配置，计算末端位置，
    绘制 x=const 截面的 y-z 工作空间分布，
    并输出建议的轨迹参数。
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
    except ImportError:
        print("matplotlib not installed.  pip install matplotlib")
        return None

    model = mujoco.MjModel.from_xml_path(PLAIN_XML)
    data  = mujoco.MjData(model)
    addr  = _build_addr_table(model, ACTIVE_JOINTS + list(LOCKED_JOINTS.keys()))
    _reset_to_home(model, data, addr)

    ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, EE_BODY)

    print(f"\n[工作空间分析] 采样 {N_per_joint}³ = {N_per_joint**3} 个配置（含碰撞过滤）...")
    t0 = time.perf_counter()

    arm_ids, obstacle_ids = _build_collision_id_sets(model)

    ranges = [addr[n][2] for n in ACTIVE_JOINTS]
    grids  = [np.linspace(lo, hi, N_per_joint) for lo, hi in ranges]

    pts        = []
    n_total    = 0
    n_collide  = 0
    for q0 in grids[0]:
        data.qpos[addr[ACTIVE_JOINTS[0]][0]] = q0
        for q1 in grids[1]:
            data.qpos[addr[ACTIVE_JOINTS[1]][0]] = q1
            for q2 in grids[2]:
                data.qpos[addr[ACTIVE_JOINTS[2]][0]] = q2
                for name, val in LOCKED_JOINTS.items():
                    data.qpos[addr[name][0]] = val
                mujoco.mj_forward(model, data)
                n_total += 1
                if _has_self_collision(model, data, arm_ids, obstacle_ids):
                    n_collide += 1
                    continue         # skip collision configurations
                pts.append(data.xpos[ee_id].copy())

    pts = np.array(pts)
    elapsed = time.perf_counter() - t0
    print(f"  Sampling done in {elapsed:.1f}s  |  "
          f"valid: {len(pts)}  |  collision-filtered: {n_collide}  "
          f"({100*n_collide/n_total:.1f}%)")
    print(f"  EE X range: [{pts[:,0].min():.3f}, {pts[:,0].max():.3f}] m")
    print(f"  EE Y range: [{pts[:,1].min():.3f}, {pts[:,1].max():.3f}] m")
    print(f"  EE Z range: [{pts[:,2].min():.3f}, {pts[:,2].max():.3f}] m")

    # 选择截面 x 值
    x_lo, x_hi = pts[:,0].min(), pts[:,0].max()
    x_planes = np.linspace(
        max(x_lo + 0.02, 0.20),
        min(x_hi - 0.02, 0.55),
        4
    )

    fig, axes = plt.subplots(1, len(x_planes), figsize=(4*len(x_planes), 5))
    fig.suptitle("G1 right-hand workspace (collision-free, shoulder_yaw=0, wrist=0)",
                 fontsize=11)

    best = None
    best_pts_count = 0

    for ax, xp in zip(axes, x_planes):
        mask  = np.abs(pts[:,0] - xp) < 0.025   # ±2.5 cm slice
        sl    = pts[mask]

        ax.scatter(sl[:,1], sl[:,2], s=1, alpha=0.25, c='steelblue',
                   label='collision-free')
        ax.set_xlabel('Y (m)'); ax.set_ylabel('Z (m)')
        ax.set_title(f'x = {xp:.2f} m\n({len(sl)} pts, collision-free)')
        ax.set_aspect('equal'); ax.grid(True, alpha=0.3)

        if len(sl) < 30:
            continue

        cy = (sl[:,1].max() + sl[:,1].min()) / 2
        cz = (sl[:,2].max() + sl[:,2].min()) / 2

        # 在该截面找最大内接圆半径（取 y/z 各方向余量最小值的 65%）
        r_max = min(
            sl[:,1].max() - cy,
            cy - sl[:,1].min(),
            sl[:,2].max() - cz,
            cz - sl[:,2].min(),
        ) * 0.65

        if r_max > 0.04:
            theta = np.linspace(0, 2*np.pi, 200)
            ax.plot(cy + r_max*np.cos(theta), cz + r_max*np.sin(theta),
                    'r-', lw=2, label=f'suggested circle r={r_max:.3f} m')
            ax.plot(cy, cz, 'r+', ms=10)
            # 内接正方形
            sq = plt.Polygon(
                [[cy-r_max*0.75, cz-r_max*0.75],
                 [cy+r_max*0.75, cz-r_max*0.75],
                 [cy+r_max*0.75, cz+r_max*0.75],
                 [cy-r_max*0.75, cz+r_max*0.75]],
                fill=False, edgecolor='orange', lw=2,
                label=f'suggested square half={r_max*0.75:.3f} m'
            )
            ax.add_patch(sq)
            ax.legend(fontsize=7)

            if len(sl) > best_pts_count:
                best_pts_count = len(sl)
                best = {
                    'x':      float(xp),
                    'cy':     float(cy),
                    'cz':     float(cz),
                    'radius': float(round(r_max, 3)),
                    'half':   float(round(r_max * 0.75, 3)),
                }

    plt.tight_layout()
    out_fig = "workspace_analysis.png"
    plt.savefig(out_fig, dpi=150, bbox_inches='tight')
    print(f"\n  截面图已保存 → {out_fig}")
    plt.show()

    if best:
        print("\n建议轨迹参数（可直接复制使用）：")
        print(f"  圆形：python g1_traj_gen.py --mode circle "
              f"--x {best['x']:.2f} --cy {best['cy']:.3f} "
              f"--cz {best['cz']:.3f} --radius {best['radius']:.3f}")
        print(f"  正方形：python g1_traj_gen.py --mode square "
              f"--x {best['x']:.2f} --cy {best['cy']:.3f} "
              f"--cz {best['cz']:.3f} --half {best['half']:.3f}")

    return best


# ══════════════════════════════════════════════
# 轨迹参数化（目标点序列）
# ══════════════════════════════════════════════

def make_circle_waypoints(cx, cy, cz, radius, N):
    """
    生成圆形目标点序列。
    从最高点 (cz+radius) 出发，逆时针（从 +x 向看：y-z 平面逆时针）。
    """
    t = np.linspace(0, 2*np.pi, N, endpoint=False)
    x = np.full(N, cx)
    y = cy + radius * np.sin(t)
    z = cz + radius * np.cos(t)
    return np.stack([x, y, z], axis=1)


def make_square_waypoints(cx, cy, cz, half, N, corner_r_frac=0.15):
    """
    Rounded square in the YZ plane (fixed x = cx), traversed clockwise
    when viewed from +X toward the robot.

    Each corner arc sweeps along the OUTSIDE of the square (short 90 deg
    turn).  Previous version used wrong angle ranges (e.g. PI/2 -> PI),
    which routed arcs through the interior and caused inward bulges at edges.
    """
    L = half
    r = min(corner_r_frac * 2 * L, L * 0.45)   # keep arcs inside the square

    seg_len = 2 * L - 2 * r
    arc_len = (np.pi / 2) * r
    total   = 4 * seg_len + 4 * arc_len
    N_seg   = max(2, round(N * seg_len / total))
    N_arc   = max(2, round(N * arc_len / total))

    pts = []

    def _seg(y0, z0, y1, z1, n):
        for y, z in zip(np.linspace(y0, y1, n, endpoint=False),
                        np.linspace(z0, z1, n, endpoint=False)):
            pts.append([cx, y, z])

    def _arc_corner(yc, zc, angle_start, n):
        """
        Exterior 90 deg turn: angle_start -> angle_start - pi/2.
        Always sweep clockwise on the circle (decreasing angle) so we never
        take the 270 deg interior path (e.g. linspace(-pi/2, pi) is wrong).
        """
        for a in np.linspace(angle_start, angle_start - np.pi / 2,
                             n, endpoint=False):
            pts.append([cx, yc + r * np.cos(a), zc + r * np.sin(a)])

    # Clockwise from +X: top (right->left) -> TL -> left -> BL -> bottom ->
    # BR -> right -> TR -> closes at start of top edge.

    # Top edge  z = cz+L
    _seg(cy - L + r, cz + L, cy + L - r, cz + L, N_seg)
    # Top-left  center (cy+L-r, cz+L-r), leave top edge at angle pi/2
    _arc_corner(cy + L - r, cz + L - r, np.pi / 2, N_arc)
    # Left edge  y = cy+L
    _seg(cy + L, cz + L - r, cy + L, cz - L + r, N_seg)
    # Bottom-left  center (cy+L-r, cz-L+r), leave left edge at angle 0
    _arc_corner(cy + L - r, cz - L + r, 0.0, N_arc)
    # Bottom edge  z = cz-L
    _seg(cy + L - r, cz - L, cy - L + r, cz - L, N_seg)
    # Bottom-right  center (cy-L+r, cz-L+r), leave bottom at angle -pi/2
    _arc_corner(cy - L + r, cz - L + r, -np.pi / 2, N_arc)
    # Right edge  y = cy-L
    _seg(cy - L, cz - L + r, cy - L, cz + L - r, N_seg)
    # Top-right  center (cy-L+r, cz+L-r), leave right edge at angle pi
    _arc_corner(cy - L + r, cz + L - r, np.pi, N_arc)

    return np.array(pts)


# ══════════════════════════════════════════════
# 逆运动学：生成完整关节轨迹
# ══════════════════════════════════════════════

def generate_ik_trajectory(waypoints, fps=100.0):
    """
    对每个目标点求 IK，输出 qpos / qvel / timestamps。
    使用 PLAIN_XML（无 sphere 约束）做纯运动学计算，速度更快。
    """
    model = mujoco.MjModel.from_xml_path(PLAIN_XML)
    data  = mujoco.MjData(model)
    addr  = _build_addr_table(model, ACTIVE_JOINTS + list(LOCKED_JOINTS.keys()))
    _reset_to_home(model, data, addr)

    solver                      = IKSolver(model, data)
    arm_ids, obstacle_ids       = _build_collision_id_sets(model)
    N                           = len(waypoints)
    dt                          = 1.0 / fps

    print(f"\n[IK] {N} waypoints at {fps:.0f} Hz  (with collision check) ...")
    t_start = time.perf_counter()

    qpos_list  = []
    errors     = []
    col_frames = []   # indices of frames that ended up in collision

    for i, wp in enumerate(waypoints):
        for name, val in LOCKED_JOINTS.items():
            data.qpos[addr[name][0]] = val

        err, ok = solver.solve(wp, max_iter=200)

        # Re-lock and forward to get final state + contacts
        for name, val in LOCKED_JOINTS.items():
            data.qpos[addr[name][0]] = val
            data.qvel[addr[name][1]] = 0.0
        mujoco.mj_forward(model, data)

        # Collision check ─ warn but still record (so trajectory stays continuous)
        if _has_self_collision(model, data, arm_ids, obstacle_ids):
            col_frames.append(i)

        qpos_list.append(data.qpos.copy())
        errors.append(err)

        if (i + 1) % 100 == 0 or i == N - 1:
            recent = np.mean(errors[max(0, i - 99):i + 1])
            print(f"  {i+1:4d}/{N}  avg err (last 100): {recent*1000:.2f} mm  "
                  f"collisions so far: {len(col_frames)}",
                  end="\r")

    print()
    elapsed = time.perf_counter() - t_start
    errors  = np.array(errors)
    print(f"  Elapsed {elapsed:.1f}s  |  "
          f"mean err {errors.mean()*1000:.2f} mm  |  "
          f"max err  {errors.max()*1000:.2f} mm  |  "
          f"not converged {(errors > 1e-3).sum()}/{N}")

    if errors.max() > 5e-3:
        print("  WARNING: max error > 5 mm – some waypoints may be outside "
              "the reachable workspace. Consider reducing radius / adjusting centre.")

    if col_frames:
        pct = 100 * len(col_frames) / N
        print(f"  WARNING: {len(col_frames)} frames ({pct:.1f}%) have self-collision. "
              f"First collision frame: {col_frames[0]}.\n"
              f"  --> Please re-run workspace analysis and choose a trajectory "
              f"centre / size that lies entirely within the collision-free region.")

    qpos_arr = np.array(qpos_list, dtype=np.float32)  # (N, nq)
    nv       = model.nv

    # 用中心差分估算 qvel（关节速度）
    qvel_arr = np.zeros((N, nv), dtype=np.float32)
    for i in range(1, N-1):
        # 仅更新铰链关节速度（跳过自由基座）
        qvel_arr[i, 6:] = (qpos_arr[i+1, 7:] - qpos_arr[i-1, 7:]) / (2*dt)
    qvel_arr[0]   = qvel_arr[1]
    qvel_arr[N-1] = qvel_arr[N-2]

    timestamps = np.arange(N, dtype=np.float32) * dt
    return qpos_arr, qvel_arr, timestamps, errors


# ══════════════════════════════════════════════
# 可视化辅助 / 保存 / 视频
# ══════════════════════════════════════════════

def _waypoint_vis_points(waypoints, max_pts=300):
    step = max(1, len(waypoints) // max_pts)
    return waypoints[::step]


def _add_waypoint_markers(scene, vis_pts):
    """Draw red target waypoints into an MjvScene (viewer or renderer)."""
    for wp in vis_pts:
        if scene.ngeom >= scene.maxgeom:
            break
        g = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(
            g,
            mujoco.mjtGeom.mjGEOM_SPHERE,
            np.array([0.007, 0.0, 0.0]),
            np.zeros(3),
            np.eye(3).flatten(),
            np.array([1.0, 0.15, 0.15, 0.7], dtype=np.float32),
        )
        g.pos[:] = wp
        scene.ngeom += 1


def _m_to_name_tag(meters: float) -> str:
    """Meters -> integer cm tag, e.g. 0.30->30, -0.15->15 (abs, for filenames)."""
    return str(int(round(abs(meters) * 100)))


def trajectory_output_stem(mode, x, cy, cz, radius=None, half=None):
    """
    Parameter-based filename stem, e.g.
      circle x=0.30 cy=-0.15 cz=0.90 r=0.07 -> recordings/g1_circle_30_15_90_7
      square x=0.30 cy=-0.15 cz=0.90 half=0.10 -> recordings/g1_square_30_15_90_10
    """
    tags = [_m_to_name_tag(x), _m_to_name_tag(cy), _m_to_name_tag(cz)]
    if mode == "circle":
        tags.append(_m_to_name_tag(radius))
    else:
        tags.append(_m_to_name_tag(half))
    return os.path.join("recordings", f"g1_{mode}_{'_'.join(tags)}")


def save_trajectory_npz(qpos_arr, qvel_arr, timestamps, traj_type, fps,
                        out_path=None):
    """Save trajectory NPZ compatible with g1_player.py."""
    if out_path is None:
        raise ValueError("out_path is required (use trajectory_output_stem)")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    np.savez_compressed(
        out_path,
        qpos=qpos_arr,
        qvel=qvel_arr,
        timestamps=timestamps,
        record_dt=np.float32(1.0 / fps),
    )
    n = len(qpos_arr)
    print(f"\n轨迹已保存 → {out_path}")
    print(f"  帧数: {n}  |  帧率: {fps:.0f} Hz  |  时长: {n/fps:.1f}s")
    print(f"  可用命令回放：python g1_player.py --traj {out_path}")
    return out_path


def render_trajectory_video(qpos_arr, qvel_arr, waypoints, video_path,
                            traj_fps=100.0, video_fps=30.0,
                            width=1280, height=720):
    """
    Offscreen render trajectory to MP4 (no viewer window needed).
    Red spheres = target waypoints; camera tracks pelvis.
    """
    if not HAS_IMAGEIO:
        raise RuntimeError("imageio not installed.  pip install imageio imageio-ffmpeg")

    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    if width > model.vis.global_.offwidth:
        model.vis.global_.offwidth = width
    if height > model.vis.global_.offheight:
        model.vis.global_.offheight = height
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=height, width=width)

    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    camera.trackbodyid = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, "pelvis"
    )
    camera.distance = 2.5
    camera.elevation = -20.0
    camera.azimuth = -140.0

    vis_pts = _waypoint_vis_points(waypoints)
    frame_step = max(1, round(traj_fps / video_fps))
    indices = list(range(0, len(qpos_arr), frame_step))
    if indices[-1] != len(qpos_arr) - 1:
        indices.append(len(qpos_arr) - 1)

    print(f"\n[Video] rendering {len(indices)} frames -> {video_path}")
    print(f"  trajectory {traj_fps:.0f} Hz, video {video_fps:.0f} fps, "
          f"{width}x{height}")

    frames = []
    try:
        for k, i in enumerate(indices):
            data.qpos[:] = qpos_arr[i]
            data.qvel[:] = qvel_arr[i]
            mujoco.mj_forward(model, data)

            renderer.update_scene(data, camera=camera)
            _add_waypoint_markers(renderer.scene, vis_pts)
            frames.append(renderer.render())

            if (k + 1) % 50 == 0 or k + 1 == len(indices):
                print(f"  {k+1}/{len(indices)} frames", end="\r")
        print()
    finally:
        renderer.close()

    os.makedirs(os.path.dirname(video_path) or ".", exist_ok=True)
    ext = os.path.splitext(video_path)[1].lower()
    if ext not in (".mp4", ".gif", ".webm"):
        video_path = video_path + ".mp4"

    try:
        imageio.mimsave(video_path, frames, fps=float(video_fps))
    except Exception as exc:
        gif_path = os.path.splitext(video_path)[0] + ".gif"
        print(f"  MP4 save failed ({exc}), falling back to GIF -> {gif_path}")
        imageio.mimsave(gif_path, frames, fps=float(video_fps))
        video_path = gif_path

    duration = len(frames) / float(video_fps)
    print(f"  Video saved -> {video_path}  ({len(frames)} frames, {duration:.1f}s)")
    return video_path


# ══════════════════════════════════════════════
# MuJoCo 预览
# ══════════════════════════════════════════════

def preview_trajectory(qpos_arr, qvel_arr, timestamps,
                       waypoints, traj_type, fps,
                       out_path=None, auto_save=False):
    """
    在 MuJoCo 场景窗口预览生成的轨迹。
    - 红色小球：目标轨迹点云
    - 机器人实时跟随
    - 按 P 键保存 NPZ
    """
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data  = mujoco.MjData(model)

    N        = len(qpos_arr)
    frame_dt = 1.0 / fps
    save_flag  = [False]

    def key_cb(keycode):
        if keycode in (ord('p'), ord('P')):
            save_flag[0] = True
            print("\n[预览] P 键按下，将在本轮播放结束后保存 NPZ ...")

    print(f"\n[预览] {N} 帧  |  {fps:.0f} Hz  |  时长 {N/fps:.1f}s")
    print("  红色点云 = 目标轨迹；机器人跟随轨迹运动")
    print("  按 P 键保存 NPZ；ESC 键退出")

    vis_pts = _waypoint_vis_points(waypoints)

    with mujoco.viewer.launch_passive(
        model, data, key_callback=key_cb
    ) as viewer:
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

                try:
                    scn = viewer.user_scn
                    scn.ngeom = 0
                    _add_waypoint_markers(scn, vis_pts)
                except Exception:
                    pass

                viewer.sync()
                elapsed = time.perf_counter() - t0
                sleep_t = max(0.0, frame_dt - elapsed)
                if sleep_t > 0:
                    time.sleep(sleep_t)

            if save_flag[0] or auto_save:
                break

            # 最多循环 3 次，防止无限等待
            if loop >= 3:
                print("\n[预览] 已循环 3 次，保持最终帧 ...")
                while viewer.is_running():
                    viewer.sync()
                    time.sleep(0.02)
                break

    if save_flag[0] or auto_save:
        if out_path is None:
            raise ValueError("out_path is required when saving from preview")
        save_trajectory_npz(qpos_arr, qvel_arr, timestamps, traj_type, fps, out_path)
    else:
        print("\n[预览] 未保存（在预览窗口按 P 键可保存）")


# ══════════════════════════════════════════════
# 主程序
# ══════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(description="G1 右手轨迹生成器")
    p.add_argument("--mode",   choices=["workspace", "circle", "square"],
                   default="workspace",
                   help="运行模式：workspace=工作空间分析; circle/square=生成轨迹")
    p.add_argument("--x",      type=float, default=None,
                   help="轨迹平面 x 坐标 (m)")
    p.add_argument("--cy",     type=float, default=None,
                   help="轨迹中心 y 坐标 (m)")
    p.add_argument("--cz",     type=float, default=None,
                   help="轨迹中心 z 坐标 (m)")
    p.add_argument("--radius", type=float, default=None,
                   help="圆半径 (m)，--mode circle 时使用")
    p.add_argument("--half",   type=float, default=None,
                   help="正方形半边长 (m)，--mode square 时使用")
    p.add_argument("--N",      type=int,   default=400,
                   help="轨迹点总数（默认 400）")
    p.add_argument("--fps",    type=float, default=100.0,
                   help="帧率 Hz（默认 100，与录制器一致）")
    p.add_argument("--save",   action="store_true",
                   help="跳过预览确认，直接保存 NPZ")
    p.add_argument("--video",  nargs="?", const="", default=None, metavar="PATH",
                   help="导出 MP4（可选路径；默认与 NPZ 同名 .mp4，如 g1_circle_30_15_90_7.mp4）")
    p.add_argument("--video-fps", type=float, default=30.0,
                   help="视频帧率（默认 30；轨迹 100Hz 时会自动降采样）")
    p.add_argument("--no-preview", action="store_true",
                   help="不打开 MuJoCo 预览窗口（配合 --video / --save 使用）")
    args = p.parse_args()

    if args.mode == "workspace":
        analyze_workspace()
        return

    # ── 生成轨迹模式 ──
    if args.x is None or args.cy is None or args.cz is None:
        print("错误：--mode circle/square 必须指定 --x --cy --cz")
        print("提示：先运行 python g1_traj_gen.py --mode workspace 获取建议参数")
        return

    if args.mode == "circle":
        if args.radius is None:
            print("错误：--mode circle 需要 --radius")
            return
        print(f"\n生成圆形轨迹：中心=({args.x:.3f},{args.cy:.3f},{args.cz:.3f})，"
              f"半径={args.radius:.3f}m，{args.N} 个点")
        waypoints = make_circle_waypoints(args.x, args.cy, args.cz, args.radius, args.N)

    else:  # square
        size = args.half if args.half is not None else args.radius
        if size is None:
            print("错误：--mode square 需要 --half（或 --radius 作为半边长）")
            return
        print(f"\n生成正方形轨迹：中心=({args.x:.3f},{args.cy:.3f},{args.cz:.3f})，"
              f"半边长={size:.3f}m，{args.N} 个点")
        waypoints = make_square_waypoints(args.x, args.cy, args.cz, size, args.N)

    # IK 求解
    qpos_arr, qvel_arr, timestamps, errors = generate_ik_trajectory(waypoints, args.fps)

    size = args.half if args.mode == "square" else None
    if args.mode == "square" and size is None:
        size = args.radius
    out_stem = trajectory_output_stem(
        args.mode, args.x, args.cy, args.cz,
        radius=args.radius, half=size,
    )
    out_npz = out_stem + ".npz"
    print(f"  输出文件名: {os.path.basename(out_npz)}")

    if args.video is not None:
        video_path = args.video if args.video else out_stem + ".mp4"
        render_trajectory_video(
            qpos_arr, qvel_arr, waypoints, video_path,
            traj_fps=args.fps, video_fps=args.video_fps,
        )

    if args.save and args.no_preview:
        save_trajectory_npz(
            qpos_arr, qvel_arr, timestamps, args.mode, args.fps, out_npz
        )
    elif not args.no_preview:
        preview_trajectory(
            qpos_arr, qvel_arr, timestamps,
            waypoints, args.mode, args.fps,
            out_path=out_npz, auto_save=args.save,
        )


if __name__ == "__main__":
    main()
