"""
G1 人形机器人轨迹回放与观测提取
=====================================
功能：
  1. 加载 g1_recorder.py 录制的 NPZ 轨迹文件
  2. 按目标帧率重采样轨迹（插值 or 降采样）：
       目标帧率 > 录制帧率 (100Hz) → 线性插值（四元数 SLERP）
       目标帧率 < 录制帧率         → 均匀降采样
  3. 在 MuJoCo 查看器中回放轨迹（可选）
  4. 逐帧计算 state 和 privileged_state，保存为 NPZ

输出 NPZ 包含：
  state             (N, 64)   dof_pos(29)+dof_vel(29)+proj_grav(3)+ang_vel(3)
  last_action       (N, 29)   全零（占位）
  privileged_state  (N, D)    body 局部位置/旋转/速度拼接
  qpos              (N, 36)   原始 qpos（供调试）
  qvel              (N, 35)   原始 qvel（供调试）
  timestamps        (N,)      轨迹时间戳

用法：
  python g1_player.py --traj recordings/g1_traj_*.npz
  python g1_player.py --traj recordings/g1_traj_*.npz --fps 50 --no-viewer
"""

import argparse
import glob
import os
import time

import mujoco
import mujoco.viewer
import numpy as np
from scipy.spatial.transform import Rotation, Slerp
from scipy.interpolate import interp1d


# ══════════════════════════════════════════════
# 常量
# ══════════════════════════════════════════════

MODEL_PATH = "g1/scene_g1_draggable.xml"

# 29 个铰链关节的默认角度（与 YAML default_joint_angles 一致，顺序与 G1 XML 中关节顺序相同）
DEFAULT_JOINT_POS = np.array([
    # 左腿 (6)
    -0.1,  0.0,  0.0,  0.3, -0.2,  0.0,
    # 右腿 (6)
    -0.1,  0.0,  0.0,  0.3, -0.2,  0.0,
    # 腰部 (3)
     0.0,  0.0,  0.0,
    # 左臂 (7)
     0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,
    # 右臂 (7)
     0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,
], dtype=np.float32)

# privileged_state 速度计算所用 dt（与 RL 训练环境一致）
PRIV_STATE_DT = 0.001


# ══════════════════════════════════════════════
# 四元数工具函数（NumPy 版，逻辑与 env.py 一致）
# ══════════════════════════════════════════════

def _as_batch_quat(q):
    q = np.asarray(q, dtype=np.float64)
    if q.ndim == 1:
        return q.reshape(1, 4), q.shape
    return q.reshape(-1, 4), q.shape


def quat_mul(a, b, w_last=True):
    """Multiply quaternions. Format: [x,y,z,w] if w_last=True, else [w,x,y,z]."""
    a, shape_a = _as_batch_quat(a)
    b, _ = _as_batch_quat(b)
    if w_last:
        x1, y1, z1, w1 = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
        x2, y2, z2, w2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    else:
        w1, x1, y1, z1 = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
        w2, x2, y2, z2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    ww = (z1 + x1) * (x2 + y2)
    yy = (w1 - y1) * (w2 + z2)
    zz = (w1 + y1) * (w2 - z2)
    xx = ww + yy + zz
    qq = 0.5 * (xx + (z1 - x1) * (x2 - y2))
    w = qq - ww + (z1 - y1) * (y2 - z2)
    x = qq - xx + (x1 + w1) * (x2 + w2)
    y = qq - yy + (w1 - x1) * (y2 + z2)
    z = qq - zz + (z1 + y1) * (w2 - x2)
    if w_last:
        out = np.stack([x, y, z, w], axis=-1)
    else:
        out = np.stack([w, x, y, z], axis=-1)
    return out.reshape(shape_a)


def quat_rotate(q, v):
    """Rotate vector v by quaternion q. q format: [x,y,z,w]."""
    q, shape_q = _as_batch_quat(q)
    v = np.asarray(v, dtype=np.float64).reshape(-1, 3)
    q_w = q[:, -1]
    q_vec = q[:, :3]
    a = v * (2.0 * q_w**2 - 1.0)[:, np.newaxis]
    b = np.cross(q_vec, v, axis=-1) * (2.0 * q_w)[:, np.newaxis]
    c = q_vec * np.sum(q_vec * v, axis=1, keepdims=True) * 2.0
    out = a + b + c
    if len(shape_q) == 1:
        return out[0]
    return out.reshape(shape_q[:-1] + (3,))


def quat_to_tan_norm(q, w_last=True):
    """Convert quaternion to 6D tangent-normal representation."""
    if not w_last:
        raise NotImplementedError("w_last=False not implemented")
    q, shape_q = _as_batch_quat(q)
    n = q.shape[0]
    ref_tan = np.zeros((n, 3), dtype=np.float64)
    ref_tan[:, 0] = 1.0
    ref_norm = np.zeros((n, 3), dtype=np.float64)
    ref_norm[:, 2] = 1.0
    tan = quat_rotate(q, ref_tan)
    norm = quat_rotate(q, ref_norm)
    out = np.concatenate([tan, norm], axis=-1)
    return out.reshape(shape_q[:-1] + (6,))


def calc_heading_quat_inv(q, w_last=True):
    """Inverse heading (yaw-only) quaternion, shape (..., 4)."""
    q = np.asarray(q, dtype=np.float64)
    if w_last:
        x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    else:
        w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    heading = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    half = -heading / 2.0
    cos_h = np.cos(half)
    sin_h = np.sin(half)
    axis = np.zeros(q.shape[:-1] + (3,), dtype=np.float64)
    axis[..., 2] = 1.0
    if w_last:
        return np.concatenate([sin_h[..., np.newaxis] * axis, cos_h[..., np.newaxis]], axis=-1)
    return np.concatenate([cos_h[..., np.newaxis], sin_h[..., np.newaxis] * axis], axis=-1)


def calc_angular_velocity(quat_cur, quat_prev, dt):
    """Angular velocity from consecutive quaternions (both [w,x,y,z])."""
    quat_cur = np.asarray(quat_cur); quat_prev = np.asarray(quat_prev)
    orig = quat_cur.shape
    if quat_cur.ndim == 1:
        quat_cur = quat_cur[None, :]; quat_prev = quat_prev[None, :]
    xyzw_cur  = quat_cur[:,  [1,2,3,0]]
    xyzw_prev = quat_prev[:, [1,2,3,0]]
    delta = Rotation.from_quat(xyzw_prev).inv() * Rotation.from_quat(xyzw_cur)
    ang_vel = delta.as_rotvec() / dt
    return ang_vel[0] if orig == (4,) else ang_vel


# ══════════════════════════════════════════════
# PrivilegedStateComputer
# ══════════════════════════════════════════════

class PrivilegedStateComputer:
    """
    将 env.py 的 get_privileged_state 封装为独立工具。
    使用同一 model 和 data 引用，每帧调用 compute() 即可。
    """

    # 排除以下 body：虚拟体 / 橡皮手 / worldbody / 球形控制点
    _EXCLUDE_PREFIXES = ("dummy", "world", "sphere")
    _EXCLUDE_SUFFIXES = ("hand",)

    def __init__(self, model: mujoco.MjModel, dt: float = PRIV_STATE_DT):
        self.model = model
        self.dt    = dt
        self.body_pos_prev  = None
        self.body_quat_prev = None
        self._valid_indices, self._body_names = self._build_valid_bodies()

    def _build_valid_bodies(self):
        indices, names = [], []
        head_idx, head_name = None, None
        for i in range(self.model.nbody):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, i) or ""
            if not name:
                indices.append(i); names.append(f"body_{i}")
                continue
            if (any(name.startswith(p) for p in self._EXCLUDE_PREFIXES) or
                    any(name.endswith(s) for s in self._EXCLUDE_SUFFIXES)):
                continue
            if name == "head_link":
                head_idx, head_name = i, name
            else:
                indices.append(i); names.append(name)
        if head_idx is not None:
            indices.append(head_idx); names.append(head_name)
        return np.array(indices), names

    def reset(self):
        self.body_pos_prev  = None
        self.body_quat_prev = None

    def compute(self, data: mujoco.MjData) -> np.ndarray:
        idx = self._valid_indices
        body_pos  = data.xpos [idx, :].copy()   # (B, 3)
        body_quat = data.xquat[idx, :].copy()   # (B, 4) w,x,y,z

        # 速度：有限差分
        if self.body_pos_prev is None:
            B = len(idx)
            body_vel     = np.zeros((B, 3))
            body_ang_vel = np.zeros((B, 3))
        else:
            body_vel     = (body_pos - self.body_pos_prev) / self.dt
            body_ang_vel = calc_angular_velocity(body_quat, self.body_quat_prev, self.dt)
        self.body_pos_prev  = body_pos
        self.body_quat_prev = body_quat

        B = len(idx)
        body_pos_b = body_pos.astype(np.float64)                          # (B, 3)
        body_rot_b = body_quat[:, [1, 2, 3, 0]].astype(np.float64)        # (B, 4) x,y,z,w
        body_vel_b = body_vel.astype(np.float64)
        body_ang_vel_b = body_ang_vel.astype(np.float64)

        root_pos = body_pos_b[0:1, :]                                     # (1, 3)
        root_rot = body_rot_b[0:1, :]                                     # (1, 4)

        heading_inv = calc_heading_quat_inv(root_rot, w_last=True)        # (1, 4)
        h_inv_expand = np.repeat(heading_inv, B, axis=0)                  # (B, 4)

        # 局部位置 (去掉 root 自身 [0,0,0])
        local_pos = body_pos_b - root_pos
        local_pos = quat_rotate(h_inv_expand, local_pos)
        local_pos_obs = local_pos[1:].reshape(-1)                         # ((B-1)*3,)

        # 局部旋转 6D
        local_rot = quat_mul(h_inv_expand, body_rot_b, w_last=True)
        local_rot_obs = quat_to_tan_norm(local_rot, w_last=True).reshape(-1)

        # 局部线速度 / 角速度
        local_vel_obs = quat_rotate(h_inv_expand, body_vel_b).reshape(-1)
        local_ang_obs = quat_rotate(h_inv_expand, body_ang_vel_b).reshape(-1)

        root_h = np.array([root_pos[0, 2]], dtype=np.float64)

        priv = np.concatenate([
            root_h, local_pos_obs, local_rot_obs, local_vel_obs, local_ang_obs
        ]).astype(np.float32)
        return priv


# ══════════════════════════════════════════════
# 轨迹重采样
# ══════════════════════════════════════════════

def resample_trajectory(qpos: np.ndarray, qvel: np.ndarray,
                        t_orig: np.ndarray, target_fps: float):
    """
    将原始轨迹（任意采样率）重采样至目标帧率。
    qpos: (N, 36)  qvel: (N, 35)  t_orig: (N,)
    返回 (qpos_new, qvel_new, t_new)
    """
    duration  = t_orig[-1] - t_orig[0]
    N_new     = max(2, int(duration * target_fps) + 1)
    t_new     = np.linspace(t_orig[0], t_orig[-1], N_new)

    # ── qpos[0:3]  位置 → 线性插值
    interp_pos = interp1d(t_orig, qpos[:, 0:3], axis=0, assume_sorted=True)
    pos_new = interp_pos(t_new)

    # ── qpos[3:7]  四元数 → SLERP
    # MuJoCo 四元数格式 [w,x,y,z]；scipy Slerp 需要 [x,y,z,w]
    quats_xyzw = qpos[:, [4, 5, 6, 3]]   # (N, 4)  x,y,z,w
    rots  = Rotation.from_quat(quats_xyzw)
    slerp = Slerp(t_orig, rots)
    quats_new_xyzw = slerp(t_new).as_quat()          # (N_new, 4) x,y,z,w
    quats_new = quats_new_xyzw[:, [3, 0, 1, 2]]      # → w,x,y,z

    # ── qpos[7:36] 关节角 → 线性插值
    interp_jnt = interp1d(t_orig, qpos[:, 7:], axis=0, assume_sorted=True)
    jnt_new = interp_jnt(t_new)

    qpos_new = np.concatenate([pos_new, quats_new, jnt_new], axis=1).astype(np.float32)

    # ── qvel 全部线性插值
    interp_qvel = interp1d(t_orig, qvel, axis=0, assume_sorted=True)
    qvel_new = interp_qvel(t_new).astype(np.float32)

    return qpos_new, qvel_new, t_new.astype(np.float32)


# ══════════════════════════════════════════════
# 单帧 state 计算
# ══════════════════════════════════════════════

def compute_state(data: mujoco.MjData) -> np.ndarray:
    """
    计算 64 维 state = [dof_pos(29), dof_vel(29), proj_grav(3), ang_vel(3)]
    """
    dof_pos = data.qpos[7:7+29].copy() - DEFAULT_JOINT_POS
    dof_vel = data.qvel[6:6+29].copy()

    root_quat = data.qpos[3:7]   # w,x,y,z
    rot = Rotation.from_quat([root_quat[1], root_quat[2], root_quat[3], root_quat[0]])
    proj_grav = rot.inv().apply(np.array([0., 0., -1.]))
    ang_vel   = rot.apply(data.qvel[3:6].copy())

    return np.concatenate([dof_pos, dof_vel, proj_grav, ang_vel]).astype(np.float32)


# ══════════════════════════════════════════════
# 主程序
# ══════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="G1 轨迹回放 + 观测提取")
    parser.add_argument("--traj",      type=str,   default=None,
                        help="轨迹 NPZ 文件路径（缺省则自动选最新）")
    parser.add_argument("--fps",       type=float, default=100.0,
                        help="回放/观测帧率（默认 100 = 不重采样）")
    parser.add_argument("--loop",      action="store_true", default=False,
                        help="循环回放（仅 viewer 模式有效）")
    parser.add_argument("--no-viewer", action="store_true", default=False,
                        help="不打开 MuJoCo 窗口（仅计算观测）")
    parser.add_argument("--out-dir",   type=str,   default="recordings",
                        help="观测 NPZ 输出目录")
    args = parser.parse_args()

    # ── 选择轨迹文件 ──
    traj_path = args.traj
    if traj_path is None:
        files = sorted(glob.glob(os.path.join("recordings", "g1_traj_*.npz")))
        if not files:
            print("录制目录中未找到轨迹文件，请先用 g1_recorder.py 录制。")
            return
        print("可用轨迹文件:")
        for i, f in enumerate(files):
            print(f"  [{i+1}] {os.path.basename(f)}")
        idx = int(input("请选择编号: ")) - 1
        if not (0 <= idx < len(files)):
            print("无效编号"); return
        traj_path = files[idx]

    print(f"\n加载轨迹: {traj_path}")
    traj = np.load(traj_path)
    qpos_raw  = traj["qpos"].astype(np.float32)    # (N, 36)
    qvel_raw  = traj["qvel"].astype(np.float32)    # (N, 35)
    t_raw     = traj["timestamps"].astype(np.float32)
    record_dt = float(traj.get("record_dt", 0.01))
    record_fps = 1.0 / record_dt

    N_raw = len(qpos_raw)
    print(f"  原始帧数: {N_raw}  |  录制帧率: {record_fps:.1f} Hz  |  时长: {t_raw[-1]-t_raw[0]:.2f}s")

    # ── 重采样 ──
    if abs(args.fps - record_fps) < 0.5:
        qpos_play, qvel_play, t_play = qpos_raw, qvel_raw, t_raw
        print(f"  目标帧率 {args.fps} Hz ≈ 录制帧率，跳过重采样")
    else:
        qpos_play, qvel_play, t_play = resample_trajectory(
            qpos_raw, qvel_raw, t_raw, args.fps
        )
        mode = "插值" if args.fps > record_fps else "降采样"
        print(f"  {mode}: {N_raw} → {len(qpos_play)} 帧 @ {args.fps:.1f} Hz")

    N = len(qpos_play)

    # ── 加载模型 ──
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data  = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    priv_computer = PrivilegedStateComputer(model, dt=PRIV_STATE_DT)

    # 预计算 privileged_state 维度
    data.qpos[:] = qpos_play[0]; data.qvel[:] = qvel_play[0]
    mujoco.mj_forward(model, data)
    sample_priv = priv_computer.compute(data)
    D_priv = len(sample_priv)
    priv_computer.reset()

    print(f"\n  state 维度: 64  |  privileged_state 维度: {D_priv}")
    print(f"  输出目录: {args.out_dir}")

    # ── 预分配输出缓冲 ──
    out_state  = np.zeros((N, 64),     dtype=np.float32)
    out_action = np.zeros((N, 29),     dtype=np.float32)  # 全零
    out_priv   = np.zeros((N, D_priv), dtype=np.float32)

    # ── 帧间隔 ──
    frame_dt = 1.0 / args.fps

    # ── 带或不带 viewer 的回放 ──
    def _run_headless():
        """无 viewer：仅计算观测并保存"""
        print("\n[无窗口模式] 正在处理轨迹...")
        for i in range(N):
            data.qpos[:] = qpos_play[i]
            data.qvel[:] = qvel_play[i]
            mujoco.mj_forward(model, data)
            out_state[i]  = compute_state(data)
            out_priv[i]   = priv_computer.compute(data)
            if (i+1) % 100 == 0 or i == N-1:
                print(f"  {i+1}/{N} 帧处理完成", end="\r")
        print()

    def _run_with_viewer():
        """带 viewer：边显示边计算"""
        with mujoco.viewer.launch_passive(model, data) as viewer:
            print("\n[回放模式] ESC 退出")
            loop_count = 0
            while viewer.is_running():
                loop_count += 1
                for i in range(N):
                    if not viewer.is_running():
                        break
                    t0 = time.perf_counter()

                    data.qpos[:] = qpos_play[i]
                    data.qvel[:] = qvel_play[i]
                    mujoco.mj_forward(model, data)

                    # 仅第一遍计算观测（避免 loop 重复写）
                    if loop_count == 1:
                        out_state[i] = compute_state(data)
                        out_priv[i]  = priv_computer.compute(data)

                    viewer.sync()

                    elapsed = time.perf_counter() - t0
                    sleep_t = max(0.0, frame_dt - elapsed)
                    if sleep_t > 0:
                        time.sleep(sleep_t)

                if not args.loop:
                    print("回放完成，按 ESC 退出")
                    while viewer.is_running():
                        viewer.sync()
                        time.sleep(0.02)
                    break

    if args.no_viewer:
        _run_headless()
    else:
        _run_with_viewer()

    # ── 保存观测 NPZ ──
    os.makedirs(args.out_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(traj_path))[0]
    out_path = os.path.join(args.out_dir, f"{base}_obs.npz")
    np.savez_compressed(
        out_path,
        state            = out_state,          # (N, 64)
        last_action      = out_action,         # (N, 29)
        privileged_state = out_priv,           # (N, D)
        # qpos             = qpos_play,          # (N, 36)
        # qvel             = qvel_play,          # (N, 35)
        # timestamps       = t_play,             # (N,)
    )
    print(f"\n观测已保存 → {out_path}")
    print(f"  state:            {out_state.shape}")
    print(f"  last_action:      {out_action.shape}")
    print(f"  privileged_state: {out_priv.shape}")

    # ── 演示：加载为 dict（接入 RL 时再转 torch 即可）──
    print("\n加载演示（numpy dict 格式）：")
    saved = np.load(out_path)
    obs = {
        "state":            saved["state"],
        "last_action":      saved["last_action"],
        "privileged_state": saved["privileged_state"],
    }
    for k, v in obs.items():
        print(f"  {k}: {v.shape}  dtype={v.dtype}")


if __name__ == "__main__":
    main()
