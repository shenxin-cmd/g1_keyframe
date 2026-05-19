"""
G1 人形机器人姿态捕获系统
================================
功能：
  - MuJoCo 交互窗口拖拽调姿（鼠标直接拖动机器人body）
  - sphere + connect 拖拽 IK：拖动7个黄色控制球
      控制点：左手末端、右手末端、左肘、右肘、骨盆、左脚、右脚
  - 腕关节锁定（W键切换）：拖手末端时双手6个腕关节保持0，
      确保画正方形/圆形等轨迹时手型不变形
  - Tk 滑动条 UI：全关节（29个铰链关节）精确调节
  - 关节阻尼动态调节（快捷键 [ / ]）
  - 全局快捷键：S=保存关键帧，H=归零姿态，W=腕锁开关，[=减阻尼，]=增阻尼

使用方法：
  conda activate mujoco
  python g1_recorder.py
"""

import mujoco
import mujoco.viewer
import numpy as np
import time
import os
import datetime
import tkinter as tk
from tkinter import ttk, messagebox
import threading

from utils.joint_utils import (
    get_joint_info, set_joint_damping, set_specific_joint_damping,
    find_related_joints, set_selected_body_joints_damping, restore_all_joint_damping,
    move_sphere_body_directly
)
from utils.ui_controller import RobotJointUI
from utils.keyframe_manager import KeyframeManager


def qpos_index(addr) -> int:
    """将 MuJoCo 3.x 的 qposadr（可能是 ndarray）转为 Python int。"""
    return int(np.asarray(addr).reshape(-1)[0])


# ──────────────────────────────────────────────
# 加载模型
# ──────────────────────────────────────────────
MODEL_PATH = "g1/scene_g1_draggable.xml"
model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)

# ──────────────────────────────────────────────
# 关节信息
# ──────────────────────────────────────────────
joint_info = get_joint_info(model)
joint_names     = joint_info['joint_names']
joint_ids       = joint_info['joint_ids']
joint_types     = joint_info['joint_types']
joint_qpos_map  = joint_info['joint_qpos_map']
joint_dim_map   = joint_info['joint_dim_map']
joint_ranges    = joint_info['joint_ranges']

# G1 hinge 关节 qpos 起始地址（用于 home 重置）
FREE_JOINT_NAME = "floating_base_joint"
HOME_PELVIS_POS  = np.array([0.0, 0.0, 0.793])
HOME_PELVIS_QUAT = np.array([1.0, 0.0, 0.0, 0.0])   # w x y z

# 仅铰链关节用于 UI 和 joint_angles 字典
hinge_joint_names = [
    name for name in joint_names
    if name and joint_types[joint_ids[name]] == mujoco.mjtJoint.mjJNT_HINGE
]

# ──────────────────────────────────────────────
# G1 关节中文描述（29个铰链关节，按部位分组）
# ──────────────────────────────────────────────
joint_descriptions = {
    # ── 左腿（6个关节）──
    "left_hip_pitch_joint":   "左髋俯仰 (前后摆腿)",
    "left_hip_roll_joint":    "左髋侧展 (内外展腿)",
    "left_hip_yaw_joint":     "左髋旋转 (内外旋)",
    "left_knee_joint":        "左膝关节 (弯曲伸直)",
    "left_ankle_pitch_joint": "左踝俯仰 (背屈/跖屈)",
    "left_ankle_roll_joint":  "左踝侧摆 (内翻/外翻)",
    # ── 右腿（6个关节）──
    "right_hip_pitch_joint":   "右髋俯仰 (前后摆腿)",
    "right_hip_roll_joint":    "右髋侧展 (内外展腿)",
    "right_hip_yaw_joint":     "右髋旋转 (内外旋)",
    "right_knee_joint":        "右膝关节 (弯曲伸直)",
    "right_ankle_pitch_joint": "右踝俯仰 (背屈/跖屈)",
    "right_ankle_roll_joint":  "右踝侧摆 (内翻/外翻)",
    # ── 腰部（3个关节）──
    "waist_yaw_joint":   "腰部旋转 (左右转体)",
    "waist_roll_joint":  "腰部侧弯 (左右侧倾)",
    "waist_pitch_joint": "腰部前后弯 (俯仰)",
    # ── 左臂（7个关节）──
    "left_shoulder_pitch_joint": "左肩俯仰 (前后抬臂)",
    "left_shoulder_roll_joint":  "左肩侧展 (内外展臂)",
    "left_shoulder_yaw_joint":   "左肩旋转 (臂内外旋)",
    "left_elbow_joint":          "左肘关节 (弯曲伸直)",
    "left_wrist_roll_joint":     "左腕旋转 (前臂旋转)",
    "left_wrist_pitch_joint":    "左腕俯仰 (掌屈/背伸)",
    "left_wrist_yaw_joint":      "左腕偏移 (桡/尺偏)",
    # ── 右臂（7个关节）──
    "right_shoulder_pitch_joint": "右肩俯仰 (前后抬臂)",
    "right_shoulder_roll_joint":  "右肩侧展 (内外展臂)",
    "right_shoulder_yaw_joint":   "右肩旋转 (臂内外旋)",
    "right_elbow_joint":          "右肘关节 (弯曲伸直)",
    "right_wrist_roll_joint":     "右腕旋转 (前臂旋转)",
    "right_wrist_pitch_joint":    "右腕俯仰 (掌屈/背伸)",
    "right_wrist_yaw_joint":      "右腕偏移 (桡/尺偏)",
}

# 打印关节信息
print("=" * 60)
print("G1 人形机器人 - 可控铰链关节:")
print("=" * 60)
for name in hinge_joint_names:
    rng = joint_ranges.get(name, None)
    rng_str = f", 限位: [{rng[0]:.2f}, {rng[1]:.2f}]" if rng else ""
    desc = joint_descriptions.get(name, "")
    print(f"  {name}: {desc}{rng_str}")
print(f"\n共 {len(hinge_joint_names)} 个铰链关节")
print("=" * 60)

# ──────────────────────────────────────────────
# 腕关节锁定（默认开启，W 键切换）
# 拖动手末端球体时，将双手6个腕关节强制锁在0位，
# 保证手末端轨迹形状纯粹由肩/肘决定，不被腕关节扰动。
# ──────────────────────────────────────────────
# 拖手末端时需要锁定的关节：双手腕（6个）+ 腰部（3个）
# 腰锁定保证上半身整体不动，手末端轨迹完全由肩/肘决定。
UPPER_LOCK_JOINT_NAMES = [
    # 双手腕（保持手型不变）
    "left_wrist_roll_joint",  "left_wrist_pitch_joint",  "left_wrist_yaw_joint",
    "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
    # 腰部（上半身不扭转/侧弯/前后弯）
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
]

# 预计算各锁定关节在 qpos / qvel 中的地址
_lock_qpos_addrs: dict[str, int] = {}
_lock_dof_addrs:  dict[str, int] = {}
for _jname in UPPER_LOCK_JOINT_NAMES:
    _jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, _jname)
    if _jid >= 0:
        _lock_qpos_addrs[_jname] = int(model.jnt_qposadr[_jid])
        _lock_dof_addrs[_jname]  = int(model.jnt_dofadr[_jid])

wrist_locked = True   # 默认开启（同时锁定腕关节 + 腰部关节）


def lock_wrist_joints():
    """将双手腕（6个）+ 腰部（3个）关节强制设为0（位置 + 速度），并重算FK"""
    for jname, qaddr in _lock_qpos_addrs.items():
        data.qpos[qaddr] = 0.0
        data.qvel[_lock_dof_addrs[jname]] = 0.0
    mujoco.mj_forward(model, data)


def toggle_wrist_lock():
    """切换腕+腰关节锁定开关"""
    global wrist_locked
    wrist_locked = not wrist_locked
    print(f"腕/腰关节锁定: {'✓ 开启（拖手末端时上半身不动）' if wrist_locked else '✗ 关闭（所有关节自由）'}")


# ──────────────────────────────────────────────
# 关节阻尼设置
# ──────────────────────────────────────────────
current_damping = 10.0
set_joint_damping(model, damping_value=current_damping, armature_value=1.0)

# ──────────────────────────────────────────────
# 关节角度状态（仅铰链关节）
# ──────────────────────────────────────────────
joint_angles = {name: 0.0 for name in hinge_joint_names}
joint_damping_backup = {}


def update_joint_angle(joint_name, angle):
    """更新指定铰链关节的角度"""
    if joint_name not in joint_qpos_map:
        return
    joint_type = joint_types[joint_ids[joint_name]]
    if joint_type in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
        data.qpos[qpos_index(joint_qpos_map[joint_name])] = angle
        joint_angles[joint_name] = angle


def reset_to_home():
    """重置机器人到初始 T 形站立姿态（零关节角，骨盆归位）"""
    # 重置浮动基座位置和朝向
    if FREE_JOINT_NAME in joint_qpos_map:
        addr = qpos_index(joint_qpos_map[FREE_JOINT_NAME])
        data.qpos[addr:addr + 3] = HOME_PELVIS_POS
        data.qpos[addr + 3:addr + 7] = HOME_PELVIS_QUAT

    # 重置所有铰链关节到0
    for name in hinge_joint_names:
        update_joint_angle(name, 0.0)

    # 同步sphere控制球位置到初始状态
    _reset_sphere_positions()
    mujoco.mj_forward(model, data)
    print("已重置到初始姿态 (T-pose)")


def _reset_sphere_positions():
    """将所有sphere控制球重置到对应机器人body的当前位置"""
    sphere_to_body = {
        "sphere_left_foot":  "left_ankle_roll_link",
        "sphere_right_foot": "right_ankle_roll_link",
        "sphere_left_elbow": "left_elbow_link",
        "sphere_right_elbow": "right_elbow_link",
        "sphere_left_hand":  "left_wrist_yaw_link",
        "sphere_right_hand": "right_wrist_yaw_link",
        "sphere_pelvis":     "pelvis",
    }
    for sphere_name, body_name in sphere_to_body.items():
        sphere_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, sphere_name)
        body_id   = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if sphere_id >= 0 and body_id >= 0:
            model.body_pos[sphere_id] = data.xpos[body_id].copy()


def reset_all_joints():
    """重置所有铰链关节到0（由UI按钮调用）"""
    reset_to_home()


# ──────────────────────────────────────────────
# 关键帧管理（离散快照，S 键保存）
# ──────────────────────────────────────────────
keyframe_manager = KeyframeManager(model, data, joint_names=hinge_joint_names)

should_save_keyframe = False

def trigger_save_keyframe():
    global should_save_keyframe
    should_save_keyframe = True


# ──────────────────────────────────────────────
# 轨迹连续录制（R 键开始/停止，100 Hz，存 NPZ）
# ──────────────────────────────────────────────
RECORDINGS_DIR = "recordings"
os.makedirs(RECORDINGS_DIR, exist_ok=True)

is_recording        = False
_rec_qpos_buf:  list = []
_rec_qvel_buf:  list = []
_rec_simtime_buf: list = []


def start_recording():
    global is_recording
    if is_recording:
        return
    _rec_qpos_buf.clear()
    _rec_qvel_buf.clear()
    _rec_simtime_buf.clear()
    is_recording = True
    print("\n[录制] ▶  开始录制轨迹（按 R 停止）")


def stop_and_save_recording():
    global is_recording
    if not is_recording:
        return
    is_recording = False
    n = len(_rec_qpos_buf)
    if n == 0:
        print("[录制] ■  无数据，未保存")
        return

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(RECORDINGS_DIR, f"g1_traj_{ts}.npz")
    np.savez_compressed(
        save_path,
        qpos=np.array(_rec_qpos_buf, dtype=np.float32),          # (N, 36)
        qvel=np.array(_rec_qvel_buf, dtype=np.float32),          # (N, 35)
        timestamps=np.array(_rec_simtime_buf, dtype=np.float32), # (N,)  仿真时间
        record_dt=np.float32(0.01),                              # 名义帧间隔
    )
    duration = _rec_simtime_buf[-1] - _rec_simtime_buf[0] if n > 1 else 0.0
    print(f"[录制] ■  停止，共 {n} 帧 / {duration:.2f}s → {save_path}")


def toggle_recording():
    if is_recording:
        stop_and_save_recording()
    else:
        start_recording()


# ──────────────────────────────────────────────
# 阻尼调节
# ──────────────────────────────────────────────
def increase_damping():
    global current_damping
    current_damping += 1.0
    set_joint_damping(model, damping_value=current_damping, armature_value=1.0)
    print(f"关节阻尼增加至: {current_damping:.1f}")


def decrease_damping():
    global current_damping
    current_damping = max(0.0, current_damping - 1.0)
    set_joint_damping(model, damping_value=current_damping, armature_value=1.0)
    print(f"关节阻尼减少至: {current_damping:.1f}")


# ──────────────────────────────────────────────
# 快捷键（Linux 下 keyboard 库需 root；默认用 Tk + MuJoCo 窗口，无需 sudo）
# ──────────────────────────────────────────────
hotkey_state = {"ctrl": False}
use_global_keyboard = False

# GLFW 控制键码（MuJoCo viewer key_callback）
_GLFW_KEY_LEFT_CTRL = 341
_GLFW_KEY_RIGHT_CTRL = 345


def setup_global_keyboard_hotkeys():
    """尝试注册 keyboard 全局热键；失败则返回 False（普通用户可继续运行）。"""
    global use_global_keyboard
    try:
        import keyboard
        keyboard.on_press_key("s", lambda _: trigger_save_keyframe())
        keyboard.on_press_key("h", lambda _: reset_to_home())
        keyboard.on_press_key("w", lambda _: toggle_wrist_lock())
        keyboard.on_press_key("r", lambda _: toggle_recording())
        keyboard.on_press_key("[", lambda _: decrease_damping())
        keyboard.on_press_key("]", lambda _: increase_damping())
        use_global_keyboard = True
        return True
    except Exception as exc:
        use_global_keyboard = False
        print(f"[快捷键] keyboard 库不可用: {exc}")
        print("[快捷键] 已改用 Tk / MuJoCo 窗口快捷键（无需 sudo，文件归当前用户）")
        return False


def bind_tk_hotkeys(root):
    """在 Tk 窗口绑定快捷键（窗口有焦点时生效）。"""
    root.bind_all("<KeyPress-s>", lambda _e: trigger_save_keyframe())
    root.bind_all("<KeyPress-S>", lambda _e: trigger_save_keyframe())
    root.bind_all("<KeyPress-h>", lambda _e: reset_to_home())
    root.bind_all("<KeyPress-H>", lambda _e: reset_to_home())
    root.bind_all("<KeyPress-w>", lambda _e: toggle_wrist_lock())
    root.bind_all("<KeyPress-W>", lambda _e: toggle_wrist_lock())
    root.bind_all("<KeyPress-r>", lambda _e: toggle_recording())
    root.bind_all("<KeyPress-R>", lambda _e: toggle_recording())
    root.bind_all("<KeyPress-bracketleft>", lambda _e: decrease_damping())
    root.bind_all("<KeyPress-bracketright>", lambda _e: increase_damping())
    root.bind_all("<Control_L>", lambda _e: hotkey_state.__setitem__("ctrl", True))
    root.bind_all("<Control_R>", lambda _e: hotkey_state.__setitem__("ctrl", True))
    root.bind_all("<KeyRelease-Control_L>", lambda _e: hotkey_state.__setitem__("ctrl", False))
    root.bind_all("<KeyRelease-Control_R>", lambda _e: hotkey_state.__setitem__("ctrl", False))


def make_mujoco_key_callback():
    """MuJoCo 查看器窗口快捷键（该窗口有焦点时生效）。"""
    def _callback(keycode):
        if keycode in (_GLFW_KEY_LEFT_CTRL, _GLFW_KEY_RIGHT_CTRL):
            hotkey_state["ctrl"] = True
            return
        if keycode in (ord("s"), ord("S")):
            trigger_save_keyframe()
        elif keycode in (ord("h"), ord("H")):
            reset_to_home()
        elif keycode in (ord("w"), ord("W")):
            toggle_wrist_lock()
        elif keycode in (ord("r"), ord("R")):
            toggle_recording()
        elif keycode == ord("["):
            decrease_damping()
        elif keycode == ord("]"):
            increase_damping()
    return _callback


def is_ctrl_pressed():
    if use_global_keyboard:
        import keyboard
        return keyboard.is_pressed("ctrl")
    return hotkey_state.get("ctrl", False)


def sphere_drag_ik_active():
    """拖黄色球体时是否启用直接 IK。无全局 keyboard 时默认开启（无需 Ctrl）。"""
    return is_ctrl_pressed() or not use_global_keyboard


setup_global_keyboard_hotkeys()


# ──────────────────────────────────────────────
# 线程共享数据
# ──────────────────────────────────────────────
class SharedData:
    def __init__(self):
        self.keyframe_count = 0
        self.running = True
        self.lock = threading.Lock()


shared_data = SharedData()


# ──────────────────────────────────────────────
# Tk UI 线程
# ──────────────────────────────────────────────
def ui_thread_function():
    ui = RobotJointUI(
        joint_names=hinge_joint_names,      # 仅传入铰链关节，过滤掉浮动基座
        joint_ids=joint_ids,
        joint_types=joint_types,
        joint_qpos_map=joint_qpos_map,
        joint_dim_map=joint_dim_map,
        joint_ranges=joint_ranges,
        joint_descriptions=joint_descriptions,
        update_joint_angle_callback=update_joint_angle,
        reset_all_joints_callback=reset_all_joints,
        save_keyframe_callback=keyframe_manager.save_keyframe,
        save_all_keyframes_callback=keyframe_manager.save_keyframes_to_file
    )

    bind_tk_hotkeys(ui.root)

    def update_ui_periodically():
        if shared_data.running:
            with shared_data.lock:
                ui.update_ui(data, shared_data.keyframe_count)
            ui.root.after(100, update_ui_periodically)

    update_ui_periodically()
    ui.start()


ui_thread = threading.Thread(target=ui_thread_function, daemon=True)
ui_thread.start()
time.sleep(0.5)   # 等待 UI 线程就绪


# ──────────────────────────────────────────────
# MuJoCo 主循环
# ──────────────────────────────────────────────
print("\nG1 姿态捕获系统已启动")
print("─" * 60)
print("  MuJoCo窗口：直接拖拽机器人body可调整姿态")
print("  Ctrl+拖拽 黄色球体：拖动手/脚/肘/骨盆末端（IK模式）")
print("  Tk滑动条窗口：精确控制每个关节角度")
print("  鼠标悬停滑动条 + 滚轮：微调关节角度")
print()
print("  快捷键（先点击 Tk 或 MuJoCo 窗口再按键）：")
if use_global_keyboard:
    print("    （已启用全局 keyboard，任意窗口均可）")
else:
    print("    （未用 sudo：在 Tk / MuJoCo 窗口内按键；拖黄球无需 Ctrl）")
print("    S    - 保存当前姿态为关键帧")
print("    H    - 重置到初始 T 形站立姿态")
print("    W    - 切换腕/腰关节锁定（默认：开启）")
print("           开启：拖手末端时腕(6)+腰(3)保持0，上半身不动")
print("           关闭：所有关节自由参与IK")
print("    R    - 开始/停止轨迹录制（自动保存 recordings/ 目录）")
print("    [    - 减少关节阻尼（更易拖动）")
print("    ]    - 增加关节阻尼（更稳定）")
print("    ESC  - 退出程序")
print("─" * 60)
print(f"  [腕/腰关节锁定]: {'开启 ✓  (腕×6 + 腰×3 保持0)' if wrist_locked else '关闭'}")
print(f"  [轨迹录制]: 就绪，按 R 开始")

is_dragging = False

with mujoco.viewer.launch_passive(
    model, data, key_callback=make_mujoco_key_callback()
) as viewer:
    while viewer.is_running():
        current_perturbation = data.xfrc_applied.copy()
        has_perturbation = np.any(np.abs(current_perturbation) > 1e-6)
        ctrl_pressed = is_ctrl_pressed()
        sphere_ik = sphere_drag_ik_active()

        # ── 拖拽开始检测 ──
        if has_perturbation and not is_dragging:
            perturbed_bodies = [
                i for i in range(model.nbody)
                if np.any(np.abs(current_perturbation[i]) > 1e-6)
            ]
            if perturbed_bodies:
                is_dragging = True
                for body_id in perturbed_bodies:
                    body_name = model.body(body_id).name
                    if "sphere" in body_name.lower() and sphere_ik:
                        print(f"\n[拖拽球体] {body_name} → IK 模式")
                    # 对机器人body降阻尼，球体跳过（让约束传递力）
                    related = set_selected_body_joints_damping(
                        model, body_id, 0.0, joint_damping_backup
                    )
                    if related:
                        print(f"\n拖拽 {body_name}，相关关节阻尼→0: {', '.join(related)}")

        # ── 拖拽结束检测 ──
        elif not has_perturbation and is_dragging:
            is_dragging = False
            restore_all_joint_damping(model, joint_damping_backup)
            joint_damping_backup.clear()
            print("\n拖拽结束，关节阻尼已恢复")

        # ── 拖拽球体时：移动球体位置（IK 驱动） ──
        if is_dragging and sphere_ik:
            for i in range(model.nbody):
                body_name = model.body(i).name
                if "sphere" in body_name.lower() and np.any(np.abs(current_perturbation[i]) > 1e-6):
                    force = current_perturbation[i][:3]
                    force_norm = np.linalg.norm(force)
                    if force_norm < 1e-6:
                        continue
                    force_dir = force / force_norm

                    current_pos = model.body_pos[i].copy()
                    move_scale = 0.01
                    new_pos = current_pos + move_scale * force_dir

                    # 限制单步最大移动距离，防止跳变
                    delta = new_pos - current_pos
                    dist = np.linalg.norm(delta)
                    if dist > 0.05:
                        new_pos = current_pos + (delta / dist) * 0.05

                    if move_sphere_body_directly(model, data, i, new_pos):
                        if dist > 0.01:
                            print(f"球体 {body_name} → {new_pos.round(3)}")

        # ── 关键帧保存 ──
        if should_save_keyframe:
            keyframe_manager.save_keyframe()
            should_save_keyframe = False
            with shared_data.lock:
                shared_data.keyframe_count = keyframe_manager.get_keyframe_count()
            print(f"已保存关键帧 #{shared_data.keyframe_count}")

        # ── 物理步进 ──
        mujoco.mj_step(model, data)

        # ── 腕关节锁定（锁定后重算FK，保证渲染与物理一致）──
        if wrist_locked:
            lock_wrist_joints()

        # ── 轨迹录制：每步追加 qpos / qvel（使用仿真时间作时间戳）──
        if is_recording:
            _rec_qpos_buf.append(data.qpos.copy())    # (36,)
            _rec_qvel_buf.append(data.qvel.copy())    # (35,)
            _rec_simtime_buf.append(float(data.time)) # 仿真时间(s)

        # ── 同步视图 ──
        viewer.sync()
        time.sleep(0.01)

    shared_data.running = False

# ── 退出时保存关键帧 ──
if keyframe_manager.get_keyframe_count() > 0:
    keyframe_manager.save_keyframes_to_file()
    print(f"\n已保存 {keyframe_manager.get_keyframe_count()} 个关键帧到文件")
