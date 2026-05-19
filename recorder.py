import mujoco
import mujoco.viewer
import numpy as np
import time
import os
import keyboard  # 需要安装：pip install keyboard
import tkinter as tk
from tkinter import ttk, messagebox
import threading

# 导入自定义模块
from utils.joint_utils import (
    get_joint_info, set_joint_damping, set_specific_joint_damping,
    find_related_joints, set_selected_body_joints_damping, restore_all_joint_damping,
    toggle_sphere_constraints, find_sphere_joint, update_sphere_position, 
    move_sphere_directly, move_sphere_body_directly
)
from utils.ui_controller import RobotJointUI
from utils.keyframe_manager import KeyframeManager

# 加载模型和数据
model_path = "go2/scene_zero_gravity_free_draggable.xml"
model = mujoco.MjModel.from_xml_path(model_path)
data = mujoco.MjData(model)

# 初始化数据
mujoco.mj_forward(model, data)

# 获取关节信息
joint_info = get_joint_info(model)
joint_names = joint_info['joint_names']
joint_ids = joint_info['joint_ids']
joint_types = joint_info['joint_types']
joint_qpos_map = joint_info['joint_qpos_map']
joint_dim_map = joint_info['joint_dim_map']
joint_ranges = joint_info['joint_ranges']

# 关节描述映射
joint_descriptions = {
    "FL_hip_joint": "左前肩关节 (里-外)",
    "FL_thigh_joint": "左前大臂 (前-后)",
    "FL_calf_joint": "左前小臂 (前-后)",
    "FR_hip_joint": "右前肩关节 (外-里)",
    "FR_thigh_joint": "右前大臂 (前-后)",
    "FR_calf_joint": "右前小臂 (前-后)",
    "RL_hip_joint": "左后肩关节 (里-外)",
    "RL_thigh_joint": "左后大臂 (前-后)",
    "RL_calf_joint": "左后小臂 (前-后)",
    "RR_hip_joint": "右后肩关节 (外-里)",
    "RR_thigh_joint": "右后大臂 (前-后)",
    "RR_calf_joint": "右后小臂 (前-后)"
}

# 打印可控制的关节
print("可控制的关节:")
for name in joint_names:
    if name:
        dim = joint_dim_map[name]
        type_str = "未知"
        if joint_types[joint_ids[name]] == mujoco.mjtJoint.mjJNT_FREE:
            type_str = "自由关节"
        elif joint_types[joint_ids[name]] == mujoco.mjtJoint.mjJNT_BALL:
            type_str = "球关节"
        elif joint_types[joint_ids[name]] == mujoco.mjtJoint.mjJNT_HINGE:
            type_str = "铰链关节"
        elif joint_types[joint_ids[name]] == mujoco.mjtJoint.mjJNT_SLIDE:
            type_str = "滑动关节"
        
        range_info = ""
        if name in joint_ranges:
            lower, upper = joint_ranges[name]
            range_info = f", 限位: [{lower:.2f}, {upper:.2f}]"
            
        print(f"  - {name}: {type_str} (维度: {dim}{range_info})")

# 设置关节阻尼
set_joint_damping(model, damping_value=10.0, armature_value=1.0)

# 创建一个字典，存储每个关节的当前角度
joint_angles = {}
for name in joint_names:
    if name:  # 确保关节名称不为空
        joint_angles[name] = 0.0

# 创建关节阻尼备份，用于拖拽时恢复
joint_damping_backup = {}

# 更新关节角度
def update_joint_angle(joint_name, angle):
    """更新关节角度"""
    if joint_name in joint_qpos_map:
        qpos_addr = joint_qpos_map[joint_name]
        joint_type = joint_types[joint_ids[joint_name]]
        
        if joint_type == mujoco.mjtJoint.mjJNT_HINGE or joint_type == mujoco.mjtJoint.mjJNT_SLIDE:
            # 只更新铰链关节和滑动关节
            data.qpos[qpos_addr] = angle
            joint_angles[joint_name] = angle

# 重置所有关节到初始位置
def reset_all_joints():
    """重置所有关节到初始位置"""
    for name in joint_names:
        if name in joint_qpos_map:
            joint_type = joint_types[joint_ids[name]]
            if joint_type == mujoco.mjtJoint.mjJNT_HINGE or joint_type == mujoco.mjtJoint.mjJNT_SLIDE:
                update_joint_angle(name, 0.0)

# 创建关键帧管理器
keyframe_manager = KeyframeManager(model, data, joint_names=joint_names)

# 触发保存关键帧
def trigger_save_keyframe():
    """触发保存关键帧"""
    global should_save_keyframe
    should_save_keyframe = True

# 增加和减少阻尼的函数
def increase_damping():
    """增加所有关节的阻尼"""
    global current_damping
    current_damping += 1.0
    set_joint_damping(model, damping_value=current_damping, armature_value=1.0)
    print(f"增加阻尼至: {current_damping}")

def decrease_damping():
    """减少所有关节的阻尼"""
    global current_damping
    current_damping = max(0.0, current_damping - 1.0)
    set_joint_damping(model, damping_value=current_damping, armature_value=1.0)
    print(f"减少阻尼至: {current_damping}")

# 初始化关键帧相关变量
should_save_keyframe = False

# 初始化当前阻尼值
current_damping = 10.0

# 设置键盘监听
keyboard.on_press_key("s", lambda _: trigger_save_keyframe())
keyboard.on_press_key("[", lambda _: decrease_damping())
keyboard.on_press_key("]", lambda _: increase_damping())

# 创建一个共享变量，用于在线程间传递数据
class SharedData:
    def __init__(self):
        self.keyframe_count = 0
        self.running = True
        self.lock = threading.Lock()

shared_data = SharedData()

# 创建UI线程函数
def ui_thread_function():
    """UI线程函数"""
    # 创建UI控制器
    ui = RobotJointUI(
        joint_names=joint_names,
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
    
    # 定义UI更新函数
    def update_ui_periodically():
        if shared_data.running:
            with shared_data.lock:
                ui.update_ui(data, shared_data.keyframe_count)
            # 每100毫秒更新一次
            ui.root.after(100, update_ui_periodically)
    
    # 启动UI更新
    update_ui_periodically()
    
    # 启动UI主循环
    ui.start()

# 创建并启动UI线程
ui_thread = threading.Thread(target=ui_thread_function)
ui_thread.daemon = True  # 设置为守护线程，这样主线程退出时UI线程也会退出
ui_thread.start()

# 等待UI线程启动
time.sleep(0.5)

# 创建查看器并开始模拟
with mujoco.viewer.launch_passive(model, data) as viewer:
    print("机器人关节控制器已启动。")
    print("提示：使用UI滑动条可以精确控制每个关节的角度")
    print("      在MuJoCo窗口中，按住Ctrl并用鼠标拖动sphere可直接修改其位置")
    print("      拖拽物体时，相关关节阻尼会自动设为0，松开后恢复")
    print("      按S键或点击'保存关键帧'按钮保存当前姿势")
    print("      按[键减少阻尼，按]键增加阻尼")
    print("      鼠标悬停在滑动条上，滚轮向上增加数值，向下减少数值")
    print("      按ESC键退出程序")
    
    # 跟踪拖拽状态
    is_dragging = False
    
    # 主循环
    while viewer.is_running():
        # 检测拖拽状态 - 使用扰动(perturbation)来判断
        # 在MuJoCo中，当用户拖拽物体时，会产生扰动
        current_perturbation = data.xfrc_applied.copy()
        
        # 检查是否有扰动（用户正在拖拽物体）
        has_perturbation = np.any(np.abs(current_perturbation) > 1e-6)
        
        # 检查是否按下了Ctrl键
        ctrl_pressed = keyboard.is_pressed('ctrl')
        
        if has_perturbation and not is_dragging:
            # 找到被扰动的物体
            perturbed_bodies = []
            for i in range(model.nbody):
                if np.any(np.abs(current_perturbation[i]) > 1e-6):
                    perturbed_bodies.append(i)
            
            if perturbed_bodies:
                is_dragging = True
                for body_id in perturbed_bodies:
                    # 获取物体名称
                    body_name = model.body(body_id).name
                    
                    # 如果是sphere，检查是否按下Ctrl键
                    if body_name == "sphere" and ctrl_pressed:
                        print(f"\n按住Ctrl拖拽sphere - 直接修改位置模式")
                    
                    # 设置相关关节阻尼为0
                    related_joints = set_selected_body_joints_damping(model, body_id, 0.0, joint_damping_backup)
                    if related_joints and related_joints != ["sphere_constraint_disabled"]:
                        print(f"\n拖拽物体 {body_name}，相关关节阻尼设为0: {', '.join(related_joints)}")
        
        elif not has_perturbation and is_dragging:
            # 拖拽结束
            is_dragging = False
            
            # 恢复关节阻尼
            restore_all_joint_damping(model, joint_damping_backup)
            print("\n拖拽结束，恢复关节阻尼")
            
            # 清空备份
            joint_damping_backup.clear()
        
        # 检查是否应该保存关键帧
        if should_save_keyframe:
            keyframe_manager.save_keyframe()
            should_save_keyframe = False
            
            # 更新共享数据
            with shared_data.lock:
                shared_data.keyframe_count = keyframe_manager.get_keyframe_count()
        
        # 如果正在拖拽 sphere 并按下了Ctrl键
        if is_dragging and ctrl_pressed:
            # 检查是否有 sphere 被拖拽
            for i in range(model.nbody):
                body_name = model.body(i).name
                if "sphere" in body_name.lower() and np.any(np.abs(current_perturbation[i]) > 1e-6):
                    # 获取当前鼠标位置（通过扰动方向估计）
                    force = current_perturbation[i][:3]  # 前3个是力
                    
                    # 归一化力向量，防止数值过大
                    force_norm = np.linalg.norm(force)
                    if force_norm > 1e-6:  # 避免除以零
                        force = force / force_norm
                    
                    # 获取当前位置
                    current_pos = model.body_pos[i].copy()
                    
                    # 根据力的方向移动 sphere，使用适当的移动比例因子
                    move_scale = 0.01  # 调整移动比例因子
                    new_pos = current_pos + move_scale * force
                    
                    # 限制移动范围，防止位置异常
                    max_distance = 0.05  # 每次最大移动距离
                    move_vector = new_pos - current_pos
                    move_distance = np.linalg.norm(move_vector)
                    if move_distance > max_distance:
                        move_vector = (move_vector / move_distance) * max_distance
                        new_pos = current_pos + move_vector
                    
                    # 直接修改sphere的位置
                    if move_sphere_body_directly(model, data, i, new_pos):
                        # 减少日志输出，只在位置变化较大时打印
                        if move_distance > 0.01:
                            print(f"直接修改 {body_name} 位置: {new_pos}")
                    else:
                        print(f"修改 {body_name} 位置失败")
        
        # 更新物理
        mujoco.mj_step(model, data)
        
        # 同步查看器
        viewer.sync()
        
        # 控制帧率
        time.sleep(0.01)
    
    # 设置运行标志为False
    shared_data.running = False
    
    # 退出时保存关键帧
    if keyframe_manager.get_keyframe_count() > 0:
        keyframe_manager.save_keyframes_to_file() 