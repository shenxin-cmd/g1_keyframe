import mujoco
import mujoco.viewer
import numpy as np
import time
import os
import keyboard  # 需要安装：pip install keyboard

# 加载模型和数据
model_path = "go2/scene_stopmotion.xml"
model = mujoco.MjModel.from_xml_path(model_path)
data = mujoco.MjData(model)

# 初始化数据 - 不使用关键帧，而是直接使用默认状态
mujoco.mj_resetData(model, data)

# 如果需要使用关键帧，请确保关键帧存在且控制器大小正确
# try:
#     keyframe_id = model.key_name2id("default_pose")
#     mujoco.mj_resetDataKeyframe(model, data, keyframe_id)
# except Exception as e:
#     print(f"警告：无法加载关键帧，使用默认姿势。错误: {e}")
#     mujoco.mj_resetData(model, data)

# 创建保存关键帧的目录
keyframe_dir = "keyframes"
os.makedirs(keyframe_dir, exist_ok=True)

# 获取所有关节信息
joint_names = []
joint_qpos_addrs = []  # 存储每个关节的qpos地址
joint_types = []

for i in range(model.njnt):
    joint_name = model.joint(i).name
    joint_type = model.joint(i).type
    joint_qposadr = model.joint(i).qposadr  # 获取关节的qpos地址
    
    joint_names.append(joint_name)
    joint_qpos_addrs.append(joint_qposadr)
    joint_types.append(joint_type)

# 创建关节名称到ID和qpos地址的映射
joint_ids = {}
joint_qpos_map = {}
for i, name in enumerate(joint_names):
    if name:  # 确保关节名称不为空
        joint_ids[name] = i
        joint_qpos_map[name] = joint_qpos_addrs[i]

# 打印可控制的关节
print("可控制的关节:", joint_names)

# 初始化关键帧列表
keyframes = []

# 创建标志，用于指示是否应该保存关键帧
should_save_keyframe = False

# 设置S键监听（保存关键帧）
def on_s_press(e):
    global should_save_keyframe
    if e.name == 's':
        print("\n检测到S键，保存当前关键帧...")
        should_save_keyframe = True

# 注册按键监听
keyboard.on_press(on_s_press)

# 创建查看器并开始记录
with mujoco.viewer.launch_passive(model, data) as viewer:
    print("开始记录关键帧。")
    print("按 S 键保存当前姿势作为关键帧")
    print("按 ESC 键退出并保存所有关键帧")
    print("提示：在MuJoCo窗口中，按住Ctrl并用鼠标拖动物体调整姿势")
    
    # 主循环
    while viewer.is_running():
        # 更新物理
        mujoco.mj_step(model, data)
        
        # 同步查看器
        viewer.sync()
        
        # 如果检测到S键被按下，保存当前关键帧
        if should_save_keyframe:
            # 创建关键帧字典，使用关节名称作为键
            keyframe = {'timestamp': time.time()}
            
            # 保存每个关节的当前角度
            for name, joint_id in joint_ids.items():
                # 使用qpos地址获取关节角度
                qpos_addr = joint_qpos_map[name]
                joint_type = joint_types[joint_id]
                
                if joint_type == mujoco.mjtJoint.mjJNT_FREE:
                    # 自由关节有7个值：3个位置 + 4个四元数
                    keyframe[name] = [data.qpos[qpos_addr + j] for j in range(7)]
                elif joint_type == mujoco.mjtJoint.mjJNT_BALL:
                    # 球关节有4个值：四元数
                    keyframe[name] = [data.qpos[qpos_addr + j] for j in range(4)]
                else:
                    # 铰链关节和滑动关节只有1个值
                    keyframe[name] = data.qpos[qpos_addr]
            
            # 添加到关键帧列表
            keyframes.append(keyframe)
            
            # 打印保存的关键帧信息
            print(f"已保存关键帧 #{len(keyframes)}")
            for name, value in keyframe.items():
                if name != 'timestamp':
                    if isinstance(value, (list, np.ndarray)):
                        print(f"  - {name}: {value}")
                    else:
                        print(f"  - {name}: 角度 {value:.4f} 弧度 ({np.degrees(value):.2f}°)")
            
            # 重置标志
            should_save_keyframe = False
        
        # 控制帧率
        time.sleep(0.01)

# 保存所有关键帧到文件
if keyframes:
    timestamp = int(time.time())
    filename = f"{keyframe_dir}/joint_keyframes_{timestamp}.npz"
    np.savez(filename, keyframes=keyframes)
    print(f"\n已保存 {len(keyframes)} 个关键帧到文件: {filename}")
else:
    print("\n未记录任何关键帧，退出程序。") 