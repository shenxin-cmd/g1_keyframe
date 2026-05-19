import mujoco
import mujoco.viewer
import numpy as np
import time
import os
import keyboard  # 需要安装：pip install keyboard

# 加载模型和数据
model_path = "go2/scene_zero_gravity.xml"
model = mujoco.MjModel.from_xml_path(model_path)
data = mujoco.MjData(model)

# 创建一个标志，用于指示是否应该退出程序
should_exit = False
should_save = False

# 设置ESC键监听（退出程序）
def on_esc_press(e):
    global should_exit
    if e.name == 'esc':
        print("检测到ESC键，准备退出并保存数据...")
        should_exit = True

# 设置S键监听（保存数据）
def on_s_press(e):
    global should_save
    if e.name == 's':
        print("检测到S键，准备保存数据...")
        should_save = True

# 注册按键监听
keyboard.on_press(on_esc_press)
keyboard.on_press(on_s_press)

# 创建查看器
with mujoco.viewer.launch_passive(model, data) as viewer:
    # 创建一个字典，用于存储不同部位的轨迹数据
    trajectories = {}

    # 获取所有可能需要跟踪的身体部位
    body_names = [model.body(i).name for i in range(model.nbody) if model.body(i).name != "world"]
    print(f"可跟踪的身体部位: {body_names}")

    # 初始化轨迹数据结构
    for body_name in body_names:
        trajectories[body_name] = []

    # 记录上一帧的位置，用于检测拖动
    last_positions = {}
    for body_name in body_names:
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body_id >= 0:
            last_positions[body_name] = np.copy(data.xpos[body_id])

    # 设置拖动检测的阈值
    drag_threshold = 0.005  # 增大阈值，使检测更容易触发
    is_dragging = False
    currently_dragged_body = None
    start_time = time.time()

    print("开始记录轨迹。")
    print("按 S 键保存当前记录的所有轨迹")
    print("按 ESC 键退出并保存数据")
    print("提示：在MuJoCo窗口中，按住Ctrl并用鼠标拖动物体")

    # 保存轨迹数据的函数
    def save_trajectories():
        output_dir = "trajectories"
        os.makedirs(output_dir, exist_ok=True)
        
        # 检查是否有任何轨迹数据
        has_data = False
        for trajectory in trajectories.values():
            if trajectory:
                has_data = True
                break
        
        if not has_data:
            print("没有轨迹数据可保存")
            return
        
        # 将所有轨迹数据保存到一个文件中
        timestamp = int(time.time())
        filename = os.path.join(output_dir, f"all_trajectories_{timestamp}.npz")
        
        try:
            # 使用npz格式保存多个数组，每个数组对应一个身体部位的轨迹
            save_dict = {}
            for body_name, trajectory in trajectories.items():
                if trajectory:  # 只保存有数据的轨迹
                    save_dict[body_name] = np.array(trajectory, dtype=float)
            
            np.savez(filename, **save_dict)
            print(f"已将所有轨迹数据保存到单个文件: {filename}")
            print(f"文件包含 {len(save_dict)} 个物体的轨迹数据")
            
            # 显示每个物体的轨迹点数量
            for body_name, trajectory in trajectories.items():
                if trajectory:
                    print(f"  - {body_name}: {len(trajectory)} 个点")
        except Exception as e:
            print(f"保存轨迹数据时出错: {e}")

    try:
        while viewer.is_running() and not should_exit:
            # 检查是否需要保存数据
            if should_save:
                print("\n正在保存轨迹数据...")
                save_trajectories()
                should_save = False
                print("继续记录轨迹...")
            
            # 更新仿真
            mujoco.mj_step(model, data)
            
            # 检查是否有物体被拖动
            for body_name in body_names:
                body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
                if body_id >= 0:
                    current_pos = np.copy(data.xpos[body_id])
                    
                    # 计算位置变化
                    pos_change = np.linalg.norm(current_pos - last_positions[body_name])
                    
                    # 如果位置变化超过阈值，认为物体被拖动
                    if pos_change > drag_threshold:
                        if not is_dragging or currently_dragged_body != body_name:
                            print(f"开始拖动 {body_name}")
                            is_dragging = True
                            currently_dragged_body = body_name
                        
                        # 记录时间和位置
                        elapsed_time = time.time() - start_time
                        # 将位置转换为列表，确保数据格式一致
                        trajectories[body_name].append([elapsed_time, current_pos[0], current_pos[1], current_pos[2]])
                        print(f"记录 {body_name} 位置: {current_pos}", end="\r")
                    
                    # 更新上一帧位置
                    last_positions[body_name] = current_pos
            
            # 如果没有物体被拖动，重置拖动状态
            if is_dragging:
                all_static = True
                for body_name in body_names:
                    if body_name == currently_dragged_body:  # 只检查当前拖动的物体
                        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
                        if body_id >= 0:
                            pos_change = np.linalg.norm(data.xpos[body_id] - last_positions[body_name])
                            if pos_change > drag_threshold:
                                all_static = False
                                break
                
                if all_static:
                    print(f"\n停止拖动 {currently_dragged_body}")
                    is_dragging = False
                    currently_dragged_body = None
            
            # 渲染场景
            viewer.sync()
            
            # 添加一个小延迟
            time.sleep(0.01)
            
    except KeyboardInterrupt:
        print("\n程序被用户中断")
    except Exception as e:
        print(f"\n程序出错: {e}")
    finally:
        # 保存所有轨迹数据
        print("\n正在保存所有轨迹数据...")
        save_trajectories()
        print("\n程序结束")
