import mujoco
import mujoco.viewer
import numpy as np
import time
import os
import glob

# 加载模型和数据
model_path = "go2/scene_zero_gravity.xml"
model = mujoco.MjModel.from_xml_path(model_path)
data = mujoco.MjData(model)

# 查找所有轨迹文件
trajectory_dir = "trajectories"
trajectory_files = glob.glob(os.path.join(trajectory_dir, "*.npz"))  # 现在查找.npz文件

if not trajectory_files:
    print("未找到轨迹文件！请先运行轨迹记录程序。")
    exit()

# 显示可用的轨迹文件
print("可用的轨迹文件:")
for i, file_path in enumerate(trajectory_files):
    file_name = os.path.basename(file_path)
    print(f"{i+1}. {file_name}")

# 让用户选择要复现的轨迹
selection = int(input("请选择要复现的轨迹文件编号: ")) - 1
if selection < 0 or selection >= len(trajectory_files):
    print("无效的选择！")
    exit()

# 加载选定的轨迹数据
trajectory_path = trajectory_files[selection]
trajectory_data = np.load(trajectory_path)  # 加载.npz文件
print(f"已加载轨迹数据文件: {os.path.basename(trajectory_path)}")
print(f"文件包含以下物体的轨迹: {trajectory_data.files}")

# 让用户选择要复现的物体
print("\n请选择要复现的物体:")
for i, body_name in enumerate(trajectory_data.files):
    traj = trajectory_data[body_name]
    print(f"{i+1}. {body_name} (包含 {len(traj)} 个点)")

body_selection = int(input("请输入物体编号: ")) - 1
if body_selection < 0 or body_selection >= len(trajectory_data.files):
    print("无效的选择！")
    exit()

# 获取选定的物体名称和轨迹
body_name = list(trajectory_data.files)[body_selection]
body_trajectory = trajectory_data[body_name]
print(f"已选择 {body_name} 的轨迹，包含 {len(body_trajectory)} 个点")
print(f"轨迹数据形状: {body_trajectory.shape}")
print(f"轨迹数据前几个点: {body_trajectory[:min(3, len(body_trajectory))]}")

# 检查轨迹数据是否足够
if len(body_trajectory) < 2:
    print("警告：轨迹数据点太少，无法进行有意义的复现。请记录更多的轨迹点。")
    print("提示：在运行轨迹记录程序时，确保拖动物体足够长的时间和距离。")
    
    # 如果用户仍然想继续，可以添加一个确认
    confirm = input("是否仍然尝试复现这个轨迹？(y/n): ")
    if confirm.lower() != 'y':
        exit()

# 检查身体部位是否存在
body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
if body_id < 0:
    print(f"错误：在模型中找不到名为 {body_name} 的身体部位！")
    exit()

# 询问播放速度
print("\n请选择播放速度:")
print("1. 原始速度 (与记录时相同)")
print("2. 0.5倍速")
print("3. 2倍速")
print("4. 自定义速度")
speed_selection = int(input("请输入选项编号: "))

speed_factor = 1.0  # 默认为原始速度
if speed_selection == 2:
    speed_factor = 0.5
elif speed_selection == 3:
    speed_factor = 2.0
elif speed_selection == 4:
    speed_factor = float(input("请输入速度倍率 (例如 0.75): "))

print(f"将以 {speed_factor}x 速度播放轨迹")

# 创建查看器
with mujoco.viewer.launch_passive(model, data) as viewer:
    # 重置模型
    mujoco.mj_resetData(model, data)
    
    # 设置初始位置
    initial_pos = body_trajectory[0][1:4]  # 第一个点的位置
    
    print(f"开始复现轨迹，按ESC键退出...")
    print(f"初始位置: {initial_pos}")
    
    # 如果只有一个点，我们可以简单地将机器人移动到那个位置
    if len(body_trajectory) == 1:
        print("只有一个轨迹点，将机器人移动到该位置并保持...")
        
        # 移动到目标位置
        target_pos = body_trajectory[0][1:4]
        
        # 循环直到用户退出
        try:
            while viewer.is_running():
                # 使用位置控制器移动身体部位
                if body_name == "base_link":
                    # 如果是机器人主体，设置自由关节的位置
                    data.qpos[0:3] = target_pos
                else:
                    # 对于其他部位，使用PD控制器
                    kp = 500.0  # 位置增益
                    kd = 50.0   # 阻尼增益
                    
                    # 计算当前位置和目标位置之间的差异
                    current_pos = data.xpos[body_id]
                    pos_error = target_pos - current_pos
                    
                    # 应用力到身体部位
                    # 注意：在新版MuJoCo中，body_vel已不存在，改用cvel获取速度
                    force = kp * pos_error - kd * data.cvel[body_id, 0:3]  # 使用cvel代替body_vel
                    data.xfrc_applied[body_id, 0:3] = force
                
                # 打印当前位置，以便用户看到是否达到目标
                current_pos = data.xpos[body_id]
                distance = np.linalg.norm(current_pos - target_pos)
                print(f"目标位置: {target_pos}, 当前位置: {current_pos}, 距离: {distance:.6f}", end="\r")
                
                # 更新仿真
                mujoco.mj_step(model, data)
                
                # 渲染场景
                viewer.sync()
                
                # 控制仿真速度
                time.sleep(0.01)
        except KeyboardInterrupt:
            print("\n用户中断了轨迹复现")
        except Exception as e:
            print(f"\n复现轨迹时出错: {e}")
        finally:
            print("\n程序结束")
        
        exit()
    
    # 获取轨迹的总时长
    total_duration = body_trajectory[-1][0] - body_trajectory[0][0]
    print(f"轨迹总时长: {total_duration:.2f} 秒")
    
    # 准备开始播放
    print("准备开始播放...")
    print("3...")
    time.sleep(1)
    print("2...")
    time.sleep(1)
    print("1...")
    time.sleep(1)
    print("开始!")
    
    # 复现轨迹的主循环（多个点的情况）
    start_time = time.time()
    last_update_time = start_time
    current_trajectory_time = 0.0
    
    try:
        # 继续循环直到轨迹结束或用户退出
        while viewer.is_running() and current_trajectory_time <= total_duration:
            # 计算当前的真实时间
            current_real_time = time.time()
            
            # 计算自上次更新以来经过的时间
            elapsed_since_last = current_real_time - last_update_time
            last_update_time = current_real_time
            
            # 根据速度因子更新轨迹时间
            current_trajectory_time += elapsed_since_last * speed_factor
            
            # 找到当前轨迹时间对应的位置
            # 使用二分查找找到最接近的时间点
            left = 0
            right = len(body_trajectory) - 1
            closest_idx = 0
            
            while left <= right:
                mid = (left + right) // 2
                if body_trajectory[mid][0] < current_trajectory_time:
                    left = mid + 1
                    closest_idx = mid
                else:
                    right = mid - 1
            
            # 如果找到的点不是最后一个点，进行插值
            if closest_idx < len(body_trajectory) - 1:
                next_idx = closest_idx + 1
                
                # 获取前后两个点
                prev_point = body_trajectory[closest_idx]
                next_point = body_trajectory[next_idx]
                
                # 计算插值因子
                prev_time = prev_point[0]
                next_time = next_point[0]
                if next_time > prev_time:  # 避免除以零
                    alpha = (current_trajectory_time - prev_time) / (next_time - prev_time)
                    alpha = max(0, min(1, alpha))  # 确保alpha在[0,1]范围内
                else:
                    alpha = 0
                
                # 线性插值计算当前位置
                prev_pos = prev_point[1:4]
                next_pos = next_point[1:4]
                target_pos = prev_pos + alpha * (next_pos - prev_pos)
            else:
                # 如果是最后一个点，直接使用它的位置
                target_pos = body_trajectory[closest_idx][1:4]
            
            # 使用位置控制器移动身体部位
            if body_name == "base_link":
                # 如果是机器人主体，设置自由关节的位置
                data.qpos[0:3] = target_pos
            else:
                # 对于其他部位，使用PD控制器
                kp = 500.0  # 位置增益
                kd = 50.0   # 阻尼增益
                
                # 计算当前位置和目标位置之间的差异
                current_pos = data.xpos[body_id]
                pos_error = target_pos - current_pos
                
                # 应用力到身体部位
                force = kp * pos_error - kd * data.cvel[body_id, 0:3]
                data.xfrc_applied[body_id, 0:3] = force
            
            # 打印进度信息
            progress = (current_trajectory_time / total_duration) * 100
            print(f"进度: {progress:.1f}%, 时间: {current_trajectory_time:.2f}/{total_duration:.2f}秒, 目标位置: {target_pos}", end="\r")
            
            # 更新仿真
            mujoco.mj_step(model, data)
            
            # 渲染场景
            viewer.sync()
            
            # 控制帧率，但不影响播放速度
            time.sleep(0.01)
            
        # 轨迹播放完成
        print("\n轨迹复现完成！")
        
        # 等待用户按ESC退出
        while viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(0.01)
                
    except KeyboardInterrupt:
        print("\n用户中断了轨迹复现")
    except Exception as e:
        print(f"\n复现轨迹时出错: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\n程序结束") 