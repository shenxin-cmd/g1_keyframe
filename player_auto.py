import mujoco
import mujoco.viewer
import numpy as np
import time
import os
import glob
import argparse
import json

# 导入自定义模块
from utils.joint_utils import set_joint_damping

def linear_interpolate(start_value, end_value, t):
    """线性插值函数
    
    Args:
        start_value: 起始值
        end_value: 结束值
        t: 插值参数 (0.0 到 1.0)
        
    Returns:
        插值结果
    """
    if isinstance(start_value, (list, np.ndarray)) and isinstance(end_value, (list, np.ndarray)):
        # 对数组进行元素级插值
        return np.array(start_value) * (1-t) + np.array(end_value) * t
    else:
        # 对标量进行插值
        return start_value + t * (end_value - start_value)

def slerp_interpolate(start_quat, end_quat, t):
    """球面线性插值 (SLERP) 用于四元数
    
    Args:
        start_quat: 起始四元数 [x, y, z, w]
        end_quat: 结束四元数 [x, y, z, w]
        t: 插值参数 (0.0 到 1.0)
        
    Returns:
        插值后的四元数
    """
    # 确保输入是numpy数组
    start_quat = np.array(start_quat)
    end_quat = np.array(end_quat)
    
    # 计算四元数之间的点积
    dot = np.sum(start_quat * end_quat)
    
    # 如果点积为负，翻转一个四元数
    # 这样可以确保走最短路径
    if dot < 0.0:
        end_quat = -end_quat
        dot = -dot
    
    # 如果四元数非常接近，使用线性插值
    if dot > 0.9995:
        result = start_quat + t * (end_quat - start_quat)
        return result / np.linalg.norm(result)
    
    # 计算插值角度
    theta_0 = np.arccos(dot)
    theta = theta_0 * t
    
    # 计算插值四元数
    sin_theta = np.sin(theta)
    sin_theta_0 = np.sin(theta_0)
    
    s0 = np.cos(theta) - dot * sin_theta / sin_theta_0
    s1 = sin_theta / sin_theta_0
    
    return s0 * start_quat + s1 * end_quat

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='关键帧插值播放器')
    parser.add_argument('--model', default="go2/scene_zero_gravity_free_draggable.xml", help='模型XML文件路径')
    parser.add_argument('--fps', type=int, default=30, help='播放帧率')
    parser.add_argument('--duration', type=float, default=1, help='每对关键帧之间的过渡时间(秒)')
    parser.add_argument('--loop', action='store_true', default=True, help='是否循环播放')
    parser.add_argument('--speed', type=float, default=1, help='回放速度倍率，大于1加速，小于1减速')
    parser.add_argument('--damping', type=float, default=10.0, help='关节阻尼值')
    args = parser.parse_args()
    
    # 计算每一步的时间间隔
    step_time = 1.0 / args.fps
    
    # 根据速度倍率调整过渡时间和步进时间
    actual_duration = args.duration / args.speed
    actual_step_time = step_time / args.speed if args.speed > 0 else step_time
    
    # 加载模型和数据
    model_path = args.model
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)
    
    # 初始化数据
    mujoco.mj_forward(model, data)
    
    # 设置关节阻尼
    set_joint_damping(model, damping_value=args.damping, armature_value=1.0)
    
    # 查找所有关键帧文件（优先查找NPZ文件）
    keyframe_dir = "keyframes"
    npz_files = glob.glob(os.path.join(keyframe_dir, "keyframes_*.npz"))
    json_files = glob.glob(os.path.join(keyframe_dir, "keyframes_*.json"))
    
    # 如果有NPZ文件，优先使用
    if npz_files:
        keyframe_files = npz_files
        file_type = "NPZ"
    elif json_files:
        keyframe_files = json_files
        file_type = "JSON"
    else:
        print("未找到关键帧文件！请先运行关键帧记录程序。")
        exit()
    
    # 显示可用的关键帧文件
    print(f"可用的关键帧文件 ({file_type} 格式):")
    for i, file_path in enumerate(keyframe_files):
        file_name = os.path.basename(file_path)
        print(f"{i+1}. {file_name}")
    
    # 让用户选择要回放的关键帧文件
    selection = int(input("请选择要回放的关键帧文件编号: ")) - 1
    if selection < 0 or selection >= len(keyframe_files):
        print("无效的选择！")
        exit()
    
    # 加载选定的关键帧数据
    keyframe_path = keyframe_files[selection]
    
    # 根据文件类型加载数据
    if file_type == "NPZ":
        try:
            data_file = np.load(keyframe_path, allow_pickle=True)
            qpos_data = data_file['qpos']
            timestamps = data_file['timestamps'] if 'timestamps' in data_file else None
            print(f"已加载 {len(qpos_data)} 个关键帧 (NPZ格式)")
        except Exception as e:
            print(f"加载NPZ关键帧失败: {e}")
            exit()
    else:  # JSON
        try:
            with open(keyframe_path, 'r', encoding='utf-8') as f:
                keyframes_json = json.load(f)
            
            # 从JSON提取qpos数据
            qpos_data = []
            timestamps = []
            for keyframe in keyframes_json:
                if 'qpos' in keyframe:
                    qpos_data.append(keyframe['qpos'])
                    timestamps.append(keyframe.get('timestamp', ''))
            
            print(f"已加载 {len(qpos_data)} 个关键帧 (JSON格式)")
        except Exception as e:
            print(f"加载JSON关键帧失败: {e}")
            exit()
    
    # 检查是否成功加载了关键帧
    if qpos_data is None or len(qpos_data) == 0:
        print("未能加载有效的关键帧数据！")
        exit()
    
    if len(qpos_data) < 2:
        print("至少需要2个关键帧才能进行插值！")
        exit()
    
    # 检查qpos数据是否与模型匹配
    if len(qpos_data[0]) != data.qpos.shape[0]:
        print(f"警告: 关键帧qpos维度 ({len(qpos_data[0])}) 与模型qpos维度 ({data.qpos.shape[0]}) 不匹配！")
        print("这可能导致回放失败或异常。")
        proceed = input("是否继续? (y/n): ")
        if proceed.lower() != 'y':
            exit()
    
    # 显示回放设置
    print(f"\n插值播放设置:")
    print(f"  帧率: {args.fps} FPS")
    print(f"  过渡时间: {args.duration} 秒")
    print(f"  播放速度: {args.speed}x")
    print(f"  循环播放: {'开启' if args.loop else '关闭'}")
    print(f"  关节阻尼: {args.damping}")
    
    # 创建查看器
    with mujoco.viewer.launch_passive(model, data) as viewer:
        print("\n关键帧插值播放器已启动")
        print("按ESC键退出")
        
        # 等待一小段时间让模型稳定
        print("等待模型稳定...")
        time.sleep(0.5)
        
        print("开始插值播放关键帧...")
        
        # 主循环
        running = True
        while running and viewer.is_running():
            # 循环播放所有关键帧
            for i in range(len(qpos_data) - 1):
                start_qpos = np.array(qpos_data[i])
                end_qpos = np.array(qpos_data[i + 1])
                
                # 显示当前过渡信息
                print(f"\n插值过渡: 关键帧 {i+1} -> {i+2}")
                if timestamps is not None and len(timestamps) > i:
                    print(f"  从: {timestamps[i]}")
                    if i+1 < len(timestamps):
                        print(f"  到: {timestamps[i+1]}")
                
                # 计算需要多少步来完成这个过渡
                steps = max(1, int(actual_duration / step_time))
                
                # 对每一步进行插值
                for step in range(steps + 1):
                    # 计算插值参数 t (0.0 到 1.0)
                    t = step / steps if steps > 0 else 1.0
                    
                    # 对qpos进行插值
                    # 注意：这里我们对整个qpos数组进行插值，包括基座位置、旋转和关节角度
                    # 对于旋转四元数部分，应该使用SLERP，但为了简化，这里使用线性插值
                    # 在实际应用中，可以分别对位置和四元数部分进行不同的插值
                    interpolated_qpos = linear_interpolate(start_qpos, end_qpos, t)
                    
                    # 应用插值后的qpos
                    data.qpos[:] = interpolated_qpos
                    
                    # 前向动力学更新
                    mujoco.mj_forward(model, data)
                    
                    # 更新查看器
                    viewer.sync()
                    
                    # 等待一段时间
                    time.sleep(actual_step_time)
                    
                    # 检查是否应该退出
                    if not viewer.is_running():
                        running = False
                        break
                
                if not running:
                    break
            
            # 如果不循环播放，就退出
            if not args.loop:
                break
        
        print("\n播放完成")
        
        # 等待用户按ESC退出
        print("按ESC键退出...")
        while viewer.is_running():
            viewer.sync()
            time.sleep(0.01)

if __name__ == "__main__":
    main() 