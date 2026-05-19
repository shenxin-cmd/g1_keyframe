import mujoco
import mujoco.viewer
import numpy as np
import time
import os
import glob
import argparse

# 导入自定义模块
from utils.joint_utils import set_joint_damping

def main():
    parser = argparse.ArgumentParser(description='直接播放MuJoCo关键帧')
    parser.add_argument('--model', default="go2/scene_zero_gravity_free_draggable.xml", help='模型XML文件路径')
    parser.add_argument('--stay-time', type=float, default=0.01, help='每个关键帧的停留时间(秒)')
    parser.add_argument('--auto-play', action='store_true', default=False, help='自动播放所有关键帧，默认按回车键播放下一帧')
    parser.add_argument('--loop', action='store_true', default=True, help='循环播放关键帧')
    args = parser.parse_args()

    # 加载模型和数据
    model_path = args.model
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)

    # 初始化数据
    mujoco.mj_forward(model, data)

    # 设置关节阻尼
    set_joint_damping(model, damping_value=10.0, armature_value=1.0)

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
            timestamps = data_file['timestamps']
            print(f"已加载 {len(qpos_data)} 个关键帧 (NPZ格式)")
        except Exception as e:
            print(f"加载NPZ关键帧失败: {e}")
            exit()
    else:  # JSON
        import json
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
    
    # 检查qpos数据是否与模型匹配
    if len(qpos_data[0]) != data.qpos.shape[0]:
        print(f"警告: 关键帧qpos维度 ({len(qpos_data[0])}) 与模型qpos维度 ({data.qpos.shape[0]}) 不匹配！")
        print("这可能导致回放失败或异常。")
        proceed = input("是否继续? (y/n): ")
        if proceed.lower() != 'y':
            exit()
    
    # 设置每个关键帧的停留时间
    stay_time = args.stay_time
    print(f"\n每个关键帧停留时间: {stay_time}秒")
    
    # 是否自动播放
    auto_play = args.auto_play
    if auto_play:
        print("模式: 自动播放")
    else:
        print("模式: 手动播放 (按Enter键显示下一帧)")
    
    # 是否循环播放
    loop_play = args.loop
    if loop_play:
        print("循环播放: 开启")
    
    # 启动查看器
    with mujoco.viewer.launch_passive(model, data) as viewer:
        print("\n关键帧播放器已启动")
        print("按ESC键退出")
        
        # 等待一小段时间让模型稳定
        print("等待模型稳定...")
        time.sleep(0.5)
        
        print("开始回放关键帧...")
        
        # 回放循环
        current_frame = 0
        while viewer.is_running():
            # 显示当前关键帧信息
            print(f"\n显示关键帧 {current_frame+1}/{len(qpos_data)}")
            if timestamps is not None and len(timestamps) > current_frame:
                print(f"时间戳: {timestamps[current_frame]}")
            
            # 应用qpos数据
            try:
                data.qpos[:] = np.array(qpos_data[current_frame], dtype=np.float64)
                mujoco.mj_forward(model, data)
            except Exception as e:
                print(f"应用关键帧失败: {e}")
                print(f"错误详情: qpos_data类型={type(qpos_data)}, 形状={qpos_data.shape if hasattr(qpos_data, 'shape') else '未知'}")
                print(f"当前帧索引: {current_frame}")
                if current_frame < len(qpos_data):
                    print(f"当前帧数据类型: {type(qpos_data[current_frame])}")
                    if hasattr(qpos_data[current_frame], 'shape'):
                        print(f"当前帧数据形状: {qpos_data[current_frame].shape}")
                break
            
            # 显示当前关键帧一段时间
            start_time = time.time()
            while time.time() - start_time < stay_time and viewer.is_running():
                # 只更新视图，不执行物理模拟
                viewer.sync()
                time.sleep(0.01)
            
            # 移动到下一帧
            if auto_play:
                # 自动模式下直接前进到下一帧
                current_frame = (current_frame + 1) % len(qpos_data) if loop_play else min(current_frame + 1, len(qpos_data) - 1)
                
                # 如果到达最后一帧且不循环，则等待用户输入
                if current_frame == len(qpos_data) - 1 and not loop_play:
                    print("\n已到达最后一帧")
                    if input("按Enter键重新开始，或输入'q'退出: ").lower() == 'q':
                        break
                    current_frame = 0
            else:
                # 手动模式下等待用户输入
                if current_frame < len(qpos_data) - 1:
                    # 不是最后一帧，等待用户按Enter继续
                    if input("按Enter键显示下一帧，或输入'q'退出: ").lower() == 'q':
                        break
                    current_frame += 1
                else:
                    # 最后一帧，询问是否重新开始
                    print("\n已到达最后一帧")
                    if input("按Enter键重新开始，或输入'q'退出: ").lower() == 'q':
                        break
                    current_frame = 0
        
        print("\n关键帧播放结束")

if __name__ == "__main__":
    main()