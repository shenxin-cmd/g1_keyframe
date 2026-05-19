import numpy as np
import json
import os
import time
import mujoco

class KeyframeManager:
    def __init__(self, model, data, joint_names=None, output_dir="keyframes"):
        self.model = model
        self.data = data
        self.joint_names = joint_names if joint_names else []
        self.output_dir = output_dir
        self.keyframes = []
        
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
    
    def save_keyframe(self):
        """保存当前姿势为关键帧"""
        # 获取当前时间作为关键帧ID
        keyframe_id = int(time.time())
        
        # 获取基座全局位置和旋转
        base_pos = self.data.qpos[:3].copy()  # 基座全局位置
        base_quat = self.data.qpos[3:7].copy()  # 基座全局旋转四元数
        
        # 获取关节角度
        joint_angles = {}
        for name in self.joint_names:
            if name:
                for i in range(self.model.njnt):
                    if self.model.joint(i).name == name:
                        qpos_addr = self.model.joint(i).qposadr
                        if isinstance(qpos_addr, np.ndarray):
                            qpos_addr = int(qpos_addr[0])
                        else:
                            qpos_addr = int(qpos_addr)
                        
                        # 获取关节角度
                        joint_angles[name] = float(self.data.qpos[qpos_addr])
                        break
        
        # 获取所有链接的全局位置和旋转
        link_poses = {}
        for i in range(1, self.model.nbody):  # 跳过世界体
            body_name = self.model.body(i).name
            
            # 获取链接的全局位置
            pos = self.data.xpos[i].copy()
            
            # 获取链接的全局旋转四元数
            # MuJoCo中旋转矩阵存储在xmat中，需要转换为四元数
            rot_matrix = self.data.xmat[i].reshape(3, 3)
            quat = self._rot_matrix_to_quat(rot_matrix)
            
            link_poses[body_name] = {
                "position": pos.tolist(),
                "quaternion": quat.tolist()
            }
        
        # 创建关键帧数据
        keyframe = {
            "id": keyframe_id,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "base_position": base_pos.tolist(),
            "base_quaternion": base_quat.tolist(),
            "joint_angles": joint_angles,
            "link_poses": link_poses,
            "qpos": self.data.qpos.copy().tolist(),
            "ctrl": self.data.ctrl.copy().tolist() if self.data.ctrl.size > 0 else []
        }
        
        # 添加到关键帧列表
        self.keyframes.append(keyframe)
        
        print(f"已保存关键帧 #{len(self.keyframes)}, ID: {keyframe_id}")
        return keyframe_id
    
    def _rot_matrix_to_quat(self, rot_matrix):
        """将旋转矩阵转换为四元数"""
        # 确保旋转矩阵是正交的
        u, s, vh = np.linalg.svd(rot_matrix)
        rot_matrix = u @ vh
        
        trace = rot_matrix[0, 0] + rot_matrix[1, 1] + rot_matrix[2, 2]
        
        if trace > 0:
            s = 0.5 / np.sqrt(trace + 1.0)
            w = 0.25 / s
            x = (rot_matrix[2, 1] - rot_matrix[1, 2]) * s
            y = (rot_matrix[0, 2] - rot_matrix[2, 0]) * s
            z = (rot_matrix[1, 0] - rot_matrix[0, 1]) * s
        elif rot_matrix[0, 0] > rot_matrix[1, 1] and rot_matrix[0, 0] > rot_matrix[2, 2]:
            s = 2.0 * np.sqrt(1.0 + rot_matrix[0, 0] - rot_matrix[1, 1] - rot_matrix[2, 2])
            w = (rot_matrix[2, 1] - rot_matrix[1, 2]) / s
            x = 0.25 * s
            y = (rot_matrix[0, 1] + rot_matrix[1, 0]) / s
            z = (rot_matrix[0, 2] + rot_matrix[2, 0]) / s
        elif rot_matrix[1, 1] > rot_matrix[2, 2]:
            s = 2.0 * np.sqrt(1.0 + rot_matrix[1, 1] - rot_matrix[0, 0] - rot_matrix[2, 2])
            w = (rot_matrix[0, 2] - rot_matrix[2, 0]) / s
            x = (rot_matrix[0, 1] + rot_matrix[1, 0]) / s
            y = 0.25 * s
            z = (rot_matrix[1, 2] + rot_matrix[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + rot_matrix[2, 2] - rot_matrix[0, 0] - rot_matrix[1, 1])
            w = (rot_matrix[1, 0] - rot_matrix[0, 1]) / s
            x = (rot_matrix[0, 2] + rot_matrix[2, 0]) / s
            y = (rot_matrix[1, 2] + rot_matrix[2, 1]) / s
            z = 0.25 * s
        
        # 返回四元数 [x, y, z, w]
        return np.array([x, y, z, w])
    
    def get_keyframe_count(self):
        """获取关键帧数量"""
        return len(self.keyframes)
    
    def save_keyframes_to_file(self, filename=None):
        """将所有关键帧保存到文件"""
        if not self.keyframes:
            print("没有关键帧可保存")
            return False
        
        if filename is None:
            # 使用当前时间作为文件名
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"keyframes_{timestamp}"
        
        # 保存为JSON格式（完整数据，但文件较大）
        json_filepath = os.path.join(self.output_dir, f"{filename}.json")
        with open(json_filepath, 'w', encoding='utf-8') as f:
            json.dump(self.keyframes, f, indent=2, ensure_ascii=False)
        
        # 同时保存为NPZ格式（更高效，但只包含关键数据）
        npz_filepath = os.path.join(self.output_dir, f"{filename}.npz")
        
        # 提取qpos和ctrl数据
        qpos_data = []
        ctrl_data = []
        timestamps = []
        
        for keyframe in self.keyframes:
            qpos_data.append(keyframe["qpos"])
            ctrl_data.append(keyframe["ctrl"] if "ctrl" in keyframe and keyframe["ctrl"] else [])
            timestamps.append(keyframe["timestamp"])
        
        # 保存为NPZ文件
        np.savez_compressed(
            npz_filepath,
            qpos=np.array(qpos_data, dtype=np.float32),
            ctrl=np.array(ctrl_data, dtype=np.float32) if any(ctrl_data) else np.array([]),
            timestamps=np.array(timestamps, dtype=str),
            joint_names=np.array(self.joint_names, dtype=str)
        )
        
        print(f"已将 {len(self.keyframes)} 个关键帧保存到:")
        print(f"  - JSON格式: {json_filepath}")
        print(f"  - NPZ格式: {npz_filepath}")
        return True
    
    def load_keyframes_from_file(self, filepath):
        """从文件加载关键帧"""
        try:
            # 根据文件扩展名选择加载方式
            if filepath.endswith('.json'):
                with open(filepath, 'r', encoding='utf-8') as f:
                    self.keyframes = json.load(f)
                print(f"已从JSON文件 {filepath} 加载 {len(self.keyframes)} 个关键帧")
            elif filepath.endswith('.npz'):
                data = np.load(filepath, allow_pickle=True)
                
                # 从NPZ文件重建关键帧数据
                qpos_data = data['qpos']
                ctrl_data = data['ctrl'] if 'ctrl' in data else []
                timestamps = data['timestamps']
                
                self.keyframes = []
                for i in range(len(qpos_data)):
                    keyframe = {
                        "id": int(time.time()) + i,
                        "timestamp": str(timestamps[i]),
                        "qpos": qpos_data[i].tolist()
                    }
                    
                    if len(ctrl_data) > i:
                        keyframe["ctrl"] = ctrl_data[i].tolist()
                    
                    self.keyframes.append(keyframe)
                
                print(f"已从NPZ文件 {filepath} 加载 {len(self.keyframes)} 个关键帧")
            else:
                print(f"不支持的文件格式: {filepath}")
                return False
            
            return True
        except Exception as e:
            print(f"加载关键帧失败: {e}")
            return False
    
    def apply_keyframe(self, keyframe_index):
        """应用指定索引的关键帧到模型"""
        if keyframe_index < 0 or keyframe_index >= len(self.keyframes):
            print(f"关键帧索引 {keyframe_index} 超出范围 [0, {len(self.keyframes)-1}]")
            return False
        
        keyframe = self.keyframes[keyframe_index]
        
        # 应用qpos
        if "qpos" in keyframe and len(keyframe["qpos"]) == self.data.qpos.shape[0]:
            self.data.qpos[:] = np.array(keyframe["qpos"])
        
        # 应用ctrl
        if "ctrl" in keyframe and len(keyframe["ctrl"]) == self.data.ctrl.shape[0]:
            self.data.ctrl[:] = np.array(keyframe["ctrl"])
        
        # 更新物理状态
        mujoco.mj_forward(self.model, self.data)
        
        print(f"已应用关键帧 #{keyframe_index}, ID: {keyframe['id']}")
        return True 