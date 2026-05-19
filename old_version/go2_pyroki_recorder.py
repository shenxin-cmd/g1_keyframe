"""Go2 Viser 动作拖拽录制器

使用Viser库实现的Go2机器人动作拖拽录制器，允许通过交互式控制点拖拽机器人关节，并保存关键帧。
提供类似3D建模软件的Gizmo控制器，支持精确的位置和旋转控制。
"""

import mujoco
import numpy as np
import time
import os
import keyboard  # 需要安装：pip install keyboard
import viser  # 需要安装：pip install viser
import trimesh  # 需要安装：pip install trimesh
from typing import Dict, List, Tuple, Any

# 加载模型和数据
model_path = "go2/scene_zero_gravity.xml"
model = mujoco.MjModel.from_xml_path(model_path)
data = mujoco.MjData(model)

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

# 创建Viser服务器用于可视化和拖拽控制
server = viser.ViserServer()
server.scene.add_grid("/ground", width=5, height=5)

# 打印访问可视化界面的提示信息
print("\n请在浏览器中访问以下地址查看可视化界面:")
print(f"http://localhost:8080")
print("如果无法访问，请检查端口是否被占用，或尝试其他端口。")
print("请保持此程序运行，不要关闭终端窗口。\n")

# 存储拖拽控制点和机器人几何体
drag_controls = {}
robot_geometries = {}
control_mode = "position"  # 默认控制模式：position 或 rotation

# 从MuJoCo模型创建Trimesh几何体
def create_mesh_from_mujoco_geom(geom_id):
    """从MuJoCo几何体创建Trimesh网格。"""
    geom_type = model.geom_type[geom_id]
    geom_size = model.geom_size[geom_id].copy()
    
    if geom_type == mujoco.mjtGeom.mjGEOM_SPHERE:
        return trimesh.creation.icosphere(radius=geom_size[0])
    elif geom_type == mujoco.mjtGeom.mjGEOM_CAPSULE:
        return trimesh.creation.capsule(radius=geom_size[0], height=geom_size[1]*2)
    elif geom_type == mujoco.mjtGeom.mjGEOM_CYLINDER:
        return trimesh.creation.cylinder(radius=geom_size[0], height=geom_size[1]*2)
    elif geom_type == mujoco.mjtGeom.mjGEOM_BOX:
        return trimesh.creation.box(extents=geom_size*2)
    else:
        # 对于其他几何体类型，使用简单的球体
        return trimesh.creation.icosphere(radius=0.05)

# 设置机器人可视化
def setup_robot_visualization():
    """设置机器人的可视化。"""
    global robot_geometries
    
    # 前向动力学计算以获取当前姿势
    mujoco.mj_forward(model, data)
    
    # 为每个几何体创建可视化
    for i in range(model.ngeom):
        geom_name = model.geom(i).name
        body_id = model.geom_bodyid[i]
        
        # 创建几何体的网格
        mesh = create_mesh_from_mujoco_geom(i)
        
        # 获取几何体的位置
        pos = data.geom_xpos[i].copy()
        
        # 获取几何体的旋转矩阵并转换为四元数
        # 在您的MuJoCo版本中，使用geom_xmat而不是geom_xquat
        rot_mat = data.geom_xmat[i].reshape(3, 3).copy()
        
        # 从旋转矩阵计算四元数
        # 使用简化的方法从旋转矩阵计算四元数
        trace = rot_mat[0, 0] + rot_mat[1, 1] + rot_mat[2, 2]
        
        if trace > 0:
            s = 0.5 / np.sqrt(trace + 1.0)
            w = 0.25 / s
            x = (rot_mat[2, 1] - rot_mat[1, 2]) * s
            y = (rot_mat[0, 2] - rot_mat[2, 0]) * s
            z = (rot_mat[1, 0] - rot_mat[0, 1]) * s
        elif rot_mat[0, 0] > rot_mat[1, 1] and rot_mat[0, 0] > rot_mat[2, 2]:
            s = 2.0 * np.sqrt(1.0 + rot_mat[0, 0] - rot_mat[1, 1] - rot_mat[2, 2])
            w = (rot_mat[2, 1] - rot_mat[1, 2]) / s
            x = 0.25 * s
            y = (rot_mat[0, 1] + rot_mat[1, 0]) / s
            z = (rot_mat[0, 2] + rot_mat[2, 0]) / s
        elif rot_mat[1, 1] > rot_mat[2, 2]:
            s = 2.0 * np.sqrt(1.0 + rot_mat[1, 1] - rot_mat[0, 0] - rot_mat[2, 2])
            w = (rot_mat[0, 2] - rot_mat[2, 0]) / s
            x = (rot_mat[0, 1] + rot_mat[1, 0]) / s
            y = 0.25 * s
            z = (rot_mat[1, 2] + rot_mat[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + rot_mat[2, 2] - rot_mat[0, 0] - rot_mat[1, 1])
            w = (rot_mat[1, 0] - rot_mat[0, 1]) / s
            x = (rot_mat[0, 2] + rot_mat[2, 0]) / s
            y = (rot_mat[1, 2] + rot_mat[2, 1]) / s
            z = 0.25 * s
        
        # 四元数格式为[w, x, y, z]
        quat = np.array([w, x, y, z])
        
        # MuJoCo四元数格式是[w,x,y,z]，而viser使用[x,y,z,w]
        wxyz = (quat[1], quat[2], quat[3], quat[0])
        
        # 尝试使用另一种方式添加颜色
        try:
            geom_node = server.scene.add_mesh_trimesh(
                f"/robot/{geom_name}",
                mesh,
                position=tuple(pos),
                wxyz=wxyz
            )
            # 尝试单独设置颜色
            try:
                geom_node.color = (0.7, 0.7, 0.7, 0.9)
            except:
                pass  # 如果不支持，忽略错误
        except Exception as e:
            print(f"无法创建几何体 {geom_name}: {e}")
            continue
        
        # 存储几何体信息
        robot_geometries[geom_name] = {
            "node": geom_node,
            "geom_id": i,
            "body_id": body_id
        }
    
    print(f"已创建 {len(robot_geometries)} 个机器人几何体可视化")

# 更新机器人可视化
def update_robot_visualization():
    """更新机器人可视化以匹配当前姿势。"""
    for geom_name, geom_info in robot_geometries.items():
        geom_id = geom_info["geom_id"]
        node = geom_info["node"]
        
        # 获取几何体的当前位置
        pos = data.geom_xpos[geom_id].copy()
        
        # 获取几何体的旋转矩阵并转换为四元数
        rot_mat = data.geom_xmat[geom_id].reshape(3, 3).copy()
        
        # 从旋转矩阵计算四元数
        trace = rot_mat[0, 0] + rot_mat[1, 1] + rot_mat[2, 2]
        
        if trace > 0:
            s = 0.5 / np.sqrt(trace + 1.0)
            w = 0.25 / s
            x = (rot_mat[2, 1] - rot_mat[1, 2]) * s
            y = (rot_mat[0, 2] - rot_mat[2, 0]) * s
            z = (rot_mat[1, 0] - rot_mat[0, 1]) * s
        elif rot_mat[0, 0] > rot_mat[1, 1] and rot_mat[0, 0] > rot_mat[2, 2]:
            s = 2.0 * np.sqrt(1.0 + rot_mat[0, 0] - rot_mat[1, 1] - rot_mat[2, 2])
            w = (rot_mat[2, 1] - rot_mat[1, 2]) / s
            x = 0.25 * s
            y = (rot_mat[0, 1] + rot_mat[1, 0]) / s
            z = (rot_mat[0, 2] + rot_mat[2, 0]) / s
        elif rot_mat[1, 1] > rot_mat[2, 2]:
            s = 2.0 * np.sqrt(1.0 + rot_mat[1, 1] - rot_mat[0, 0] - rot_mat[2, 2])
            w = (rot_mat[0, 2] - rot_mat[2, 0]) / s
            x = (rot_mat[0, 1] + rot_mat[1, 0]) / s
            y = 0.25 * s
            z = (rot_mat[1, 2] + rot_mat[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + rot_mat[2, 2] - rot_mat[0, 0] - rot_mat[1, 1])
            w = (rot_mat[1, 0] - rot_mat[0, 1]) / s
            x = (rot_mat[0, 2] + rot_mat[2, 0]) / s
            y = (rot_mat[1, 2] + rot_mat[2, 1]) / s
            z = 0.25 * s
        
        # 四元数格式为[w, x, y, z]
        quat = np.array([w, x, y, z])
        
        # MuJoCo四元数格式是[w,x,y,z]，而viser使用[x,y,z,w]
        wxyz = (quat[1], quat[2], quat[3], quat[0])
        
        # 更新节点位置
        node.position = tuple(pos)
        node.wxyz = wxyz

# 设置拖拽控制点
def setup_drag_controls():
    """设置拖拽控制点。"""
    global drag_controls
    
    # 为每个脚创建拖拽控制点
    foot_links = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
    
    for link_name in foot_links:
        try:
            # 查找对应的body ID
            body_id = None
            for i in range(model.nbody):
                if model.body(i).name == link_name:
                    body_id = i
                    break
            
            if body_id is None:
                print(f"找不到 {link_name} 的body ID")
                continue
            
            # 获取当前位置和方向
            pos = data.xpos[body_id].copy()
            quat = data.xquat[body_id].copy()  # [w, x, y, z]
            
            # MuJoCo四元数格式是[w,x,y,z]，而viser使用[x,y,z,w]
            wxyz = (quat[1], quat[2], quat[3], quat[0])
            
            # 添加拖拽控制点
            control = server.scene.add_transform_controls(
                f"/{link_name}_control",
                scale=0.15,  # 增大控制点尺寸，使其更容易操作
                position=tuple(pos),
                wxyz=wxyz
            )
            
            # 不再尝试添加文本标签，因为不支持
            
            # 存储控制点信息
            drag_controls[link_name] = {
                "control": control,
                "body_id": body_id,
                "original_pos": pos.copy(),
                "original_quat": quat.copy()
            }
            
            print(f"已为 {link_name} 创建拖拽控制点")
        except Exception as e:
            print(f"无法为 {link_name} 创建控制点: {e}")

# 应用拖拽控制点的位置到机器人
def apply_drag_controls():
    """尝试将拖拽控制点的位置应用到机器人上。"""
    # 创建临时数据
    temp_data = mujoco.MjData(model)
    
    # 复制当前数据
    for i in range(len(data.qpos)):
        temp_data.qpos[i] = data.qpos[i]
    
    # 前向动力学
    mujoco.mj_forward(model, temp_data)
    
    # 尝试使用简单的IK求解
    try:
        if control_mode == "position":
            # 应用位置控制
            for _ in range(5):  # 多次迭代以获得更好的结果
                for link_name, control_info in drag_controls.items():
                    body_id = control_info["body_id"]
                    control = control_info["control"]
                    
                    # 获取目标位置和旋转
                    target_pos = np.array(control.position)
                    target_wxyz = np.array(control.wxyz)
                    
                    # 将viser四元数[x,y,z,w]转换为MuJoCo四元数[w,x,y,z]
                    target_quat = np.array([target_wxyz[3], target_wxyz[0], target_wxyz[1], target_wxyz[2]])
                    
                    # 获取当前位置和旋转
                    current_pos = temp_data.xpos[body_id].copy()
                    current_quat = temp_data.xquat[body_id].copy()
                    
                    # 计算位置误差
                    pos_error = target_pos - current_pos
                    
                    # 计算位置雅可比矩阵
                    jacp = np.zeros((3, model.nv))
                    jacr = np.zeros((3, model.nv))
                    mujoco.mj_jacBody(model, temp_data, jacp, jacr, body_id)
                    
                    # 使用伪逆求解关节速度
                    jac_pinv = np.linalg.pinv(jacp)
                    qvel = jac_pinv @ pos_error * 0.5  # 缩小步长以增加稳定性
                    
                    # 更新关节位置
                    temp_data.qvel[:] = qvel
                    mujoco.mj_integratePos(model, temp_data.qpos, temp_data.qvel, 0.1)
                    
                    # 前向动力学
                    mujoco.mj_forward(model, temp_data)
        
            # 将求解结果复制回主数据
            for i in range(len(data.qpos)):
                data.qpos[i] = temp_data.qpos[i]
            
            mujoco.mj_forward(model, data)
            
            # 更新控制点状态显示
            update_control_status()
        
        elif control_mode == "rotation":
            # 应用旋转控制
            for _ in range(5):  # 多次迭代以获得更好的结果
                for link_name, control_info in drag_controls.items():
                    body_id = control_info["body_id"]
                    control = control_info["control"]
                    
                    # 获取目标旋转
                    target_quat = np.array(control.wxyz)
                    
                    # 获取当前旋转
                    current_quat = data.xquat[body_id].copy()
                    
                    # 计算旋转误差
                    quat_error = np.quaternion(target_quat[3], target_quat[0], target_quat[1], target_quat[2]) * np.quaternion(current_quat[3], -current_quat[0], -current_quat[1], -current_quat[2])
                    
                    # 计算旋转雅可比矩阵
                    jacp = np.zeros((3, model.nv))
                    jacr = np.zeros((3, model.nv))
                    mujoco.mj_jacBody(model, data, jacp, jacr, body_id)
                    
                    # 使用伪逆求解关节速度
                    jac_pinv = np.linalg.pinv(jacp)
                    qvel = jac_pinv @ quat_error.imag * 0.5  # 缩小步长以增加稳定性
                    
                    # 更新关节位置
                    temp_data.qvel[:] = qvel
                    mujoco.mj_integratePos(model, temp_data.qpos, temp_data.qvel, 0.1)
                    
                    # 前向动力学
                    mujoco.mj_forward(model, temp_data)
            
            # 将求解结果复制回主数据
            for i in range(len(data.qpos)):
                data.qpos[i] = temp_data.qpos[i]
            
            mujoco.mj_forward(model, data)
            
            # 更新控制点状态显示
            update_control_status()
        
    except Exception as e:
        print(f"应用拖拽控制失败: {e}")

# 更新控制点状态显示
def update_control_status():
    """更新控制点状态显示。"""
    for link_name, control_info in drag_controls.items():
        body_id = control_info["body_id"]
        original_pos = control_info["original_pos"]
        
        # 获取当前位置
        current_pos = data.xpos[body_id].copy()
        
        # 计算位移
        displacement = np.linalg.norm(current_pos - original_pos)
        
        # 更新标签显示位移信息
        try:
            label = server.scene.get_by_path(f"/{link_name}_label")
            label.text = f"{link_name}\n位移: {displacement:.3f}m"
            
            # 根据位移大小改变标签颜色
            if displacement > 0.1:
                label.color = (1.0, 0.5, 0.0)  # 橙色表示大位移
            elif displacement > 0.05:
                label.color = (1.0, 1.0, 0.0)  # 黄色表示中等位移
            else:
                label.color = (0.0, 1.0, 0.0)  # 绿色表示小位移
        except:
            pass  # 忽略错误

# 添加UI控件
keyframe_counter = server.gui.add_text("已保存关键帧", initial_value="0", disabled=True)
save_button = server.gui.add_button("保存关键帧")
position_mode_button = server.gui.add_button("位置控制模式")
rotation_mode_button = server.gui.add_button("旋转控制模式")
reset_button = server.gui.add_button("重置位置")
exit_button = server.gui.add_button("退出并保存")
joint_angles_text = server.gui.add_text("关节角度", initial_value="", disabled=True)

# 更新关节角度显示
def update_joint_angles_display():
    """更新关节角度显示。"""
    try:
        text = "关节角度:\n"
        for name, joint_id in joint_ids.items():
            qpos_addr = joint_qpos_map[name]
            joint_type = joint_types[joint_id]
            
            if joint_type == mujoco.mjtJoint.mjJNT_FREE or joint_type == mujoco.mjtJoint.mjJNT_BALL:
                text += f"{name}: [复杂关节]\n"
            else:
                # 确保angle是标量 - 使用索引0确保获取单个值
                try:
                    angle_value = data.qpos[qpos_addr]
                    # 如果是数组，取第一个元素
                    if hasattr(angle_value, "__len__") and len(angle_value) > 0:
                        angle = float(angle_value[0])
                    else:
                        angle = float(angle_value)
                    text += f"{name}: {angle:.2f} rad ({np.degrees(angle):.1f}°)\n"
                except Exception as e:
                    text += f"{name}: [错误: {e}]\n"
        
        # 尝试更新文本
        try:
            if joint_angles_text is not None:
                joint_angles_text.value = text
        except Exception as e:
            print(f"无法更新关节角度显示: {e}")
    except Exception as e:
        print(f"更新关节角度显示时出错: {e}")

# 切换控制模式
def toggle_control_mode(mode):
    """切换控制模式。"""
    global control_mode
    control_mode = mode
    
    # 由于可能不支持直接设置show_translation和show_rotation，
    # 我们只记录当前模式，并在apply_drag_controls中使用
    if mode == "position":
        print(f"已切换到位置控制模式")
    elif mode == "rotation":
        print(f"已切换到旋转控制模式")

# 重置控制点位置
def reset_control_positions():
    """重置控制点位置到原始状态。"""
    for link_name, control_info in drag_controls.items():
        body_id = control_info["body_id"]
        control = control_info["control"]
        original_pos = control_info["original_pos"]
        original_quat = control_info["original_quat"]
        
        # 重置控制点位置
        control.position = tuple(original_pos)
        
        # MuJoCo四元数格式是[w,x,y,z]，而viser使用[x,y,z,w]
        wxyz = (original_quat[1], original_quat[2], original_quat[3], original_quat[0])
        control.wxyz = wxyz
    
    print("已重置所有控制点位置")

# 保存当前关键帧
def save_keyframe():
    """保存当前关键帧。"""
    global keyframes
    
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
    
    # 更新关键帧计数器
    keyframe_counter.value = str(len(keyframes))
    
    # 打印保存的关键帧信息
    print(f"已保存关键帧 #{len(keyframes)}")
    for name, value in keyframe.items():
        if name != 'timestamp':
            if isinstance(value, (list, np.ndarray)):
                print(f"  - {name}: {value}")
            else:
                print(f"  - {name}: 角度 {value:.4f} 弧度 ({np.degrees(value):.2f}°)")
    
    # 在界面上显示保存成功的消息
    status_text = server.gui.add_text("状态", initial_value="已保存关键帧！", disabled=True)
    # 2秒后清除消息
    def clear_status():
        time.sleep(2)
        try:
            status_text.value = ""  # 尝试使用value属性
        except:
            try:
                status_text.text = ""  # 如果失败，尝试使用text属性
            except:
                pass  # 如果都失败，忽略错误
    
    import threading
    threading.Thread(target=clear_status, daemon=True).start()

# 保存所有关键帧到文件
def save_all_keyframes():
    """保存所有关键帧到文件。"""
    if keyframes:
        timestamp = int(time.time())
        filename = f"{keyframe_dir}/joint_keyframes_{timestamp}.npz"
        np.savez(filename, keyframes=keyframes)
        print(f"\n已保存 {len(keyframes)} 个关键帧到文件: {filename}")
        return filename
    else:
        print("\n未记录任何关键帧。")
        return None

# 设置机器人可视化和拖拽控制点
setup_robot_visualization()
setup_drag_controls()

# 默认使用位置控制模式
toggle_control_mode("position")

print("开始记录关键帧。")
print("按 S 键保存当前姿势作为关键帧")
print("点击'退出并保存'按钮退出并保存所有关键帧")
print("提示：使用拖拽控制点调整机器人姿势")
print("      - 红色轴控制X方向")
print("      - 绿色轴控制Y方向")
print("      - 蓝色轴控制Z方向")
print("\n等待用户连接到可视化界面...(请在浏览器中打开http://localhost:8080)")

# 在主循环中检查按钮状态
def check_button_clicks():
    """检查按钮点击状态"""
    global running
    
    # 尝试不同的方法检查按钮点击
    try:
        # 方法1: 使用on_click属性
        if hasattr(save_button, "on_click") and save_button.on_click:
            save_keyframe()
            save_button.on_click = False
            
        if hasattr(position_mode_button, "on_click") and position_mode_button.on_click:
            toggle_control_mode("position")
            position_mode_button.on_click = False
            
        if hasattr(rotation_mode_button, "on_click") and rotation_mode_button.on_click:
            toggle_control_mode("rotation")
            rotation_mode_button.on_click = False
            
        if hasattr(reset_button, "on_click") and reset_button.on_click:
            reset_control_positions()
            reset_button.on_click = False
            
        if hasattr(exit_button, "on_click") and exit_button.on_click:
            running = False
            exit_button.on_click = False
    except:
        pass
    
    try:
        # 方法2: 使用value属性
        if hasattr(save_button, "value") and save_button.value:
            save_keyframe()
            save_button.value = False
            
        if hasattr(position_mode_button, "value") and position_mode_button.value:
            toggle_control_mode("position")
            position_mode_button.value = False
            
        if hasattr(rotation_mode_button, "value") and rotation_mode_button.value:
            toggle_control_mode("rotation")
            rotation_mode_button.value = False
            
        if hasattr(reset_button, "value") and reset_button.value:
            reset_control_positions()
            reset_button.value = False
            
        if hasattr(exit_button, "value") and exit_button.value:
            running = False
            exit_button.value = False
    except:
        pass

# 主循环
try:
    running = True
    print("程序已启动，请在浏览器中打开可视化界面")
    while running:
        # 尝试应用拖拽控制
        apply_drag_controls()
        
        # 更新机器人可视化
        update_robot_visualization()
        
        # 更新关节角度显示
        update_joint_angles_display()
        
        # 检查按钮点击
        check_button_clicks()
        
        # 如果检测到S键被按下，保存当前关键帧
        if should_save_keyframe:
            save_keyframe()
            should_save_keyframe = False
        
        # 控制帧率
        time.sleep(0.01)

except KeyboardInterrupt:
    print("\n检测到Ctrl+C，正在退出...")

finally:
    # 保存所有关键帧到文件
    filename = save_all_keyframes()
    if filename:
        print(f"程序已退出。关键帧已保存到: {filename}")
    else:
        print("程序已退出。未保存任何关键帧。")
    
    # 添加一个提示，告诉用户可视化服务器将继续运行一段时间
    print("\n可视化服务器将继续运行30秒，您可以继续查看可视化界面。")
    print("30秒后程序将自动退出，或者您可以按Ctrl+C立即退出。")
    try:
        time.sleep(30)  # 等待30秒
    except KeyboardInterrupt:
        pass
    print("程序已完全退出。")