import mujoco
import mujoco.viewer
import time

def display_scene(model_path, duration=None):
    """
    加载并显示MuJoCo场景
    
    Args:
        model_path: 模型XML文件的路径
        duration: 显示持续时间（秒），None表示一直显示直到窗口关闭
    """
    # 加载模型
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)
    
    # 设置初始状态
    data.qpos[0:3] = [0, 0, 0.45]  # 位置
    data.qpos[3:7] = [1, 0, 0, 0]  # 四元数
    
    # 设置腿部关节位置
    leg_joints = [8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
    leg_pos = [0, 0.9, -1.8, 0, 0.9, -1.8, 0, 0.9, -1.8, 0, 0.9, -1.8]
    for i, pos in zip(leg_joints, leg_pos):
        data.qpos[i] = pos
    
    # 初始化模拟
    mujoco.mj_forward(model, data)
    
    # 查找base_link的ID
    base_link_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
    if base_link_id != -1:
        print(f"base_link的ID: {base_link_id}")
        print(f"base_link的初始位置: {data.xpos[base_link_id]}")
    
    # 启动查看器
    print("正在启动MuJoCo查看器...")
    print("提示: 按ESC键退出，鼠标左键旋转视角，鼠标右键平移视角，滚轮缩放")
    
    with mujoco.viewer.launch_passive(model, data) as viewer:
        # 设置相机视角
        viewer.cam.distance = 3.0
        viewer.cam.azimuth = 45.0
        viewer.cam.elevation = -20.0
        
        # 模拟循环
        start_time = time.time()
        while viewer.is_running():
            # 步进模拟
            mujoco.mj_step(model, data)
            
            # 更新查看器
            viewer.sync()
            
            # 控制模拟速度
            time.sleep(0.01)
            
            # 如果设置了持续时间，检查是否应该退出
            if duration is not None and time.time() - start_time > duration:
                break

if __name__ == "__main__":
    # 模型文件路径
    model_path = "go2/scene_zero_gravity_free.xml"
    
    # 显示场景
    display_scene(model_path) 