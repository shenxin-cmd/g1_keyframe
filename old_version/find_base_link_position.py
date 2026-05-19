import mujoco
import numpy as np

def find_base_link_position(model_path):
    """
    加载MuJoCo模型并查找base_link的中心坐标
    
    Args:
        model_path: 模型XML文件的路径
    
    Returns:
        base_link的中心坐标
    """
    # 加载模型
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)
    
    # 初始化模拟
    mujoco.mj_forward(model, data)
    
    # 查找base_link的ID
    base_link_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
    
    if base_link_id == -1:
        print("错误：找不到名为'base_link'的刚体")
        return None
    
    # 获取base_link的位置
    base_link_pos = data.xpos[base_link_id].copy()
    
    return base_link_pos

if __name__ == "__main__":
    # 模型文件路径
    model_path = "go2/scene_zero_gravity.xml"
    
    # 查找base_link位置
    position = find_base_link_position(model_path)
    
    if position is not None:
        print(f"base_link的中心坐标为: {position}")
        print(f"X: {position[0]:.6f}")
        print(f"Y: {position[1]:.6f}")
        print(f"Z: {position[2]:.6f}") 