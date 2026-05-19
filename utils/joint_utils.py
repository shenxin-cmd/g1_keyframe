import numpy as np
import mujoco

def get_joint_info(model):
    """获取模型中所有关节的信息"""
    joint_names = []
    joint_qpos_addrs = []  # 存储每个关节的qpos地址
    joint_types = []
    joint_dims = []  # 存储每个关节的维度
    joint_ranges = {}  # 存储每个关节的限位范围
    
    for i in range(model.njnt):
        joint_name = model.joint(i).name
        
        # 跳过 sphere_joint
        if joint_name == "sphere_joint":
            continue
            
        joint_type = model.joint(i).type
        joint_qposadr = model.joint(i).qposadr  # 获取关节的qpos地址
        
        # 确定关节的维度
        if joint_type == mujoco.mjtJoint.mjJNT_FREE:
            joint_dim = 7  # 自由关节：3个位置 + 4个四元数
        elif joint_type == mujoco.mjtJoint.mjJNT_BALL:
            joint_dim = 4  # 球关节：4个四元数
        else:
            joint_dim = 1  # 铰链关节或滑动关节：1个值
        
        joint_names.append(joint_name)
        joint_qpos_addrs.append(joint_qposadr)
        joint_types.append(joint_type)
        joint_dims.append(joint_dim)
        
        # 获取关节限位范围
        if joint_type == mujoco.mjtJoint.mjJNT_HINGE or joint_type == mujoco.mjtJoint.mjJNT_SLIDE:
            lower_limit = model.jnt_range[i][0]
            upper_limit = model.jnt_range[i][1]
            if lower_limit != 0 or upper_limit != 0:  # 只有当限位不全为0时才记录
                joint_ranges[joint_name] = (lower_limit, upper_limit)
    
    # 创建关节ID映射
    joint_ids = {name: i for i, name in enumerate(joint_names)}
    
    # 创建关节qpos地址映射
    joint_qpos_map = {name: addr for name, addr in zip(joint_names, joint_qpos_addrs)}
    
    # 创建关节维度映射
    joint_dim_map = {name: dim for name, dim in zip(joint_names, joint_dims)}
    
    return {
        'joint_names': joint_names,
        'joint_ids': joint_ids,
        'joint_types': joint_types,
        'joint_qpos_map': joint_qpos_map,
        'joint_dim_map': joint_dim_map,
        'joint_ranges': joint_ranges
    }

def set_joint_damping(model, damping_value=10, armature_value=1.0):
    """设置所有关节的阻尼和惯性"""
    for i in range(model.njnt):
        # 获取关节类型
        joint_type = model.jnt_type[i]
        
        # 自由关节(类型为0)有6个DOF，需要特殊处理
        if joint_type == 0:  # mjtJoint.mjJNT_FREE
            # 自由关节的阻尼和惯性需要为每个DOF单独设置
            for dof in range(6):
                dof_adr = model.jnt_dofadr[i] + dof
                model.dof_damping[dof_adr] = damping_value * 2  # 自由关节使用更大的阻尼
                model.dof_armature[dof_adr] = armature_value * 2
        else:
            # 其他类型关节(铰链、滑动等)
            dof_adr = model.jnt_dofadr[i]
            model.dof_damping[dof_adr] = damping_value
            model.dof_armature[dof_adr] = armature_value

def set_specific_joint_damping(model, joint_name, damping_value, armature_value=None):
    """设置特定关节的阻尼和惯性"""
    # 查找关节ID
    for i in range(model.njnt):
        if model.joint(i).name == joint_name:
            dof_adr = model.jnt_dofadr[i]
            model.dof_damping[dof_adr] = damping_value
            if armature_value is not None:
                model.dof_armature[dof_adr] = armature_value
            return True
    return False

def find_related_joints(model, body_id):
    """根据选中的物体ID找到相关的关节"""
    related_joints = []
    
    # 检查每个关节，看它是否连接到选中的物体
    for i in range(model.njnt):
        joint = model.joint(i)
        # 获取关节连接的物体ID
        joint_body_id = joint.bodyid
        
        # 如果joint_body_id是数组，则使用相等性检查
        if isinstance(joint_body_id, np.ndarray):
            if np.any(joint_body_id == body_id):
                related_joints.append(joint.name)
        else:
            # 如果是单个值，直接比较
            if joint_body_id == body_id:
                related_joints.append(joint.name)
    
    return related_joints

def toggle_sphere_constraints(model, data, enable=True):
    """启用或禁用与sphere相关的约束
    
    参数:
        model: MuJoCo模型
        data: MuJoCo数据
        enable: 是否启用约束，True为启用，False为禁用
    
    返回:
        是否成功修改了约束
    """
    # 查找sphere的body_id
    sphere_id = -1
    for i in range(model.nbody):
        if model.body(i).name == "sphere":
            sphere_id = i
            break
    
    if sphere_id == -1:
        print("未找到sphere物体")
        return False
    
    # 标记是否修改了约束
    modified = False
    
    # 如果要启用约束，先保存sphere当前位置
    if enable:
        # 获取sphere当前位置
        sphere_pos = data.xpos[sphere_id].copy()
        sphere_quat = data.xquat[sphere_id].copy()
        print(f"启用约束时sphere位置: {sphere_pos}")
        
        # 先禁用所有sphere的weld约束
        for i in range(model.neq):
            try:
                obj1_id = model.eq_obj1id[i]
                obj2_id = model.eq_obj2id[i]
                eq_type = model.eq_type[i]
                
                obj1_name = model.body(obj1_id).name if obj1_id >= 0 else "world"
                obj2_name = model.body(obj2_id).name if obj2_id >= 0 else "world"
                
                if eq_type == 1 and ((obj1_id == sphere_id and obj2_name == "world") or 
                                     (obj2_id == sphere_id and obj1_name == "world")):
                    data.eq_active[i] = 0
                    print(f"临时禁用 sphere 的 weld 约束 (ID: {i})")
            except Exception as e:
                print(f"处理约束 {i} 时出错: {e}")
        
        # 手动将sphere移动到保存的位置
        for i in range(model.njnt):
            if model.joint(i).name == "sphere_joint":
                qpos_addr = int(model.joint(i).qposadr)
                data.qpos[qpos_addr:qpos_addr+3] = sphere_pos
                data.qpos[qpos_addr+3:qpos_addr+7] = sphere_quat
                print(f"更新sphere位置为: {sphere_pos}")
                break
        
        # 更新物理状态
        mujoco.mj_forward(model, data)
        
        # 启用weld约束
        for i in range(model.neq):
            try:
                obj1_id = model.eq_obj1id[i]
                obj2_id = model.eq_obj2id[i]
                eq_type = model.eq_type[i]
                
                obj1_name = model.body(obj1_id).name if obj1_id >= 0 else "world"
                obj2_name = model.body(obj2_id).name if obj2_id >= 0 else "world"
                
                if eq_type == 1 and ((obj1_id == sphere_id and obj2_name == "world") or 
                                     (obj2_id == sphere_id and obj1_name == "world")):
                    data.eq_active[i] = 1
                    modified = True
                    print(f"启用 sphere 的 weld 约束 (ID: {i})")
            except Exception as e:
                print(f"处理约束 {i} 时出错: {e}")
    else:
        # 禁用约束
        for i in range(model.neq):
            try:
                obj1_id = model.eq_obj1id[i]
                obj2_id = model.eq_obj2id[i]
                eq_type = model.eq_type[i]
                
                # 打印约束类型和对象
                obj1_name = model.body(obj1_id).name if obj1_id >= 0 else "world"
                obj2_name = model.body(obj2_id).name if obj2_id >= 0 else "world"
                print(f"检查约束 {i}: 类型={eq_type}, {obj1_name} 连接到 {obj2_name}")
                
                # 只处理weld约束（类型为1）
                if eq_type == 1:  # mjEQ_WELD
                    # 检查是否是sphere的weld约束
                    if (obj1_id == sphere_id or obj2_id == sphere_id):
                        # 设置约束的激活状态
                        data.eq_active[i] = 0
                        modified = True
                        print(f"禁用 sphere 的 weld 约束 (ID: {i})")
            except Exception as e:
                print(f"处理约束 {i} 时出错: {e}")
    
    return modified or ["sphere_constraint_disabled"]

def set_selected_body_joints_damping(model, body_id, damping_value=0.0, joint_damping_backup=None):
    """设置选中物体相关关节的阻尼"""
    if joint_damping_backup is None:
        joint_damping_backup = {}
    
    # 获取物体名称
    body_name = model.body(body_id).name
    
    # 特殊处理所有sphere物体（检查名称是否包含"sphere"）
    if "sphere" in body_name.lower():
        # 对于sphere，我们不需要找关节，直接返回空列表
        return []
        
    related_joints = find_related_joints(model, body_id)
    
    # 如果找到相关关节，设置它们的阻尼
    for joint_name in related_joints:
        # 备份当前阻尼值（如果尚未备份）
        if joint_name not in joint_damping_backup:
            for i in range(model.njnt):
                if model.joint(i).name == joint_name:
                    dof_adr = model.jnt_dofadr[i]
                    joint_damping_backup[joint_name] = model.dof_damping[dof_adr]
                    break
        
        # 设置阻尼为指定值
        set_specific_joint_damping(model, joint_name, damping_value)
    
    return related_joints

def restore_all_joint_damping(model, joint_damping_backup):
    """恢复所有关节的原始阻尼值"""
    for joint_name, damping_value in joint_damping_backup.items():
        if "sphere_constraint_disabled" not in joint_name:
            set_specific_joint_damping(model, joint_name, damping_value)

def find_sphere_joint(model):
    """查找 sphere 的关节"""
    for i in range(model.njnt):
        joint = model.joint(i)
        joint_name = joint.name
        if joint_name == "sphere_joint":
            return i
    return -1

def update_sphere_position(model, data, new_position):
    """更新 sphere 的位置"""
    # 查找 sphere 关节
    joint_id = find_sphere_joint(model)
    if joint_id >= 0:
        # 获取关节的 qpos 地址
        qpos_addr = model.joint(joint_id).qposadr
        # 更新位置 (前3个值是位置)
        data.qpos[qpos_addr:qpos_addr+3] = new_position
        # 重新计算前向动力学
        mujoco.mj_forward(model, data)
        return True
    return False

def move_sphere_directly(model, data, body_id, new_position):
    """直接移动sphere到新位置
    
    参数:
        model: MuJoCo模型
        data: MuJoCo数据
        body_id: sphere的body ID
        new_position: 新位置 [x, y, z]
    
    返回:
        是否成功移动
    """
    # 确认是sphere
    body_name = model.body(body_id).name
    if body_name != "sphere":
        print(f"错误：尝试移动非sphere物体: {body_name}")
        return False
    
    # 查找sphere的自由关节
    for i in range(model.njnt):
        if model.joint(i).name == "sphere_joint":
            # 获取关节的qpos地址
            qpos_addr = model.joint(i).qposadr
            
            # 确保qpos_addr是整数
            if isinstance(qpos_addr, np.ndarray):
                qpos_addr = int(qpos_addr[0])
            else:
                qpos_addr = int(qpos_addr)
            
            # 保存当前四元数
            current_quat = data.qpos[qpos_addr+3:qpos_addr+7].copy()
            
            # 更新位置 (前3个值是位置)
            data.qpos[qpos_addr:qpos_addr+3] = new_position
            
            # 保持四元数不变 (后4个值是四元数)
            data.qpos[qpos_addr+3:qpos_addr+7] = current_quat
            
            # 更新物理
            mujoco.mj_forward(model, data)
            return True
    
    print("未找到sphere的自由关节")
    return False

def move_sphere_body_directly(model, data, body_id, new_position):
    """直接修改sphere物体的位置（适用于没有关节的sphere）
    
    参数:
        model: MuJoCo模型
        data: MuJoCo数据
        body_id: sphere的body ID
        new_position: 新位置 [x, y, z]
    
    返回:
        是否成功移动
    """
    # 获取物体名称
    body_name = model.body(body_id).name
    
    # 检查是否是sphere类型的物体（名称中包含"sphere"）
    if "sphere" not in body_name.lower():
        print(f"错误：尝试移动非sphere物体: {body_name}")
        return False
    
    try:
        # 直接修改body的位置
        # 注意：这种方法会在下一次mj_step时被重置，除非sphere没有关节
        model.body_pos[body_id] = new_position
        
        # 更新物理状态
        mujoco.mj_forward(model, data)
        print(f"成功移动 {body_name} 到位置: {new_position}")
        return True
    except Exception as e:
        print(f"移动 {body_name} 失败: {e}")
        return False

def find_sphere_by_name(model, sphere_name):
    """根据名称查找sphere的body ID
    
    参数:
        model: MuJoCo模型
        sphere_name: sphere的名称
    
    返回:
        sphere的body ID，如果未找到则返回-1
    """
    for i in range(model.nbody):
        if model.body(i).name == sphere_name:
            return i
    return -1

def get_all_spheres(model):
    """获取模型中所有sphere物体的ID和名称
    
    参数:
        model: MuJoCo模型
    
    返回:
        包含所有sphere物体ID和名称的字典 {body_id: body_name}
    """
    spheres = {}
    for i in range(model.nbody):
        body_name = model.body(i).name
        if "sphere" in body_name.lower():
            spheres[i] = body_name
    return spheres

def move_specific_sphere(model, data, sphere_name, new_position):
    """移动指定名称的sphere到新位置
    
    参数:
        model: MuJoCo模型
        data: MuJoCo数据
        sphere_name: sphere的名称
        new_position: 新位置 [x, y, z]
    
    返回:
        是否成功移动
    """
    body_id = find_sphere_by_name(model, sphere_name)
    if body_id >= 0:
        return move_sphere_body_directly(model, data, body_id, new_position)
    else:
        print(f"未找到名为 {sphere_name} 的sphere")
        return False 