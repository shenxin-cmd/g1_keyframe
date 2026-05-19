import mujoco
import mujoco.viewer
import numpy as np
import time
import tkinter as tk
from tkinter import ttk, messagebox
import threading
import os
import keyboard  # 需要安装：pip install keyboard

class JointController:
    def __init__(self, model_path="go2/scene_zero_gravity_free.xml"):
        # 加载模型和数据
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        
        # 初始化数据
        mujoco.mj_forward(self.model, self.data)
        
        # 获取所有关节信息
        self.joint_names = []
        self.joint_qpos_addrs = []  # 存储每个关节的qpos地址
        self.joint_types = []
        self.joint_dims = []  # 存储每个关节的维度
        
        for i in range(self.model.njnt):
            joint_name = self.model.joint(i).name
            joint_type = self.model.joint(i).type
            joint_qposadr = self.model.joint(i).qposadr  # 获取关节的qpos地址
            
            # 确定关节的维度
            if joint_type == mujoco.mjtJoint.mjJNT_FREE:
                joint_dim = 7  # 自由关节：3个位置 + 4个四元数
            elif joint_type == mujoco.mjtJoint.mjJNT_BALL:
                joint_dim = 4  # 球关节：4个四元数
            else:
                joint_dim = 1  # 铰链关节或滑动关节：1个值
            
            self.joint_names.append(joint_name)
            self.joint_qpos_addrs.append(joint_qposadr)
            self.joint_types.append(joint_type)
            self.joint_dims.append(joint_dim)
        
        # 创建关节名称到ID和qpos地址的映射
        self.joint_ids = {}
        self.joint_qpos_map = {}
        self.joint_dim_map = {}  # 添加维度映射
        for i, name in enumerate(self.joint_names):
            if name:  # 确保关节名称不为空
                self.joint_ids[name] = i
                self.joint_qpos_map[name] = self.joint_qpos_addrs[i]
                self.joint_dim_map[name] = self.joint_dims[i]
        
        # 打印可控制的关节
        print("可控制的关节:")
        for name in self.joint_names:
            if name:
                dim = self.joint_dim_map[name]
                type_str = "未知"
                if self.joint_types[self.joint_ids[name]] == mujoco.mjtJoint.mjJNT_FREE:
                    type_str = "自由关节"
                elif self.joint_types[self.joint_ids[name]] == mujoco.mjtJoint.mjJNT_BALL:
                    type_str = "球关节"
                elif self.joint_types[self.joint_ids[name]] == mujoco.mjtJoint.mjJNT_HINGE:
                    type_str = "铰链关节"
                elif self.joint_types[self.joint_ids[name]] == mujoco.mjtJoint.mjJNT_SLIDE:
                    type_str = "滑动关节"
                print(f"  - {name}: {type_str} (维度: {dim})")
        
        # 设置关节阻尼
        self.set_joint_damping(damping_value=10.0, armature_value=1.0)
        
        # 创建一个字典，存储每个关节的当前角度
        self.joint_angles = {}
        for name in self.joint_names:
            if name:  # 确保关节名称不为空
                self.joint_angles[name] = 0.0
        
        # 创建一个事件，用于通知主线程UI已经准备好
        self.ui_ready = threading.Event()
        
        # 创建UI线程
        self.ui_thread = threading.Thread(target=self.create_ui)
        self.ui_thread.daemon = True  # 设置为守护线程，这样主线程退出时UI线程也会退出
        
        # 跟踪拖拽状态
        self.is_dragging = False
        self.joint_damping_backup = {}
        
        # 关键帧相关
        self.keyframes = []
        self.should_save_keyframe = False
        
        # 确保关键帧目录存在
        self.keyframe_dir = "keyframes"
        if not os.path.exists(self.keyframe_dir):
            os.makedirs(self.keyframe_dir)
    
    def set_joint_damping(self, damping_value=10.0, armature_value=1.0):
        """设置所有关节的阻尼和惯性"""
        for i in range(self.model.njnt):
            # 获取关节类型
            joint_type = self.model.jnt_type[i]
            
            # 自由关节(类型为0)有6个DOF，需要特殊处理
            if joint_type == 0:  # mjtJoint.mjJNT_FREE
                # 自由关节的阻尼和惯性需要为每个DOF单独设置
                for dof in range(6):
                    dof_adr = self.model.jnt_dofadr[i] + dof
                    self.model.dof_damping[dof_adr] = damping_value * 2  # 自由关节使用更大的阻尼
                    self.model.dof_armature[dof_adr] = armature_value * 2
            else:
                # 其他类型关节(铰链、滑动等)
                dof_adr = self.model.jnt_dofadr[i]
                self.model.dof_damping[dof_adr] = damping_value
                self.model.dof_armature[dof_adr] = armature_value
    
    def set_specific_joint_damping(self, joint_name, damping_value, armature_value=None):
        """设置特定关节的阻尼和惯性"""
        # 查找关节ID
        for i in range(self.model.njnt):
            if self.model.joint(i).name == joint_name:
                dof_adr = self.model.jnt_dofadr[i]
                self.model.dof_damping[dof_adr] = damping_value
                if armature_value is not None:
                    self.model.dof_armature[dof_adr] = armature_value
                return True
        return False
    
    def find_related_joints(self, body_id):
        """根据选中的物体ID找到相关的关节"""
        related_joints = []
        
        # 检查每个关节，看它是否连接到选中的物体
        for i in range(self.model.njnt):
            joint = self.model.joint(i)
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
    
    def set_selected_body_joints_damping(self, body_id, damping_value=0.0):
        """设置选中物体相关关节的阻尼"""
        related_joints = self.find_related_joints(body_id)
        
        # 如果找到相关关节，设置它们的阻尼
        for joint_name in related_joints:
            # 备份当前阻尼值（如果尚未备份）
            if joint_name not in self.joint_damping_backup:
                for i in range(self.model.njnt):
                    if self.model.joint(i).name == joint_name:
                        dof_adr = self.model.jnt_dofadr[i]
                        self.joint_damping_backup[joint_name] = self.model.dof_damping[dof_adr]
                        break
            
            # 设置阻尼为指定值
            self.set_specific_joint_damping(joint_name, damping_value)
        
        return related_joints
    
    def restore_all_joint_damping(self):
        """恢复所有关节的原始阻尼值"""
        for joint_name, damping_value in self.joint_damping_backup.items():
            self.set_specific_joint_damping(joint_name, damping_value)
    
    def update_joint_angle(self, joint_name, angle):
        """更新关节角度"""
        if joint_name in self.joint_qpos_map:
            qpos_addr = self.joint_qpos_map[joint_name]
            joint_type = self.joint_types[self.joint_ids[joint_name]]
            
            if joint_type == mujoco.mjtJoint.mjJNT_HINGE or joint_type == mujoco.mjtJoint.mjJNT_SLIDE:
                # 只更新铰链关节和滑动关节
                self.data.qpos[qpos_addr] = angle
                self.joint_angles[joint_name] = angle
    
    def save_keyframe(self):
        """保存当前姿势作为关键帧"""
        # 创建关键帧字典，使用关节名称作为键
        keyframe = {'timestamp': time.time()}
        
        # 保存每个关节的当前角度
        for name in self.joint_names:
            if name:  # 确保关节名称不为空
                # 使用qpos地址获取关节角度
                qpos_addr = self.joint_qpos_map[name]
                joint_type = self.joint_types[self.joint_ids[name]]
                
                if joint_type == mujoco.mjtJoint.mjJNT_FREE:
                    # 自由关节有7个值：3个位置 + 4个四元数
                    keyframe[name] = [float(self.data.qpos[qpos_addr + j]) for j in range(7)]
                elif joint_type == mujoco.mjtJoint.mjJNT_BALL:
                    # 球关节有4个值：四元数
                    keyframe[name] = [float(self.data.qpos[qpos_addr + j]) for j in range(4)]
                else:
                    # 铰链关节和滑动关节只有1个值
                    keyframe[name] = float(self.data.qpos[qpos_addr])
        
        # 添加到关键帧列表
        self.keyframes.append(keyframe)
        
        # 打印保存的关键帧信息
        print(f"已保存关键帧 #{len(self.keyframes)}")
        for name, value in keyframe.items():
            if name != 'timestamp':
                if isinstance(value, (list, np.ndarray)):
                    print(f"  - {name}: {value}")
                else:
                    print(f"  - {name}: 角度 {value:.4f} 弧度 ({np.degrees(value):.2f}°)")
        
        return len(self.keyframes)
    
    def save_keyframes_to_file(self):
        """保存所有关键帧到文件"""
        if self.keyframes:
            timestamp = int(time.time())
            filename = f"{self.keyframe_dir}/joint_keyframes_{timestamp}.npz"
            np.savez(filename, keyframes=self.keyframes)
            print(f"\n已保存 {len(self.keyframes)} 个关键帧到文件: {filename}")
            return filename
        else:
            print("\n未记录任何关键帧。")
            return None
    
    def create_ui(self):
        """创建UI界面"""
        root = tk.Tk()
        root.title("机器人关节控制器")
        
        # 设置窗口大小和位置
        root.geometry("400x800+50+50")
        
        # 创建一个框架，包含所有控件
        main_frame = ttk.Frame(root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # 创建一个画布和滚动条，用于显示大量滑动条
        canvas = tk.Canvas(main_frame)
        scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # 为每个关节创建一个滑动条
        sliders = {}
        for name in self.joint_names:
            if name and name not in ["root"]:  # 排除根关节
                joint_type = self.joint_types[self.joint_ids[name]]
                
                # 只为铰链关节和滑动关节创建滑动条
                if joint_type == mujoco.mjtJoint.mjJNT_HINGE or joint_type == mujoco.mjtJoint.mjJNT_SLIDE:
                    # 创建一个框架，包含标签和滑动条
                    frame = ttk.Frame(scrollable_frame, padding="5")
                    frame.pack(fill=tk.X, padx=5, pady=5)
                    
                    # 创建标签
                    label = ttk.Label(frame, text=f"{name}:")
                    label.pack(side=tk.LEFT)
                    
                    # 创建一个变量，用于存储滑动条的值
                    var = tk.DoubleVar(value=0.0)
                    
                    # 创建滑动条
                    slider = ttk.Scale(
                        frame,
                        from_=-3.14,  # 最小值（-180度）
                        to=3.14,      # 最大值（180度）
                        orient="horizontal",
                        variable=var,
                        length=200
                    )
                    slider.pack(side=tk.LEFT, fill=tk.X, expand=True)
                    
                    # 创建一个标签，显示当前值
                    value_label = ttk.Label(frame, text="0.00")
                    value_label.pack(side=tk.RIGHT)
                    
                    # 存储滑动条和标签
                    sliders[name] = (slider, var, value_label)
                    
                    # 添加回调函数，当滑动条值改变时更新关节角度
                    def update_callback(name=name):
                        angle = sliders[name][1].get()
                        self.update_joint_angle(name, angle)
                        # 更新值标签
                        sliders[name][2].config(text=f"{angle:.2f}")
                    
                    slider.config(command=lambda x, n=name: update_callback(n))
        
        # 创建一个框架，包含控制按钮
        button_frame = ttk.Frame(main_frame, padding="5")
        button_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # 创建重置按钮
        reset_button = ttk.Button(
            button_frame,
            text="重置所有关节",
            command=self.reset_all_joints
        )
        reset_button.pack(side=tk.LEFT, padx=5)
        
        # 创建保存关键帧按钮
        save_keyframe_button = ttk.Button(
            button_frame,
            text="保存关键帧",
            command=lambda: self.save_keyframe_ui(root)
        )
        save_keyframe_button.pack(side=tk.LEFT, padx=5)
        
        # 创建保存所有关键帧到文件按钮
        save_all_button = ttk.Button(
            button_frame,
            text="保存所有关键帧",
            command=lambda: self.save_all_keyframes_ui(root)
        )
        save_all_button.pack(side=tk.LEFT, padx=5)
        
        # 显示当前关键帧数量的标签
        self.keyframe_count_label = ttk.Label(button_frame, text="关键帧: 0")
        self.keyframe_count_label.pack(side=tk.RIGHT, padx=5)
        
        # 创建一个函数，用于更新UI上的滑动条
        def update_ui():
            for name, (slider, var, value_label) in sliders.items():
                if name in self.joint_qpos_map:
                    qpos_addr = self.joint_qpos_map[name]
                    joint_type = self.joint_types[self.joint_ids[name]]
                    
                    if joint_type == mujoco.mjtJoint.mjJNT_HINGE or joint_type == mujoco.mjtJoint.mjJNT_SLIDE:
                        # 只更新铰链关节和滑动关节
                        current_angle = float(self.data.qpos[qpos_addr])  # 确保是浮点数
                        var.set(current_angle)
                        value_label.config(text=f"{current_angle:.2f}")
            
            # 更新关键帧计数
            self.keyframe_count_label.config(text=f"关键帧: {len(self.keyframes)}")
            
            # 每100毫秒更新一次
            root.after(100, update_ui)
        
        # 开始更新UI
        update_ui()
        
        # 设置键盘快捷键
        root.bind("<s>", lambda e: self.save_keyframe_ui(root))
        root.bind("<S>", lambda e: self.save_keyframe_ui(root))
        
        # 通知主线程UI已经准备好
        self.ui_ready.set()
        
        # 启动UI主循环
        root.mainloop()
    
    def save_keyframe_ui(self, root):
        """从UI保存关键帧"""
        keyframe_num = self.save_keyframe()
        messagebox.showinfo("保存关键帧", f"已保存关键帧 #{keyframe_num}")
    
    def save_all_keyframes_ui(self, root):
        """从UI保存所有关键帧到文件"""
        if not self.keyframes:
            messagebox.showwarning("保存关键帧", "没有关键帧可保存！")
            return
        
        filename = self.save_keyframes_to_file()
        if filename:
            messagebox.showinfo("保存关键帧", f"已保存 {len(self.keyframes)} 个关键帧到文件:\n{filename}")
    
    def reset_all_joints(self):
        """重置所有关节到初始位置"""
        for name in self.joint_names:
            if name in self.joint_qpos_map:
                joint_type = self.joint_types[self.joint_ids[name]]
                if joint_type == mujoco.mjtJoint.mjJNT_HINGE or joint_type == mujoco.mjtJoint.mjJNT_SLIDE:
                    self.update_joint_angle(name, 0.0)
    
    def run(self):
        """运行控制器"""
        # 启动UI线程
        self.ui_thread.start()
        
        # 等待UI准备好
        self.ui_ready.wait()
        
        # 设置键盘监听
        keyboard.on_press_key("s", lambda _: self.trigger_save_keyframe())
        
        # 创建查看器并开始模拟
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            print("机器人关节控制器已启动。")
            print("提示：使用UI滑动条可以精确控制每个关节的角度")
            print("      在MuJoCo窗口中，按住Ctrl并用鼠标拖动物体调整姿势")
            print("      拖拽物体时，相关关节阻尼会自动设为0，松开后恢复")
            print("      按S键或点击'保存关键帧'按钮保存当前姿势")
            print("      按ESC键退出程序")
            
            # 主循环
            while viewer.is_running():
                # 检测拖拽状态 - 使用扰动(perturbation)来判断
                # 在MuJoCo中，当用户拖拽物体时，会产生扰动
                current_perturbation = self.data.xfrc_applied.copy()
                
                # 检查是否有扰动（用户正在拖拽物体）
                has_perturbation = np.any(np.abs(current_perturbation) > 1e-6)
                
                if has_perturbation and not self.is_dragging:
                    # 找到被扰动的物体
                    perturbed_bodies = []
                    for i in range(self.model.nbody):
                        if np.any(np.abs(current_perturbation[i]) > 1e-6):
                            perturbed_bodies.append(i)
                    
                    if perturbed_bodies:
                        self.is_dragging = True
                        for body_id in perturbed_bodies:
                            # 设置相关关节阻尼为0
                            related_joints = self.set_selected_body_joints_damping(body_id, 0.0)
                            if related_joints:
                                print(f"\n拖拽物体 {self.model.body(body_id).name}，相关关节阻尼设为0: {', '.join(related_joints)}")
                
                elif not has_perturbation and self.is_dragging:
                    # 拖拽结束
                    self.is_dragging = False
                    
                    # 恢复关节阻尼
                    self.restore_all_joint_damping()
                    print("\n拖拽结束，恢复关节阻尼")
                    
                    # 清空备份
                    self.joint_damping_backup.clear()
                
                # 检查是否应该保存关键帧
                if self.should_save_keyframe:
                    self.save_keyframe()
                    self.should_save_keyframe = False
                
                # 更新物理
                mujoco.mj_step(self.model, self.data)
                
                # 同步查看器
                viewer.sync()
                
                # 控制帧率
                time.sleep(0.01)
            
            # 退出时保存关键帧
            if self.keyframes:
                self.save_keyframes_to_file()
    
    def trigger_save_keyframe(self):
        """触发保存关键帧"""
        self.should_save_keyframe = True

if __name__ == "__main__":
    controller = JointController()
    controller.run() 