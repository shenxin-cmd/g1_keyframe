import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import threading
import time

class RobotJointUI:
    def __init__(self, joint_names, joint_ids, joint_types, joint_qpos_map, joint_dim_map, 
                 joint_ranges, joint_descriptions, update_joint_angle_callback, 
                 reset_all_joints_callback, save_keyframe_callback, save_all_keyframes_callback):
        """初始化机器人关节UI控制器"""
        self.joint_names = joint_names
        self.joint_ids = joint_ids
        self.joint_types = joint_types
        self.joint_qpos_map = joint_qpos_map
        self.joint_dim_map = joint_dim_map
        self.joint_ranges = joint_ranges
        self.joint_descriptions = joint_descriptions
        
        # 回调函数
        self.update_joint_angle = update_joint_angle_callback
        self.reset_all_joints = reset_all_joints_callback
        self.save_keyframe = save_keyframe_callback
        self.save_all_keyframes = save_all_keyframes_callback
        
        # UI元素
        self.root = None
        self.sliders = {}
        self.keyframe_count_label = None
        self.current_focus = [None]  # 使用列表以便在回调中修改
        
        # 创建UI
        self.create_ui()
    
    def create_ui(self):
        """创建UI界面"""
        self.root = tk.Tk()
        self.root.title("机器人关节控制器")
        
        # 设置窗口大小和位置
        self.root.geometry("600x900+50+50")
        
        # 设置默认字体大小
        import tkinter.font as tkfont
        default_font = tkfont.nametofont("TkDefaultFont")
        default_font.configure(size=12)
        self.root.option_add("*Font", default_font)
        
        # 创建一个框架，包含所有控件
        main_frame = ttk.Frame(self.root, padding="15")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # 创建一个画布和滚动条，用于显示大量滑动条
        canvas = tk.Canvas(main_frame)
        scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        # 配置滚动区域
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        # 创建一个窗口，将scrollable_frame放入canvas
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # 放置画布和滚动条
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # 为每个关节创建一个滑动条
        for name in self.joint_names:
            if name and name not in ["root"]:  # 排除根关节
                joint_type = self.joint_types[self.joint_ids[name]]
                
                # 只为铰链关节和滑动关节创建滑动条
                if joint_type == 0 or joint_type == 3:  # mjJNT_HINGE or mjJNT_SLIDE
                    # 创建一个框架，包含标签和滑动条
                    frame = ttk.Frame(scrollable_frame, padding="8")
                    frame.pack(fill=tk.X, padx=8, pady=4)
                    
                    # 创建标签，添加关节描述
                    label_text = name
                    if name in self.joint_descriptions:
                        label_text = self.joint_descriptions[name]
                    
                    # 创建标签放在上方，使用大字体
                    label = ttk.Label(frame, text=label_text, font=("TkDefaultFont", 14))
                    label.pack(side=tk.TOP, anchor=tk.W)
                    
                    # 创建一个子框架用于放置滑动条和值标签
                    slider_frame = ttk.Frame(frame)
                    slider_frame.pack(fill=tk.X, expand=True, pady=5)
                    
                    # 创建变量，用于存储滑动条的值
                    var = tk.DoubleVar()
                    var.set(0.0)  # 初始值
                    
                    # 获取关节的限位范围
                    lower_limit, upper_limit = self.joint_ranges.get(name, (-3.14, 3.14))
                    
                    # 创建滑动条，使用关节的实际限位范围
                    slider = ttk.Scale(
                        slider_frame,
                        from_=lower_limit,
                        to=upper_limit,
                        orient=tk.HORIZONTAL,
                        variable=var,
                        length=250
                    )
                    slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=8)
                    
                    # 创建值标签，显示当前值和限位范围，使用稍大字体
                    value_label = ttk.Label(slider_frame, text=f"0.00 [{lower_limit:.2f}, {upper_limit:.2f}]", font=("TkDefaultFont", 12))
                    value_label.pack(side=tk.RIGHT)
                    
                    # 存储滑动条和标签
                    self.sliders[name] = (slider, var, value_label)
                    
                    # 添加回调函数，当滑动条值改变时更新关节角度
                    def update_callback(name=name):
                        angle = self.sliders[name][1].get()
                        self.update_joint_angle(name, angle)
                        # 更新值标签，显示当前值
                        lower, upper = self.joint_ranges.get(name, (-3.14, 3.14))
                        self.sliders[name][2].config(text=f"{angle:.2f} [{lower:.2f}, {upper:.2f}]")
                    
                    slider.config(command=lambda x, n=name: update_callback(n))
                    
                    # 设置鼠标进入滑动条时的焦点
                    slider.bind("<Enter>", lambda e, n=name: self.current_focus.__setitem__(0, n))
        
        # 创建一个框架，包含控制按钮，放在界面最下方
        button_frame = ttk.Frame(main_frame, padding="10")
        button_frame.pack(fill=tk.X, padx=8, pady=10, side=tk.BOTTOM)
        
        # 创建按钮样式
        style = ttk.Style()
        style.configure("Big.TButton", font=("TkDefaultFont", 14), padding=8)
        
        # 显示当前关键帧数量的标签，使用大字体
        self.keyframe_count_label = ttk.Label(button_frame, text="关键帧: 0", font=("TkDefaultFont", 14))
        self.keyframe_count_label.pack(side=tk.TOP, anchor=tk.W, pady=5)
        
        # 创建重置按钮，左对齐
        reset_button = ttk.Button(
            button_frame,
            text="重置所有关节",
            command=self.reset_all_joints,
            style="Big.TButton"
        )
        reset_button.pack(side=tk.TOP, anchor=tk.W, pady=5, fill=tk.X)
        
        # 创建保存关键帧按钮，左对齐
        save_keyframe_button = ttk.Button(
            button_frame,
            text="保存关键帧",
            command=lambda: self.save_keyframe_ui(),
            style="Big.TButton"
        )
        save_keyframe_button.pack(side=tk.TOP, anchor=tk.W, pady=5, fill=tk.X)
        
        # 创建保存所有关键帧到文件按钮，左对齐
        save_all_button = ttk.Button(
            button_frame,
            text="保存所有关键帧",
            command=lambda: self.save_all_keyframes_ui(),
            style="Big.TButton"
        )
        save_all_button.pack(side=tk.TOP, anchor=tk.W, pady=5, fill=tk.X)
        
        # 绑定鼠标滚轮事件到整个窗口
        self.root.bind("<MouseWheel>", self.on_mousewheel)  # Windows
        self.root.bind("<Button-4>", self.on_mousewheel)    # Linux/macOS向上滚动
        self.root.bind("<Button-5>", self.on_mousewheel)    # Linux/macOS向下滚动
        
        # 设置键盘快捷键
        self.root.bind("<s>", lambda e: self.save_keyframe_ui())
        self.root.bind("<S>", lambda e: self.save_keyframe_ui())
    
    def on_mousewheel(self, event):
        """鼠标滚轮事件处理函数"""
        # 确定鼠标位置下的滑动条
        for name, (slider, var, value_label) in self.sliders.items():
            # 检查鼠标是否在滑动条上
            x, y = event.x_root, event.y_root
            slider_x = slider.winfo_rootx()
            slider_y = slider.winfo_rooty()
            slider_width = slider.winfo_width()
            slider_height = slider.winfo_height()
            
            # 如果鼠标在滑动条区域内
            if (slider_x <= x <= slider_x + slider_width and 
                slider_y <= y <= slider_y + slider_height):
                self.current_focus[0] = name
                break
        
        # 如果有聚焦的滑动条，调整其值
        if self.current_focus[0]:
            slider, var, value_label = self.sliders[self.current_focus[0]]
            current_value = var.get()
            
            # 获取关节的限位范围
            lower_limit, upper_limit = self.joint_ranges.get(self.current_focus[0], (-3.14, 3.14))
            
            # 计算步长（范围的0.5%，更精细的控制）
            step = (upper_limit - lower_limit) * 0.005
            
            # 根据滚轮方向调整值
            # 在Windows上，event.delta为正表示向上滚动
            # 在Linux/macOS上，Button-4是向上滚动，Button-5是向下滚动
            if hasattr(event, 'delta'):  # Windows
                delta = event.delta
                if delta > 0:  # 向上滚动
                    new_value = min(current_value + step, upper_limit)
                else:  # 向下滚动
                    new_value = max(current_value - step, lower_limit)
            elif hasattr(event, 'num'):  # Linux/macOS
                if event.num == 4:  # 向上滚动
                    new_value = min(current_value + step, upper_limit)
                else:  # event.num == 5，向下滚动
                    new_value = max(current_value - step, lower_limit)
            
            # 设置新值
            var.set(new_value)
            
            # 更新关节角度
            self.update_joint_angle(self.current_focus[0], new_value)
            
            # 更新值标签
            value_label.config(text=f"{new_value:.2f} [{lower_limit:.2f}, {upper_limit:.2f}]")
    
    def update_ui(self, data, keyframe_count):
        """更新UI上的滑动条和关键帧计数"""
        for name, (slider, var, value_label) in self.sliders.items():
            if name in self.joint_qpos_map:
                qpos_addr = self.joint_qpos_map[name]
                joint_type = self.joint_types[self.joint_ids[name]]
                
                if joint_type == 0 or joint_type == 3:  # mjJNT_HINGE or mjJNT_SLIDE
                    # 只更新铰链关节和滑动关节
                    current_angle = float(data.qpos[qpos_addr])  # 确保是浮点数
                    var.set(current_angle)
                    # 更新值标签，显示当前值和限位范围
                    lower, upper = self.joint_ranges.get(name, (-3.14, 3.14))
                    value_label.config(text=f"{current_angle:.2f} [{lower:.2f}, {upper:.2f}]")
        
        # 更新关键帧计数
        self.keyframe_count_label.config(text=f"关键帧: {keyframe_count}")
    
    def save_keyframe_ui(self):
        """从UI保存关键帧"""
        keyframe_num = self.save_keyframe()
        messagebox.showinfo("保存关键帧", f"已保存关键帧 #{keyframe_num}")
    
    def save_all_keyframes_ui(self):
        """从UI保存所有关键帧到文件"""
        keyframe_count = self.save_all_keyframes()
        if keyframe_count == 0:
            messagebox.showwarning("保存关键帧", "没有关键帧可保存！")
        else:
            messagebox.showinfo("保存关键帧", f"已保存 {keyframe_count} 个关键帧到文件")
    
    def start(self):
        """启动UI主循环"""
        self.root.mainloop()
    
    def update_periodically(self, data, keyframe_count):
        """定期更新UI"""
        self.update_ui(data, keyframe_count)
        # 每100毫秒更新一次
        self.root.after(100, lambda: self.update_periodically(data, keyframe_count)) 