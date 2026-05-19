import mujoco
import numpy as np
import glfw
import time
from enum import Enum

class GizmoType(Enum):
    TRANSLATE = 0
    ROTATE = 1
    MIXED = 2

class Gizmo:
    def __init__(self, id=0, pos=None, scale=1.0, type=GizmoType.TRANSLATE, visible=True, color=None):
        self.id = id
        self.pos = np.zeros(3) if pos is None else np.array(pos)
        self.axis = np.eye(3)  # 默认为单位矩阵，表示标准坐标系
        self.scale = scale
        self.type = type
        self.visible = visible
        self.color = np.array([1.0, 1.0, 1.0, 1.0]) if color is None else np.array(color)
        self.selected_axis = -1  # -1表示未选中任何轴

class MujocoGizmoViewer:
    def __init__(self, model_path):
        # 初始化GLFW
        if not glfw.init():
            raise Exception("GLFW初始化失败")
        
        # 创建窗口
        self.window = glfw.create_window(1200, 900, "MuJoCo Gizmo Controller", None, None)
        if not self.window:
            glfw.terminate()
            raise Exception("窗口创建失败")
        
        glfw.make_context_current(self.window)
        glfw.swap_interval(1)
        
        # 加载MuJoCo模型
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        
        # 初始化场景和相机
        self.scene = mujoco.MjvScene(self.model, maxgeom=10000)
        self.camera = mujoco.MjvCamera()
        self.camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.camera.distance = 2.0
        self.camera.azimuth = 90.0
        self.camera.elevation = -45.0
        
        # 初始化渲染选项
        self.opt = mujoco.MjvOption()
        
        # 初始化上下文
        self.context = mujoco.MjrContext(self.model, mujoco.mjtFontScale.mjFONTSCALE_150)
        
        # 鼠标交互状态
        self.button_left = False
        self.button_right = False
        self.button_middle = False
        self.lastx = 0
        self.lasty = 0
        
        # 注册回调
        glfw.set_key_callback(self.window, self.keyboard_callback)
        glfw.set_cursor_pos_callback(self.window, self.mouse_move_callback)
        glfw.set_mouse_button_callback(self.window, self.mouse_button_callback)
        glfw.set_scroll_callback(self.window, self.scroll_callback)
        
        # 创建Gizmo列表
        self.gizmos = []
        
        # 添加一个默认的Gizmo
        self.add_gizmo(Gizmo(id=0, pos=[0, 0, 0.5], scale=0.2))
        
        # 当前选中的Gizmo
        self.selected_gizmo_idx = -1
        
        # 射线相关
        self.ray_start = np.zeros(3)
        self.ray_dir = np.zeros(3)
        
        # 渲染相关
        self.viewport = mujoco.MjrRect(0, 0, 0, 0)
    
    def add_gizmo(self, gizmo):
        """添加一个Gizmo到列表中"""
        self.gizmos.append(gizmo)
    
    def remove_gizmo(self, gizmo_id):
        """根据ID移除Gizmo"""
        for i, gizmo in enumerate(self.gizmos):
            if gizmo.id == gizmo_id:
                self.gizmos.pop(i)
                return True
        return False
    
    def keyboard_callback(self, window, key, scancode, act, mods):
        """键盘回调函数"""
        if act == glfw.PRESS and key == glfw.KEY_ESCAPE:
            glfw.set_window_should_close(window, True)
    
    def mouse_button_callback(self, window, button, act, mods):
        """鼠标按钮回调函数"""
        # 更新鼠标按钮状态
        if button == glfw.MOUSE_BUTTON_LEFT:
            self.button_left = (act == glfw.PRESS)
            
            # 鼠标点击时，检测是否选中Gizmo
            if self.button_left:
                self.update_mouse_ray()
                self.select_gizmo()
            else:
                # 释放鼠标时，取消选择
                if self.selected_gizmo_idx >= 0:
                    self.gizmos[self.selected_gizmo_idx].selected_axis = -1
        
        elif button == glfw.MOUSE_BUTTON_RIGHT:
            self.button_right = (act == glfw.PRESS)
        
        elif button == glfw.MOUSE_BUTTON_MIDDLE:
            self.button_middle = (act == glfw.PRESS)
        
        # 记录鼠标位置
        x, y = glfw.get_cursor_pos(window)
        self.lastx, self.lasty = x, y
    
    def mouse_move_callback(self, window, x, y):
        """鼠标移动回调函数"""
        dx = x - self.lastx
        dy = y - self.lasty
        
        # 更新鼠标射线
        self.update_mouse_ray()
        
        # 处理Gizmo拖拽
        if self.button_left and self.selected_gizmo_idx >= 0:
            gizmo = self.gizmos[self.selected_gizmo_idx]
            if gizmo.selected_axis >= 0:
                self.drag_gizmo(gizmo, dx, dy)
        
        # 处理相机旋转
        elif self.button_right:
            self.camera.azimuth += 0.5 * dx
            self.camera.elevation = max(min(self.camera.elevation + 0.5 * dy, 90), -90)
        
        # 处理相机平移
        elif self.button_middle:
            forward = np.array([np.cos(np.radians(self.camera.azimuth)) * np.cos(np.radians(self.camera.elevation)),
                               np.sin(np.radians(self.camera.azimuth)) * np.cos(np.radians(self.camera.elevation)),
                               np.sin(np.radians(self.camera.elevation))])
            right = np.array([-np.sin(np.radians(self.camera.azimuth)), 
                             np.cos(np.radians(self.camera.azimuth)), 
                             0])
            up = np.cross(right, forward)
            
            self.camera.lookat[0] += 0.01 * (-dx * right[0] + dy * up[0])
            self.camera.lookat[1] += 0.01 * (-dx * right[1] + dy * up[1])
            self.camera.lookat[2] += 0.01 * (-dx * right[2] + dy * up[2])
        
        self.lastx, self.lasty = x, y
    
    def scroll_callback(self, window, xoffset, yoffset):
        """滚轮回调函数"""
        self.camera.distance *= 0.9 if yoffset > 0 else 1.1
    
    def update_mouse_ray(self):
        """更新鼠标射线"""
        # 获取当前视口大小
        width, height = glfw.get_window_size(self.window)
        self.viewport.width = width
        self.viewport.height = height
        
        # 获取鼠标位置
        x, y = glfw.get_cursor_pos(self.window)
        
        # 计算鼠标射线
        self.ray_start, self.ray_dir = self.get_ray_from_screen(x, height - y)
    
    def get_ray_from_screen(self, viewport_x, viewport_y):
        """从屏幕坐标计算射线"""
        width, height = self.viewport.width, self.viewport.height
        
        # 创建视口
        viewport = mujoco.MjrRect(0, 0, width, height)
        
        # 计算射线
        ray_start = np.zeros(3)
        ray_dir = np.zeros(3)
        
        mujoco.mjv_select(self.model, self.data, self.opt,
                         viewport.width / viewport.height,
                         viewport_x, viewport_y,
                         self.scene, ray_start, ray_dir)
        
        return ray_start, ray_dir
    
    def select_gizmo(self):
        """选择Gizmo"""
        min_dist = float('inf')
        selected_idx = -1
        selected_axis = -1
        
        for i, gizmo in enumerate(self.gizmos):
            if not gizmo.visible:
                continue
            
            # 检测中心点
            dist_to_center = self.ray_sphere_intersection(self.ray_start, self.ray_dir, 
                                                         gizmo.pos, gizmo.scale * 0.05)
            if dist_to_center >= 0 and dist_to_center < min_dist:
                min_dist = dist_to_center
                selected_idx = i
                selected_axis = -1  # 中心点
            
            # 检测三个轴
            for axis_idx in range(3):
                axis_start = gizmo.pos
                axis_dir = gizmo.axis[axis_idx] * gizmo.scale * 0.2
                axis_end = axis_start + axis_dir
                
                dist_to_axis = self.ray_cylinder_intersection(self.ray_start, self.ray_dir,
                                                            axis_start, axis_end,
                                                            gizmo.scale * 0.02)
                if dist_to_axis >= 0 and dist_to_axis < min_dist:
                    min_dist = dist_to_axis
                    selected_idx = i
                    selected_axis = axis_idx
        
        # 更新选中状态
        if selected_idx >= 0:
            self.selected_gizmo_idx = selected_idx
            self.gizmos[selected_idx].selected_axis = selected_axis
        else:
            self.selected_gizmo_idx = -1
    
    def ray_sphere_intersection(self, ray_start, ray_dir, sphere_center, sphere_radius):
        """射线与球体相交检测"""
        oc = ray_start - sphere_center
        a = np.dot(ray_dir, ray_dir)
        b = 2.0 * np.dot(oc, ray_dir)
        c = np.dot(oc, oc) - sphere_radius * sphere_radius
        discriminant = b * b - 4 * a * c
        
        if discriminant < 0:
            return -1.0
        else:
            return (-b - np.sqrt(discriminant)) / (2.0 * a)
    
    def ray_cylinder_intersection(self, ray_start, ray_dir, cyl_start, cyl_end, cyl_radius):
        """射线与圆柱体相交检测（简化版）"""
        # 圆柱体轴向
        cyl_dir = cyl_end - cyl_start
        cyl_length = np.linalg.norm(cyl_dir)
        if cyl_length < 1e-6:
            return -1.0
        
        cyl_dir = cyl_dir / cyl_length
        
        # 计算射线与圆柱体轴的最近点
        w = ray_start - cyl_start
        a = np.dot(ray_dir, ray_dir) - np.dot(ray_dir, cyl_dir)**2
        b = np.dot(ray_dir, w) - np.dot(ray_dir, cyl_dir) * np.dot(w, cyl_dir)
        c = np.dot(w, w) - np.dot(w, cyl_dir)**2 - cyl_radius**2
        
        if abs(a) < 1e-6:
            return -1.0
        
        discriminant = b * b - a * c
        
        if discriminant < 0:
            return -1.0
        
        t = (-b - np.sqrt(discriminant)) / a
        if t < 0:
            return -1.0
        
        # 检查交点是否在圆柱体长度范围内
        intersection = ray_start + t * ray_dir
        projection = np.dot(intersection - cyl_start, cyl_dir)
        
        if projection < 0 or projection > cyl_length:
            return -1.0
        
        return t
    
    def drag_gizmo(self, gizmo, dx, dy):
        """拖拽Gizmo"""
        if gizmo.selected_axis < 0:
            return
        
        # 获取选中的轴
        axis = gizmo.axis[gizmo.selected_axis]
        
        # 计算拖拽平面
        # 平面法向量为相机方向与轴的叉积
        cam_dir = self.camera.lookat - (self.camera.lookat + 
                                       np.array([np.cos(np.radians(self.camera.azimuth)) * np.cos(np.radians(self.camera.elevation)),
                                                np.sin(np.radians(self.camera.azimuth)) * np.cos(np.radians(self.camera.elevation)),
                                                np.sin(np.radians(self.camera.elevation))]) * self.camera.distance)
        cam_dir = cam_dir / np.linalg.norm(cam_dir)
        
        plane_normal = np.cross(cam_dir, axis)
        plane_normal = plane_normal / np.linalg.norm(plane_normal)
        
        # 计算射线与平面的交点
        t = self.ray_plane_intersection(self.ray_start, self.ray_dir, gizmo.pos, plane_normal)
        if t >= 0:
            intersection = self.ray_start + t * self.ray_dir
            
            # 计算交点在轴上的投影
            projection = np.dot(intersection - gizmo.pos, axis) * axis
            
            # 更新Gizmo位置
            gizmo.pos += projection * 0.01 * (dx + dy)
    
    def ray_plane_intersection(self, ray_start, ray_dir, plane_point, plane_normal):
        """射线与平面相交检测"""
        denom = np.dot(ray_dir, plane_normal)
        if abs(denom) < 1e-6:
            return -1.0
        
        t = np.dot(plane_point - ray_start, plane_normal) / denom
        return t if t >= 0 else -1.0
    
    def render_gizmos(self):
        """渲染所有Gizmo"""
        # 创建一个新的场景来存储Gizmo
        gizmo_scene = mujoco.MjvScene(self.model, maxgeom=100)
        
        for gizmo in self.gizmos:
            if not gizmo.visible:
                continue
            
            # 添加中心球体
            sphere = mujoco.MjvGeom()
            sphere.type = mujoco.mjtGeom.mjGEOM_SPHERE
            
            # 确保所有数组都是正确的numpy数组类型和形状
            size = np.array([gizmo.scale * 0.05, 0, 0], dtype=np.float64)
            pos = np.array(gizmo.pos, dtype=np.float64)
            # 矩阵必须是一维数组，长度为9
            mat = np.eye(3, dtype=np.float64).flatten()
            rgba = np.array(gizmo.color, dtype=np.float32)
            
            # 如果是选中状态，增加透明度
            if gizmo.id == self.selected_gizmo_idx:
                rgba[3] = 0.7
            
            # 使用正确类型的数组初始化几何体
            mujoco.mjv_initGeom(sphere, mujoco.mjtGeom.mjGEOM_SPHERE, size, 
                               pos, mat, rgba)
            # 将几何体添加到gizmo_scene
            if gizmo_scene.ngeom < gizmo_scene.maxgeom:
                gizmo_scene.geoms[gizmo_scene.ngeom] = sphere
                gizmo_scene.ngeom += 1
            
            # 添加三个轴向箭头
            for axis_idx in range(3):
                arrow = mujoco.MjvGeom()
                arrow.type = mujoco.mjtGeom.mjGEOM_ARROW
                
                # 准备数据
                size = np.array([gizmo.scale * 0.02, gizmo.scale * 0.2, 0], dtype=np.float64)
                pos = np.array(gizmo.pos, dtype=np.float64)
                
                # 设置箭头方向 - 创建旋转矩阵
                mat = np.eye(3, dtype=np.float64)
                axis_dir = gizmo.axis[axis_idx]
                
                # 设置第一列为轴方向
                mat[:, 0] = axis_dir
                
                # 确保矩阵是正交的
                if axis_idx == 0:  # X轴
                    mat[:, 1] = np.array([0, 1, 0])
                    mat[:, 2] = np.array([0, 0, 1])
                elif axis_idx == 1:  # Y轴
                    mat[:, 0] = np.array([0, 1, 0])
                    mat[:, 1] = np.array([-1, 0, 0])
                    mat[:, 2] = np.array([0, 0, 1])
                else:  # Z轴
                    mat[:, 0] = np.array([0, 0, 1])
                    mat[:, 1] = np.array([1, 0, 0])
                    mat[:, 2] = np.array([0, 1, 0])
                
                # 将矩阵展平为一维数组
                mat = mat.flatten()
                
                # 设置颜色 (RGB对应XYZ轴)
                rgba = np.zeros(4, dtype=np.float32)
                rgba[axis_idx] = 1.0
                rgba[3] = 1.0
                
                # 如果是选中的轴，增加亮度
                if gizmo.selected_axis == axis_idx:
                    rgba[:3] = np.minimum(rgba[:3] * 1.5, 1.0)
                
                # 使用正确类型的数组初始化几何体
                mujoco.mjv_initGeom(arrow, mujoco.mjtGeom.mjGEOM_ARROW, size,
                                   pos, mat, rgba)
                # 将几何体添加到gizmo_scene
                if gizmo_scene.ngeom < gizmo_scene.maxgeom:
                    gizmo_scene.geoms[gizmo_scene.ngeom] = arrow
                    gizmo_scene.ngeom += 1
        
        return gizmo_scene
    
    def render(self):
        """渲染场景"""
        # 更新视口
        width, height = glfw.get_window_size(self.window)
        self.viewport.width = width
        self.viewport.height = height
        
        # 更新场景
        mujoco.mjv_updateScene(self.model, self.data, self.opt, None, self.camera, 
                              mujoco.mjtCatBit.mjCAT_ALL, self.scene)
        
        # 获取Gizmo场景
        gizmo_scene = self.render_gizmos()
        
        # 渲染主场景
        mujoco.mjr_render(self.viewport, self.scene, self.context)
        
        # 渲染Gizmo场景
        mujoco.mjr_render(self.viewport, gizmo_scene, self.context)
    
    def run(self):
        """运行主循环"""
        while not glfw.window_should_close(self.window):
            # 模拟物理
            mujoco.mj_step(self.model, self.data)
            
            # 渲染
            self.render()
            
            # 交换缓冲区
            glfw.swap_buffers(self.window)
            
            # 处理事件
            glfw.poll_events()
            
            # 控制帧率
            time.sleep(0.01)
        
        # 清理资源
        glfw.terminate()

if __name__ == "__main__":
    # 使用模型
    model_path = "go2/scene_zero_gravity.xml"
    
    try:
        viewer = MujocoGizmoViewer(model_path)
        viewer.run()
    except Exception as e:
        print(f"错误: {e}")
        glfw.terminate() 