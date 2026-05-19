下面是一套在 MuJoCo 源码内部集成并可视化 Gizmo 控制器的实现方案，包含整体架构、关键模块改动和示例代码片段，供参考和落地。

---

## 1. 总体架构

1. **Gizmo 数据结构**

   * 在 `mjData` 中新增一份 `Gizmo` 结构，保存当前控制器状态（如目标点、力/扭矩向量、交互句柄等）。
   * 定义 `mjGizmo` 类型，包含位置、朝向、可视化属性（颜色、尺寸）、及控制句柄 ID。

2. **控制器核心**

   * 在控制环（`mj_step` 或自定义控制回调）中更新 `mjGizmo` 信息。
   * 将用户输入或算法输出映射到 Gizmo（例如鼠标拖拽、键盘切换等）。

3. **渲染集成**

   * 修改渲染管线，将 `mjGizmo` 渲染为额外的几何体（箭头、球体、坐标系等）。
   * 扩展 `mjvScene` 和 `mjrContext`，支持自定义可视化对象列表。

---

## 2. 数据结构扩展

```c
// 在 include/mjdata.h 中
typedef struct {
  int id;                   // Gizmo 唯一 ID
  mjtNum pos[3];            // 中心位置
  mjtNum axis[3][3];        // 本地坐标系三个轴向
  mjtNum scale;             // 缩放因子
  mjtByte type;             // 类型：平移/旋转/混合
  mjtByte visible;          // 可见开关
  mjtRGBA color;            // RGBA 颜色
} mjGizmo;

struct _mjData {
  …;
  mjGizmo* gizmo;           // 动态数组
  int ngizmo;               // 数量
  …;
};
```

* 在 `mj_makeData`/`mj_freeData` 中分配与释放 `gizmo` 数组。
* 通过 API （如 `mj_addGizmo(mjData*, mjGizmo)`、`mj_removeGizmo`）管理 Gizmo 列表。

---

## 3. 控制器回调

1. **注册回调**

   ```c
   // 用户代码中
   void my_controller(const mjModel* m, mjData* d) {
     // 根据键盘/鼠标更新 gizmo[0]
     if (mouse_button_down) {
       d->gizmo[0].visible = 1;
       // 将鼠标映射到世界坐标
       mjtNum ray_start[3], ray_dir[3];
       mjv_mouseRay(&vopt, &camera, rayp, ray_start, ray_dir);
       // 计算交点，更新 gizmo 位置
       intersect_plane(ray_start, ray_dir, d->gizmo[0].axis[2], d->gizmo[0].pos);
     }
   }

   // 在初始化时
   mj_setCallback(m, d, mjCB_CONTROL, my_controller);
   ```

2. **更新 Gizmo 属性**

   * 动态调整 `axis` 用于显示旋转轴。
   * 根据控制算法在 `mj_step` 之后插入可视化数据。

---

## 4. 渲染管线改造

MuJoCo 渲染分两步：

* CPU 端构建 `mjvScene`（几何体列表）。
* GPU 端由 `mjr_render` 绘制。

### 4.1 扩展 `mjvScene`

在 `mjv_makeScene` / `mjv_updateScene` 中追加处理 Gizmo：

```c
// 在 mjv_updateScene 的末尾
for (int i=0; i<d->ngizmo; i++) {
  if (!d->gizmo[i].visible) continue;

  // 添加球体（中心点）
  mjvGeom* geom = &scn->geoms[scn->ngeom++];
  geom->type    = mjGEOM_SPHERE;
  geom->dataid  = 0;
  geom->rgba[0] = d->gizmo[i].color[0];
  geom->rgba[1] = d->gizmo[i].color[1];
  geom->rgba[2] = d->gizmo[i].color[2];
  geom->rgba[3] = d->gizmo[i].color[3];
  geom->size[0] = d->gizmo[i].scale * 0.05;
  memcpy(geom->pos, d->gizmo[i].pos, 3*sizeof(mjtNum));

  // 添加三根轴向箭头
  for (int a=0; a<3; a++) {
    mjvGeom* arrow = &scn->geoms[scn->ngeom++];
    arrow->type   = mjGEOM_ARROW;
    arrow->dataid = 0;
    // 颜色映射 X:红 Y:绿 Z:蓝
    arrow->rgba[a] = 1.0;
    arrow->rgba[3] = d->gizmo[i].color[3];
    arrow->size[0] = d->gizmo[i].scale * 0.02; // 半径
    arrow->size[1] = d->gizmo[i].scale * 0.2;  // 长度
    // 起点 gizmo.pos，方向 axis[a]
    mju_addTo3(arrow->pos, d->gizmo[i].pos, d->gizmo[i].axis[a], 0.0);
    memcpy(arrow->mat, d->gizmo[i].axis[a], 3*sizeof(mjtNum));
  }
}
```

### 4.2 调整 `mjr_render`

通常无需额外修改，只要 `mjvScene` 中已有几何体，绘制即可自动兼容。

---

## 5. 用户交互

* **鼠标拾取**：

  * 使用 `mjv_makeRay` + `mjv_select` 实现对 Gizmo 轴或手柄的点击检测。
  * 高亮当前选中轴，可通过修改 `rgba` 或增加半透明模型。

* **拖拽操作**：

  * 拖动过程中实时调用控制器回调，更新 `mjData` 中目标位置/角度。
  * 在 `mjv_updateScene` 中根据新的 target 绘制指示线（`mjGEOM_CYLINDER`）或点线。

---

## 6. 性能优化

1. **批量几何**

   * 将相同类型的小几何（如所有 Gizmo 箭头）合并为一个 `mjGEOM_MESH`，减少 draw call。

2. **LOD（细节层次）**

   * 根据相机距离切换 Gizmo 细节，近距离绘制完整箭头，远距离只绘制中心点。

3. **双缓冲更新**

   * 将 Gizmo 数据更新从主渲染线程异步到控制线程，避免卡顿。

---

## 7. 示例调用流程

1. **初始化**

   ```c
   mj_activate("mjkey.txt");
   mjModel* m = mj_loadXML("model.xml", 0, 0, 0);
   mjData* d  = mj_makeData(m);

   // 添加一个平移 Gizmo
   mjGizmo g = { .id=0, .scale=1.0, .type=mjGIZMO_TRANSLATE, .visible=1 };
   mju_zero3(g.pos);
   mju_unit3(g.axis[0], 0); mju_unit3(g.axis[1], 1); mju_unit3(g.axis[2], 2);
   d->gizmo = malloc(sizeof(mjGizmo));
   d->gizmo[0] = g; d->ngizmo = 1;
   ```

2. **主循环**

   ```c
   while (!glfwWindowShouldClose(window)) {
     mj_step(m, d);
     mjv_updateScene(&vopt, &camerainfo, &scn);
     mjr_render(window, &scn, &con);
     glfwSwapBuffers(window);
     glfwPollEvents();
   }
   ```

---

通过以上方案，你可以在 MuJoCo 内部无缝集成可视化 Gizmo 控制器，并兼顾渲染效果与运行效率。根据项目需要，可进一步丰富 Gizmo 类型（旋转环、混合手柄）及交互逻辑。希望对你的实现有所帮助！
