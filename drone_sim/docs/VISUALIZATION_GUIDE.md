# 增强版可视化系统使用指南

## 概述

增强版可视化系统提供了全面的实时可视化功能，帮助您直观地理解无人机的感知、建图和规划过程。

## 功能特性

### 1. 无人机视锥体显示 ✓
- **功能**：实时显示相机的视野范围（FOV）
- **显示内容**：
  - 青色线框表示视锥体边界
  - 视锥深度：5米
  - 视场角：90度
- **作用**：直观看到无人机当前能观测到的区域

### 2. 深度点云显示 ✓
- **功能**：显示当前帧的深度传感器数据
- **显示内容**：
  - 彩色点云（颜色表示距离）
  - 使用 viridis 色图（蓝色=近，黄色=远）
  - 自动下采样以提高性能
- **位置**：左下角 3D 视图
- **作用**：查看当前帧的原始感知数据

### 3. 地图累积过程显示 ✓
- **功能**：用热力图显示体素被观测的次数
- **显示内容**：
  - 红色热力图（深红=多次观测，浅红=少次观测）
  - 障碍物颜色深浅反映观测置信度
- **位置**：主 3D 视图
- **作用**：理解地图是如何逐步构建的，哪些区域被多次确认

### 4. RRT* 搜索树显示 ✓
- **功能**：显示路径规划的搜索过程
- **显示内容**：
  - 灰色半透明线条表示搜索树的边
  - 显示 RRT* 如何探索空间
- **位置**：主 3D 视图（叠加在地图上）
- **作用**：理解规划器如何找到路径

### 5. 实时动画 + 视频录制 ✓
- **功能**：
  - 实时显示规划过程
  - 自动录制每一帧
  - 飞行结束后保存为 MP4 视频
- **视频参数**：
  - 默认帧率：10 FPS
  - 分辨率：2000×1000 像素
  - 格式：MP4（需要 ffmpeg）
- **作用**：回放分析、论文展示、演示汇报

## 界面布局

```
┌─────────────────────────────────────────────────────────────┐
│  Receding Horizon Planning - Enhanced Visualization          │
├──────────────────────────────┬──────────────┬────────────────┤
│                              │              │                │
│                              │  XY Plane    │  XZ Plane      │
│   Main 3D View               │  (Top View)  │  (Side View)   │
│   - 障碍物热力图              │              │                │
│   - 视锥体                    │              │                │
│   - RRT*搜索树                │              │                │
│   - 规划路径                  │              │                │
│   - 执行轨迹                  │              │                │
│                              │              │                │
├──────────────────────────────┼──────────────┴────────────────┤
│                              │                                │
│  Depth Point Cloud           │  Statistics                    │
│  - 当前帧点云                 │  - 迭代次数                     │
│  - 距离着色                   │  - 地图统计                     │
│                              │  - 路径信息                     │
│                              │  - 性能指标                     │
└──────────────────────────────┴────────────────────────────────┘
```

## 使用方法

### 基础使用

在 `fly_planned_path.py` 中已经配置好：

```python
receding_config = {
    'visualize': True,         # 启用可视化
    'enhanced_viz': True,      # 使用增强版
    'save_video': True,        # 保存视频
    'video_fps': 10            # 视频帧率
}
```

### 运行测试

```bash
# 确保 AirSim 正在运行
cd drone_sim
python fly_planned_path.py
```

### 输出文件

运行结束后会生成：
- `receding_horizon_result.png` - 最终状态截图
- `receding_horizon_video.mp4` - 完整过程视频

## 性能优化

### 已实现的优化

1. **点云下采样**
   - 障碍物：最多显示 5000 个点
   - 深度点云：每 4 个点采样 1 个
   - XY/XZ 视图：最多 3000 个点

2. **智能更新**
   - 只在数据变化时重绘
   - 使用 `plt.pause(0.01)` 而非 `plt.show()`

3. **帧率控制**
   - 可视化更新：实时（~10-20ms）
   - 视频帧率：10 FPS（可调整）

### 性能监控

可视化器会自动记录更新时间：

```python
viz_stats = visualizer.get_performance_stats()
print(f"Average update time: {viz_stats['avg_update_time']:.1f}ms")
print(f"Total frames: {viz_stats['total_frames']}")
```

## 自定义配置

### 修改视锥体参数

在 `enhanced_visualizer.py` 中：

```python
def _draw_camera_frustum(self, ax, position, orientation,
                        fov_deg=90.0,    # 视场角
                        depth=5.0):      # 视锥深度
```

### 修改热力图颜色

```python
scatter = self.ax_3d.scatter(
    ...,
    cmap='hot',      # 可选: 'viridis', 'plasma', 'coolwarm'
    ...
)
```

### 修改视频参数

```python
receding_config = {
    'video_fps': 15,           # 提高帧率
    'save_video': True,
}
```

## 故障排除

### 问题 1：视频保存失败

**错误信息：** `Failed to save video: ... ffmpeg not found`

**解决方案：**
1. 安装 ffmpeg：
   ```bash
   # Windows (使用 chocolatey)
   choco install ffmpeg

   # 或下载：https://ffmpeg.org/download.html
   ```
2. 确保 ffmpeg 在 PATH 中
3. 重启终端

### 问题 2：可视化窗口卡顿

**症状：** 窗口无响应或更新很慢

**解决方案：**
1. 降低点云密度：
   ```python
   subsample = 8  # 增加到 16
   ```
2. 关闭 RRT* 树显示（如果不需要）
3. 使用简单版可视化器：
   ```python
   'enhanced_viz': False
   ```

### 问题 3：内存占用过高

**症状：** 长时间运行后内存不足

**解决方案：**
1. 限制帧数：
   ```python
   if len(self.frames) > 500:  # 最多保存 500 帧
       self.frames = self.frames[-500:]
   ```
2. 降低图像分辨率
3. 定期清理旧帧

## 与简单版可视化器对比

| 功能 | 简单版 | 增强版 |
|------|--------|--------|
| 3D 地图显示 | ✓ | ✓ |
| XY/XZ 平面投影 | ✓ | ✓ |
| 执行轨迹 | ✓ | ✓ |
| 视锥体显示 | ✗ | ✓ |
| 深度点云 | ✗ | ✓ |
| 观测热力图 | ✗ | ✓ |
| RRT* 搜索树 | ✗ | ✓ |
| 统计信息面板 | ✗ | ✓ |
| 视频录制 | ✗ | ✓ |
| 性能开销 | 低 | 中 |

## 最佳实践

### 1. 论文/报告使用

- 启用所有功能
- 使用高帧率（15-20 FPS）
- 保存高分辨率截图（dpi=300）

### 2. 实时调试

- 关闭视频录制（节省内存）
- 使用简单版可视化器
- 降低更新频率

### 3. 演示展示

- 启用视频录制
- 使用中等帧率（10 FPS）
- 确保所有标签清晰可见

## 技术细节

### 坐标系转换

增强版可视化器正确处理了多个坐标系：

1. **相机坐标系**：Z前, X右, Y下
2. **机体坐标系**：X前, Y右, Z下
3. **世界坐标系**：NED (X北, Y东, Z下)

### 数据流

```
深度图 → 相机点云 → 世界点云 → 可视化
  ↓
地图更新 → 观测计数 → 热力图
  ↓
RRT*规划 → 搜索树 → 树结构显示
```

## 示例输出

### 控制台输出

```
[5] Initializing map manager and planner...
  Using Enhanced Visualizer
    ✓ Initialization complete

[7] Starting receding horizon planning...

============================================================
Iteration 1
============================================================
Current position: (0.00, 0.00, -3.00)
Distance to global goal: 17.00m
Map update: +234 voxels, total 234 occupied
Local goal: (6.00, 0.00, -3.00)
Path planned: 8 waypoints
Executing 3 waypoints...

  Performance:
    perception: 45.2ms
    mapping: 38.7ms
    planning: 52.1ms
    execution: 1250.3ms
    TOTAL: 1386.3ms
```

### 视频内容

- 0:00-0:10 - 起飞和初始建图
- 0:10-0:30 - 发现障碍物，规划绕行
- 0:30-0:50 - 执行路径，持续更新地图
- 0:50-1:00 - 到达目标，完成任务

---

**创建日期：** 2026-01-19
**版本：** 1.0
**状态：** 已实现并集成
