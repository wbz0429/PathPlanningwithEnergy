# 简洁3D可视化 (Simple 3D Visualization)

## 概述

这是一个**简洁清晰的单视图3D可视化系统**，专注展示：
- 🚁 无人机飞行轨迹（绿色粗线）
- 🔴 实时累积的障碍物点云（红色点）
- 🔵 当前规划路径（蓝色虚线）
- 🟠 无人机当前位置（橙色球体 + 方向箭头）
- 🟣 目标位置（紫色星形）

**特点**：
- ✅ 单个清晰的3D视图，无多图叠加
- ✅ 动态旋转视角，展示不同角度
- ✅ 实时累积障碍物点云（模拟深度感知）
- ✅ 流畅的动画效果
- ✅ 适合演示和答辩

## 生成的文件

```
drone_sim/
├── simple_3d_final.png              # 最终图像 (125KB)
├── simple_3d_visualization.mp4      # 视频 (761KB, 5 FPS)
└── simple_3d_frames/                # 图像序列 (20帧)
    ├── frame_0000.png
    ├── frame_0001.png
    ├── ...
    └── frame_0019.png
```

## 可视化效果

### 单个3D视图展示：
- **障碍物点云**：红色半透明点，随着无人机移动逐渐累积
- **执行轨迹**：绿色粗线，显示无人机已飞行的路径
- **规划路径**：蓝色虚线，显示当前规划的局部路径
- **无人机**：橙色球体，带有朝向目标的方向箭头
- **目标**：紫色星形标记

### 动态效果：
- 视角缓慢旋转（每帧旋转2度）
- 障碍物点云逐帧累积
- 轨迹不断延伸
- 规划路径实时更新

## 使用方法

### 1. 生成新的可视化

```bash
cd drone_sim
python generate_simple_visualization.py
```

**输出**:
- `simple_3d_frames/` - 20帧图像序列
- `simple_3d_final.png` - 最终静态图像
- `simple_3d_visualization.mp4` - 动态视频

### 2. 在真实AirSim飞行中使用

修改 `fly_planned_path.py`：

```python
from visualization.simple_3d_visualizer import Simple3DVisualizer

# 初始化可视化器
visualizer = Simple3DVisualizer(figsize=(12, 10), dpi=100)

# 在滚动规划循环中
accumulated_obstacles = []  # 累积观测到的障碍物

for iteration in range(max_iterations):
    # ... 获取深度图和位置 ...

    # 累积障碍物点云
    observed_obstacles = get_obstacles_from_depth(depth_image)
    accumulated_obstacles.extend(observed_obstacles)

    # 添加帧
    frame_data = {
        'iteration': iteration,
        'obstacles': np.array(accumulated_obstacles),
        'drone_pos': current_pos,
        'trajectory': executed_trajectory,
        'planned_path': current_path,
        'goal': global_goal,
        'info': f"Obstacles: {len(accumulated_obstacles)} points"
    }
    visualizer.add_frame(frame_data)

# 保存结果
visualizer.save_all_frames('output_frames')
visualizer.create_video('output.mp4', fps=5)
```

### 3. 自定义配置

#### 修改图像尺寸和分辨率

```python
visualizer = Simple3DVisualizer(
    figsize=(16, 12),  # 更大的图像
    dpi=150            # 更高的分辨率
)
```

#### 修改颜色

编辑 `simple_3d_visualizer.py` 中的 `self.colors`：

```python
self.colors = {
    'obstacle': '#FF0000',    # 红色
    'trajectory': '#00FF00',  # 绿色
    'planned': '#0000FF',     # 蓝色
    'drone': '#FFA500',       # 橙色
    'goal': '#800080',        # 紫色
}
```

#### 修改视角旋转速度

在 `render_frame()` 中修改：

```python
self.ax.view_init(elev=25, azim=45 + frame_idx * 5)  # 更快旋转
```

#### 修改显示范围

```python
margin = 20  # 显示无人机周围20m范围
self.ax.set_xlim(x_center - margin, x_center + margin)
```

## 技术特性

### 1. 实时障碍物累积
- 模拟深度相机的视场角（FOV 90度）
- 只观测前方和视场内的障碍物
- 逐帧累积，模拟真实的建图过程
- 简单去重（距离阈值0.3m）

### 2. 动态视角
- 视角随帧数缓慢旋转
- 自动以无人机为中心
- 保持合适的显示范围

### 3. 清晰的视觉层次
- 障碍物：半透明，不遮挡轨迹
- 轨迹：粗线，高优先级显示
- 规划路径：虚线，区分已执行和计划
- 无人机和目标：最高优先级，始终可见

### 4. 性能优化
- 点云下采样（最多5000点）
- 高效的去重算法
- 合理的图像分辨率

## 与复杂版本的对比

| 特性 | 简洁版 (Simple) | 复杂版 (Paper-Quality) |
|------|----------------|----------------------|
| 视图数量 | 1个3D视图 | 5个子图 |
| 文件大小 | 125KB | 514KB |
| 清晰度 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| 信息量 | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| 适用场景 | 演示、答辩 | 论文插图 |
| 生成速度 | 快 | 慢 |

## 使用建议

### 1. 答辩演示
- 使用视频文件 `simple_3d_visualization.mp4`
- 5 FPS 的帧率适合讲解
- 清晰展示建图和规划过程

### 2. 海报展示
- 使用最终图像 `simple_3d_final.png`
- 添加标注说明各个元素
- 可以选择中间某一帧展示建图过程

### 3. 论文插图
- 选择关键帧（开始、中间、结束）
- 使用图像编辑软件添加标签
- 可以并排放置多帧对比

### 4. 在线展示
- 将视频上传到视频平台
- 或转换为GIF动图（体积更小）

## 创建GIF动图

```bash
# 使用FFmpeg创建GIF
ffmpeg -i simple_3d_visualization.mp4 -vf "fps=5,scale=800:-1:flags=lanczos" -loop 0 simple_3d.gif

# 或从图像序列创建
ffmpeg -framerate 5 -i simple_3d_frames/frame_%04d.png -vf "scale=800:-1:flags=lanczos" -loop 0 simple_3d.gif
```

## 故障排除

### 1. 视频播放问题
**症状**: 视频无法播放
**解决**: 使用VLC播放器或转换格式
```bash
ffmpeg -i simple_3d_visualization.mp4 -c:v libx264 -preset slow -crf 22 output_compatible.mp4
```

### 2. 图像显示不完整
**症状**: 轨迹或障碍物被裁剪
**解决**: 增加显示范围
```python
margin = 25  # 增大显示范围
```

### 3. 点云过于密集
**症状**: 障碍物点太多，影响性能
**解决**: 减少下采样阈值
```python
if len(obstacles) > 3000:  # 从5000改为3000
    indices = np.random.choice(len(obstacles), 3000, replace=False)
```

### 4. 视角不理想
**症状**: 视角太高或太低
**解决**: 调整仰角
```python
self.ax.view_init(elev=30, azim=45 + frame_idx * 2)  # 调整elev参数
```

## 示例输出说明

当前生成的可视化展示了：
- **20次迭代**的滚动规划过程
- **630个障碍物点**的实时累积
- **21个航点**的执行轨迹
- **动态规划路径**的更新过程
- **缓慢旋转的视角**（总共旋转40度）

## 下一步改进

可能的增强功能：
1. 添加速度矢量显示
2. 显示ESDF等值面
3. 添加时间戳和统计信息
4. 支持多无人机可视化
5. 添加交互式控制（暂停、回放）

## 参考

- 可视化器实现: `visualization/simple_3d_visualizer.py`
- 生成脚本: `generate_simple_visualization.py`
- 主项目文档: `CLAUDE.md`
