# 论文级可视化系统 (Paper-Quality Visualization)

## 概述

本系统提供IROS论文风格的高质量可视化，展示**增量式建图 + ESDF距离场 + RRT*路径规划**的完整过程。

## 生成的文件

### 1. 图像序列
- **位置**: `paper_visualization_frames/`
- **文件**: `frame_0000.png` 到 `frame_0014.png` (15帧)
- **分辨率**: 3000×1800 像素 @ 150 DPI
- **大小**: 每帧约400-500KB

### 2. 最终合成图像
- **文件**: `paper_visualization_final.png`
- **大小**: 514KB
- **用途**: 论文插图、演示文稿

### 3. 视频文件
- **文件**: `paper_visualization.mp4`
- **大小**: 774KB
- **帧率**: 2 FPS
- **编码**: H.264
- **用途**: 动态演示、答辩展示

## 可视化布局

```
┌─────────────────┬─────────────────┬─────────────────┐
│  3D Occupancy   │  ESDF XY Slice  │  ESDF XZ Slice  │
│  Map + Path     │  (Top View)     │  (Side View)    │
│                 │                 │                 │
│  - 障碍物点云    │  - 距离场热力图   │  - 距离场热力图   │
│  - 当前位置      │  - 规划路径      │  - 规划路径      │
│  - 局部/全局目标 │  - 位置标记      │  - 位置标记      │
│  - 规划路径      │                 │                 │
│  - 执行轨迹      │                 │                 │
├─────────────────┼─────────────────┴─────────────────┤
│  RRT* Tree      │  Top View with ESDF Heatmap       │
│  Growth         │                                   │
│                 │  - ESDF热力图背景                  │
│  - 树的节点和边  │  - 障碍物点云                      │
│  - 最终路径高亮  │  - 已执行轨迹（绿色粗线）           │
│  - 起点/终点     │  - 当前规划路径（蓝色虚线）         │
│                 │  - 当前位置和目标                  │
└─────────────────┴───────────────────────────────────┘
```

## 颜色方案（论文级配色）

| 元素 | 颜色 | 用途 |
|------|------|------|
| 障碍物 | 红色 (#E74C3C) | 占据体素 |
| 自由空间 | 蓝色 (#3498DB) | 已知空闲区域 |
| 未知空间 | 灰色 (#95A5A6) | 未观测区域 |
| 执行轨迹 | 绿色 (#2ECC71) | 已飞行路径 |
| 当前位置 | 橙色 (#F39C12) | 无人机位置 |
| 全局目标 | 紫色 (#9B59B6) | 最终目标 |
| RRT树 | 深灰 (#34495E) | 采样树结构 |
| 规划路径 | 蓝色虚线 | 当前规划 |

## 使用方法

### 1. 生成新的可视化

```bash
cd drone_sim
python generate_paper_visualization.py
```

**输出**:
- `paper_visualization_frames/` - 图像序列
- `paper_visualization_final.png` - 最终图像
- `paper_visualization.mp4` - 视频文件

### 2. 使用真实AirSim数据

修改 `fly_planned_path.py` 以使用论文级可视化器：

```python
from visualization.paper_quality_visualizer import PaperQualityVisualizer

# 初始化可视化器
visualizer = PaperQualityVisualizer(figsize=(20, 12), dpi=150)

# 在滚动规划循环中添加帧
frame_data = {
    'iteration': iteration,
    'map_manager': map_manager,
    'current_pos': current_pos,
    'local_goal': local_goal,
    'global_goal': global_goal,
    'current_path': path,
    'rrt_tree': rrt_tree,  # 可选
    'executed_trajectory': executed_trajectory,
    'map_stats': map_stats
}
visualizer.add_frame(frame_data)

# 保存结果
visualizer.save_all_frames('output_frames', prefix='frame')
visualizer.render_frame(-1, save_path='final_result.png')
```

### 3. 从图像序列创建视频

如果FFmpeg不可用，可以手动创建视频：

```bash
# 使用FFmpeg
ffmpeg -framerate 2 -i paper_visualization_frames/frame_%04d.png \
       -c:v libx264 -pix_fmt yuv420p -crf 18 output.mp4

# 使用更高质量
ffmpeg -framerate 2 -i paper_visualization_frames/frame_%04d.png \
       -c:v libx264 -pix_fmt yuv420p -crf 15 -preset slow output_hq.mp4
```

## 技术特性

### 1. 增量式地图可视化
- 体素逐帧累积显示
- 障碍物点云动态增长
- 支持下采样显示（避免过多点）

### 2. ESDF距离场
- 热力图显示（红-黄-绿配色）
- XY平面切片（俯视图）
- XZ平面切片（侧视图）
- 距离范围：-2m 到 8m

### 3. RRT*树可视化
- 显示所有采样节点
- 显示树的边连接
- 最终路径高亮显示
- 节点数量统计

### 4. 轨迹跟踪
- 已执行轨迹（绿色粗线）
- 当前规划路径（蓝色虚线）
- 局部目标和全局目标标记

## 性能优化

### 点云下采样
- 3D视图：最多8000个点
- 2D视图：最多5000个点
- 随机采样保持分布均匀

### 渲染优化
- 使用matplotlib的`tight_layout`
- 图像压缩（PNG格式）
- 视频编码（H.264, CRF 18）

## 论文使用建议

### 1. 单帧图像
- 使用 `paper_visualization_final.png`
- 分辨率：3000×1800 @ 150 DPI
- 适合：论文插图、海报

### 2. 多帧对比
- 选择关键帧：`frame_0000.png`, `frame_0007.png`, `frame_0014.png`
- 展示建图过程的演化
- 适合：方法说明、结果展示

### 3. 动态演示
- 使用 `paper_visualization.mp4`
- 适合：答辩演示、在线展示

### 4. 图像编辑
- 使用Inkscape/Adobe Illustrator添加标注
- 调整字体大小以适应论文格式
- 添加子图标签 (a), (b), (c), (d)

## 自定义配置

### 修改图像尺寸

```python
visualizer = PaperQualityVisualizer(
    figsize=(24, 14),  # 更大的图像
    dpi=200            # 更高的分辨率
)
```

### 修改颜色方案

编辑 `paper_quality_visualizer.py` 中的 `self.colors` 字典：

```python
self.colors = {
    'obstacle': '#YOUR_COLOR',
    'path': '#YOUR_COLOR',
    # ...
}
```

### 修改布局

修改 `GridSpec` 参数：

```python
self.gs = GridSpec(
    2, 3,                    # 行数, 列数
    figure=self.fig,
    hspace=0.3,              # 垂直间距
    wspace=0.3               # 水平间距
)
```

## 故障排除

### 1. FFmpeg不可用
**症状**: 视频创建失败
**解决**:
- 安装FFmpeg: `conda install ffmpeg`
- 或手动使用图像序列创建视频

### 2. 内存不足
**症状**: 程序崩溃或变慢
**解决**:
- 减少点云采样数量
- 降低图像分辨率
- 减少帧数

### 3. Unicode编码错误
**症状**: Windows下打印特殊字符失败
**解决**: 已修复，使用ASCII字符替代

### 4. 图像显示问题
**症状**: 布局重叠或不正确
**解决**: 调整 `figsize` 和 `GridSpec` 参数

## 示例输出

当前生成的可视化展示了：
- **15次迭代**的滚动规划过程
- **630个障碍物体素**的增量式建图
- **ESDF距离场**的实时更新
- **RRT*树**的生长和路径规划
- **执行轨迹**的动态演化

## 参考文献

可视化风格参考：
- IROS/ICRA论文标准
- IEEE Robotics & Automation Magazine
- 经典路径规划论文的可视化方法

## 联系方式

如有问题或建议，请参考项目主README或CLAUDE.md文件。
