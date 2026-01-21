# 已知问题与待解决事项

本文档记录当前系统存在的技术问题、局限性及其根本原因分析。

---

## 1. 地面深度信息误识别问题 [未解决]

### 问题描述
深度相机向下倾斜时会观测到地面，系统将地面点云错误识别为高度 8-9m 的障碍物墙壁，导致路径规划失败或产生不必要的绕行。

### 问题根源分析

#### 1.1 坐标变换链路
```
深度图像素 → 相机坐标系 → 机体坐标系 → 世界坐标系(NED)
```

相关代码位置：
- `utils/transforms.py:65-119` - `depth_image_to_camera_points()` 深度图转点云
- `utils/transforms.py:29-62` - `transform_camera_to_world()` 坐标变换
- `mapping/incremental_map.py:134-187` - `_accumulate_points()` 点云累积

#### 1.2 当前的地面过滤逻辑（不完善）

```python
# mapping/incremental_map.py:149-166
# 地面过滤参数
drone_z = camera_pos[2]  # 负值，如-3表示3米高
ground_threshold = -0.5  # Z > -0.5 的点认为是地面

for point in points_world:
    point_z = point[2]
    # 过滤地面点（Z接近0或为正的点）
    if point_z > ground_threshold:
        continue  # 跳过地面点
```

**问题**：
1. `ground_threshold = -0.5` 是硬编码的固定值，只过滤 Z > -0.5m 的点
2. 当无人机在 3m 高度（Z = -3.0）时，地面点的 Z 坐标应该接近 0
3. 但如果坐标变换有误差，或者地面不平坦，地面点可能被计算为 Z = -8 或 -9
4. 这些点会被错误地标记为障碍物

#### 1.3 可能的误差来源

**A. 深度图到相机坐标系转换**
```python
# utils/transforms.py:111-117
x_cam = (u - cx) * z / fx
y_cam = (v - cy) * z / fy
z_cam = z

# 相机坐标系: Z前, X右, Y下
points_camera = np.stack([x_cam, y_cam, z_cam], axis=1)
```
- 假设相机内参 `fx`, `fy` 基于 FOV 计算，可能与 AirSim 实际相机参数不匹配
- 深度值 `z` 是沿光轴的距离还是欧几里得距离？需要确认

**B. 相机到机体坐标系变换**
```python
# utils/transforms.py:50-56
R_body_camera = np.array([
    [0, 0, 1],   # 机体X = 相机Z
    [1, 0, 0],   # 机体Y = 相机X
    [0, 1, 0]    # 机体Z = 相机Y
])
```
- 假设相机安装在机体正前方，无俯仰角
- 如果相机有向下的安装角度，此变换矩阵需要修正

**C. 机体到世界坐标系变换**
```python
# utils/transforms.py:58-60
R_world_body = quaternion_to_rotation_matrix(drone_orientation)
points_world = (R_world_body @ points_body.T).T + drone_position
```
- 依赖 AirSim 返回的四元数姿态
- 如果无人机有俯仰角，地面点会被投影到错误的高度

### 排查步骤建议

1. **验证深度值含义**：确认 AirSim 深度图返回的是光轴距离还是欧几里得距离
2. **检查相机安装角度**：确认相机是否有向下的俯仰角，如有需要在变换中补偿
3. **打印调试信息**：在 `_accumulate_points()` 中打印地面点的原始深度值和变换后的世界坐标
4. **可视化点云**：将原始点云和变换后的点云分别可视化，对比分析

### 临时解决方案

当前代码使用了两个保护机制，但不能根本解决问题：

```python
# mapping/incremental_map.py:108-124
def _clear_around_drone(self, drone_pos, radius=2.5):
    """清除无人机周围的障碍物标记"""
    # 防止无人机被错误标记在障碍物内

# mapping/incremental_map.py:126-132
def _ensure_drone_safe(self, drone_pos):
    """确保ESDF中无人机位置是安全的"""
    # 如果ESDF显示在障碍物内，再次清除
```

---

## 2. 感知层面局限性

### 2.1 单相机有限视场 [设计局限]

**问题描述**：仅有 90° 前向深度相机，侧面和后方是盲区。

**影响**：
- 无人机转向后，之前观测到的障碍物可能丢失
- 侧向接近的障碍物无法检测

**代码位置**：
- `planning/config.py:18` - `fov_deg: float = 90.0`

**可能的改进**：
- 添加多相机支持（左右侧向相机）
- 或使用 360° 激光雷达数据

### 2.2 无回环检测 [设计局限]

**问题描述**：长距离飞行会有地图漂移累积，没有回环校正机制。

**影响**：
- 长时间飞行后，地图与真实环境可能出现偏差
- 重复访问同一区域时，障碍物位置可能不一致

**当前状态**：系统假设 AirSim 提供的位姿是准确的，未实现 SLAM 回环检测。

---

## 3. 建图层面问题

### 3.1 静态环境假设 [设计局限]

**问题描述**：系统假设环境是静态的，不处理动态障碍物。

**代码体现**：
```python
# mapping/incremental_map.py:176-178
# 标记为占据后不会自动清除
if self.voxel_grid.grid[idx] != 1:
    self.voxel_grid.grid[idx] = 1
    new_count += 1
```

**影响**：
- 移动的障碍物（人、车辆）会在地图中留下"幽灵"轨迹
- 可能导致路径规划产生不必要的绕行

### 3.2 内存管理 [潜在问题]

**问题描述**：虽然有滑动窗口机制，但长时间运行仍可能内存增长。

**代码位置**：
```python
# mapping/incremental_map.py:221-254
def _update_map_center(self, drone_position):
    if np.linalg.norm(shift) > 10.0:  # 移动超过10米
        self._prune_distant_voxels(drone_position, max_distance=30.0)

def _prune_distant_voxels(self, drone_position, max_distance):
    # 清理距离无人机过远的体素
```

**潜在问题**：
- `observation_count` 数组始终保持完整大小
- 如果飞行范围超出初始 grid 范围，需要重新初始化

---

## 4. 规划层面问题

### 4.1 RRT* 局部最优 [算法局限]

**问题描述**：RRT* 是采样算法，在复杂环境中可能陷入局部最小值。

**代码位置**：
```python
# planning/rrt_star.py - RRT* 实现
# planning/receding_horizon.py:150-180 - 备选目标策略
```

**当前缓解措施**：
```python
# 尝试备选目标角度
for angle in [-30, 30, -45, 45]:
    # 旋转方向尝试找到可行路径
```

**局限**：备选策略不保证成功，在死胡同环境中可能完全失败。

### 4.2 无全局拓扑引导 [设计局限]

**问题描述**：没有全局路径引导，完全依赖局部规划向目标方向探索。

**影响**：
- 在 U 形障碍物中可能无法找到出路
- 需要多次尝试才能绕过大型障碍物

**可能的改进**：
- 添加全局 A* 或 Dijkstra 路径作为引导
- 实现拓扑地图记录已探索区域

---

## 5. 代码层面问题

### 5.1 可视化脚本冗余 [代码质量]

**问题描述**：存在功能重叠的可视化脚本。

| 文件 | 功能 | 状态 |
|------|------|------|
| `generate_simple_visualization.py` | 模拟数据 + Simple3DVisualizer | 可删除 |
| `generate_paper_visualization.py` | 模拟数据 + PaperQualityVisualizer | 保留 |
| `fly_with_simple_visualization.py` | AirSim飞行 + Simple3DVisualizer | 与 fly_planned_path.py 重叠 |

**建议**：保留 `generate_paper_visualization.py` 用于论文图表生成，其他可考虑删除或合并。

### 5.2 硬编码参数 [代码质量]

**问题描述**：部分参数硬编码在代码中，不便于调试。

**示例**：
```python
# mapping/incremental_map.py:154
ground_threshold = -0.5  # 应该放入 PlanningConfig

# mapping/incremental_map.py:158
drone_protection_radius = 2.0  # 应该放入 PlanningConfig
```

---

## 6. 性能相关

### 6.1 当前性能基准

| 模块 | 耗时 | 代码位置 |
|------|------|----------|
| 深度图转点云 | ~5ms | `utils/transforms.py:65-119` |
| 点云累积 | ~15ms | `mapping/incremental_map.py:134-187` |
| ESDF 计算 | ~40ms | `mapping/esdf.py` |
| RRT* 规划 | ~50ms | `planning/rrt_star.py` |
| **总周期** | **~100-150ms** | - |

### 6.2 性能瓶颈

- ESDF 计算使用 `scipy.ndimage.distance_transform_edt`，对大栅格较慢
- RRT* 迭代次数固定为 3000，可能过多或过少

---

## 优先级排序

| 优先级 | 问题 | 影响程度 | 修复难度 |
|--------|------|----------|----------|
| **P0** | 地面误识别 | 高 - 导致规划失败 | 中 - 需要排查坐标变换 |
| P1 | RRT* 局部最优 | 中 - 复杂环境失败 | 高 - 需要算法改进 |
| P2 | 单相机盲区 | 中 - 侧向障碍物 | 高 - 需要硬件/仿真支持 |
| P3 | 静态环境假设 | 低 - 仿真环境静态 | 中 - 需要动态更新机制 |
| P4 | 代码冗余 | 低 - 不影响功能 | 低 - 删除即可 |

---

## 更新日志

- **2026-01-21**: 初始版本，记录所有已知问题
