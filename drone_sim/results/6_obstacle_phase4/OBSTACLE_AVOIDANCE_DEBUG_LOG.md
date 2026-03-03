# 障碍物避障调试日志

> 时间：2025-02-25
> 背景：Phase 2 滚动规划架构已完成，Phase 3 能量感知规划已实现，现在在 AirSim Blocks 场景中测试真实避障飞行

---

## 1. 任务描述

- 起点：X=0, Y=0, Z=-3（NED 坐标，Z=-3 表示 3m 高度）
- 目标：X=35, Y=0, Z=-3（穿过建筑群到达对面）
- 场景：Blocks 场景，三排建筑物
  - Row 1 (X=23~33)：实心墙，宽约 40m（Y=-21.5 到 18.5）
  - Row 2 (X=33~43)：有 20m 宽缝隙（Y=-11.5 到 8.5）
  - Row 3 (X=43~53)：同上
  - OrangeBall 在 Y=33 附近，需避开
- 感知：单个前向深度相机，90° FOV

---

## 2. 核心问题链

### 问题 A：感知盲区（根本原因）

无人机从 X=0 出发，到达 X≈16~22 时面对 Row 1 的墙。此时：
- 距墙约 1~7m，90° FOV 只能看到正前方一小片墙面
- 墙宽 40m，但相机只能看到 ±几米的范围
- 墙面大部分区域在地图中标记为 **unknown（grid=0）**，不是 occupied（grid=1）

### 问题 B：ESDF 把 unknown 当 free

`esdf.py` 第 33-35 行：
```python
# 空闲区域：包括明确标记为空闲的(-1)和未知的(0)
free = (grid != 1).astype(np.float32)
```
→ unknown 区域的 ESDF 距离值正常（>safety_margin），RRT* 认为可以安全穿越

### 问题 C：RRT* 穿越 unknown 区域

RRT* 的 `_is_valid_point` 和 `_is_collision_free` 只检查 ESDF 距离，不检查体素状态
→ 规划出的路径直接穿过未探索的墙面

### 问题 D：碰撞检测拦截 → 死循环

执行路径时，碰撞检测发现路径不安全（因为飞近了能看到墙了），拦截执行
→ 下一轮 RRT* 还是规划穿墙路径 → 又被拦截 → 无限循环

### 问题 E：Z 坐标漂移

碰撞拦截后调用 `drone.hover()`，无人机悬停但 Z 不断上升：
- 从 -2.94 → -2.50 → -2.15 → -1.43 → -0.71 → 0.02（地面）
- 原因：AirSim 的 hover 行为 + 没有主动高度修正

### 问题 F：wall-follow 来回摇摆

加了 wall-follow 策略后，每次迭代重新评估侧移方向：
- Iteration 8：选 left，移到 Y=9
- Iteration 9：选 right，移回 Y=5.9
- 来回摇摆，无法持续朝一个方向走到墙边缘

---

## 3. 已实施的修复

### 修复 1：Yaw 扫描（解决感知盲区）

**文件**：`planning/receding_horizon.py` → `_yaw_scan()`

当前方被墙挡住时，原地转头扫描 5 个角度（-90°, -45°, 0°, +45°, +90°），覆盖约 270°。每个角度拍摄深度图并更新地图。

**效果**：扫描后 occupied voxels 从 ~1800 增加到 ~2500，但仍然只能看到有限范围的墙面。

### 修复 2：Wall-follow 策略（绕墙）

**文件**：`planning/receding_horizon.py` → `_wall_follow_step()`

当前方被墙挡住时，不用 RRT*，直接沿墙壁平行移动：
- 计算垂直于前进方向的左右两个方向
- 评估哪个方向更快找到墙边缘
- 生成侧移路径（3~5m 一步）

### 修复 3：Wall-follow 方向记忆（防摇摆）

**文件**：`planning/receding_horizon.py`

- 新增 `_wall_follow_direction` 状态变量
- 首次选择方向后记住，后续迭代继续使用同一方向
- 只有在真正取得前进进度（`_compute_forward_progress` > best + 1m）时才清除记忆
- 当前方向走不通时才重新选择

### 修复 4：RRT* 拒绝 unknown 区域（核心修复）

**文件**：`planning/rrt_star.py`

在 `_is_valid_point()` 和 `_is_collision_free()` 中增加体素状态检查：
```python
idx = self.voxel_grid.world_to_grid(point)
if self.voxel_grid.is_valid_index(idx):
    if self.voxel_grid.grid[idx] == 0:  # unknown
        return False  # 不允许通行
```
→ RRT* 只能在已确认空闲（grid==-1）的区域规划路径

### 修复 5：Z 高度自动修正

**文件**：`planning/receding_horizon.py`

每次迭代开始时检查 Z 偏差：
```python
if abs(current_pos[2] - global_goal[2]) > 0.5:
    fix_pos = current_pos.copy()
    fix_pos[2] = global_goal[2]
    self.drone.move_to_position(fix_pos, velocity=self.flight_velocity)
```

### 修复 6：卡住检测

**文件**：`planning/receding_horizon.py` → `_detect_stuck()`

- 记录最近 6 次迭代的位置
- 如果总移动距离 > 8m 但净位移 < 3m → 判定为来回摇摆
- 如果总移动距离 < 1m → 判定为原地卡住

### 修复 7：其他参数调整

| 参数 | 原值 | 新值 | 原因 |
|------|------|------|------|
| safety_margin | 2.0m | 1.0m | 无人机起飞后距墙 1.8m，margin=2.0 导致起点就"在障碍物内" |
| drone_protection_radius | 2.0m | 0.8m | 地图清除半径太大，把附近的墙也清掉了 |
| execution_ratio | 0.5 | 0.3 | 每次只执行 30% 路径，更频繁地重新规划 |
| flight_velocity | 2.0 | 1.5 m/s | 降低速度，给碰撞检测更多反应时间 |
| 目标 Y 坐标 | 25.0 | 0.0 | Y=25 会撞 OrangeBall，Y=0 利用 Row2/3 的缝隙 |

---

## 4. 两次飞行测试结果

### 测试 1（修复 1-2 后）

- 无人机到达 X≈22, Y≈8 附近
- wall-follow 来回摇摆（left → right → left）
- Z 从 -2.94 漂到 0.02
- RRT* 反复规划穿墙路径被拦截
- 26 个航点，95m 总距离，380 秒后超时失败

### 测试 2（修复 3 后，方向记忆）

- wall-follow 方向记忆生效，持续向 left 移动
- 到达 Y≈15-18（接近墙边缘 Y=18.5）
- 但 Iteration 6 时 RRT* 接管（2454 次迭代才找到路径），规划了一条绕到 Y=17.6 的路径
- 之后又陷入 X≈22, Y≈8-13 区域的死循环
- wall-follow 方向被重置后重新选择，又开始摇摆
- Z 漂移问题依然存在
- 33 次迭代后仍在 X≈22 附近打转

---

## 5. 当前状态（未测试）

修复 4-6 已实现但尚未测试。关键改动是 **RRT* 拒绝 unknown 区域**。

---

## 6. 下一步尝试思路

### 思路 A：测试当前修复（优先）

直接跑一次看 unknown 区域检查的效果。预期行为：
- wall-follow 持续向一个方向移动
- yaw 扫描逐步揭示墙的边缘
- RRT* 只在已探索的空闲空间规划，不再穿墙
- 到达墙边缘后，RRT* 规划绕过去的路径

**潜在风险**：unknown 检查可能太严格，导致 RRT* 在已探索区域内也找不到路径（因为 yaw 扫描覆盖有限，很多区域还是 unknown）。如果出现这种情况，需要放宽检查条件。

### 思路 B：Bug2 算法（如果思路 A 失败）

参考经典 Bug2 算法，实现状态机：
1. **GO_TO_GOAL**：直线飞向目标
2. **WALL_FOLLOW**：碰到墙后沿墙面移动，保持固定偏移
3. **LEAVE_WALL**：当再次穿过起点-目标连线（M-line）且比上次更近目标时，切回 GO_TO_GOAL

Bug2 的优势：
- 数学上保证能到达目标（如果路径存在）
- 不依赖全局地图，只需要局部感知
- 不会陷入 RRT* 的 unknown 区域问题

实现要点：
- 沿墙移动时保持 1-2m 的固定偏移距离
- 用深度相机实时检测墙面距离
- M-line 判断：检查当前位置是否在起点到目标的直线上

### 思路 C：Frontier-based 探索（更复杂但更通用）

在 voxel grid 中找 frontier（空闲体素与 unknown 体素的边界），优先探索朝目标方向的 frontier：
1. 检测所有 frontier 体素
2. 聚类成 frontier 区域
3. 选择最佳 frontier：`score = goal_proximity × 0.7 + info_gain × 0.3`
4. 导航到 frontier → 探索 → 重复直到目标可达

### 思路 D：放宽 unknown 检查（折中方案）

如果完全拒绝 unknown 太严格，可以改为：
- 距离已知 occupied 体素 3m 以内的 unknown → 视为 occupied
- 距离已知 occupied 体素 3m 以外的 unknown → 视为 free（乐观假设）
- 这样 RRT* 不会穿越墙面附近的 unknown，但可以穿越远处的 unknown

---

## 7. 关键文件清单

| 文件 | 作用 | 本次改动 |
|------|------|----------|
| `planning/receding_horizon.py` | 主控制循环 | wall-follow、方向记忆、Z修正、卡住检测 |
| `planning/rrt_star.py` | RRT* 路径规划 | unknown 区域检查 |
| `planning/config.py` | 配置参数 | safety_margin 等调整 |
| `mapping/esdf.py` | ESDF 距离场 | 未改（unknown 当 free 的根源在这里） |
| `mapping/voxel_grid.py` | 体素栅格 | 未改（grid: 0=unknown, 1=occupied, -1=free） |
| `mapping/incremental_map.py` | 地图管理 | protection_radius 调整 |
| `control/drone_interface.py` | AirSim 接口 | 新增 set_yaw() |
| `fly_planned_path.py` | 飞行入口脚本 | 目标改为 Y=0 |

---

## 8. 场景几何参考

```
Y轴 (East)
  ^
  |
  |  OrangeBall (Y≈33)
18.5 +---------+---------+---------+
  |  | Row 1   | Row 2   | Row 3   |
  |  | X=23~33 | X=33~43 | X=43~53 |
8.5  |  |         +----+----+----+----+
  |  |              GAP (20m wide)
-11.5|  |         +----+----+----+----+
  |  |         | Row 2   | Row 3   |
-21.5+---------+---------+---------+
  |
  +--+----+----+----+----+----+----> X轴 (North)
     0   10   20   30   40   50

起点: (0, 0, -3)
目标: (35, 0, -3)
```

Row 1 是实心墙（Y=-21.5 到 18.5），必须绕过边缘。
Row 2/3 中间有 20m 缝隙（Y=-11.5 到 8.5），目标 Y=0 正好在缝隙中间。
关键挑战：先绕过 Row 1 的墙，然后穿过 Row 2/3 的缝隙。

---

## 9. 调试命令

```bash
# 重置 AirSim 无人机
python -c "import airsim; c=airsim.MultirotorClient(); c.confirmConnection(); c.reset()"

# 运行飞行测试
python fly_planned_path.py

# 不连 AirSim 的离线测试（用 ground truth 地图）
python test_mapping_and_planning.py

# 查询场景物体位置
python query_scene_objects.py
```
