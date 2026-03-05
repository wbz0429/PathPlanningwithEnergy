"""
RRT* - Rapidly-exploring Random Tree Star algorithm
支持能量感知的路径规划
"""

import numpy as np
from typing import List, Optional, TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from mapping.voxel_grid import VoxelGrid
    from mapping.esdf import ESDF
    from .config import PlanningConfig


class EnergyAwareCostFunction:
    """
    能量感知代价函数

    Cost = w_e × (energy/E_ref) + w_d × (dist/D_ref) + w_t × (time/T_ref)
    """

    def __init__(self, config: 'PlanningConfig', energy_model=None):
        """
        初始化代价函数

        Args:
            config: 规划配置
            energy_model: 能耗模型（PhysicsEnergyModel 或 HybridEnergyModel）
        """
        self.config = config
        self.energy_model = energy_model

        # 如果没有提供能耗模型，尝试导入默认的物理模型
        if self.energy_model is None and config.energy_aware:
            try:
                from energy.physics_model import PhysicsEnergyModel
                self.energy_model = PhysicsEnergyModel()
            except ImportError:
                print("  [Warning] Energy model not available, using distance-only cost")

    def compute_cost(self, from_point: np.ndarray, to_point: np.ndarray) -> float:
        """
        计算两点之间的代价

        Args:
            from_point: 起点
            to_point: 终点

        Returns:
            归一化加权代价
        """
        # 计算距离
        distance = np.linalg.norm(to_point - from_point)

        if distance < 1e-6:
            return 0.0

        # 如果不启用能量感知或没有能耗模型，返回纯距离代价
        if not self.config.energy_aware or self.energy_model is None:
            return distance

        # 计算时间
        time = distance / self.config.flight_velocity

        # 计算能耗
        energy, _ = self.energy_model.compute_energy_for_segment(
            from_point, to_point, self.config.flight_velocity
        )

        # 归一化并加权
        cost = (
            self.config.weight_energy * (energy / self.config.energy_ref) +
            self.config.weight_distance * (distance / self.config.distance_ref) +
            self.config.weight_time * (time / self.config.time_ref)
        )

        return cost

    def compute_path_cost(self, path: List[np.ndarray]) -> float:
        """计算整条路径的代价"""
        total_cost = 0.0
        for i in range(len(path) - 1):
            total_cost += self.compute_cost(path[i], path[i + 1])
        return total_cost


class RRTStar:
    """
    RRT* 路径规划算法
    改进版：局部采样 + 偏向采样 + 能量感知代价 + 更好的绕障能力
    """

    def __init__(self, voxel_grid: 'VoxelGrid', esdf: 'ESDF', config: 'PlanningConfig',
                 energy_model=None):
        """
        初始化 RRT* 规划器

        Args:
            voxel_grid: 体素栅格地图
            esdf: ESDF 距离场
            config: 规划配置
            energy_model: 能耗模型（可选），用于能量感知规划
        """
        self.voxel_grid = voxel_grid
        self.esdf = esdf
        self.config = config

        # 规划空间边界
        self.bounds_min = np.array(config.origin)
        self.bounds_max = self.bounds_min + np.array(config.grid_size) * config.voxel_size

        # 存储搜索树边（用于可视化）
        self.tree_edges = []

        # 能量感知代价函数
        self.cost_function = EnergyAwareCostFunction(config, energy_model)

        # 记录规划统计信息
        self.last_plan_stats = {}

    def plan(self, start: np.ndarray, goal: np.ndarray) -> Optional[List[np.ndarray]]:
        """
        规划从 start 到 goal 的路径

        Args:
            start: 起点坐标
            goal: 终点坐标

        Returns:
            路径点列表，如果规划失败返回 None
        """
        # 确保 ESDF 已计算
        if self.esdf.distance_field is None:
            self.esdf.compute()

        # 清空搜索树
        self.tree_edges = []

        # 检查起点 - 使用更宽松的检查，允许从危险位置逃离
        start_safe = self._is_valid_point(start)
        if not start_safe:
            start_dist = self.esdf.get_distance(start)
            # 如果起点只是略微不安全（在障碍物边缘），仍然尝试规划
            if start_dist > 0:  # 不在障碍物内部
                print(f"  [RRT*] Start marginally unsafe (dist={start_dist:.2f}m), attempting escape planning")
            else:
                print(f"  [RRT*] Start inside obstacle (dist={start_dist:.2f}m)")
                return None

        # 检查终点 - 使用更宽松的检查
        goal_safe = self._is_valid_point(goal)
        if not goal_safe:
            print(f"  [RRT*] Goal in obstacle, will find nearest safe point")

        # 计算局部采样范围（以起点和终点为中心）
        self._compute_sampling_bounds(start, goal)

        # 初始化树
        nodes = [start.copy()]
        parents = {0: -1}
        costs = {0: 0.0}

        goal_idx = None
        best_dist_to_goal = np.inf
        best_node_idx = 0

        for i in range(self.config.max_iterations):
            # 智能采样
            sample = self._smart_sample(start, goal, nodes, i)

            # 找最近节点
            nearest_idx = self._nearest_node(nodes, sample)
            nearest = nodes[nearest_idx]

            # 向采样点扩展
            new_point = self._steer(nearest, sample)

            # 碰撞检测 - 对于起点不安全的情况，放宽第一步的检查
            if not start_safe and len(nodes) == 1:
                # 第一步：只检查新点是否安全，不检查路径
                if not self._is_valid_point(new_point):
                    continue
            else:
                if not self._is_collision_free(nearest, new_point):
                    continue

            # RRT* 重连接
            new_idx = len(nodes)
            nodes.append(new_point)

            # 记录边（用于可视化）
            self.tree_edges.append((nearest.copy(), new_point.copy()))

            # 找附近节点
            near_indices = self._near_nodes(nodes, new_point)

            # 选择最优父节点 - 使用能量感知代价
            min_cost = costs[nearest_idx] + self.cost_function.compute_cost(nearest, new_point)
            min_parent = nearest_idx

            for near_idx in near_indices:
                if near_idx == new_idx:
                    continue
                near_node = nodes[near_idx]
                if self._is_collision_free(near_node, new_point):
                    new_cost = costs[near_idx] + self.cost_function.compute_cost(near_node, new_point)
                    if new_cost < min_cost:
                        min_cost = new_cost
                        min_parent = near_idx

            parents[new_idx] = min_parent
            costs[new_idx] = min_cost

            # 重连接附近节点 - 使用能量感知代价
            for near_idx in near_indices:
                if near_idx == new_idx:
                    continue
                near_node = nodes[near_idx]
                if self._is_collision_free(new_point, near_node):
                    new_cost = costs[new_idx] + self.cost_function.compute_cost(new_point, near_node)
                    if new_cost < costs[near_idx]:
                        parents[near_idx] = new_idx
                        costs[near_idx] = new_cost

            # 记录最接近目标的节点
            dist_to_goal = np.linalg.norm(new_point - goal)
            if dist_to_goal < best_dist_to_goal:
                best_dist_to_goal = dist_to_goal
                best_node_idx = new_idx

            # 检查是否到达目标
            if dist_to_goal < self.config.step_size * 1.5:
                if goal_safe and self._is_collision_free(new_point, goal):
                    goal_idx = len(nodes)
                    nodes.append(goal.copy())
                    parents[goal_idx] = new_idx
                    costs[goal_idx] = costs[new_idx] + self.cost_function.compute_cost(new_point, goal)
                    print(f"  [RRT*] Path found! Iterations: {i+1}")
                    break
                elif not goal_safe and dist_to_goal < self.config.step_size * 2:
                    # 目标不安全，但已经足够接近
                    goal_idx = new_idx
                    print(f"  [RRT*] Reached near goal (dist={dist_to_goal:.2f}m), iterations: {i+1}")
                    break

        # 如果没找到完整路径，但有接近目标的节点
        if goal_idx is None:
            if best_dist_to_goal < self.config.step_size * 3:
                goal_idx = best_node_idx
                print(f"  [RRT*] Using best node (dist to goal: {best_dist_to_goal:.2f}m)")
            else:
                print(f"  [RRT*] Failed after {self.config.max_iterations} iterations (best dist: {best_dist_to_goal:.2f}m)")
                return None

        # 回溯路径
        path = []
        idx = goal_idx
        while idx != -1:
            path.append(nodes[idx])
            idx = parents[idx]
        path.reverse()

        # 路径平滑
        path = self._smooth_path(path)

        # 记录规划统计信息
        self._compute_plan_stats(path)

        return path

    def _compute_plan_stats(self, path: List[np.ndarray]):
        """计算并记录规划统计信息"""
        if path is None or len(path) < 2:
            self.last_plan_stats = {}
            return

        # 计算路径总距离
        total_distance = sum(
            np.linalg.norm(path[i+1] - path[i])
            for i in range(len(path) - 1)
        )

        # 计算路径总代价
        total_cost = self.cost_function.compute_path_cost(path)

        # 计算能耗（如果有能耗模型）
        total_energy = 0.0
        total_time = 0.0
        if self.cost_function.energy_model is not None:
            for i in range(len(path) - 1):
                energy, time = self.cost_function.energy_model.compute_energy_for_segment(
                    path[i], path[i+1], self.config.flight_velocity
                )
                total_energy += energy
                total_time += time

        self.last_plan_stats = {
            'path_length': len(path),
            'total_distance': total_distance,
            'total_cost': total_cost,
            'total_energy_joules': total_energy,
            'total_time_seconds': total_time,
            'energy_aware': self.config.energy_aware
        }

    def get_plan_stats(self) -> dict:
        """获取上次规划的统计信息"""
        return self.last_plan_stats

    def _compute_sampling_bounds(self, start: np.ndarray, goal: np.ndarray):
        """计算局部采样范围"""
        # 以起点和终点的中心为基准，扩展一定范围
        center = (start + goal) / 2
        dist = np.linalg.norm(goal - start)
        margin = max(dist * 0.8, 5.0)  # 至少5米的采样范围

        self.local_bounds_min = np.maximum(
            self.bounds_min,
            center - margin
        )
        self.local_bounds_max = np.minimum(
            self.bounds_max,
            center + margin
        )

        # 确保Z轴范围合理（保持在当前高度附近）
        z_margin = 3.0
        self.local_bounds_min[2] = max(self.bounds_min[2], min(start[2], goal[2]) - z_margin)
        self.local_bounds_max[2] = min(self.bounds_max[2], max(start[2], goal[2]) + z_margin)

    def _smart_sample(self, start: np.ndarray, goal: np.ndarray,
                      nodes: List[np.ndarray], iteration: int) -> np.ndarray:
        """
        智能采样策略：
        - 20% 概率采样目标点
        - 30% 概率在目标方向两侧采样（用于绕障）
        - 50% 概率在局部范围内随机采样

        能量感知模式下会更多探索不同高度
        """
        r = np.random.random()

        # 能量感知模式下，减少高度锁定的概率
        height_lock_prob = 0.3 if self.config.energy_aware else 0.7

        if r < 0.2:
            # 直接采样目标
            return goal.copy()

        elif r < 0.5:
            # 在目标方向的两侧采样（帮助绕过障碍物）
            direction = goal - start
            dist = np.linalg.norm(direction)
            if dist > 0:
                direction = direction / dist

            # 随机选择左侧或右侧
            angle = np.random.uniform(30, 90) * (1 if np.random.random() > 0.5 else -1)
            angle_rad = np.radians(angle)

            # 在XY平面旋转
            cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
            rotated_dir = np.array([
                cos_a * direction[0] - sin_a * direction[1],
                sin_a * direction[0] + cos_a * direction[1],
                direction[2]
            ])

            # 随机距离
            sample_dist = np.random.uniform(1.0, min(dist, self.config.step_size * 4))
            sample = start + rotated_dir * sample_dist

            # 能量感知模式：有概率探索不同高度
            if np.random.random() < height_lock_prob:
                sample[2] = start[2]  # 保持高度
            else:
                # 随机高度变化（向上或向下）
                z_variation = np.random.uniform(-3.0, 3.0)
                sample[2] = np.clip(start[2] + z_variation,
                                   self.local_bounds_min[2],
                                   self.local_bounds_max[2])

            # 确保在边界内
            sample = np.clip(sample, self.local_bounds_min, self.local_bounds_max)
            return sample

        else:
            # 局部随机采样
            sample = np.random.uniform(self.local_bounds_min, self.local_bounds_max)
            # 根据模式决定高度锁定概率
            if np.random.random() < height_lock_prob:
                sample[2] = start[2]
            return sample

    def _random_sample(self) -> np.ndarray:
        """随机采样（在局部范围内）"""
        return np.random.uniform(self.local_bounds_min, self.local_bounds_max)

    def _nearest_node(self, nodes: List[np.ndarray], point: np.ndarray) -> int:
        """找最近节点"""
        distances = [np.linalg.norm(node - point) for node in nodes]
        return int(np.argmin(distances))

    def _near_nodes(self, nodes: List[np.ndarray], point: np.ndarray) -> List[int]:
        """找附近节点"""
        indices = []
        for i, node in enumerate(nodes):
            if np.linalg.norm(node - point) < self.config.search_radius:
                indices.append(i)
        return indices

    def _steer(self, from_point: np.ndarray, to_point: np.ndarray) -> np.ndarray:
        """向目标点扩展一步"""
        direction = to_point - from_point
        distance = np.linalg.norm(direction)
        if distance < self.config.step_size:
            return to_point.copy()
        return from_point + direction / distance * self.config.step_size

    def _is_valid_point(self, point: np.ndarray) -> bool:
        """
        检查点是否有效（在边界内且不在障碍物中）
        Unknown 区域处理：
        - ESDF距离 >= unknown_safe_threshold → 允许（远离已知障碍物，乐观通行）
        - ESDF距离 < unknown_safe_threshold → 拒绝（靠近已知障碍物，保守拒绝）
        """
        if not all(self.bounds_min[i] <= point[i] <= self.bounds_max[i] for i in range(3)):
            return False
        if not self.esdf.is_safe(point, self.config.safety_margin):
            return False
        idx = self.voxel_grid.world_to_grid(point)
        if self.voxel_grid.is_valid_index(idx):
            if self.voxel_grid.grid[idx] == 0:
                # unknown 区域 — 根据 ESDF 距离决定是否允许
                esdf_dist = self.esdf.get_distance(point)
                if esdf_dist < self.config.unknown_safe_threshold:
                    return False  # 靠近已知障碍物，保守拒绝
                # 远离已知障碍物，乐观允许通行
        return True

    def _is_collision_free(self, from_point: np.ndarray, to_point: np.ndarray) -> bool:
        """
        检查路径段是否无碰撞
        Unknown 区域：ESDF距离 >= threshold → 允许，否则拒绝
        """
        direction = to_point - from_point
        distance = np.linalg.norm(direction)
        if distance < 1e-6:
            return True

        # 沿路径采样检查（每 0.2m 采样一次，比体素尺寸更密集）
        check_interval = self.config.voxel_size * 0.4  # 0.2m
        num_checks = max(3, int(distance / check_interval))
        for i in range(num_checks + 1):
            t = i / num_checks
            point = from_point + t * direction
            # ESDF 安全检查
            if not self.esdf.is_safe(point, self.config.safety_margin):
                return False
            # unknown 区域检查：根据 ESDF 距离决定
            idx = self.voxel_grid.world_to_grid(point)
            if self.voxel_grid.is_valid_index(idx):
                if self.voxel_grid.grid[idx] == 0:
                    esdf_dist = self.esdf.get_distance(point)
                    if esdf_dist < self.config.unknown_safe_threshold:
                        return False
        return True

    def _smooth_path(self, path: List[np.ndarray]) -> List[np.ndarray]:
        """
        路径平滑：先适度简化，再 B-spline 拟合
        """
        if len(path) <= 2:
            return path

        # 第一步：适度简化（保留更多中间点用于 B-spline）
        # 不是贪心跳到最远，而是每次跳固定距离
        simplified = [path[0]]
        i = 0
        max_skip_dist = 10.0  # 每次最多跳10米

        while i < len(path) - 1:
            # 从远到近找第一个可达且距离不超过 max_skip_dist 的点
            best_j = i + 1
            for j in range(len(path) - 1, i, -1):
                dist = np.linalg.norm(path[j] - path[i])
                if dist <= max_skip_dist and self._is_collision_free(path[i], path[j]):
                    best_j = j
                    break
            simplified.append(path[best_j])
            i = best_j

        # 第二步：B-spline 平滑（如果简化后还有足够的点）
        if len(simplified) >= 4:
            try:
                smoothed = self._bspline_smooth(simplified)
                return smoothed
            except Exception as e:
                # B-spline 失败则返回简化路径
                print(f"  [RRT*] B-spline failed: {e}, using simplified path")
                return simplified
        else:
            # 点太少，直接返回简化路径
            return simplified

    def _bspline_smooth(self, path: List[np.ndarray], num_samples: int = None) -> List[np.ndarray]:
        """
        使用 B-spline 平滑路径

        Args:
            path: 原始路径点
            num_samples: 采样点数量（默认为原路径长度的2倍）

        Returns:
            平滑后的路径
        """
        from scipy.interpolate import splprep, splev

        if len(path) < 4:
            return path

        # 转换为数组
        path_array = np.array(path)

        # 计算路径总长度
        path_length = sum(
            np.linalg.norm(path[i+1] - path[i])
            for i in range(len(path) - 1)
        )

        # 采样点数量：每0.3m一个点
        if num_samples is None:
            num_samples = max(len(path) * 2, int(path_length / 0.3))

        # B-spline 拟合（k=3 表示三次样条）
        # s=0 表示精确通过所有控制点，s>0 允许偏差
        tck, u = splprep([path_array[:, 0], path_array[:, 1], path_array[:, 2]],
                         s=0.1, k=min(3, len(path) - 1))

        # 在 [0, 1] 区间均匀采样
        u_new = np.linspace(0, 1, num_samples)
        smooth_points = splev(u_new, tck)

        # 转换回列表格式
        smoothed = [np.array([x, y, z]) for x, y, z in zip(*smooth_points)]

        # 碰撞检查：如果平滑后的路径有碰撞，回退到原路径
        for i in range(len(smoothed) - 1):
            if not self._is_collision_free(smoothed[i], smoothed[i+1]):
                print("  [RRT*] B-spline smoothing caused collision, using simplified path")
                return path

        return smoothed
