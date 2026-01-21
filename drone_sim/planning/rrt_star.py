"""
RRT* - Rapidly-exploring Random Tree Star algorithm
"""

import numpy as np
from typing import List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from mapping.voxel_grid import VoxelGrid
    from mapping.esdf import ESDF
    from .config import PlanningConfig


class RRTStar:
    """
    RRT* 路径规划算法
    改进版：局部采样 + 偏向采样 + 更好的绕障能力
    """

    def __init__(self, voxel_grid: 'VoxelGrid', esdf: 'ESDF', config: 'PlanningConfig'):
        self.voxel_grid = voxel_grid
        self.esdf = esdf
        self.config = config

        # 规划空间边界
        self.bounds_min = np.array(config.origin)
        self.bounds_max = self.bounds_min + np.array(config.grid_size) * config.voxel_size

        # 存储搜索树边（用于可视化）
        self.tree_edges = []

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

            # 选择最优父节点
            min_cost = costs[nearest_idx] + np.linalg.norm(new_point - nearest)
            min_parent = nearest_idx

            for near_idx in near_indices:
                if near_idx == new_idx:
                    continue
                near_node = nodes[near_idx]
                if self._is_collision_free(near_node, new_point):
                    new_cost = costs[near_idx] + np.linalg.norm(new_point - near_node)
                    if new_cost < min_cost:
                        min_cost = new_cost
                        min_parent = near_idx

            parents[new_idx] = min_parent
            costs[new_idx] = min_cost

            # 重连接附近节点
            for near_idx in near_indices:
                if near_idx == new_idx:
                    continue
                near_node = nodes[near_idx]
                if self._is_collision_free(new_point, near_node):
                    new_cost = costs[new_idx] + np.linalg.norm(near_node - new_point)
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
                    costs[goal_idx] = costs[new_idx] + np.linalg.norm(goal - new_point)
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

        return path

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
        """
        r = np.random.random()

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
            sample[2] = start[2]  # 保持高度

            # 确保在边界内
            sample = np.clip(sample, self.local_bounds_min, self.local_bounds_max)
            return sample

        else:
            # 局部随机采样
            sample = np.random.uniform(self.local_bounds_min, self.local_bounds_max)
            # 偏向保持当前高度
            if np.random.random() < 0.7:
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
        """检查点是否有效（在边界内且不在障碍物中）"""
        if not all(self.bounds_min[i] <= point[i] <= self.bounds_max[i] for i in range(3)):
            return False
        return self.esdf.is_safe(point, self.config.safety_margin)

    def _is_collision_free(self, from_point: np.ndarray, to_point: np.ndarray) -> bool:
        """检查路径段是否无碰撞"""
        direction = to_point - from_point
        distance = np.linalg.norm(direction)
        if distance < 1e-6:
            return True

        # 沿路径采样检查
        num_checks = max(2, int(distance / (self.config.voxel_size * 0.5)))
        for i in range(num_checks + 1):
            t = i / num_checks
            point = from_point + t * direction
            if not self.esdf.is_safe(point, self.config.safety_margin):
                return False
        return True

    def _smooth_path(self, path: List[np.ndarray]) -> List[np.ndarray]:
        """简单的路径平滑"""
        if len(path) <= 2:
            return path

        smoothed = [path[0]]
        i = 0
        while i < len(path) - 1:
            # 尝试跳过中间点
            j = len(path) - 1
            while j > i + 1:
                if self._is_collision_free(path[i], path[j]):
                    break
                j -= 1
            smoothed.append(path[j])
            i = j

        return smoothed
