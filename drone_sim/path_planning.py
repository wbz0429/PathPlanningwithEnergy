"""
path_planning.py - 基于 OctoMap + ESDF + RRT* 的路径规划模块
可行性验证版本

流程: 深度图 → 点云 → 3D体素栅格(OctoMap) → ESDF距离场 → RRT*路径规划
"""

import numpy as np
from scipy import ndimage
from dataclasses import dataclass
from typing import List, Tuple, Optional
import heapq


@dataclass
class PlanningConfig:
    """规划配置参数"""
    # 体素栅格参数
    voxel_size: float = 0.5          # 体素大小（米）
    grid_size: Tuple[int, int, int] = (80, 80, 40)  # 栅格尺寸 (x, y, z)
    origin: Tuple[float, float, float] = (-20.0, -20.0, 0.0)  # 栅格原点

    # 相机参数
    fov_deg: float = 90.0            # 视场角
    max_depth: float = 25.0          # 最大有效深度

    # RRT* 参数
    step_size: float = 1.0           # RRT 步长
    max_iterations: int = 3000       # 最大迭代次数
    goal_sample_rate: float = 0.1    # 目标采样概率
    search_radius: float = 3.0       # 重连接搜索半径

    # 安全参数
    safety_margin: float = 1.0       # 安全边距（米）


class VoxelGrid:
    """
    简化版 3D 体素栅格地图（OctoMap 简化实现）
    使用 numpy 3D 数组代替八叉树
    """

    def __init__(self, config: PlanningConfig):
        self.config = config
        self.voxel_size = config.voxel_size
        self.grid_size = config.grid_size
        self.origin = np.array(config.origin)

        # 初始化栅格: 0=未知, 1=占据, -1=空闲
        self.grid = np.zeros(config.grid_size, dtype=np.int8)

        # 相机内参
        self.fov_rad = np.radians(config.fov_deg)

    def world_to_grid(self, point: np.ndarray) -> Tuple[int, int, int]:
        """世界坐标转栅格索引"""
        idx = ((point - self.origin) / self.voxel_size).astype(int)
        return tuple(idx)

    def grid_to_world(self, idx: Tuple[int, int, int]) -> np.ndarray:
        """栅格索引转世界坐标（体素中心）"""
        return self.origin + (np.array(idx) + 0.5) * self.voxel_size

    def is_valid_index(self, idx: Tuple[int, int, int]) -> bool:
        """检查索引是否有效"""
        return all(0 <= idx[i] < self.grid_size[i] for i in range(3))

    def update_from_pointcloud(self, points: np.ndarray, camera_pos: np.ndarray):
        """
        从点云更新栅格地图

        Args:
            points: Nx3 点云数组（世界坐标）
            camera_pos: 相机位置
        """
        for point in points:
            idx = self.world_to_grid(point)
            if self.is_valid_index(idx):
                self.grid[idx] = 1  # 标记为占据

        # 简化处理：将相机到点云之间的体素标记为空闲
        # 这里用简化的方法，实际应该用 ray casting

    def update_from_depth_image(self, depth_image: np.ndarray,
                                 camera_pos: np.ndarray,
                                 camera_rotation: np.ndarray = None):
        """
        从深度图更新栅格地图

        Args:
            depth_image: HxW 深度图（米）
            camera_pos: 相机世界坐标位置
            camera_rotation: 相机旋转矩阵（可选）
        """
        h, w = depth_image.shape

        # 计算相机内参
        fx = w / (2 * np.tan(self.fov_rad / 2))
        fy = fx
        cx, cy = w / 2, h / 2

        # 下采样以提高效率
        subsample = 4
        depth_sub = depth_image[::subsample, ::subsample]
        h_sub, w_sub = depth_sub.shape

        # 生成像素坐标网格
        u = np.arange(0, w, subsample)
        v = np.arange(0, h, subsample)
        u, v = np.meshgrid(u, v)

        # 展平
        z = depth_sub.flatten()
        u = u.flatten()
        v = v.flatten()

        # 过滤有效深度
        valid = (z > 0) & (z < self.config.max_depth)
        z = z[valid]
        u = u[valid]
        v = v[valid]

        if len(z) == 0:
            return

        # 计算相机坐标系下的 3D 点
        x_cam = (u - cx) * z / fx
        y_cam = (v - cy) * z / fy
        z_cam = z

        # 转换到世界坐标
        # AirSim 坐标系: X-前, Y-右, Z-下
        # 这里假设相机朝向 X 正方向
        if camera_rotation is not None:
            points_cam = np.stack([z_cam, x_cam, y_cam], axis=1)
            points_world = (camera_rotation @ points_cam.T).T + camera_pos
        else:
            # 简化：假设相机朝向 X 正方向
            points_world = np.stack([
                camera_pos[0] + z_cam,      # X = 前方深度
                camera_pos[1] + x_cam,      # Y = 左右
                camera_pos[2] + y_cam       # Z = 上下
            ], axis=1)

        # 更新栅格
        occupied_count = 0
        for point in points_world:
            idx = self.world_to_grid(point)
            if self.is_valid_index(idx):
                self.grid[idx] = 1
                occupied_count += 1

        return occupied_count

    def is_occupied(self, point: np.ndarray) -> bool:
        """检查某点是否被占据"""
        idx = self.world_to_grid(point)
        if not self.is_valid_index(idx):
            return True  # 超出边界视为占据
        return self.grid[idx] == 1

    def get_occupied_voxels(self) -> np.ndarray:
        """获取所有占据体素的世界坐标"""
        occupied_indices = np.argwhere(self.grid == 1)
        if len(occupied_indices) == 0:
            return np.array([])
        return np.array([self.grid_to_world(tuple(idx)) for idx in occupied_indices])


class ESDF:
    """
    欧几里得符号距离场 (Euclidean Signed Distance Field)
    使用 scipy 的距离变换实现
    """

    def __init__(self, voxel_grid: VoxelGrid):
        self.voxel_grid = voxel_grid
        self.distance_field = None
        self.gradient_field = None

    def compute(self):
        """计算 ESDF"""
        grid = self.voxel_grid.grid
        voxel_size = self.voxel_grid.voxel_size

        # 创建二值占据图
        occupied = (grid == 1).astype(np.float32)
        free = 1 - occupied

        # 计算到最近障碍物的距离
        # distance_transform_edt 计算每个点到最近 0 值点的距离
        dist_to_obstacle = ndimage.distance_transform_edt(free) * voxel_size

        # 计算到最近空闲区域的距离（用于障碍物内部）
        dist_to_free = ndimage.distance_transform_edt(occupied) * voxel_size

        # 符号距离场：空闲区域为正，障碍物内部为负
        self.distance_field = dist_to_obstacle - dist_to_free

        # 计算梯度（用于势场法等）
        self.gradient_field = np.gradient(self.distance_field, voxel_size)

        return self.distance_field

    def get_distance(self, point: np.ndarray) -> float:
        """获取某点到最近障碍物的距离"""
        if self.distance_field is None:
            self.compute()

        idx = self.voxel_grid.world_to_grid(point)
        if not self.voxel_grid.is_valid_index(idx):
            return -1.0  # 超出边界
        return self.distance_field[idx]

    def get_gradient(self, point: np.ndarray) -> np.ndarray:
        """获取某点的距离场梯度"""
        if self.gradient_field is None:
            self.compute()

        idx = self.voxel_grid.world_to_grid(point)
        if not self.voxel_grid.is_valid_index(idx):
            return np.zeros(3)
        return np.array([g[idx] for g in self.gradient_field])

    def is_safe(self, point: np.ndarray, margin: float) -> bool:
        """检查某点是否安全（距离障碍物足够远）"""
        return self.get_distance(point) > margin


class RRTStar:
    """
    RRT* 路径规划算法
    """

    def __init__(self, voxel_grid: VoxelGrid, esdf: ESDF, config: PlanningConfig):
        self.voxel_grid = voxel_grid
        self.esdf = esdf
        self.config = config

        # 规划空间边界
        self.bounds_min = np.array(config.origin)
        self.bounds_max = self.bounds_min + np.array(config.grid_size) * config.voxel_size

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

        # 检查起点和终点
        if not self._is_valid_point(start):
            print(f"起点无效: {start}")
            return None
        if not self._is_valid_point(goal):
            print(f"终点无效: {goal}")
            return None

        # 初始化树
        nodes = [start.copy()]
        parents = {0: -1}
        costs = {0: 0.0}

        goal_idx = None

        for i in range(self.config.max_iterations):
            # 采样
            if np.random.random() < self.config.goal_sample_rate:
                sample = goal.copy()
            else:
                sample = self._random_sample()

            # 找最近节点
            nearest_idx = self._nearest_node(nodes, sample)
            nearest = nodes[nearest_idx]

            # 向采样点扩展
            new_point = self._steer(nearest, sample)

            # 碰撞检测
            if not self._is_collision_free(nearest, new_point):
                continue

            # RRT* 重连接
            new_idx = len(nodes)
            nodes.append(new_point)

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

            # 检查是否到达目标
            if np.linalg.norm(new_point - goal) < self.config.step_size:
                if self._is_collision_free(new_point, goal):
                    goal_idx = len(nodes)
                    nodes.append(goal.copy())
                    parents[goal_idx] = new_idx
                    costs[goal_idx] = costs[new_idx] + np.linalg.norm(goal - new_point)
                    print(f"找到路径! 迭代次数: {i+1}")
                    break

        if goal_idx is None:
            print(f"规划失败，未能在 {self.config.max_iterations} 次迭代内找到路径")
            return None

        # 回溯路径
        path = []
        idx = goal_idx
        while idx != -1:
            path.append(nodes[idx])
            idx = parents[idx]
        path.reverse()

        # 路径平滑（可选）
        path = self._smooth_path(path)

        return path

    def _random_sample(self) -> np.ndarray:
        """随机采样"""
        return np.random.uniform(self.bounds_min, self.bounds_max)

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


class PathPlanner:
    """
    路径规划器 - 整合所有组件
    """

    def __init__(self, config: PlanningConfig = None):
        self.config = config or PlanningConfig()
        self.voxel_grid = VoxelGrid(self.config)
        self.esdf = ESDF(self.voxel_grid)
        self.rrt = RRTStar(self.voxel_grid, self.esdf, self.config)

    def update_map(self, depth_image: np.ndarray, camera_pos: np.ndarray,
                   camera_rotation: np.ndarray = None) -> int:
        """
        更新地图

        Returns:
            更新的占据体素数量
        """
        count = self.voxel_grid.update_from_depth_image(
            depth_image, camera_pos, camera_rotation
        )
        # 重新计算 ESDF
        self.esdf.compute()
        return count

    def plan_path(self, start: np.ndarray, goal: np.ndarray) -> Optional[List[np.ndarray]]:
        """规划路径"""
        return self.rrt.plan(start, goal)

    def get_distance_to_obstacle(self, point: np.ndarray) -> float:
        """获取某点到障碍物的距离"""
        return self.esdf.get_distance(point)

    def is_path_safe(self, path: List[np.ndarray]) -> bool:
        """检查路径是否安全"""
        for i in range(len(path) - 1):
            if not self.rrt._is_collision_free(path[i], path[i+1]):
                return False
        return True


# 可视化辅助函数
def visualize_planning_result(voxel_grid: VoxelGrid, esdf: ESDF,
                               path: List[np.ndarray] = None,
                               start: np.ndarray = None,
                               goal: np.ndarray = None):
    """可视化规划结果"""
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure(figsize=(15, 5))

    # 1. 3D 占据栅格
    ax1 = fig.add_subplot(131, projection='3d')
    occupied = voxel_grid.get_occupied_voxels()
    if len(occupied) > 0:
        ax1.scatter(occupied[:, 0], occupied[:, 1], occupied[:, 2],
                   c='red', s=1, alpha=0.3, label='Obstacles')

    if path is not None:
        path_arr = np.array(path)
        ax1.plot(path_arr[:, 0], path_arr[:, 1], path_arr[:, 2],
                'b-', linewidth=2, label='Path')
        ax1.scatter(path_arr[:, 0], path_arr[:, 1], path_arr[:, 2],
                   c='blue', s=20)

    if start is not None:
        ax1.scatter(*start, c='green', s=100, marker='o', label='Start')
    if goal is not None:
        ax1.scatter(*goal, c='orange', s=100, marker='*', label='Goal')

    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title('3D Voxel Grid & Path')
    ax1.legend()

    # 2. ESDF 切片（XY 平面）
    ax2 = fig.add_subplot(132)
    if esdf.distance_field is not None:
        # 取中间 Z 层
        z_mid = esdf.distance_field.shape[2] // 2
        slice_xy = esdf.distance_field[:, :, z_mid].T
        im = ax2.imshow(slice_xy, cmap='RdYlGn', origin='lower',
                       extent=[voxel_grid.origin[0],
                              voxel_grid.origin[0] + voxel_grid.grid_size[0] * voxel_grid.voxel_size,
                              voxel_grid.origin[1],
                              voxel_grid.origin[1] + voxel_grid.grid_size[1] * voxel_grid.voxel_size])
        plt.colorbar(im, ax=ax2, label='Distance (m)')

        if path is not None:
            path_arr = np.array(path)
            ax2.plot(path_arr[:, 0], path_arr[:, 1], 'b-', linewidth=2)
        if start is not None:
            ax2.scatter(start[0], start[1], c='green', s=100, marker='o')
        if goal is not None:
            ax2.scatter(goal[0], goal[1], c='orange', s=100, marker='*')

    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_title(f'ESDF Slice (Z={z_mid})')

    # 3. ESDF 切片（XZ 平面）
    ax3 = fig.add_subplot(133)
    if esdf.distance_field is not None:
        y_mid = esdf.distance_field.shape[1] // 2
        slice_xz = esdf.distance_field[:, y_mid, :].T
        im = ax3.imshow(slice_xz, cmap='RdYlGn', origin='lower',
                       extent=[voxel_grid.origin[0],
                              voxel_grid.origin[0] + voxel_grid.grid_size[0] * voxel_grid.voxel_size,
                              voxel_grid.origin[2],
                              voxel_grid.origin[2] + voxel_grid.grid_size[2] * voxel_grid.voxel_size])
        plt.colorbar(im, ax=ax3, label='Distance (m)')

        if path is not None:
            path_arr = np.array(path)
            ax3.plot(path_arr[:, 0], path_arr[:, 2], 'b-', linewidth=2)
        if start is not None:
            ax3.scatter(start[0], start[2], c='green', s=100, marker='o')
        if goal is not None:
            ax3.scatter(goal[0], goal[2], c='orange', s=100, marker='*')

    ax3.set_xlabel('X')
    ax3.set_ylabel('Z')
    ax3.set_title(f'ESDF Slice (Y={y_mid})')

    plt.tight_layout()
    return fig
