"""
Incremental Map Manager - 增量式全局地图管理器
"""

import numpy as np
import time
from typing import Tuple, Dict
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mapping.voxel_grid import VoxelGrid
from mapping.esdf import ESDF
from planning.config import PlanningConfig
from utils.transforms import depth_image_to_camera_points, transform_camera_to_world


class IncrementalMapManager:
    """
    增量式全局地图管理器
    - 累积多帧深度图构建全局地图
    - 处理坐标变换（相机 → 世界）
    - 维护局部滑动窗口
    """

    def __init__(self, config: PlanningConfig):
        self.voxel_grid = VoxelGrid(config)
        self.esdf = ESDF(self.voxel_grid)
        self.config = config

        # 新增：观测计数（用于置信度）
        self.observation_count = np.zeros(config.grid_size, dtype=np.uint8)

        # 新增：地图中心（用于滑动窗口）
        self.map_center = np.array([0.0, 0.0, 0.0])

        # 统计信息
        self.total_updates = 0
        self.total_occupied_voxels = 0

    def update(self, depth_image: np.ndarray,
               drone_position: np.ndarray,
               drone_orientation: np.ndarray) -> Dict:
        """
        累积更新全局地图

        Args:
            depth_image: HxW 深度图（米）
            drone_position: 无人机位置 [x, y, z]
            drone_orientation: 无人机姿态四元数 [w, x, y, z]

        Returns:
            stats: {
                'new_occupied': int,  # 新增占据体素数
                'total_occupied': int,
                'update_time_ms': float
            }
        """
        start_time = time.time()

        # 0. 清除无人机当前位置周围的障碍物（防止被之前的观测错误标记）
        self._clear_around_drone(drone_position, radius=2.5)

        # 1. 深度图 → 相机坐标系点云
        points_camera = depth_image_to_camera_points(
            depth_image,
            fov_deg=self.config.fov_deg,
            subsample=4,
            max_depth=self.config.max_depth
        )

        if len(points_camera) == 0:
            return {
                'new_occupied': 0,
                'total_occupied': self.total_occupied_voxels,
                'update_time_ms': 0.0
            }

        # 2. 相机坐标系 → 世界坐标系
        points_world = transform_camera_to_world(
            points_camera, drone_position, drone_orientation
        )

        # 3. 累积更新体素栅格（关键：不覆盖，而是叠加）
        new_occupied = self._accumulate_points(points_world, drone_position)

        # 4. 更新地图中心（滑动窗口）
        self._update_map_center(drone_position)

        # 5. 重新计算 ESDF
        self.esdf.compute()

        # 6. 确保无人机位置安全（如果ESDF显示在障碍物内，再次清除）
        self._ensure_drone_safe(drone_position)

        # 7. 更新统计信息
        self.total_updates += 1
        self.total_occupied_voxels = np.sum(self.voxel_grid.grid == 1)

        update_time = (time.time() - start_time) * 1000  # ms

        return {
            'new_occupied': new_occupied,
            'total_occupied': self.total_occupied_voxels,
            'update_time_ms': update_time
        }

    def _clear_around_drone(self, drone_pos: np.ndarray, radius: float = 2.5):
        """清除无人机周围的障碍物标记"""
        drone_idx = self.voxel_grid.world_to_grid(drone_pos)
        radius_voxels = int(radius / self.config.voxel_size) + 1

        for dx in range(-radius_voxels, radius_voxels + 1):
            for dy in range(-radius_voxels, radius_voxels + 1):
                for dz in range(-radius_voxels, radius_voxels + 1):
                    idx = (drone_idx[0] + dx, drone_idx[1] + dy, drone_idx[2] + dz)
                    if self.voxel_grid.is_valid_index(idx):
                        # 计算实际距离
                        world_pos = self.voxel_grid.grid_to_world(idx)
                        dist = np.linalg.norm(world_pos - drone_pos)
                        if dist < radius:
                            # 清除障碍物标记，设为空闲
                            if self.voxel_grid.grid[idx] == 1:
                                self.voxel_grid.grid[idx] = -1  # 标记为空闲

    def _ensure_drone_safe(self, drone_pos: np.ndarray):
        """确保ESDF中无人机位置是安全的"""
        if self.esdf.distance_field is not None:
            dist = self.esdf.get_distance(drone_pos)
            if dist < 0:  # 在障碍物内
                self._clear_around_drone(drone_pos, radius=3.0)
                self.esdf.compute()  # 重新计算ESDF

    def _accumulate_points(self, points_world: np.ndarray,
                          camera_pos: np.ndarray) -> int:
        """
        累积点云到体素栅格
        - 新观测到的占据体素：标记为占据
        - 相机到点云之间的体素：标记为空闲（如果之前未被占据）
        - 已占据的体素：保持占据状态（不会被覆盖）
        - 过滤地面点：只保留与无人机高度相近的障碍物
        - 保护无人机周围区域：不标记为障碍物

        Returns:
            新增占据体素数量
        """
        new_count = 0

        # 地面过滤参数
        # NED坐标系：Z负值表示高度，Z=0是地面
        drone_z = camera_pos[2]  # 负值，如-3表示3米高

        # 地面阈值：Z > -0.5 的点认为是地面（0.5米以下）
        ground_threshold = -0.5

        # 无人机保护半径：不在无人机周围标记障碍物
        # 这防止深度传感器噪声导致无人机被标记为在障碍物内
        drone_protection_radius = 2.0

        # 标记占据体素
        for point in points_world:
            point_z = point[2]

            # 过滤地面点（Z接近0或为正的点）
            if point_z > ground_threshold:
                continue  # 跳过地面点

            # 保护无人机周围区域：跳过距离无人机太近的点
            dist_to_drone = np.linalg.norm(point - camera_pos)
            if dist_to_drone < drone_protection_radius:
                continue  # 跳过无人机附近的点

            idx = self.voxel_grid.world_to_grid(point)
            if self.voxel_grid.is_valid_index(idx):
                # 标记为占据
                if self.voxel_grid.grid[idx] != 1:
                    self.voxel_grid.grid[idx] = 1
                    new_count += 1

                # 增加观测计数
                if self.observation_count[idx] < 255:
                    self.observation_count[idx] += 1

        # Ray casting: 标记空闲空间（简化版）
        self._mark_free_space(camera_pos, points_world)

        return new_count

    def _mark_free_space(self, camera_pos: np.ndarray,
                        points: np.ndarray):
        """
        简化的 ray casting：标记相机到障碍物之间的空闲空间
        """
        # 下采样点云以提高效率
        if len(points) > 100:
            sampled_indices = np.random.choice(len(points), 100, replace=False)
            sampled_points = points[sampled_indices]
        else:
            sampled_points = points

        for point in sampled_points:
            # 沿射线采样
            direction = point - camera_pos
            distance = np.linalg.norm(direction)
            if distance < 0.1:
                continue

            direction = direction / distance

            # 每隔 0.5m 采样一个点
            num_samples = int(distance / 0.5)
            for i in range(1, num_samples):  # 跳过起点和终点
                sample_point = camera_pos + direction * (i * 0.5)
                idx = self.voxel_grid.world_to_grid(sample_point)

                if self.voxel_grid.is_valid_index(idx):
                    # 只有未被占据的体素才标记为空闲
                    if self.voxel_grid.grid[idx] == 0:
                        self.voxel_grid.grid[idx] = -1

    def _update_map_center(self, drone_position: np.ndarray):
        """
        更新地图中心，实现滑动窗口
        如果无人机移动超过阈值，清理远处的体素
        """
        shift = drone_position - self.map_center

        if np.linalg.norm(shift) > 10.0:  # 移动超过10米
            # 清理距离当前位置超过 30 米的体素
            self._prune_distant_voxels(drone_position, max_distance=30.0)
            self.map_center = drone_position.copy()

    def _prune_distant_voxels(self, drone_position: np.ndarray, max_distance: float):
        """
        清理距离无人机过远的体素
        """
        # 获取所有占据体素的索引
        occupied_indices = np.argwhere(self.voxel_grid.grid == 1)

        if len(occupied_indices) == 0:
            return

        pruned_count = 0
        for idx in occupied_indices:
            world_pos = self.voxel_grid.grid_to_world(tuple(idx))
            distance = np.linalg.norm(world_pos - drone_position)

            if distance > max_distance:
                self.voxel_grid.grid[tuple(idx)] = 0
                self.observation_count[tuple(idx)] = 0
                pruned_count += 1

        if pruned_count > 0:
            print(f"  Pruned {pruned_count} distant voxels")

    def get_map_stats(self) -> Dict:
        """获取地图统计信息"""
        return {
            'total_updates': self.total_updates,
            'occupied_voxels': self.total_occupied_voxels,
            'free_voxels': np.sum(self.voxel_grid.grid == -1),
            'unknown_voxels': np.sum(self.voxel_grid.grid == 0),
            'map_center': self.map_center.copy()
        }
