"""
VoxelGrid - 3D voxel grid map (simplified OctoMap implementation)
"""

import numpy as np
from typing import Tuple
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from planning.config import PlanningConfig


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
