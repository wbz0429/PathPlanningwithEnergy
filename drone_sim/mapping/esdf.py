"""
ESDF - Euclidean Signed Distance Field
"""

import numpy as np
from scipy import ndimage
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .voxel_grid import VoxelGrid


class ESDF:
    """
    欧几里得符号距离场 (Euclidean Signed Distance Field)
    使用 scipy 的距离变换实现
    """

    def __init__(self, voxel_grid: 'VoxelGrid'):
        self.voxel_grid = voxel_grid
        self.distance_field = None
        self.gradient_field = None

    def compute(self):
        """计算 ESDF"""
        grid = self.voxel_grid.grid
        voxel_size = self.voxel_grid.voxel_size

        # 创建二值占据图
        # grid: 0=未知, 1=占据, -1=空闲
        occupied = (grid == 1).astype(np.float32)

        # 空闲区域：包括明确标记为空闲的(-1)和未知的(0)
        # 这样未知区域不会被当作障碍物
        free = (grid != 1).astype(np.float32)

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
