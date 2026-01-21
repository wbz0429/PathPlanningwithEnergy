"""
Planning configuration parameters
"""

from dataclasses import dataclass
from typing import Tuple


@dataclass
class PlanningConfig:
    """规划配置参数"""
    # 体素栅格参数
    voxel_size: float = 0.5          # 体素大小（米）
    grid_size: Tuple[int, int, int] = (80, 80, 40)  # 栅格尺寸 (x, y, z)
    origin: Tuple[float, float, float] = (-20.0, -20.0, -10.0)  # 栅格原点（调整Z轴）

    # 相机参数
    fov_deg: float = 90.0            # 视场角
    max_depth: float = 25.0          # 最大有效深度

    # RRT* 参数
    step_size: float = 1.5           # RRT 步长（增大以加快搜索）
    max_iterations: int = 3000       # 最大迭代次数
    goal_sample_rate: float = 0.2    # 目标采样概率（提高）
    search_radius: float = 4.0       # 重连接搜索半径（增大）

    # 安全参数
    safety_margin: float = 0.8       # 安全边距（米）- 从1.0降低到0.8
