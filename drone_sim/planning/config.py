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
    # 注意：以下参数基于特定测试场景调优，不同环境可能需要重新调整
    # max_iterations 越大规划越慢但成功率越高；goal_sample_rate 越大收敛越快但探索越少
    step_size: float = 1.5           # RRT 步长（增大以加快搜索）
    max_iterations: int = 5000       # 最大迭代次数（原值3000，复杂场景下需要更多迭代）
    goal_sample_rate: float = 0.4    # 目标采样概率（原值0.2，提高以加速收敛，但可能降低复杂环境探索能力）
    search_radius: float = 4.0       # 重连接搜索半径（增大）

    # 安全参数
    safety_margin: float = 1.0       # 安全边距（米）- 覆盖体素误差0.25m + 无人机半径0.3m + 缓冲0.45m
    unknown_safe_threshold: float = 2.0  # Unknown区域安全阈值：ESDF距离>=此值的unknown视为可通行

    # 能量感知规划参数
    energy_aware: bool = True        # 是否启用能量感知规划
    flight_velocity: float = 2.0     # 规划时假设的飞行速度 (m/s)

    # 代价函数权重（归一化后）
    weight_energy: float = 0.6       # 能耗权重（最重要）
    weight_distance: float = 0.3     # 距离权重
    weight_time: float = 0.1         # 时间权重

    # 归一化参考值
    energy_ref: float = 500.0        # 能耗参考值 (J)，约10米水平飞行
    distance_ref: float = 10.0       # 距离参考值 (m)
    time_ref: float = 5.0            # 时间参考值 (s)
