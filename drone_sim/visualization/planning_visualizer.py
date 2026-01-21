"""
Planning Visualizer - 规划过程可视化
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import List, Optional
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class PlanningVisualizer:
    """
    实时可视化规划过程
    - 3D 地图累积
    - 当前路径和局部目标
    - 已执行轨迹
    """

    def __init__(self):
        plt.ion()  # 交互模式
        self.fig = plt.figure(figsize=(18, 6))

        self.ax_map = self.fig.add_subplot(131, projection='3d')
        self.ax_xy = self.fig.add_subplot(132)
        self.ax_xz = self.fig.add_subplot(133)

        self.executed_trajectory = []
        self.update_count = 0

    def update(self, **kwargs):
        """
        更新可视化

        Args:
            map_manager: IncrementalMapManager
            current_pos: 当前位置
            local_goal: 局部目标
            global_goal: 全局目标
            current_path: 当前规划路径
            executed_waypoints: 已执行的路径点
        """
        map_manager = kwargs['map_manager']
        current_pos = kwargs['current_pos']
        local_goal = kwargs['local_goal']
        global_goal = kwargs['global_goal']
        current_path = kwargs['current_path']
        executed_waypoints = kwargs['executed_waypoints']

        self.update_count += 1

        # 累积执行轨迹
        self.executed_trajectory.extend(executed_waypoints)

        # 清空
        self.ax_map.clear()
        self.ax_xy.clear()
        self.ax_xz.clear()

        # === 1. 3D 地图视图 ===
        occupied = map_manager.voxel_grid.get_occupied_voxels()
        if len(occupied) > 0:
            # 下采样显示（避免太多点）
            if len(occupied) > 5000:
                indices = np.random.choice(len(occupied), 5000, replace=False)
                occupied = occupied[indices]

            self.ax_map.scatter(occupied[:, 0], occupied[:, 1], occupied[:, 2],
                              c='red', s=1, alpha=0.3, label='Obstacles')

        # 当前位置
        self.ax_map.scatter(*current_pos, c='blue', s=100, marker='o', label='Current', zorder=10)

        # 局部目标
        self.ax_map.scatter(*local_goal, c='orange', s=100, marker='*', label='Local Goal', zorder=10)

        # 全局目标
        self.ax_map.scatter(*global_goal, c='green', s=150, marker='*', label='Global Goal', zorder=10)

        # 当前规划路径
        if current_path and len(current_path) > 0:
            path_arr = np.array(current_path)
            self.ax_map.plot(path_arr[:, 0], path_arr[:, 1], path_arr[:, 2],
                           'b--', linewidth=2, alpha=0.5, label='Planned Path')

        # 已执行轨迹
        if len(self.executed_trajectory) > 0:
            traj_arr = np.array(self.executed_trajectory)
            self.ax_map.plot(traj_arr[:, 0], traj_arr[:, 1], traj_arr[:, 2],
                           'g-', linewidth=3, label='Executed', zorder=5)

        self.ax_map.legend(loc='upper right', fontsize=8)
        self.ax_map.set_xlabel('X (m)')
        self.ax_map.set_ylabel('Y (m)')
        self.ax_map.set_zlabel('Z (m)')
        self.ax_map.set_title(f'3D Map & Path (Update {self.update_count})')

        # 设置视角
        self.ax_map.view_init(elev=20, azim=45)

        # === 2. XY 平面投影（俯视图）===
        if len(occupied) > 0:
            self.ax_xy.scatter(occupied[:, 0], occupied[:, 1],
                             c='red', s=1, alpha=0.3, label='Obstacles')

        # 当前位置
        self.ax_xy.scatter(current_pos[0], current_pos[1],
                          c='blue', s=100, marker='o', label='Current', zorder=10)

        # 局部目标
        self.ax_xy.scatter(local_goal[0], local_goal[1],
                          c='orange', s=100, marker='*', label='Local Goal', zorder=10)

        # 全局目标
        self.ax_xy.scatter(global_goal[0], global_goal[1],
                          c='green', s=150, marker='*', label='Global Goal', zorder=10)

        # 当前规划路径
        if current_path and len(current_path) > 0:
            path_arr = np.array(current_path)
            self.ax_xy.plot(path_arr[:, 0], path_arr[:, 1],
                          'b--', linewidth=2, alpha=0.5, label='Planned')

        # 已执行轨迹
        if len(self.executed_trajectory) > 0:
            traj_arr = np.array(self.executed_trajectory)
            self.ax_xy.plot(traj_arr[:, 0], traj_arr[:, 1],
                          'g-', linewidth=3, label='Executed', zorder=5)

        self.ax_xy.set_xlabel('X (m)')
        self.ax_xy.set_ylabel('Y (m)')
        self.ax_xy.set_title('Top View (XY Plane)')
        self.ax_xy.legend(loc='upper right', fontsize=8)
        self.ax_xy.grid(True, alpha=0.3)
        self.ax_xy.axis('equal')

        # === 3. XZ 平面投影（侧视图）===
        if len(occupied) > 0:
            self.ax_xz.scatter(occupied[:, 0], occupied[:, 2],
                             c='red', s=1, alpha=0.3, label='Obstacles')

        # 当前位置
        self.ax_xz.scatter(current_pos[0], current_pos[2],
                          c='blue', s=100, marker='o', label='Current', zorder=10)

        # 局部目标
        self.ax_xz.scatter(local_goal[0], local_goal[2],
                          c='orange', s=100, marker='*', label='Local Goal', zorder=10)

        # 全局目标
        self.ax_xz.scatter(global_goal[0], global_goal[2],
                          c='green', s=150, marker='*', label='Global Goal', zorder=10)

        # 当前规划路径
        if current_path and len(current_path) > 0:
            path_arr = np.array(current_path)
            self.ax_xz.plot(path_arr[:, 0], path_arr[:, 2],
                          'b--', linewidth=2, alpha=0.5, label='Planned')

        # 已执行轨迹
        if len(self.executed_trajectory) > 0:
            traj_arr = np.array(self.executed_trajectory)
            self.ax_xz.plot(traj_arr[:, 0], traj_arr[:, 2],
                          'g-', linewidth=3, label='Executed', zorder=5)

        self.ax_xz.set_xlabel('X (m)')
        self.ax_xz.set_ylabel('Z (m)')
        self.ax_xz.set_title('Side View (XZ Plane)')
        self.ax_xz.legend(loc='upper right', fontsize=8)
        self.ax_xz.grid(True, alpha=0.3)
        self.ax_xz.invert_yaxis()  # Z 轴向下为正

        plt.tight_layout()
        plt.pause(0.01)

    def save_figure(self, filename: str):
        """保存当前图像"""
        self.fig.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"✓ Saved visualization to {filename}")

    def close(self):
        """关闭可视化窗口"""
        plt.ioff()
        plt.close(self.fig)
