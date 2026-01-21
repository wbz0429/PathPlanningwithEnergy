"""
Paper-Quality Visualizer - IROS风格的论文级可视化
展示增量式建图 + ESDF + RRT*的完整过程
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.patches import Circle
import matplotlib.patches as mpatches
from typing import List, Optional, Dict, Tuple
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class PaperQualityVisualizer:
    """
    论文级可视化器 - IROS风格

    布局：
    ┌─────────────┬─────────────┬─────────────┐
    │  3D Map     │  ESDF XY    │  ESDF XZ    │
    │  + Path     │  Slice      │  Slice      │
    ├─────────────┼─────────────┴─────────────┤
    │  RRT* Tree  │  Top View (XY)            │
    │  Growth     │  with ESDF heatmap        │
    └─────────────┴───────────────────────────┘
    """

    def __init__(self, figsize=(20, 12), dpi=150):
        """
        初始化可视化器

        Args:
            figsize: 图像尺寸
            dpi: 分辨率
        """
        plt.style.use('seaborn-v0_8-paper')  # 论文风格

        self.fig = plt.figure(figsize=figsize, dpi=dpi)
        self.gs = GridSpec(2, 3, figure=self.fig, hspace=0.3, wspace=0.3)

        # 创建子图
        self.ax_3d = self.fig.add_subplot(self.gs[0, 0], projection='3d')
        self.ax_esdf_xy = self.fig.add_subplot(self.gs[0, 1])
        self.ax_esdf_xz = self.fig.add_subplot(self.gs[0, 2])
        self.ax_rrt = self.fig.add_subplot(self.gs[1, 0])
        self.ax_top = self.fig.add_subplot(self.gs[1, 1:])

        # 数据存储
        self.frames = []  # 存储每一帧的数据
        self.current_frame = 0

        # 颜色方案（论文级配色）
        self.colors = {
            'obstacle': '#E74C3C',      # 红色 - 障碍物
            'free': '#3498DB',          # 蓝色 - 自由空间
            'unknown': '#95A5A6',       # 灰色 - 未知
            'path': '#2ECC71',          # 绿色 - 路径
            'current': '#F39C12',       # 橙色 - 当前位置
            'goal': '#9B59B6',          # 紫色 - 目标
            'rrt_tree': '#34495E',      # 深灰 - RRT树
            'rrt_sample': '#E67E22',    # 橙红 - 采样点
        }

    def add_frame(self, frame_data: Dict):
        """
        添加一帧数据

        Args:
            frame_data: {
                'iteration': int,
                'map_manager': IncrementalMapManager,
                'current_pos': np.ndarray,
                'local_goal': np.ndarray,
                'global_goal': np.ndarray,
                'current_path': List[np.ndarray],
                'rrt_tree': Optional[Dict],  # {'nodes': [], 'edges': []}
                'executed_trajectory': List[np.ndarray],
                'map_stats': Dict
            }
        """
        self.frames.append(frame_data)

    def render_frame(self, frame_idx: int, save_path: Optional[str] = None):
        """
        渲染指定帧

        Args:
            frame_idx: 帧索引
            save_path: 保存路径（可选）
        """
        if frame_idx >= len(self.frames):
            print(f"Warning: Frame {frame_idx} does not exist")
            return

        frame = self.frames[frame_idx]

        # 清空所有子图
        self.ax_3d.clear()
        self.ax_esdf_xy.clear()
        self.ax_esdf_xz.clear()
        self.ax_rrt.clear()
        self.ax_top.clear()

        # 渲染各个子图
        self._render_3d_map(frame)
        self._render_esdf_slices(frame)
        self._render_rrt_tree(frame)
        self._render_top_view(frame)

        # 添加全局标题
        iteration = frame.get('iteration', frame_idx)
        map_stats = frame.get('map_stats', )
        self.fig.suptitle(
            f"Receding Horizon Planning - Iteration {iteration}\n"
            f"Occupied: {map_stats.get('total_occupied', 0)} voxels | "
            f"Free: {map_stats.get('free_voxels', 0)} voxels | "
            f"Unknown: {map_stats.get('unknown_voxels', 0)} voxels",
            fontsize=14, fontweight='bold'
        )

        plt.tight_layout(rect=[0, 0, 1, 0.96])

        if save_path:
            self.fig.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"[OK] Saved frame {frame_idx} to {save_path}")
        else:
            plt.pause(0.01)

    def _render_3d_map(self, frame: Dict):
        """渲染3D地图视图"""
        ax = self.ax_3d
        map_manager = frame['map_manager']
        current_pos = frame['current_pos']
        local_goal = frame.get('local_goal')
        global_goal = frame['global_goal']
        current_path = frame.get('current_path', [])
        executed_traj = frame.get('executed_trajectory', [])

        # 获取占据体素
        occupied = map_manager.voxel_grid.get_occupied_voxels()
        if len(occupied) > 0:
            # 下采样显示
            if len(occupied) > 8000:
                indices = np.random.choice(len(occupied), 8000, replace=False)
                occupied = occupied[indices]

            ax.scatter(occupied[:, 0], occupied[:, 1], occupied[:, 2],
                      c=self.colors['obstacle'], s=2, alpha=0.4,
                      label='Obstacles', depthshade=True)

        # 已执行轨迹
        if len(executed_traj) > 0:
            traj_arr = np.array(executed_traj)
            ax.plot(traj_arr[:, 0], traj_arr[:, 1], traj_arr[:, 2],
                   color=self.colors['path'], linewidth=3,
                   label='Executed', zorder=5, alpha=0.8)

        # 当前规划路径
        if current_path and len(current_path) > 0:
            path_arr = np.array(current_path)
            ax.plot(path_arr[:, 0], path_arr[:, 1], path_arr[:, 2],
                   color=self.colors['rrt_tree'], linestyle='--',
                   linewidth=2, alpha=0.6, label='Planned Path')

        # 当前位置
        ax.scatter(*current_pos, c=self.colors['current'], s=200,
                  marker='o', edgecolors='black', linewidths=2,
                  label='Current', zorder=10)

        # 局部目标
        if local_goal is not None:
            ax.scatter(*local_goal, c=self.colors['rrt_sample'], s=150,
                      marker='*', edgecolors='black', linewidths=1.5,
                      label='Local Goal', zorder=10)

        # 全局目标
        ax.scatter(*global_goal, c=self.colors['goal'], s=200,
                  marker='*', edgecolors='black', linewidths=2,
                  label='Global Goal', zorder=10)

        ax.set_xlabel('X (m)', fontsize=10, fontweight='bold')
        ax.set_ylabel('Y (m)', fontsize=10, fontweight='bold')
        ax.set_zlabel('Z (m)', fontsize=10, fontweight='bold')
        ax.set_title('3D Occupancy Map', fontsize=12, fontweight='bold', pad=10)
        ax.legend(loc='upper right', fontsize=8, framealpha=0.9)
        ax.view_init(elev=25, azim=45)
        ax.grid(True, alpha=0.3)

    def _render_esdf_slices(self, frame: Dict):
        """渲染ESDF切片"""
        map_manager = frame['map_manager']
        current_pos = frame['current_pos']
        global_goal = frame['global_goal']
        current_path = frame.get('current_path', [])
        config = map_manager.config

        esdf = map_manager.esdf
        if esdf.distance_field is None:
            return

        # XY切片（俯视图）
        z_idx = int((current_pos[2] - config.origin[2]) / config.voxel_size)
        z_idx = np.clip(z_idx, 0, config.grid_size[2] - 1)
        slice_xy = esdf.distance_field[:, :, z_idx].T

        extent_xy = [
            config.origin[0],
            config.origin[0] + config.grid_size[0] * config.voxel_size,
            config.origin[1],
            config.origin[1] + config.grid_size[1] * config.voxel_size
        ]

        im1 = self.ax_esdf_xy.imshow(slice_xy, cmap='RdYlGn', origin='lower',
                                     extent=extent_xy, vmin=-2, vmax=8, alpha=0.9)

        # 添加路径
        if current_path and len(current_path) > 0:
            path_arr = np.array(current_path)
            self.ax_esdf_xy.plot(path_arr[:, 0], path_arr[:, 1],
                                color=self.colors['path'], linewidth=2.5,
                                label='Path', zorder=5)

        # 添加位置标记
        self.ax_esdf_xy.scatter(current_pos[0], current_pos[1],
                               c=self.colors['current'], s=150, marker='o',
                               edgecolors='black', linewidths=2, zorder=10)
        self.ax_esdf_xy.scatter(global_goal[0], global_goal[1],
                               c=self.colors['goal'], s=150, marker='*',
                               edgecolors='black', linewidths=2, zorder=10)

        self.ax_esdf_xy.set_xlabel('X (m)', fontsize=10, fontweight='bold')
        self.ax_esdf_xy.set_ylabel('Y (m)', fontsize=10, fontweight='bold')
        self.ax_esdf_xy.set_title(f'ESDF XY Slice (Z={z_idx})',
                                 fontsize=12, fontweight='bold')
        self.ax_esdf_xy.grid(True, alpha=0.3, linestyle='--')
        cbar1 = plt.colorbar(im1, ax=self.ax_esdf_xy, fraction=0.046, pad=0.04)
        cbar1.set_label('Distance (m)', fontsize=9)

        # XZ切片（侧视图）
        y_idx = int((current_pos[1] - config.origin[1]) / config.voxel_size)
        y_idx = np.clip(y_idx, 0, config.grid_size[1] - 1)
        slice_xz = esdf.distance_field[:, y_idx, :].T

        extent_xz = [
            config.origin[0],
            config.origin[0] + config.grid_size[0] * config.voxel_size,
            config.origin[2],
            config.origin[2] + config.grid_size[2] * config.voxel_size
        ]

        im2 = self.ax_esdf_xz.imshow(slice_xz, cmap='RdYlGn', origin='lower',
                                     extent=extent_xz, vmin=-2, vmax=8, alpha=0.9)

        # 添加路径
        if current_path and len(current_path) > 0:
            path_arr = np.array(current_path)
            self.ax_esdf_xz.plot(path_arr[:, 0], path_arr[:, 2],
                                color=self.colors['path'], linewidth=2.5, zorder=5)

        # 添加位置标记
        self.ax_esdf_xz.scatter(current_pos[0], current_pos[2],
                               c=self.colors['current'], s=150, marker='o',
                               edgecolors='black', linewidths=2, zorder=10)
        self.ax_esdf_xz.scatter(global_goal[0], global_goal[2],
                               c=self.colors['goal'], s=150, marker='*',
                               edgecolors='black', linewidths=2, zorder=10)

        self.ax_esdf_xz.set_xlabel('X (m)', fontsize=10, fontweight='bold')
        self.ax_esdf_xz.set_ylabel('Z (m)', fontsize=10, fontweight='bold')
        self.ax_esdf_xz.set_title(f'ESDF XZ Slice (Y={y_idx})',
                                 fontsize=12, fontweight='bold')
        self.ax_esdf_xz.grid(True, alpha=0.3, linestyle='--')
        self.ax_esdf_xz.invert_yaxis()
        cbar2 = plt.colorbar(im2, ax=self.ax_esdf_xz, fraction=0.046, pad=0.04)
        cbar2.set_label('Distance (m)', fontsize=9)

    def _render_rrt_tree(self, frame: Dict):
        """渲染RRT*树的生长过程"""
        ax = self.ax_rrt
        rrt_tree = frame.get('rrt_tree')
        current_pos = frame['current_pos']
        local_goal = frame.get('local_goal')
        current_path = frame.get('current_path', [])

        if rrt_tree is None:
            ax.text(0.5, 0.5, 'RRT* Tree\n(Not Available)',
                   ha='center', va='center', fontsize=12,
                   transform=ax.transAxes, color='gray')
            ax.set_title('RRT* Tree Growth', fontsize=12, fontweight='bold')
            return

        nodes = rrt_tree.get('nodes', [])
        edges = rrt_tree.get('edges', [])

        # 绘制树的边
        for edge in edges:
            parent, child = edge
            ax.plot([parent[0], child[0]], [parent[1], child[1]],
                   color=self.colors['rrt_tree'], linewidth=0.5,
                   alpha=0.4, zorder=1)

        # 绘制树的节点
        if len(nodes) > 0:
            nodes_arr = np.array(nodes)
            ax.scatter(nodes_arr[:, 0], nodes_arr[:, 1],
                      c=self.colors['rrt_tree'], s=3, alpha=0.6, zorder=2)

        # 绘制最终路径（高亮）
        if current_path and len(current_path) > 0:
            path_arr = np.array(current_path)
            ax.plot(path_arr[:, 0], path_arr[:, 1],
                   color=self.colors['path'], linewidth=3,
                   label='Final Path', zorder=5, alpha=0.9)

        # 起点和终点
        ax.scatter(current_pos[0], current_pos[1],
                  c=self.colors['current'], s=150, marker='o',
                  edgecolors='black', linewidths=2,
                  label='Start', zorder=10)

        if local_goal is not None:
            ax.scatter(local_goal[0], local_goal[1],
                      c=self.colors['rrt_sample'], s=150, marker='*',
                      edgecolors='black', linewidths=2,
                      label='Goal', zorder=10)

        ax.set_xlabel('X (m)', fontsize=10, fontweight='bold')
        ax.set_ylabel('Y (m)', fontsize=10, fontweight='bold')
        ax.set_title(f'RRT* Tree ({len(nodes)} nodes)',
                    fontsize=12, fontweight='bold')
        ax.legend(loc='upper right', fontsize=8, framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_aspect('equal')

    def _render_top_view(self, frame: Dict):
        """渲染俯视图（带ESDF热力图）"""
        ax = self.ax_top
        map_manager = frame['map_manager']
        current_pos = frame['current_pos']
        global_goal = frame['global_goal']
        current_path = frame.get('current_path', [])
        executed_traj = frame.get('executed_trajectory', [])
        config = map_manager.config

        # ESDF热力图背景
        esdf = map_manager.esdf
        if esdf.distance_field is not None:
            z_idx = int((current_pos[2] - config.origin[2]) / config.voxel_size)
            z_idx = np.clip(z_idx, 0, config.grid_size[2] - 1)
            slice_xy = esdf.distance_field[:, :, z_idx].T

            extent = [
                config.origin[0],
                config.origin[0] + config.grid_size[0] * config.voxel_size,
                config.origin[1],
                config.origin[1] + config.grid_size[1] * config.voxel_size
            ]

            im = ax.imshow(slice_xy, cmap='RdYlGn', origin='lower',
                          extent=extent, vmin=-2, vmax=8, alpha=0.5)
            cbar = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
            cbar.set_label('ESDF Distance (m)', fontsize=9)

        # 障碍物点云
        occupied = map_manager.voxel_grid.get_occupied_voxels()
        if len(occupied) > 0:
            ax.scatter(occupied[:, 0], occupied[:, 1],
                      c=self.colors['obstacle'], s=2, alpha=0.3,
                      label='Obstacles', zorder=2)

        # 已执行轨迹
        if len(executed_traj) > 0:
            traj_arr = np.array(executed_traj)
            ax.plot(traj_arr[:, 0], traj_arr[:, 1],
                   color=self.colors['path'], linewidth=4,
                   label='Executed Trajectory', zorder=5, alpha=0.9)

        # 当前规划路径
        if current_path and len(current_path) > 0:
            path_arr = np.array(current_path)
            ax.plot(path_arr[:, 0], path_arr[:, 1],
                   color=self.colors['rrt_tree'], linestyle='--',
                   linewidth=2.5, label='Planned Path', zorder=4, alpha=0.7)

        # 当前位置
        ax.scatter(current_pos[0], current_pos[1],
                  c=self.colors['current'], s=250, marker='o',
                  edgecolors='black', linewidths=2.5,
                  label='Current Position', zorder=10)

        # 全局目标
        ax.scatter(global_goal[0], global_goal[1],
                  c=self.colors['goal'], s=300, marker='*',
                  edgecolors='black', linewidths=2.5,
                  label='Global Goal', zorder=10)

        ax.set_xlabel('X (m)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Y (m)', fontsize=11, fontweight='bold')
        ax.set_title('Top View with ESDF Heatmap',
                    fontsize=13, fontweight='bold', pad=10)
        ax.legend(loc='upper right', fontsize=9, framealpha=0.95)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_aspect('equal')

    def save_all_frames(self, output_dir: str, prefix: str = 'frame'):
        """
        保存所有帧为图像序列

        Args:
            output_dir: 输出目录
            prefix: 文件名前缀
        """
        os.makedirs(output_dir, exist_ok=True)

        print(f"\nSaving {len(self.frames)} frames to {output_dir}...")
        for i in range(len(self.frames)):
            output_path = os.path.join(output_dir, f'{prefix}_{i:04d}.png')
            self.render_frame(i, save_path=output_path)

        print(f"[OK] All frames saved!")

    def create_video(self, output_path: str, fps: int = 2):
        """
        创建视频（需要ffmpeg）

        Args:
            output_path: 输出视频路径
            fps: 帧率
        """
        if len(self.frames) == 0:
            print("No frames to create video")
            return

        print(f"\nCreating video with {len(self.frames)} frames at {fps} fps...")

        # 创建动画
        def update(frame_idx):
            self.render_frame(frame_idx)
            return []

        anim = FuncAnimation(self.fig, update, frames=len(self.frames),
                           interval=1000/fps, blit=True)

        # 保存视频
        writer = FFMpegWriter(fps=fps, bitrate=5000)
        anim.save(output_path, writer=writer)

        print(f"[OK] Video saved to {output_path}")

    def close(self):
        """关闭图形"""
        plt.close(self.fig)
