"""
Simple 3D Visualizer - 简洁的3D动态可视化
展示无人机轨迹 + 实时障碍物点云 + 规划路径
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation, FFMpegWriter
import matplotlib.patches as mpatches
from typing import List, Optional, Dict
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class Simple3DVisualizer:
    """
    简洁的3D可视化器
    - 单个3D视图
    - 实时更新障碍物点云
    - 无人机轨迹
    - 当前规划路径
    """

    def __init__(self, figsize=(12, 10), dpi=100):
        """
        初始化可视化器

        Args:
            figsize: 图像尺寸
            dpi: 分辨率
        """
        self.fig = plt.figure(figsize=figsize, dpi=dpi)
        self.ax = self.fig.add_subplot(111, projection='3d')

        # 数据存储
        self.frames = []

        # 颜色配置
        self.colors = {
            'obstacle': '#E74C3C',      # 红色 - 障碍物
            'trajectory': '#2ECC71',    # 绿色 - 已执行轨迹
            'planned': '#3498DB',       # 蓝色 - 规划路径
            'drone': '#F39C12',         # 橙色 - 无人机
            'goal': '#9B59B6',          # 紫色 - 目标
        }

        # 设置背景
        self.ax.set_facecolor('white')
        self.fig.patch.set_facecolor('white')

    def add_frame(self, frame_data: Dict):
        """
        添加一帧数据

        Args:
            frame_data: {
                'iteration': int,
                'obstacles': np.ndarray,  # Nx3 障碍物点云
                'drone_pos': np.ndarray,  # 当前位置
                'trajectory': List[np.ndarray],  # 已执行轨迹
                'planned_path': Optional[List[np.ndarray]],  # 规划路径
                'goal': np.ndarray,  # 目标位置
                'info': str  # 附加信息
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
        self.ax.clear()

        # === 1. 绘制障碍物点云 ===
        obstacles = frame.get('obstacles')
        if obstacles is not None and len(obstacles) > 0:
            # 下采样显示
            if len(obstacles) > 5000:
                indices = np.random.choice(len(obstacles), 5000, replace=False)
                obstacles_show = obstacles[indices]
            else:
                obstacles_show = obstacles

            self.ax.scatter(
                obstacles_show[:, 0],
                obstacles_show[:, 1],
                obstacles_show[:, 2],
                c=self.colors['obstacle'],
                s=5,
                alpha=0.4,
                label=f'Obstacles ({len(obstacles)} points)',
                depthshade=True
            )

        # === 2. 绘制已执行轨迹 ===
        trajectory = frame.get('trajectory', [])
        if len(trajectory) > 1:
            traj_arr = np.array(trajectory)
            self.ax.plot(
                traj_arr[:, 0],
                traj_arr[:, 1],
                traj_arr[:, 2],
                color=self.colors['trajectory'],
                linewidth=4,
                label='Executed Trajectory',
                zorder=10,
                alpha=0.9
            )

        # === 3. 绘制规划路径 ===
        planned_path = frame.get('planned_path')
        if planned_path is not None and len(planned_path) > 1:
            path_arr = np.array(planned_path)
            self.ax.plot(
                path_arr[:, 0],
                path_arr[:, 1],
                path_arr[:, 2],
                color=self.colors['planned'],
                linestyle='--',
                linewidth=3,
                label='Planned Path',
                zorder=9,
                alpha=0.7
            )

        # === 4. 绘制无人机当前位置 ===
        drone_pos = frame.get('drone_pos')
        if drone_pos is not None:
            self.ax.scatter(
                drone_pos[0],
                drone_pos[1],
                drone_pos[2],
                c=self.colors['drone'],
                s=300,
                marker='o',
                edgecolors='black',
                linewidths=2,
                label='Drone',
                zorder=15
            )

            # 添加无人机方向指示（简化为朝向目标）
            goal = frame.get('goal')
            if goal is not None:
                direction = goal - drone_pos
                direction = direction / np.linalg.norm(direction) * 1.5
                self.ax.quiver(
                    drone_pos[0], drone_pos[1], drone_pos[2],
                    direction[0], direction[1], direction[2],
                    color=self.colors['drone'],
                    arrow_length_ratio=0.3,
                    linewidth=2,
                    alpha=0.8
                )

        # === 5. 绘制目标位置 ===
        goal = frame.get('goal')
        if goal is not None:
            self.ax.scatter(
                goal[0],
                goal[1],
                goal[2],
                c=self.colors['goal'],
                s=400,
                marker='*',
                edgecolors='black',
                linewidths=2,
                label='Goal',
                zorder=15
            )

        # === 6. 设置坐标轴 ===
        self.ax.set_xlabel('X (m)', fontsize=12, fontweight='bold')
        self.ax.set_ylabel('Y (m)', fontsize=12, fontweight='bold')
        self.ax.set_zlabel('Z (m)', fontsize=12, fontweight='bold')

        # === 7. 设置视角 ===
        self.ax.view_init(elev=25, azim=45 + frame_idx * 2)  # 缓慢旋转视角

        # === 8. 设置坐标轴范围（自动调整） ===
        if drone_pos is not None:
            # 以无人机为中心，显示周围20m范围
            x_center, y_center, z_center = drone_pos
            margin = 15
            self.ax.set_xlim(x_center - margin, x_center + margin)
            self.ax.set_ylim(y_center - margin, y_center + margin)
            self.ax.set_zlim(z_center - margin, z_center + margin)

        # === 9. 添加网格 ===
        self.ax.grid(True, alpha=0.3, linestyle='--')

        # === 10. 添加图例 ===
        self.ax.legend(loc='upper left', fontsize=10, framealpha=0.9)

        # === 11. 添加标题和信息 ===
        iteration = frame.get('iteration', frame_idx)
        info = frame.get('info', '')
        title = f'Iteration {iteration}\n{info}'
        self.ax.set_title(title, fontsize=14, fontweight='bold', pad=20)

        # === 12. 保存或显示 ===
        plt.tight_layout()

        if save_path:
            self.fig.savefig(save_path, dpi=100, bbox_inches='tight', facecolor='white')
            print(f"[OK] Saved frame {frame_idx} to {save_path}")
        else:
            plt.pause(0.01)

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

    def create_video(self, output_path: str, fps: int = 5):
        """
        创建视频

        Args:
            output_path: 输出视频路径
            fps: 帧率
        """
        if len(self.frames) == 0:
            print("No frames to create video")
            return

        print(f"\nCreating video with {len(self.frames)} frames at {fps} fps...")

        def update(frame_idx):
            self.render_frame(frame_idx)
            return []

        anim = FuncAnimation(
            self.fig,
            update,
            frames=len(self.frames),
            interval=1000/fps,
            blit=False
        )

        try:
            writer = FFMpegWriter(fps=fps, bitrate=3000)
            anim.save(output_path, writer=writer)
            print(f"[OK] Video saved to {output_path}")
        except Exception as e:
            print(f"[X] Video creation failed: {e}")
            print("    You can create video from frames using:")
            print(f"    ffmpeg -framerate {fps} -i frames/frame_%04d.png -c:v libx264 -pix_fmt yuv420p {output_path}")

    def show_interactive(self):
        """
        显示交互式动画（需要在支持GUI的环境中运行）
        """
        if len(self.frames) == 0:
            print("No frames to show")
            return

        def update(frame_idx):
            self.render_frame(frame_idx)
            return []

        anim = FuncAnimation(
            self.fig,
            update,
            frames=len(self.frames),
            interval=200,  # 200ms per frame
            blit=False,
            repeat=True
        )

        plt.show()

    def close(self):
        """关闭图形"""
        plt.close(self.fig)
