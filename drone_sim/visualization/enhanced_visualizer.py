"""
Enhanced Planning Visualizer - 增强版规划可视化器
支持：
1. 无人机视锥体显示
2. 深度点云显示
3. 地图累积过程（观测次数热力图）
4. RRT*搜索树结构
5. 实时动画 + 视频录制
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d, Axes3D
import matplotlib.animation as animation
from typing import List, Optional, Dict, Tuple
import time
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class Arrow3D(FancyArrowPatch):
    """3D箭头类"""
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        FancyArrowPatch.draw(self, renderer)

    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        return np.min(zs)


class EnhancedPlanningVisualizer:
    """
    增强版规划可视化器
    """

    def __init__(self, save_video: bool = True, video_fps: int = 10):
        """
        Args:
            save_video: 是否保存视频
            video_fps: 视频帧率
        """
        self._save_video_enabled = save_video  # 重命名属性避免与方法冲突
        self.video_fps = video_fps

        # 创建图形窗口
        plt.ion()
        self.fig = plt.figure(figsize=(20, 10))
        self.fig.suptitle('Receding Horizon Planning - Enhanced Visualization',
                         fontsize=16, fontweight='bold')

        # 创建子图
        # 主3D视图（左上，大）
        self.ax_3d = self.fig.add_subplot(2, 3, (1, 4), projection='3d')

        # XY平面俯视图（右上）
        self.ax_xy = self.fig.add_subplot(2, 3, 2)

        # XZ平面侧视图（右中）
        self.ax_xz = self.fig.add_subplot(2, 3, 3)

        # 深度点云视图（左下）
        self.ax_depth = self.fig.add_subplot(2, 3, 5, projection='3d')

        # 统计信息（右下）
        self.ax_stats = self.fig.add_subplot(2, 3, 6)
        self.ax_stats.axis('off')

        # 数据存储
        self.executed_trajectory = []
        self.current_depth_points = None
        self.rrt_tree_edges = []
        self.observation_counts = None

        # 视频录制
        self.frames = []
        self.frame_times = []

        # 性能统计
        self.update_times = []

    def update(self,
               map_manager,
               current_pos: np.ndarray,
               current_ori: np.ndarray,
               local_goal: np.ndarray,
               global_goal: np.ndarray,
               current_path: Optional[List[np.ndarray]],
               executed_waypoints: List[np.ndarray],
               depth_points: Optional[np.ndarray] = None,
               rrt_tree: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None,
               stats: Optional[Dict] = None):
        """
        更新可视化

        Args:
            map_manager: 地图管理器
            current_pos: 当前位置
            current_ori: 当前姿态（四元数）
            local_goal: 局部目标
            global_goal: 全局目标
            current_path: 当前规划路径
            executed_waypoints: 已执行航点
            depth_points: 当前帧深度点云
            rrt_tree: RRT*搜索树边（可选）
            stats: 统计信息（可选）
        """
        start_time = time.time()

        # 更新轨迹
        self.executed_trajectory.extend(executed_waypoints)

        # 更新深度点云
        if depth_points is not None:
            self.current_depth_points = depth_points

        # 更新RRT树
        if rrt_tree is not None:
            self.rrt_tree_edges = rrt_tree

        # 获取地图数据
        voxel_grid = map_manager.voxel_grid
        occupied_voxels = self._get_occupied_voxels(voxel_grid)

        # 获取观测次数（用于热力图）
        self.observation_counts = map_manager.observation_count

        # === 1. 主3D视图 ===
        self._draw_main_3d_view(
            occupied_voxels, current_pos, current_ori,
            local_goal, global_goal, current_path, voxel_grid
        )

        # === 2. XY平面俯视图 ===
        self._draw_xy_view(
            occupied_voxels, current_pos, local_goal,
            global_goal, current_path
        )

        # === 3. XZ平面侧视图 ===
        self._draw_xz_view(
            occupied_voxels, current_pos, local_goal,
            global_goal, current_path
        )

        # === 4. 深度点云视图 ===
        self._draw_depth_view(current_pos, current_ori)

        # === 5. 统计信息 ===
        self._draw_stats(stats)

        # 刷新显示
        plt.tight_layout()
        plt.pause(0.01)

        # 保存帧（用于视频）
        if self._save_video_enabled:
            self._capture_frame()

        # 记录更新时间
        elapsed = (time.time() - start_time) * 1000
        self.update_times.append(elapsed)

    def _draw_main_3d_view(self, occupied_voxels, current_pos, current_ori,
                           local_goal, global_goal, current_path, voxel_grid):
        """绘制主3D视图"""
        self.ax_3d.clear()

        # 1. 障碍物（用观测次数着色）
        if len(occupied_voxels) > 0:
            # 下采样以提高性能
            if len(occupied_voxels) > 5000:
                indices = np.random.choice(len(occupied_voxels), 5000, replace=False)
                occupied_sample = occupied_voxels[indices]
            else:
                occupied_sample = occupied_voxels

            # 获取每个占据体素的观测次数
            obs_counts = []
            for voxel in occupied_sample:
                idx = voxel_grid.world_to_grid(voxel)
                if voxel_grid.is_valid_index(idx):
                    obs_counts.append(self.observation_counts[idx])
                else:
                    obs_counts.append(1)

            obs_counts = np.array(obs_counts)

            # 归一化到0-1
            if obs_counts.max() > 0:
                obs_counts_norm = obs_counts / obs_counts.max()
            else:
                obs_counts_norm = obs_counts

            # 使用热力图颜色
            scatter = self.ax_3d.scatter(
                occupied_sample[:, 0],
                occupied_sample[:, 1],
                occupied_sample[:, 2],
                c=obs_counts_norm,
                cmap='hot',
                s=5,
                alpha=0.3,
                label='Obstacles (heat: obs count)'
            )

        # 2. 无人机视锥体
        self._draw_camera_frustum(self.ax_3d, current_pos, current_ori)

        # 3. 当前位置（无人机）
        self.ax_3d.scatter(*current_pos, c='blue', s=200, marker='o',
                          edgecolors='black', linewidths=2, label='Drone', zorder=10)

        # 4. 局部目标
        self.ax_3d.scatter(*local_goal, c='orange', s=150, marker='*',
                          edgecolors='black', linewidths=1, label='Local Goal', zorder=9)

        # 5. 全局目标
        self.ax_3d.scatter(*global_goal, c='green', s=200, marker='*',
                          edgecolors='black', linewidths=2, label='Global Goal', zorder=9)

        # 6. RRT*搜索树
        if len(self.rrt_tree_edges) > 0:
            for parent, child in self.rrt_tree_edges:
                self.ax_3d.plot([parent[0], child[0]],
                               [parent[1], child[1]],
                               [parent[2], child[2]],
                               'gray', alpha=0.1, linewidth=0.5)

        # 7. 当前规划路径
        if current_path and len(current_path) > 0:
            path_arr = np.array(current_path)
            self.ax_3d.plot(path_arr[:, 0], path_arr[:, 1], path_arr[:, 2],
                           'b--', linewidth=2, alpha=0.6, label='Planned Path')

        # 8. 已执行轨迹
        if len(self.executed_trajectory) > 0:
            traj_arr = np.array(self.executed_trajectory)
            self.ax_3d.plot(traj_arr[:, 0], traj_arr[:, 1], traj_arr[:, 2],
                           'g-', linewidth=3, label='Executed Trajectory', zorder=8)

        # 设置
        self.ax_3d.set_xlabel('X (m)')
        self.ax_3d.set_ylabel('Y (m)')
        self.ax_3d.set_zlabel('Z (m)')
        self.ax_3d.set_title('3D Map View (with observation heatmap)')
        self.ax_3d.legend(loc='upper right', fontsize=8)

        # 设置视角
        self.ax_3d.view_init(elev=30, azim=45)

    def _draw_camera_frustum(self, ax, position, orientation,
                            fov_deg=90.0, depth=5.0):
        """
        绘制相机视锥体

        Args:
            ax: 3D坐标轴
            position: 相机位置
            orientation: 相机姿态（四元数 [w, x, y, z]）
            fov_deg: 视场角（度）
            depth: 视锥深度
        """
        from utils.transforms import quaternion_to_rotation_matrix

        # 计算旋转矩阵
        R_world_body = quaternion_to_rotation_matrix(orientation)

        # 相机到机体的旋转
        R_body_camera = np.array([
            [0, 0, 1],
            [1, 0, 0],
            [0, 1, 0]
        ])

        R_world_camera = R_world_body @ R_body_camera

        # 计算视锥的四个角点（相机坐标系）
        half_fov = np.radians(fov_deg / 2)
        tan_half_fov = np.tan(half_fov)

        # 近平面和远平面的半宽/半高
        far_half = depth * tan_half_fov

        # 视锥角点（相机坐标系）
        frustum_points_camera = np.array([
            [0, 0, 0],  # 相机中心
            [far_half, far_half, depth],    # 右上
            [-far_half, far_half, depth],   # 左上
            [-far_half, -far_half, depth],  # 左下
            [far_half, -far_half, depth],   # 右下
        ])

        # 转换到世界坐标系
        frustum_points_world = (R_world_camera @ frustum_points_camera.T).T + position

        # 绘制视锥边
        center = frustum_points_world[0]
        for i in range(1, 5):
            ax.plot([center[0], frustum_points_world[i][0]],
                   [center[1], frustum_points_world[i][1]],
                   [center[2], frustum_points_world[i][2]],
                   'cyan', alpha=0.5, linewidth=1.5)

        # 绘制远平面矩形
        for i in range(1, 5):
            next_i = i + 1 if i < 4 else 1
            ax.plot([frustum_points_world[i][0], frustum_points_world[next_i][0]],
                   [frustum_points_world[i][1], frustum_points_world[next_i][1]],
                   [frustum_points_world[i][2], frustum_points_world[next_i][2]],
                   'cyan', alpha=0.5, linewidth=1.5)

    def _draw_xy_view(self, occupied_voxels, current_pos, local_goal,
                     global_goal, current_path):
        """绘制XY平面俯视图"""
        self.ax_xy.clear()

        # 障碍物
        if len(occupied_voxels) > 0:
            # 下采样
            if len(occupied_voxels) > 3000:
                indices = np.random.choice(len(occupied_voxels), 3000, replace=False)
                occupied_sample = occupied_voxels[indices]
            else:
                occupied_sample = occupied_voxels

            self.ax_xy.scatter(occupied_sample[:, 0], occupied_sample[:, 1],
                              c='red', s=1, alpha=0.2, label='Obstacles')

        # 当前位置
        self.ax_xy.scatter(current_pos[0], current_pos[1],
                          c='blue', s=100, marker='o', label='Drone')

        # 局部目标
        self.ax_xy.scatter(local_goal[0], local_goal[1],
                          c='orange', s=80, marker='*', label='Local Goal')

        # 全局目标
        self.ax_xy.scatter(global_goal[0], global_goal[1],
                          c='green', s=100, marker='*', label='Global Goal')

        # 规划路径
        if current_path and len(current_path) > 0:
            path_arr = np.array(current_path)
            self.ax_xy.plot(path_arr[:, 0], path_arr[:, 1],
                           'b--', linewidth=2, alpha=0.5)

        # 执行轨迹
        if len(self.executed_trajectory) > 0:
            traj_arr = np.array(self.executed_trajectory)
            self.ax_xy.plot(traj_arr[:, 0], traj_arr[:, 1],
                           'g-', linewidth=2)

        self.ax_xy.set_xlabel('X (m)')
        self.ax_xy.set_ylabel('Y (m)')
        self.ax_xy.set_title('XY Plane (Top View)')
        self.ax_xy.legend(loc='upper right', fontsize=7)
        self.ax_xy.grid(True, alpha=0.3)
        self.ax_xy.set_aspect('equal')

    def _draw_xz_view(self, occupied_voxels, current_pos, local_goal,
                     global_goal, current_path):
        """绘制XZ平面侧视图"""
        self.ax_xz.clear()

        # 障碍物
        if len(occupied_voxels) > 0:
            # 下采样
            if len(occupied_voxels) > 3000:
                indices = np.random.choice(len(occupied_voxels), 3000, replace=False)
                occupied_sample = occupied_voxels[indices]
            else:
                occupied_sample = occupied_voxels

            self.ax_xz.scatter(occupied_sample[:, 0], occupied_sample[:, 2],
                              c='red', s=1, alpha=0.2, label='Obstacles')

        # 当前位置
        self.ax_xz.scatter(current_pos[0], current_pos[2],
                          c='blue', s=100, marker='o', label='Drone')

        # 局部目标
        self.ax_xz.scatter(local_goal[0], local_goal[2],
                          c='orange', s=80, marker='*', label='Local Goal')

        # 全局目标
        self.ax_xz.scatter(global_goal[0], global_goal[2],
                          c='green', s=100, marker='*', label='Global Goal')

        # 规划路径
        if current_path and len(current_path) > 0:
            path_arr = np.array(current_path)
            self.ax_xz.plot(path_arr[:, 0], path_arr[:, 2],
                           'b--', linewidth=2, alpha=0.5)

        # 执行轨迹
        if len(self.executed_trajectory) > 0:
            traj_arr = np.array(self.executed_trajectory)
            self.ax_xz.plot(traj_arr[:, 0], traj_arr[:, 2],
                           'g-', linewidth=2)

        self.ax_xz.set_xlabel('X (m)')
        self.ax_xz.set_ylabel('Z (m)')
        self.ax_xz.set_title('XZ Plane (Side View)')
        self.ax_xz.legend(loc='upper right', fontsize=7)
        self.ax_xz.grid(True, alpha=0.3)
        self.ax_xz.set_aspect('equal')
        self.ax_xz.invert_yaxis()  # Z轴向下为正

    def _draw_depth_view(self, current_pos, current_ori):
        """绘制深度点云视图"""
        self.ax_depth.clear()

        if self.current_depth_points is not None and len(self.current_depth_points) > 0:
            # 计算点到相机的距离（用于着色）
            distances = np.linalg.norm(self.current_depth_points - current_pos, axis=1)

            # 下采样以提高性能
            subsample = 4
            points_sub = self.current_depth_points[::subsample]
            distances_sub = distances[::subsample]

            # 绘制点云
            scatter = self.ax_depth.scatter(
                points_sub[:, 0],
                points_sub[:, 1],
                points_sub[:, 2],
                c=distances_sub,
                cmap='viridis',
                s=2,
                alpha=0.6
            )

            # 添加颜色条
            try:
                plt.colorbar(scatter, ax=self.ax_depth, label='Distance (m)',
                            shrink=0.5, pad=0.1)
            except:
                pass  # 如果colorbar已存在，忽略错误

        # 当前位置
        self.ax_depth.scatter(*current_pos, c='red', s=100, marker='o',
                             label='Drone')

        self.ax_depth.set_xlabel('X (m)')
        self.ax_depth.set_ylabel('Y (m)')
        self.ax_depth.set_zlabel('Z (m)')
        self.ax_depth.set_title('Current Depth Point Cloud')
        self.ax_depth.legend(loc='upper right', fontsize=7)

    def _draw_stats(self, stats):
        """绘制统计信息"""
        self.ax_stats.clear()
        self.ax_stats.axis('off')

        if stats is None:
            return

        # 构建统计文本
        text_lines = [
            "=== Statistics ===",
            "",
            f"Iteration: {stats.get('iteration', 0)}",
            f"Distance to goal: {stats.get('dist_to_goal', 0):.2f}m",
            "",
            "--- Map ---",
            f"Occupied voxels: {stats.get('occupied_voxels', 0)}",
            f"Free voxels: {stats.get('free_voxels', 0)}",
            f"Unknown voxels: {stats.get('unknown_voxels', 0)}",
            f"New occupied: +{stats.get('new_occupied', 0)}",
            "",
            "--- Path ---",
            f"Waypoints: {stats.get('path_waypoints', 0)}",
            f"Executing: {stats.get('executing_waypoints', 0)}",
            "",
            "--- Performance ---",
            f"Perception: {stats.get('perf_perception', 0):.1f}ms",
            f"Mapping: {stats.get('perf_mapping', 0):.1f}ms",
            f"Planning: {stats.get('perf_planning', 0):.1f}ms",
            f"Execution: {stats.get('perf_execution', 0):.1f}ms",
            f"Total: {stats.get('perf_total', 0):.1f}ms",
        ]

        # 显示文本
        text = '\n'.join(text_lines)
        self.ax_stats.text(0.1, 0.95, text,
                          transform=self.ax_stats.transAxes,
                          fontsize=9,
                          verticalalignment='top',
                          fontfamily='monospace',
                          bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    def _get_occupied_voxels(self, voxel_grid):
        """获取占据体素的世界坐标"""
        occupied_indices = np.argwhere(voxel_grid.grid == 1)

        if len(occupied_indices) == 0:
            return np.array([])

        # 转换为世界坐标
        occupied_voxels = []
        for idx in occupied_indices:
            world_pos = voxel_grid.grid_to_world(tuple(idx))
            occupied_voxels.append(world_pos)

        return np.array(occupied_voxels)

    def _capture_frame(self):
        """捕获当前帧（用于视频）"""
        # 将当前图形转换为numpy数组
        self.fig.canvas.draw()
        frame = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype=np.uint8)
        frame = frame.reshape(self.fig.canvas.get_width_height()[::-1] + (3,))

        self.frames.append(frame)
        self.frame_times.append(time.time())

    def save_figure(self, filename: str):
        """保存当前图形"""
        self.fig.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"  Saved figure: {filename}")

    def save_video(self, filename: str):
        """保存视频"""
        if len(self.frames) == 0:
            print("  No frames to save")
            return

        print(f"\n  Saving video: {filename}")
        print(f"    Total frames: {len(self.frames)}")
        print(f"    FPS: {self.video_fps}")

        # 使用matplotlib的动画功能
        fig_video = plt.figure(figsize=(20, 10))
        ax_video = fig_video.add_subplot(111)
        ax_video.axis('off')

        im = ax_video.imshow(self.frames[0])

        def update_frame(i):
            im.set_array(self.frames[i])
            return [im]

        anim = animation.FuncAnimation(
            fig_video, update_frame,
            frames=len(self.frames),
            interval=1000/self.video_fps,
            blit=True
        )

        # 保存为MP4
        try:
            anim.save(filename, writer='ffmpeg', fps=self.video_fps, dpi=100)
            print(f"  Video saved successfully!")
        except Exception as e:
            print(f"  Failed to save video: {e}")
            print(f"  Make sure ffmpeg is installed")

        plt.close(fig_video)

    def close(self):
        """关闭可视化"""
        plt.ioff()
        plt.close(self.fig)

    def get_performance_stats(self):
        """获取可视化性能统计"""
        if len(self.update_times) == 0:
            return {}

        return {
            'avg_update_time': np.mean(self.update_times),
            'max_update_time': np.max(self.update_times),
            'min_update_time': np.min(self.update_times),
            'total_frames': len(self.frames)
        }
