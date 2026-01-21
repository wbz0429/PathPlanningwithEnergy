"""
Receding Horizon Planner - 滚动规划控制器
"""

import numpy as np
import time
from typing import List, Optional, Dict
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mapping.incremental_map import IncrementalMapManager
from planning.rrt_star import RRTStar
from control.drone_interface import DroneInterface
from utils.performance import PerformanceMonitor


class RecedingHorizonPlanner:
    """
    滚动规划控制器
    - 维护全局目标
    - 循环：感知 → 更新地图 → 局部规划 → 执行部分路径
    """

    def __init__(self,
                 map_manager: IncrementalMapManager,
                 drone: DroneInterface,
                 config: Dict):
        self.map_manager = map_manager
        self.drone = drone

        # 滚动规划参数
        self.local_horizon = config.get('local_horizon', 6.0)  # 米
        self.execution_ratio = config.get('execution_ratio', 0.4)
        self.replan_threshold = config.get('replan_threshold', 1.5)
        self.goal_tolerance = config.get('goal_tolerance', 1.0)
        self.max_iterations = config.get('max_iterations', 50)
        self.flight_velocity = config.get('flight_velocity', 2.0)

        # RRT* 规划器
        self.rrt = RRTStar(
            map_manager.voxel_grid,
            map_manager.esdf,
            map_manager.config
        )

        # 可视化器（可选）
        self.visualizer = None
        visualize_mode = config.get('visualize', False)
        if visualize_mode:
            try:
                # 尝试使用增强版可视化器
                if config.get('enhanced_viz', True):
                    from visualization.enhanced_visualizer import EnhancedPlanningVisualizer
                    self.visualizer = EnhancedPlanningVisualizer(
                        save_video=config.get('save_video', True),
                        video_fps=config.get('video_fps', 10)
                    )
                    print("  Using Enhanced Visualizer")
                else:
                    from visualization.planning_visualizer import PlanningVisualizer
                    self.visualizer = PlanningVisualizer()
                    print("  Using Simple Visualizer")
            except ImportError as e:
                print(f"  Warning: Visualization not available: {e}")

        # 性能监控
        self.perf_monitor = PerformanceMonitor()

        # 执行轨迹记录
        self.executed_trajectory = []

    def plan_and_execute(self, global_goal: np.ndarray) -> bool:
        """
        主循环：滚动规划到全局目标

        Args:
            global_goal: 全局目标位置 [x, y, z]

        Returns:
            success: 是否成功到达目标
        """
        print("\n" + "=" * 60)
        print("Receding Horizon Planning Started")
        print("=" * 60)

        iteration = 0
        current_path = None

        while iteration < self.max_iterations:
            iteration += 1
            print(f"\n{'='*60}")
            print(f"Iteration {iteration}")
            print(f"{'='*60}")

            # === 1. 获取当前状态 ===
            with self.perf_monitor.measure('perception'):
                current_pos, current_ori = self.drone.get_pose()
                depth_image = self.drone.get_depth_image()

            print(f"Current position: ({current_pos[0]:.2f}, {current_pos[1]:.2f}, {current_pos[2]:.2f})")

            # === 2. 检查是否到达全局目标 ===
            dist_to_goal = np.linalg.norm(current_pos - global_goal)
            print(f"Distance to global goal: {dist_to_goal:.2f}m")

            if dist_to_goal < self.goal_tolerance:
                print("\n" + "=" * 60)
                print("[SUCCESS] Reached global goal!")
                print("=" * 60)
                self.perf_monitor.print_summary()

                # 保存视频（如果是增强版可视化器）
                if self.visualizer and hasattr(self.visualizer, 'frames') and len(self.visualizer.frames) > 0:
                    print("\n  Saving visualization video...")
                    if callable(getattr(self.visualizer, 'save_video', None)):
                        self.visualizer.save_video('receding_horizon_video.mp4')

                return True

            # === 3. 更新全局地图 ===
            with self.perf_monitor.measure('mapping'):
                map_stats = self.map_manager.update(
                    depth_image, current_pos, current_ori
                )

            print(f"Map update: +{map_stats['new_occupied']} voxels, "
                  f"total {map_stats['total_occupied']} occupied")

            # === 4. 选择局部目标 ===
            local_goal = self._select_local_goal(current_pos, global_goal, map_stats)
            print(f"Local goal: ({local_goal[0]:.2f}, {local_goal[1]:.2f}, {local_goal[2]:.2f})")

            # === 5. 规划局部路径 ===
            with self.perf_monitor.measure('planning'):
                current_path = self.rrt.plan(current_pos, local_goal)

            if current_path is None:
                print("[FAILED] Planning failed! Trying alternative local goals...")
                # 尝试备选目标
                current_path = self._try_alternative_goals(current_pos, global_goal)

                if current_path is None:
                    print("[FAILED] All alternatives failed! Stopping.")
                    self.perf_monitor.print_summary()
                    return False

            print(f"Path planned: {len(current_path)} waypoints")

            # === 6. 执行部分路径 ===
            # 跳过起点（第一个点通常是当前位置）
            if len(current_path) > 1:
                path_to_execute = current_path[1:]  # 跳过起点
            else:
                path_to_execute = current_path

            execution_length = max(1, int(len(path_to_execute) * self.execution_ratio))
            waypoints_to_execute = path_to_execute[:execution_length]

            print(f"Executing {len(waypoints_to_execute)} waypoints...")

            with self.perf_monitor.measure('execution'):
                for i, wp in enumerate(waypoints_to_execute):
                    # 强制保持目标高度，防止无人机下沉
                    wp_fixed = wp.copy()
                    wp_fixed[2] = global_goal[2]  # 使用全局目标的高度

                    print(f"  -> Waypoint {i+1}/{len(waypoints_to_execute)}: "
                          f"({wp_fixed[0]:.2f}, {wp_fixed[1]:.2f}, {wp_fixed[2]:.2f})")
                    self.drone.move_to_position(wp_fixed, velocity=self.flight_velocity)
                    self.executed_trajectory.append(wp_fixed)

            # === 7. 可视化 ===
            if self.visualizer:
                # 准备深度点云（如果是增强版可视化器）
                depth_points = None
                if hasattr(self.visualizer, 'current_depth_points'):
                    # 增强版可视化器：转换深度图为点云
                    from utils.transforms import depth_image_to_camera_points, transform_camera_to_world
                    depth_points_camera = depth_image_to_camera_points(
                        depth_image, fov_deg=90.0, subsample=8
                    )
                    depth_points = transform_camera_to_world(
                        depth_points_camera, current_pos, current_ori
                    )

                # 准备RRT树（如果需要）
                rrt_tree = None
                if hasattr(self.rrt, 'tree_edges'):
                    rrt_tree = self.rrt.tree_edges

                # 准备统计信息
                map_stats_full = self.map_manager.get_map_stats()
                stats = {
                    'iteration': iteration,
                    'dist_to_goal': dist_to_goal,
                    'occupied_voxels': map_stats_full['occupied_voxels'],
                    'free_voxels': map_stats_full['free_voxels'],
                    'unknown_voxels': map_stats_full['unknown_voxels'],
                    'new_occupied': map_stats['new_occupied'],
                    'path_waypoints': len(current_path) if current_path else 0,
                    'executing_waypoints': len(waypoints_to_execute),
                    'perf_perception': self.perf_monitor.get_average('perception'),
                    'perf_mapping': self.perf_monitor.get_average('mapping'),
                    'perf_planning': self.perf_monitor.get_average('planning'),
                    'perf_execution': self.perf_monitor.get_average('execution'),
                    'perf_total': self.perf_monitor.get_total_average(),
                }

                # 调用可视化更新
                if hasattr(self.visualizer, 'current_depth_points'):
                    # 增强版可视化器
                    self.visualizer.update(
                        map_manager=self.map_manager,
                        current_pos=current_pos,
                        current_ori=current_ori,
                        local_goal=local_goal,
                        global_goal=global_goal,
                        current_path=current_path,
                        executed_waypoints=waypoints_to_execute,
                        depth_points=depth_points,
                        rrt_tree=rrt_tree,
                        stats=stats
                    )
                else:
                    # 简单版可视化器
                    self.visualizer.update(
                        map_manager=self.map_manager,
                        current_pos=current_pos,
                        local_goal=local_goal,
                        global_goal=global_goal,
                        current_path=current_path,
                        executed_waypoints=waypoints_to_execute
                    )

            # === 8. 性能报告 ===
            self.perf_monitor.print_summary()

        print("\n" + "=" * 60)
        print(f"[FAILED] Max iterations ({self.max_iterations}) reached!")
        print("=" * 60)

        # 保存视频（如果是增强版可视化器）
        if self.visualizer and hasattr(self.visualizer, 'save_video'):
            print("\n  Saving visualization video...")
            self.visualizer.save_video('receding_horizon_video.mp4')

        return False

    def _select_local_goal(self,
                          current_pos: np.ndarray,
                          global_goal: np.ndarray,
                          map_stats: Dict) -> np.ndarray:
        """
        选择局部目标点
        - 基础策略：朝向全局目标，距离 = local_horizon
        - 安全检查：确保局部目标不在障碍物内
        - 自动调整：如果目标不安全，尝试找到安全的替代点
        """
        direction = global_goal - current_pos
        distance = np.linalg.norm(direction)

        if distance < self.local_horizon:
            # 如果全局目标很近，直接作为局部目标
            local_goal = global_goal.copy()
        else:
            # 基础局部目标
            direction = direction / distance
            local_goal = current_pos + direction * self.local_horizon

        # 强制保持当前高度（Z轴）
        local_goal[2] = current_pos[2]

        # 检查局部目标是否安全（不在障碍物内）
        if self.map_manager.esdf.distance_field is not None:
            safety_dist = self.map_manager.esdf.get_distance(local_goal)
            if safety_dist < self.map_manager.config.safety_margin:
                print(f"  [WARNING] Local goal unsafe (dist={safety_dist:.2f}m), searching for safe alternative...")
                # 尝试找到安全的局部目标
                safe_goal = self._find_safe_local_goal(current_pos, global_goal)
                if safe_goal is not None:
                    local_goal = safe_goal
                    print(f"  [OK] Found safe local goal: ({local_goal[0]:.2f}, {local_goal[1]:.2f}, {local_goal[2]:.2f})")

        return local_goal

    def _find_safe_local_goal(self,
                              current_pos: np.ndarray,
                              global_goal: np.ndarray) -> Optional[np.ndarray]:
        """
        在障碍物周围找到安全的局部目标
        尝试不同角度和距离的组合
        """
        direction = global_goal - current_pos
        direction[2] = 0  # 只在XY平面旋转
        dist = np.linalg.norm(direction[:2])
        if dist > 0:
            direction = direction / np.linalg.norm(direction)

        # 尝试不同角度（优先选择偏离较小的角度）
        angles = [15, -15, 30, -30, 45, -45, 60, -60, 75, -75, 90, -90]
        distances = [self.local_horizon, self.local_horizon * 0.8, self.local_horizon * 0.6]

        best_goal = None
        best_score = -np.inf

        for angle in angles:
            for dist in distances:
                rotated_dir = self._rotate_direction_yaw(direction, angle)
                candidate = current_pos + rotated_dir * dist
                candidate[2] = current_pos[2]  # 保持高度

                # 检查安全性
                safety_dist = self.map_manager.esdf.get_distance(candidate)
                if safety_dist > self.map_manager.config.safety_margin:
                    # 计算得分：安全距离 + 朝向目标的进度
                    progress = np.dot(candidate - current_pos, global_goal - current_pos)
                    score = safety_dist * 0.3 + progress * 0.7

                    if score > best_score:
                        best_score = score
                        best_goal = candidate

        return best_goal

    def _sample_candidate_goals(self,
                                current_pos: np.ndarray,
                                base_direction: np.ndarray) -> List[np.ndarray]:
        """
        在基础方向附近采样候选局部目标
        """
        candidates = []

        # 基础方向
        candidates.append(current_pos + base_direction * self.local_horizon)

        # 左右偏移 15 度
        for angle in [-15, 15]:
            rotated_dir = self._rotate_direction_yaw(base_direction, angle)
            candidates.append(current_pos + rotated_dir * self.local_horizon)

        return candidates

    def _rotate_direction_yaw(self, direction: np.ndarray, angle_deg: float) -> np.ndarray:
        """
        绕 Z 轴旋转方向向量（yaw 旋转）
        """
        angle_rad = np.radians(angle_deg)
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)

        # 2D 旋转矩阵（XY 平面）
        rotated = np.array([
            cos_a * direction[0] - sin_a * direction[1],
            sin_a * direction[0] + cos_a * direction[1],
            direction[2]
        ])

        # 归一化
        return rotated / np.linalg.norm(rotated)

    def _select_best_candidate(self,
                               candidates: List[np.ndarray],
                               global_goal: np.ndarray,
                               current_pos: np.ndarray) -> np.ndarray:
        """
        评估候选点，选择最佳局部目标
        评分 = 0.6 * 朝向全局目标进度 + 0.3 * 信息增益 + 0.1 * 安全性
        """
        best_score = -np.inf
        best_candidate = candidates[0]

        for candidate in candidates:
            # 1. 进度得分
            progress = np.dot(
                candidate - current_pos,
                global_goal - current_pos
            ) / np.linalg.norm(global_goal - current_pos)

            # 2. 信息增益（简化：检查视野内未知体素数量）
            info_gain = self._estimate_information_gain(candidate)

            # 3. 安全性（距离最近障碍物的距离）
            safety = self.map_manager.esdf.get_distance(candidate)

            score = 0.6 * progress + 0.3 * info_gain + 0.1 * min(safety, 5.0) / 5.0

            if score > best_score:
                best_score = score
                best_candidate = candidate

        return best_candidate

    def _estimate_information_gain(self, position: np.ndarray) -> float:
        """
        估计从某位置观测的信息增益（简化版）
        返回 0-1 之间的值
        """
        # 简化实现：检查周围未知体素的比例
        voxel_grid = self.map_manager.voxel_grid
        idx = voxel_grid.world_to_grid(position)

        if not voxel_grid.is_valid_index(idx):
            return 0.0

        # 检查周围 5x5x5 的区域
        unknown_count = 0
        total_count = 0

        for dx in range(-2, 3):
            for dy in range(-2, 3):
                for dz in range(-2, 3):
                    check_idx = (idx[0] + dx, idx[1] + dy, idx[2] + dz)
                    if voxel_grid.is_valid_index(check_idx):
                        total_count += 1
                        if voxel_grid.grid[check_idx] == 0:
                            unknown_count += 1

        if total_count == 0:
            return 0.0

        return unknown_count / total_count

    def _try_alternative_goals(self,
                               current_pos: np.ndarray,
                               global_goal: np.ndarray) -> Optional[List[np.ndarray]]:
        """
        尝试备选局部目标
        包括更大角度和后退机制
        """
        direction = global_goal - current_pos
        direction[2] = 0  # XY平面
        norm = np.linalg.norm(direction[:2])
        if norm > 0:
            direction = direction / np.linalg.norm(direction)

        # 尝试不同角度（从小到大）和距离
        angles = [-30, 30, -45, 45, -60, 60, -75, 75, -90, 90, -120, 120, -150, 150, 180]
        distances = [self.local_horizon * 0.7, self.local_horizon * 0.5, self.local_horizon * 0.4]

        for angle in angles:
            for distance in distances:
                rotated_dir = self._rotate_direction_yaw(direction, angle)
                alternative_goal = current_pos + rotated_dir * distance
                alternative_goal[2] = current_pos[2]  # 保持高度

                # 先检查目标点是否安全
                if self.map_manager.esdf.distance_field is not None:
                    safety_dist = self.map_manager.esdf.get_distance(alternative_goal)
                    if safety_dist < self.map_manager.config.safety_margin * 0.5:
                        continue  # 跳过不安全的目标

                path = self.rrt.plan(current_pos, alternative_goal)
                if path is not None:
                    print(f"  [OK] Found alternative path (angle={angle} deg, dist={distance:.1f}m)")
                    return path

        # 最后尝试：后退
        print("  [WARNING] Trying retreat strategy...")
        retreat_path = self._try_retreat(current_pos, global_goal)
        if retreat_path is not None:
            return retreat_path

        return None

    def _try_retreat(self,
                     current_pos: np.ndarray,
                     global_goal: np.ndarray) -> Optional[List[np.ndarray]]:
        """
        后退策略：当前方被完全阻挡时，尝试后退到更开阔的位置
        """
        direction = global_goal - current_pos
        direction[2] = 0
        norm = np.linalg.norm(direction[:2])
        if norm > 0:
            direction = direction / np.linalg.norm(direction)

        # 后退方向（与目标相反）
        retreat_dir = -direction

        # 尝试不同的后退距离
        for retreat_dist in [2.0, 3.0, 4.0]:
            retreat_goal = current_pos + retreat_dir * retreat_dist
            retreat_goal[2] = current_pos[2]

            # 检查后退点是否安全
            if self.map_manager.esdf.distance_field is not None:
                safety_dist = self.map_manager.esdf.get_distance(retreat_goal)
                if safety_dist < self.map_manager.config.safety_margin:
                    continue

            path = self.rrt.plan(current_pos, retreat_goal)
            if path is not None:
                print(f"  [OK] Found retreat path (dist={retreat_dist:.1f}m)")
                return path

        return None

    def get_trajectory(self) -> List[np.ndarray]:
        """获取已执行的轨迹"""
        return self.executed_trajectory.copy()
