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
from planning.dubins_3d import Dubins3DParams, dubins_3d_blend_junction
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
                 config: Dict,
                 energy_model=None):
        self.map_manager = map_manager
        self.drone = drone

        # 滚动规划参数
        self.local_horizon = config.get('local_horizon', 6.0)  # 米
        self.execution_ratio = config.get('execution_ratio', 0.4)
        self.replan_threshold = config.get('replan_threshold', 1.5)
        self.goal_tolerance = config.get('goal_tolerance', 1.0)
        self.max_iterations = config.get('max_iterations', 50)
        self.flight_velocity = config.get('flight_velocity', 2.0)
        self.timeout_seconds = config.get('timeout_seconds', 300)  # 默认5分钟超时

        # 能耗模型（用于能量感知规划）
        self.energy_model = energy_model

        # RRT* 规划器（传入能耗模型）
        self.rrt = RRTStar(
            map_manager.voxel_grid,
            map_manager.esdf,
            map_manager.config,
            energy_model=energy_model
        )

        # 能耗统计
        self.total_energy_consumed = 0.0
        self.energy_history = []

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

        # 实时可视化回调（外部设置）
        self.realtime_callback = None

        # 路径缓存与复用：保存上次规划的完整路径
        self._cached_path = None  # 上次规划的完整路径
        self._cached_path_index = 0  # 当前执行到的索引
        self._last_flight_direction = None  # 上次飞行方向（用于方向连续性）

        # wall-follow 状态：记住侧移方向，避免来回摇摆
        self._wall_follow_direction = None  # 'left' or 'right', 一旦选定就坚持
        self._wall_follow_blocked_count = 0  # wall-follow 连续被 blocked 的次数
        self._wall_follow_cooldown = 0  # 方向重置后冷却计数，防止立即重新进入 wall-follow
        self._wall_follow_iterations = 0  # 当前方向连续 wall-follow 的次数
        self._wall_follow_max_iterations = 15  # 同方向最大 wall-follow 次数（墙宽40m，每步5m，需要至少8步）

        # 圆弧平滑参数（3D Dubins 替代 B-spline）
        self.arc_smoothing = config.get('arc_smoothing', True)
        self.arc_overlap_points = config.get('arc_overlap_points', 3)
        self._executed_tail = []  # 最近执行的N个航点，用于衔接平滑

        # 3D Dubins 参数
        self.dubins_params = Dubins3DParams(
            turning_radius=config.get('dubins_turning_radius', 1.5),
            max_climb_angle=config.get('dubins_max_climb_angle', 30.0),
            sample_distance=config.get('dubins_sample_distance', 0.3),
        )

        # yaw 扫描降频
        self._yaw_scan_interval = 3  # 每3轮 blocked 扫描一次
        self._blocked_since_last_scan = 0  # 距上次扫描的 blocked 轮数

        # 卡住检测：记录最近位置历史，检测原地打转
        self._position_history = []  # 最近 N 次迭代的位置
        self._stuck_counter = 0  # 连续卡住计数
        self._best_forward_x = -np.inf  # 朝目标方向的最大前进距离

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
        print(f"  Timeout: {self.timeout_seconds}s")
        print("=" * 60)

        iteration = 0
        current_path = None
        start_time = time.time()

        while iteration < self.max_iterations:
            # 超时检查
            elapsed = time.time() - start_time
            if elapsed > self.timeout_seconds:
                print(f"\n[TIMEOUT] {self.timeout_seconds}s elapsed, stopping.")
                break

            iteration += 1
            print(f"\n{'='*60}")
            print(f"Iteration {iteration}")
            print(f"{'='*60}")

            # === 1. 获取当前状态 ===
            with self.perf_monitor.measure('perception'):
                current_pos, current_ori = self.drone.get_pose()
                depth_image = self.drone.get_depth_image()

            print(f"Current position: ({current_pos[0]:.2f}, {current_pos[1]:.2f}, {current_pos[2]:.2f})")

            # 初始化 _best_forward_x（第一次迭代时）
            if self._best_forward_x == -np.inf:
                self._best_forward_x = self._compute_forward_progress(
                    current_pos, global_goal)
                print(f"  [INIT] best_forward_x = {self._best_forward_x:.1f}m")

            # === 1.3 高度修正 ===
            # 在航点执行时通过 move_to_position 的 Z 分量持续修正

            # === 1.5 卡住检测 ===
            is_stuck = self._detect_stuck(current_pos, global_goal)
            if is_stuck:
                print(f"  [STUCK] Detected oscillation! stuck_counter={self._stuck_counter}")
                if self._stuck_counter >= 3:
                    # 连续卡住3次，强制翻转 wall-follow 方向
                    old_dir = self._wall_follow_direction
                    if old_dir == 'left':
                        self._wall_follow_direction = 'right'
                    elif old_dir == 'right':
                        self._wall_follow_direction = 'left'
                    else:
                        self._wall_follow_direction = 'right'  # 默认尝试 right
                    self._wall_follow_iterations = 0
                    self._wall_follow_blocked_count = 0
                    self._stuck_counter = 0
                    self._position_history.clear()
                    print(f"  [STUCK] Flipping wall-follow direction: {old_dir} -> {self._wall_follow_direction}")

            # === 2. 检查是否到达全局目标 ===
            dist_to_goal = np.linalg.norm(current_pos - global_goal)
            print(f"Distance to global goal: {dist_to_goal:.2f}m")

            if dist_to_goal < self.goal_tolerance:
                print("\n" + "=" * 60)
                print("[SUCCESS] Reached global goal!")
                print("=" * 60)
                self.perf_monitor.print_summary()
                self._print_energy_summary()

                # 保存视频（如果是增强版可视化器）
                if self.visualizer and hasattr(self.visualizer, 'frames') and len(self.visualizer.frames) > 0:
                    print("\n  Saving visualization video...")
                    if callable(getattr(self.visualizer, 'save_video', None)):
                        self.visualizer.save_video('receding_horizon_video.mp4')

                return True

            # === 3. 更新全局地图（含 yaw 扫描） ===
            with self.perf_monitor.measure('mapping'):
                # 先用当前朝向更新
                map_stats = self.map_manager.update(
                    depth_image, current_pos, current_ori
                )

                # 检查前方是否被挡住 — 如果是，执行 yaw 扫描获取侧方信息
                forward_blocked = self._is_forward_blocked(current_pos, global_goal)

                # 无论前方是否 blocked，都检查前进进度并更新 best_forward_x
                if self._wall_follow_direction is not None:
                    forward_progress = self._compute_forward_progress(
                        current_pos, global_goal)
                    if forward_progress > self._best_forward_x:
                        self._best_forward_x = forward_progress

                    # 只有当前方不再被挡住且 X 已超过墙体范围时才重置方向
                    if not forward_blocked and current_pos[0] > 33.0:
                        print(f"  [WALL-FOLLOW] Forward clear and past Row1 (X={current_pos[0]:.1f})! Resetting direction memory")
                        self._wall_follow_direction = None
                        self._wall_follow_cooldown = 1
                        self._wall_follow_iterations = 0
                        self._blocked_since_last_scan = 0
                    elif not forward_blocked:
                        # X+ 通畅但还没绕过墙，继续 wall-follow
                        print(f"  [WALL-FOLLOW] Forward seems clear but X={current_pos[0]:.1f}<33, keeping wall-follow")

                if forward_blocked:
                    self._blocked_since_last_scan += 1
                    if self._blocked_since_last_scan >= self._yaw_scan_interval:
                        print("  [SCAN] Forward blocked, performing yaw scan...")
                        self._yaw_scan(current_pos, global_goal)
                        self._blocked_since_last_scan = 0
                    else:
                        print(f"  [SCAN] Skipping yaw scan ({self._blocked_since_last_scan}/{self._yaw_scan_interval})")

            print(f"Map update: +{map_stats['new_occupied']} voxels, "
                  f"total {map_stats['total_occupied']} occupied")

            # === 4. 选择局部目标 ===
            # 如果前方被墙挡住，使用沿墙侧移策略而不是 RRT*
            # 冷却期内：沿 X+ 方向前进（沿墙边飞行），绕过障碍后再转向目标
            if self._wall_follow_cooldown > 0:
                self._wall_follow_cooldown -= 1
                if forward_blocked:
                    # 冷却期内，沿纯 X+ 方向前进（沿墙边飞行）
                    forward_target = current_pos.copy()
                    forward_target[0] += self.local_horizon
                    forward_target[2] = current_pos[2]

                    # 检查纯 X+ 方向是否安全
                    x_blocked = False
                    for d in np.arange(0.5, self.local_horizon, 1.0):
                        check_pt = current_pos.copy()
                        check_pt[0] += d
                        dist = self.map_manager.esdf.get_distance(check_pt)
                        if dist < self.map_manager.config.safety_margin:
                            x_blocked = True
                            break

                    if not x_blocked:
                        print(f"  [WALL-FOLLOW] Cooldown: flying forward (X+) along wall edge")
                        current_path = [current_pos.copy(), forward_target]
                    else:
                        print(f"  [WALL-FOLLOW] Cooldown: X+ blocked too, using RRT*")
                        forward_blocked = False
                else:
                    pass  # forward not blocked, use normal RRT*
            elif forward_blocked:
                # === 前方被挡 ===
                # 如果已经在 wall-follow 状态，直接继续，不浪费时间尝试 RRT*
                if self._wall_follow_direction is not None:
                    print(f"  [BLOCKED] Already in wall-follow ({self._wall_follow_direction}), skipping RRT*")
                else:
                    # 首次 blocked，尝试 RRT* 找缝隙
                    print(f"  [BLOCKED] Forward blocked, trying RRT* first before wall-follow...")
                    local_goal = self._select_local_goal(current_pos, global_goal, map_stats)
                    print(f"  RRT* local goal: ({local_goal[0]:.2f}, {local_goal[1]:.2f}, {local_goal[2]:.2f})")

                    with self.perf_monitor.measure('planning'):
                        rrt_path = self.rrt.plan(current_pos, local_goal)

                    if rrt_path is not None and len(rrt_path) >= 2:
                        # RRT* 找到了路径（可能穿过缝隙），使用它
                        current_path = rrt_path
                        forward_blocked = False
                        # 如果 RRT* 成功，重置 wall-follow 状态
                        self._wall_follow_direction = None
                        self._wall_follow_iterations = 0
                        self._wall_follow_blocked_count = 0
                        print(f"  [BLOCKED] RRT* found path through gap! {len(rrt_path)} waypoints")
                    else:
                        # RRT* 也失败了，进入 wall-follow
                        print(f"  [BLOCKED] RRT* failed, falling back to wall-follow")

                if forward_blocked:
                    # 检查 Y 偏移是否已经足够大
                    y_offset_from_start = abs(current_pos[1] - global_goal[1])
                    if y_offset_from_start > 22.0:
                        print(f"  [WALL-FOLLOW] Y offset={y_offset_from_start:.1f}m > 22m, switching to X+ advance")
                        x_target = current_pos.copy()
                        x_target[0] += self.local_horizon
                        x_target[2] = global_goal[2]
                        x_safe = True
                        for d in np.arange(0.5, self.local_horizon, 1.0):
                            check_pt = current_pos.copy()
                            check_pt[0] += d
                            if self.map_manager.esdf.distance_field is not None:
                                dist = self.map_manager.esdf.get_distance(check_pt)
                                if dist < self.map_manager.config.safety_margin:
                                    x_safe = False
                                    break
                        if x_safe:
                            current_path = [current_pos.copy(), x_target]
                            forward_blocked = False
                        else:
                            pass

                    if forward_blocked:
                        if self._wall_follow_blocked_count >= 3:
                            # wall-follow 连续 blocked 太多次，翻转方向
                            old_dir = self._wall_follow_direction
                            self._wall_follow_direction = 'right' if old_dir == 'left' else 'left'
                            self._wall_follow_blocked_count = 0
                            self._wall_follow_iterations = 0
                            print(f"  [WALL-FOLLOW] Blocked {old_dir} 3 times, flipping to {self._wall_follow_direction}")

                        if self._wall_follow_iterations >= self._wall_follow_max_iterations:
                            old_dir = self._wall_follow_direction
                            self._wall_follow_direction = 'right' if old_dir == 'left' else 'left' if old_dir == 'right' else None
                            self._wall_follow_iterations = 0
                            print(f"  [WALL-FOLLOW] Max iterations ({self._wall_follow_max_iterations}) "
                                  f"in {old_dir} direction, flipping to {self._wall_follow_direction}")

                        wall_follow_path = self._wall_follow_step(current_pos, global_goal)
                        if wall_follow_path is not None:
                            current_path = wall_follow_path
                            self._wall_follow_blocked_count = 0
                            self._wall_follow_iterations += 1
                            print(f"[WALL-FOLLOW] Lateral move: {len(current_path)} waypoints "
                                  f"(iter {self._wall_follow_iterations}/{self._wall_follow_max_iterations})")
                        else:
                            self._wall_follow_blocked_count += 1
                            forward_blocked = False
                            print(f"  [WALL-FOLLOW] Blocked count: {self._wall_follow_blocked_count}/3")

            if not forward_blocked:
                # === 4.5 Y 回归模式 ===
                # 绕过 Row1 后 Y 偏移大，用增量式回归策略
                # 策略：先朝 Y=0 方向走，如果被墙挡住就沿墙外侧（保持安全距离）
                #       飞 X+ 直到超过墙体 X 范围，再转向 Y=0
                y_offset = abs(current_pos[1] - global_goal[1])
                if y_offset > 8.0 and current_pos[0] > 30.0:
                    y_dir = 1.0 if current_pos[1] < global_goal[1] else -1.0
                    import time as _time

                    print(f"  [Y-RETURN] Y offset={y_offset:.1f}m, pos=({current_pos[0]:.1f}, {current_pos[1]:.1f})")

                    # 如果 Y 偏移很大且还在障碍物 X 范围内（X<55），
                    # 优先沿 X+ 飞行，不尝试 Y-return（避免贴墙卡住）
                    if y_offset > 15.0 and current_pos[0] < 55.0:
                        print(f"  [Y-RETURN] Large Y offset + still in obstacle zone (X<55), prioritizing X+ advance")
                        current_path = None
                        # 尝试 X+ 前进，同时微量远离墙体
                        for step_x in [8.0, 6.0, 5.0, 3.0]:
                            t = current_pos.copy()
                            t[0] += step_x
                            # 微量远离墙体（增加 Y 偏移方向的距离）
                            away_dir = 1.0 if current_pos[1] > 0 else -1.0
                            t[1] += away_dir * 1.0
                            t[2] = global_goal[2]
                            if self.map_manager.esdf.distance_field is not None:
                                d = self.map_manager.esdf.get_distance(t)
                                if d >= self.map_manager.config.safety_margin and self._check_path_safe(current_pos, t):
                                    print(f"  [Y-RETURN] X+ advance to ({t[0]:.1f}, {t[1]:.1f})")
                                    current_path = [current_pos.copy(), t]
                                    break
                        # 如果带微量偏移的 X+ 也不行，试纯 X+
                        if current_path is None:
                            for step_x in [8.0, 5.0, 3.0]:
                                t = current_pos.copy()
                                t[0] += step_x
                                t[2] = global_goal[2]
                                if self.map_manager.esdf.distance_field is not None:
                                    d = self.map_manager.esdf.get_distance(t)
                                    if d >= self.map_manager.config.safety_margin and self._check_path_safe(current_pos, t):
                                        print(f"  [Y-RETURN] Pure X+ to ({t[0]:.1f}, {t[1]:.1f})")
                                        current_path = [current_pos.copy(), t]
                                        break
                        if current_path is None:
                            # X+ 也走不了，尝试远离墙体
                            for step_away in [3.0, 2.0]:
                                t = current_pos.copy()
                                away_dir = 1.0 if current_pos[1] > 0 else -1.0
                                t[1] += away_dir * step_away
                                t[0] += 1.0
                                t[2] = global_goal[2]
                                if self.map_manager.esdf.distance_field is not None:
                                    d = self.map_manager.esdf.get_distance(t)
                                    if d >= self.map_manager.config.safety_margin and self._check_path_safe(current_pos, t):
                                        print(f"  [Y-RETURN] Move away from wall to ({t[0]:.1f}, {t[1]:.1f})")
                                        current_path = [current_pos.copy(), t]
                                        break
                        if current_path is None:
                            print(f"  [Y-RETURN] All X+ strategies failed, hovering")
                            current_path = [current_pos.copy(), current_pos.copy()]
                    else:
                        # Y 偏移较小或已过障碍物区域，正常 Y-return
                        # 先朝 Y 回归方向扫描
                        y_return_yaw = 90.0 if y_dir > 0 else -90.0
                        scan_yaws = [y_return_yaw + offset for offset in [-45, 0, 45]]
                        for yaw in scan_yaws:
                            self.drone.set_yaw(yaw, duration=0.3)
                            _time.sleep(0.15)
                            pos, ori = self.drone.get_pose()
                            depth = self.drone.get_depth_image()
                            self.map_manager.update(depth, pos, ori)

                        # 尝试多种回归策略，按优先级排列
                        current_path = None

                        # 策略1: 直接朝 Y 方向走（大步 → 小步）
                        for step_y in [5.0, 3.0, 2.0, 1.0]:
                            t = current_pos.copy()
                            t[1] += y_dir * step_y
                            t[0] += 0.5  # 微量 X 前进
                            t[2] = global_goal[2]
                            if self.map_manager.esdf.distance_field is not None:
                                d = self.map_manager.esdf.get_distance(t)
                                if d >= self.map_manager.config.safety_margin and self._check_path_safe(current_pos, t):
                                    print(f"  [Y-RETURN] Direct Y step {step_y:.0f}m to ({t[0]:.1f}, {t[1]:.1f})")
                                    current_path = [current_pos.copy(), t]
                                    break

                        # 策略2: 斜向移动（Y + X 同时前进，绕过墙角）
                        if current_path is None:
                            for step_y, step_x in [(3.0, 4.0), (2.0, 5.0), (1.0, 6.0)]:
                                t = current_pos.copy()
                                t[1] += y_dir * step_y
                                t[0] += step_x
                                t[2] = global_goal[2]
                                if self.map_manager.esdf.distance_field is not None:
                                    d = self.map_manager.esdf.get_distance(t)
                                    if d >= self.map_manager.config.safety_margin and self._check_path_safe(current_pos, t):
                                        print(f"  [Y-RETURN] Diagonal to ({t[0]:.1f}, {t[1]:.1f})")
                                        current_path = [current_pos.copy(), t]
                                        break

                        # 策略3: 纯 X+ 前进
                        if current_path is None:
                            for step_x in [5.0, 3.0]:
                                t = current_pos.copy()
                                t[0] += step_x
                                t[1] += y_dir * 0.5
                                t[2] = global_goal[2]
                                if self.map_manager.esdf.distance_field is not None:
                                    d = self.map_manager.esdf.get_distance(t)
                                    if d >= self.map_manager.config.safety_margin and self._check_path_safe(current_pos, t):
                                        print(f"  [Y-RETURN] X+ step to ({t[0]:.1f}, {t[1]:.1f})")
                                        current_path = [current_pos.copy(), t]
                                        break

                        if current_path is None:
                            # 所有增量策略失败，fallback 到 RRT* 规划
                            print(f"  [Y-RETURN] Incremental strategies failed, falling back to RRT*")
                            local_goal = self._select_local_goal(current_pos, global_goal, map_stats)
                            print(f"  Local goal: ({local_goal[0]:.2f}, {local_goal[1]:.2f}, {local_goal[2]:.2f})")
                            with self.perf_monitor.measure('planning'):
                                current_path = self.rrt.plan(current_pos, local_goal)
                            if current_path is None:
                                current_path = self._try_alternative_goals(current_pos, global_goal)
                            if current_path is None:
                                print(f"  [Y-RETURN] RRT* also failed, hovering")
                                current_path = [current_pos.copy(), current_pos.copy()]

                    # 恢复朝向
                    direction_to_goal = global_goal - current_pos
                    base_yaw = np.degrees(np.arctan2(direction_to_goal[1], direction_to_goal[0]))
                    self.drone.set_yaw(base_yaw, duration=0.3)
                    _time.sleep(0.1)
                else:
                    # === 路径缓存复用机制 ===
                    # 检查是否可以复用上次规划的剩余路径
                    reuse_cached = False
                    if self._cached_path is not None and self._cached_path_index < len(self._cached_path):
                        remaining_path = self._cached_path[self._cached_path_index:]
                        if len(remaining_path) > 2:
                            # 检查剩余路径是否仍然有效（无碰撞）
                            path_valid = True
                            for i in range(len(remaining_path) - 1):
                                if not self._check_path_safe(remaining_path[i], remaining_path[i+1]):
                                    path_valid = False
                                    break

                            if path_valid:
                                # 检查起点是否接近当前位置
                                dist_to_path_start = np.linalg.norm(current_pos - remaining_path[0])
                                if dist_to_path_start < 2.0:
                                    # 检查路径方向：第一步必须朝目标前进（不能往回飞）
                                    if len(remaining_path) >= 2:
                                        step_dir = remaining_path[1] - current_pos
                                        goal_dir = global_goal - current_pos
                                        # 投影到目标方向，必须为正（前进）
                                        forward_proj = np.dot(step_dir[:2], goal_dir[:2])
                                        if forward_proj < 0:
                                            print(f"  [PATH-REUSE] Cached path goes backward, replanning")
                                        else:
                                            print(f"  [PATH-REUSE] Using cached path ({len(remaining_path)} waypoints remaining)")
                                            current_path = remaining_path
                                            reuse_cached = True
                                    else:
                                        print(f"  [PATH-REUSE] Using cached path ({len(remaining_path)} waypoints remaining)")
                                        current_path = remaining_path
                                        reuse_cached = True

                    if not reuse_cached:
                        # 正常 RRT* 规划
                        local_goal = self._select_local_goal(current_pos, global_goal, map_stats)
                        print(f"Local goal: ({local_goal[0]:.2f}, {local_goal[1]:.2f}, {local_goal[2]:.2f})")

                        with self.perf_monitor.measure('planning'):
                            # 正常 RRT* 规划
                            current_path = self.rrt.plan(current_pos, local_goal)

                        if current_path is None:
                            print("[FAILED] Planning failed! Trying alternative local goals...")
                            current_path = self._try_alternative_goals(current_pos, global_goal)

                            if current_path is None:
                                print("[FAILED] All alternatives failed! Stopping.")
                                self.perf_monitor.print_summary()
                                return False

                        # 圆弧平滑：在已执行轨迹和新路径衔接处做 B-spline 过渡
                        if (self.arc_smoothing
                                and len(self._executed_tail) >= 2
                                and current_path is not None
                                and len(current_path) >= 2):
                            current_path = self._blend_junction(self._executed_tail, current_path)

                        # 缓存新规划的路径
                        self._cached_path = current_path
                        self._cached_path_index = 0

            print(f"Path planned: {len(current_path)} waypoints")

            # === 6. 执行部分路径 ===
            # 跳过起点（第一个点通常是当前位置）
            if len(current_path) > 1:
                path_to_execute = current_path[1:]  # 跳过起点
            else:
                path_to_execute = current_path

            execution_length = max(1, int(len(path_to_execute) * self.execution_ratio))
            waypoints_to_execute = path_to_execute[:execution_length]

            # 更新缓存路径索引
            if self._cached_path is not None:
                self._cached_path_index += execution_length

            print(f"Executing {len(waypoints_to_execute)} waypoints...")

            with self.perf_monitor.measure('execution'):
                abort_execution = False
                for i, wp in enumerate(waypoints_to_execute):
                    # 高度处理：允许 Dubins 爬升角产生的 Z 变化，但限制在目标高度附近
                    wp_fixed = wp.copy()
                    wp_fixed[2] = np.clip(wp[2], global_goal[2] - 2.0, global_goal[2] + 2.0)
                    actual_pos, actual_ori = self.drone.get_pose()

                    # === 方向检查：跳过往回飞的航点 ===
                    step_vec = wp_fixed - actual_pos
                    goal_vec = global_goal - actual_pos
                    forward_proj = np.dot(step_vec[:2], goal_vec[:2])
                    if forward_proj < 0 and np.linalg.norm(step_vec[:2]) > 0.5:
                        print(f"  [SKIP] Waypoint {i+1} goes backward, skipping")
                        continue

                    print(f"  -> Waypoint {i+1}/{len(waypoints_to_execute)}: "
                          f"({wp_fixed[0]:.2f}, {wp_fixed[1]:.2f}, {wp_fixed[2]:.2f})")

                    # === 执行前碰撞检测 ===
                    # 获取当前真实位置，检查到下一个航点的路径是否安全
                    actual_pos, actual_ori = self.drone.get_pose()
                    if self.map_manager.esdf.distance_field is not None:
                        # 检查当前位置的安全距离
                        current_safety = self.map_manager.esdf.get_distance(actual_pos)
                        if current_safety < self.map_manager.config.safety_margin * 0.5:
                            print(f"  [DANGER] Current position unsafe (dist={current_safety:.2f}m), aborting execution!")
                            self.drone.hover()
                            abort_execution = True
                            break

                        # 检查到下一个航点的路径是否无碰撞
                        if not self._check_path_safe(actual_pos, wp_fixed):
                            print(f"  [DANGER] Path to waypoint {i+1} blocked, aborting execution!")
                            self.drone.hover()
                            abort_execution = True
                            break

                    # 计算并累积能耗（如果有能耗模型）
                    if self.energy_model is not None and len(self.executed_trajectory) > 0:
                        prev_pos = self.executed_trajectory[-1]
                        segment_energy, _ = self.energy_model.compute_energy_for_segment(
                            prev_pos, wp_fixed, self.flight_velocity
                        )
                        self.total_energy_consumed += segment_energy
                        self.energy_history.append({
                            'from': prev_pos.copy(),
                            'to': wp_fixed.copy(),
                            'energy': segment_energy
                        })

                    self.drone.move_to_position(wp_fixed, velocity=self.flight_velocity)
                    self.executed_trajectory.append(wp_fixed)

                    # 维护圆弧平滑所需的执行尾部
                    self._executed_tail.append(wp_fixed.copy())
                    if len(self._executed_tail) > self.arc_overlap_points + 1:
                        self._executed_tail = self._executed_tail[-(self.arc_overlap_points + 1):]

                    # 更新飞行方向（用于下次规划的方向连续性）
                    if len(self.executed_trajectory) >= 2:
                        direction = self.executed_trajectory[-1] - self.executed_trajectory[-2]
                        direction_norm = np.linalg.norm(direction)
                        if direction_norm > 0.1:
                            self._last_flight_direction = direction / direction_norm

                    # === 执行后安全检查 ===
                    # 到达航点后，快速感知并更新地图，检查前方是否安全
                    if i < len(waypoints_to_execute) - 1:
                        post_pos, post_ori = self.drone.get_pose()
                        post_depth = self.drone.get_depth_image()
                        self.map_manager.update(post_depth, post_pos, post_ori)

                        post_safety = self.map_manager.esdf.get_distance(post_pos)
                        if post_safety < self.map_manager.config.safety_margin:
                            print(f"  [WARNING] Post-move safety low ({post_safety:.2f}m), triggering replan")
                            abort_execution = True
                            break

                if abort_execution:
                    print("  [REPLAN] Execution aborted, will replan in next iteration")

            # === 7. 实时可视化回调 ===
            if self.realtime_callback is not None:
                try:
                    obstacles = self.map_manager.voxel_grid.get_occupied_voxels()
                    self.realtime_callback(current_pos, obstacles)
                except Exception as e:
                    print(f"  [WARNING] Realtime callback error: {e}")

            # === 8. 可视化 ===
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
                    'total_energy': self.total_energy_consumed,
                    'energy_aware': self.map_manager.config.energy_aware if hasattr(self.map_manager.config, 'energy_aware') else False,
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
            if self.energy_model is not None:
                print(f"  Total energy consumed: {self.total_energy_consumed:.1f} J")

        print("\n" + "=" * 60)
        print(f"[FAILED] Max iterations ({self.max_iterations}) reached!")
        print("=" * 60)
        self._print_energy_summary()

        # 保存视频（如果是增强版可视化器）
        if self.visualizer and hasattr(self.visualizer, 'save_video'):
            print("\n  Saving visualization video...")
            self.visualizer.save_video('receding_horizon_video.mp4')

        return False

    def _print_energy_summary(self):
        """打印能耗统计摘要"""
        if self.energy_model is None or self.total_energy_consumed == 0:
            return

        print("\n" + "-" * 40)
        print("Energy Consumption Summary")
        print("-" * 40)
        print(f"  Total energy: {self.total_energy_consumed:.1f} J ({self.total_energy_consumed/3600:.3f} Wh)")

        if len(self.executed_trajectory) > 1:
            total_dist = sum(
                np.linalg.norm(self.executed_trajectory[i+1] - self.executed_trajectory[i])
                for i in range(len(self.executed_trajectory) - 1)
            )
            print(f"  Total distance: {total_dist:.1f} m")
            print(f"  Energy per meter: {self.total_energy_consumed/total_dist:.1f} J/m")

            # 估算电池消耗
            battery_wh = self.energy_model.params.battery_voltage * self.energy_model.params.battery_capacity / 1000
            battery_used_pct = (self.total_energy_consumed / 3600) / battery_wh * 100
            print(f"  Battery used: {battery_used_pct:.1f}%")

    def _blend_junction(self, executed_tail: List[np.ndarray],
                        new_path: List[np.ndarray]) -> List[np.ndarray]:
        """
        3D Dubins 平滑：在已执行轨迹末尾和新规划路径开头之间做 CSC Dubins 过渡，
        消除硬拐点，使航向和爬升角连续。

        Args:
            executed_tail: 已飞过的末尾 2-3 个点
            new_path: 新规划的路径

        Returns:
            平滑后的路径（替换了开头部分），或原路径（平滑失败时）
        """
        return dubins_3d_blend_junction(
            executed_tail, new_path,
            self.dubins_params,
            safety_check=self._check_path_safe,
        )

    def get_energy_stats(self) -> dict:
        """获取能耗统计信息"""
        if len(self.executed_trajectory) < 2:
            return {'total_energy_joules': 0, 'total_energy_wh': 0,
                    'total_distance': 0, 'energy_per_meter': 0,
                    'num_segments': 0, 'energy_history': []}

        total_dist = sum(
            np.linalg.norm(self.executed_trajectory[i+1] - self.executed_trajectory[i])
            for i in range(len(self.executed_trajectory) - 1)
        )

        return {
            'total_energy_joules': self.total_energy_consumed,
            'total_energy_wh': self.total_energy_consumed / 3600,
            'total_distance': total_dist,
            'energy_per_meter': self.total_energy_consumed / total_dist if total_dist > 0 else 0,
            'num_segments': len(self.energy_history),
            'energy_history': self.energy_history
        }

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
            # Y偏移过大时，优先回归目标Y轴（穿过缺口）
            y_offset = abs(current_pos[1] - global_goal[1])
            if y_offset > 10.0:
                # 检查当前 Y 方向是否有已知墙体挡住回归路径
                # 如果有，先沿 X+ 前进绕过墙体，再回归 Y
                y_dir = -1.0 if current_pos[1] > global_goal[1] else 1.0
                y_return_blocked = False
                for d in np.arange(1.0, min(y_offset, 8.0), 1.0):
                    check_pt = current_pos.copy()
                    check_pt[1] += y_dir * d
                    if self.map_manager.esdf.distance_field is not None:
                        dist = self.map_manager.esdf.get_distance(check_pt)
                        if dist < self.map_manager.config.safety_margin:
                            y_return_blocked = True
                            break

                if y_return_blocked:
                    # Y 方向被墙挡住，先沿 X+ 前进
                    local_goal = current_pos.copy()
                    local_goal[0] += self.local_horizon
                    local_goal[1] += y_dir * 2.0  # 少量 Y 修正
                    print(f"  [CENTERLINE] Y offset={y_offset:.1f}m but Y-return blocked, flying X+ first")
                else:
                    # Y 方向通畅，直接回归
                    local_goal = current_pos.copy()
                    local_goal[1] += y_dir * self.local_horizon
                    local_goal[0] += 2.0  # 少量X前进
                    print(f"  [CENTERLINE] Y offset={y_offset:.1f}m, biasing toward goal Y")
            else:
                # 基础局部目标
                direction = direction / distance
                local_goal = current_pos + direction * self.local_horizon

        # Z 轴：逐步趋近 global_goal 高度（受爬升角约束）
        max_dz = self.local_horizon * np.tan(np.radians(self.dubins_params.max_climb_angle))
        target_dz = global_goal[2] - current_pos[2]
        local_goal[2] = current_pos[2] + np.clip(target_dz, -max_dz, max_dz)

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

        # 尝试不同角度（优先选择偏离较小的角度，扩展到 ±180°）
        angles = [15, -15, 30, -30, 45, -45, 60, -60, 75, -75, 90, -90,
                  105, -105, 120, -120, 135, -135, 150, -150, 165, -165, 180]
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

    def _check_path_safe(self, from_pos: np.ndarray, to_pos: np.ndarray) -> bool:
        """
        检查从当前位置到目标航点的路径是否安全
        沿路径每 0.3m 采样一次，检查 ESDF 距离
        跳过起点附近 1m 的检查（无人机已经在那里，无法改变）
        """
        direction = to_pos - from_pos
        distance = np.linalg.norm(direction)
        if distance < 0.1:
            return True

        # 跳过起点附近的检查：如果无人机当前位置已经在 marginally unsafe 区域，
        # 不应阻止它移动到更安全的位置
        skip_distance = min(1.5, distance * 0.3)  # 跳过前 1.5m 或路径的 30%

        # 每 0.3m 检查一次（比规划时更密集）
        num_checks = max(3, int(distance / 0.3))
        for i in range(num_checks + 1):
            t = i / num_checks
            along_dist = t * distance
            if along_dist < skip_distance:
                continue  # 跳过起点附近
            point = from_pos + t * direction
            safety_dist = self.map_manager.esdf.get_distance(point)
            if safety_dist < self.map_manager.config.safety_margin:
                return False
        return True

    def _is_forward_blocked(self, current_pos: np.ndarray, global_goal: np.ndarray,
                             check_range: float = None) -> bool:
        """
        检查朝向全局目标的前方是否被障碍物挡住
        同时检查纯 X+ 方向——如果 X+ 方向通畅，说明已绕过墙壁边缘
        """
        if self.map_manager.esdf.distance_field is None:
            return False

        if check_range is None:
            check_range = self.local_horizon

        direction = global_goal - current_pos
        direction[2] = 0  # 只看 XY 平面
        norm = np.linalg.norm(direction[:2])
        if norm < 0.1:
            return False
        direction = direction / np.linalg.norm(direction)

        # 检查1：沿目标方向
        goal_blocked = False
        for d in np.arange(0.5, check_range, 1.0):
            point = current_pos + direction * d
            point[2] = current_pos[2]
            dist = self.map_manager.esdf.get_distance(point)
            if dist < self.map_manager.config.safety_margin:
                goal_blocked = True
                break

        if not goal_blocked:
            return False

        # 检查2：纯 X+ 方向是否也被挡
        x_forward = np.array([1.0, 0.0, 0.0])
        x_blocked = False
        for d in np.arange(0.5, check_range, 1.0):
            point = current_pos + x_forward * d
            point[2] = current_pos[2]
            dist = self.map_manager.esdf.get_distance(point)
            if dist < self.map_manager.config.safety_margin:
                x_blocked = True
                break

        if not x_blocked:
            # X+ 通畅但目标方向被挡 — 只有当无人机已经绕过墙边缘
            # （X 坐标已超过墙的 X 范围）时才认为不 blocked
            # Row 1 墙在 X=23.1~33.1，必须 X > 33 才算真正绕过
            if current_pos[0] > 33.0:
                print(f"  [FORWARD] Past Row1 (X={current_pos[0]:.1f}>33), X+ clear — not blocked")
                return False
            else:
                print(f"  [FORWARD] X+ clear but still beside Row1 (X={current_pos[0]:.1f}<33), still blocked")
                return True

        return True

    def _wall_follow_step(self, current_pos: np.ndarray, global_goal: np.ndarray) -> Optional[List[np.ndarray]]:
        """
        沿墙侧移策略：当前方被墙挡住时，沿墙壁平行移动寻找边缘。

        关键改进：
        - 一旦选定侧移方向就坚持，不来回摇摆
        - 每步侧移 5m，快速找到墙的边缘
        - 当侧移后前方变通畅时，清除方向记忆，恢复正常规划
        """
        direction = global_goal - current_pos
        direction[2] = 0
        norm = np.linalg.norm(direction[:2])
        if norm < 0.1:
            return None
        forward = direction / np.linalg.norm(direction)

        # 计算左右两个侧移方向（垂直于前进方向）
        left = np.array([-forward[1], forward[0], 0.0])
        right = np.array([forward[1], -forward[0], 0.0])

        # 如果已经有记忆的侧移方向，继续使用
        if self._wall_follow_direction is not None:
            if self._wall_follow_direction == 'left':
                best_dir, best_name = left, 'left'
            else:
                best_dir, best_name = right, 'right'
        else:
            # 首次选择：选择墙更短的一侧（更快找到边缘）
            # 通过检查两侧哪个方向更早遇到空旷区域来判断
            # 改进：优先选择已知空闲的路径，避免盲目进入未知区域
            best_dir = None
            best_score = -np.inf

            for lateral_dir, name in [(left, "left"), (right, "right")]:
                score = 0.0
                known_free_count = 0  # 统计已知空闲的点数
                unknown_count = 0     # 统计未知的点数

                for d in [3.0, 6.0, 10.0, 15.0, 20.0, 25.0]:
                    check_point = current_pos + lateral_dir * d
                    check_point[2] = current_pos[2]

                    # 侧移后前方是否通畅？
                    forward_point = check_point + forward * 4.0
                    forward_point[2] = current_pos[2]

                    lateral_safety = self.map_manager.esdf.get_distance(check_point)
                    forward_safety = self.map_manager.esdf.get_distance(forward_point)

                    # 检查体素状态，unknown 空间只给部分分数
                    lateral_idx = self.map_manager.voxel_grid.world_to_grid(check_point)
                    forward_idx = self.map_manager.voxel_grid.world_to_grid(forward_point)

                    lateral_known_free = (
                        self.map_manager.voxel_grid.is_valid_index(lateral_idx)
                        and self.map_manager.voxel_grid.grid[lateral_idx] == -1
                    )
                    forward_known_free = (
                        self.map_manager.voxel_grid.is_valid_index(forward_idx)
                        and self.map_manager.voxel_grid.grid[forward_idx] == -1
                    )

                    # 统计已知空闲和未知的点
                    if lateral_known_free:
                        known_free_count += 1
                    elif lateral_safety > self.map_manager.config.safety_margin:
                        unknown_count += 1

                    if forward_known_free:
                        known_free_count += 2  # 前方权重更高

                    # 评分：已知空闲给高分，未知给低分
                    if lateral_safety > self.map_manager.config.safety_margin:
                        score += 1.5 if lateral_known_free else 0.2  # 降低未知空间的分数
                    if forward_safety > self.map_manager.config.safety_margin:
                        score += 4.0 if forward_known_free else 0.5  # 大幅降低未知空间的分数

                # 额外奖励：如果有很多已知空闲点，说明这个方向已经被探索过
                if known_free_count > 3:
                    score += known_free_count * 0.5

                # 惩罚：如果大部分是未知空间，可能是盲区
                if unknown_count > known_free_count:
                    score -= unknown_count * 0.3

                print(f"  [WALL-FOLLOW] {name} score: {score:.1f} (known_free={known_free_count}, unknown={unknown_count})")
                if score > best_score:
                    best_score = score
                    best_dir = lateral_dir
                    best_name = name

            if best_dir is None:
                return None

            # 记住选择的方向
            self._wall_follow_direction = best_name
            print(f"  [WALL-FOLLOW] Chose {best_name} direction (will persist)")

        # 生成侧移路径：斜向前进（侧移 + X前进），避免纯侧移导致Y偏移过大
        # forward 方向是朝目标的，我们需要纯 X+ 方向作为前进分量
        x_forward = np.array([1.0, 0.0, 0.0])

        for step_size in [5.0, 4.0, 3.0, 2.0, 1.0]:
            # 尝试1: 斜向前进（侧移 + X前进3m），绕墙的同时推进
            target = current_pos + best_dir * step_size + x_forward * 3.0
            target[2] = current_pos[2]

            target_safety = self.map_manager.esdf.get_distance(target)
            if target_safety >= self.map_manager.config.safety_margin:
                if self._check_path_safe(current_pos, target):
                    print(f"  [WALL-FOLLOW] Diagonal {best_name} {step_size:.0f}m + X+3m, "
                          f"target=({target[0]:.1f}, {target[1]:.1f}, {target[2]:.1f})")
                    return self._extend_wall_follow_path(current_pos, target, step_size)

            # 尝试2: 纯侧移（如果斜向不行）
            target2 = current_pos + best_dir * step_size
            target2[2] = current_pos[2]

            target2_safety = self.map_manager.esdf.get_distance(target2)
            if target2_safety >= self.map_manager.config.safety_margin:
                if self._check_path_safe(current_pos, target2):
                    print(f"  [WALL-FOLLOW] Moving {best_name} {step_size:.0f}m along wall, "
                          f"target=({target2[0]:.1f}, {target2[1]:.1f}, {target2[2]:.1f})")
                    return self._extend_wall_follow_path(current_pos, target2, step_size)

            # 尝试3: 斜向后退（侧移 + 后退2m），绕过墙角
            backward = -x_forward
            target_diag = current_pos + best_dir * step_size + backward * 2.0
            target_diag[2] = current_pos[2]

            diag_safety = self.map_manager.esdf.get_distance(target_diag)
            if diag_safety >= self.map_manager.config.safety_margin:
                if self._check_path_safe(current_pos, target_diag):
                    print(f"  [WALL-FOLLOW] Diagonal {best_name} {step_size:.0f}m + back 2m, "
                          f"target=({target_diag[0]:.1f}, {target_diag[1]:.1f}, {target_diag[2]:.1f})")
                    return self._extend_wall_follow_path(current_pos, target_diag, step_size)

        # 当前方向暂时走不通，但保留方向记忆（不清除！）
        # 返回 None 让主循环回退到 RRT*，下次 wall-follow 仍用同一方向
        print(f"  [WALL-FOLLOW] {best_name} temporarily blocked, keeping direction for next attempt")
        return None

    def _extend_wall_follow_path(self, start: np.ndarray, first_target: np.ndarray,
                                  step_size: float) -> List[np.ndarray]:
        """
        将 wall-follow 的两点路径扩展为多步路径，一次走更远。
        沿 start->first_target 方向继续追加 2~3 个安全航点。
        """
        path = [start.copy(), first_target.copy()]
        extend_dir = first_target - start
        extend_norm = np.linalg.norm(extend_dir)
        if extend_norm < 0.1:
            return path
        extend_dir = extend_dir / extend_norm

        for _ in range(3):
            next_pt = path[-1] + extend_dir * step_size
            next_pt[2] = start[2]
            safety = self.map_manager.esdf.get_distance(next_pt)
            if safety >= self.map_manager.config.safety_margin and self._check_path_safe(path[-1], next_pt):
                path.append(next_pt)
            else:
                break

        if len(path) > 2:
            print(f"  [WALL-FOLLOW] Extended to {len(path)} waypoints")
        return path

    def _yaw_scan(self, current_pos: np.ndarray, global_goal: np.ndarray):
        """
        原地转头扫描，用多个 yaw 角度拍摄深度图并更新地图
        解决单相机 90° FOV 看不到侧方障碍物边界的问题
        """
        import time
        from utils.transforms import quaternion_to_rotation_matrix

        # 计算朝向目标的基准 yaw
        direction = global_goal - current_pos
        base_yaw = np.degrees(np.arctan2(direction[1], direction[0]))

        # 扫描角度：左右各扫 90°，共覆盖约 270°
        scan_yaws = [base_yaw + offset for offset in [-90, -45, 0, 45, 90]]

        for yaw in scan_yaws:
            self.drone.set_yaw(yaw, duration=0.3)
            time.sleep(0.2)

            pos, ori = self.drone.get_pose()
            depth = self.drone.get_depth_image()
            self.map_manager.update(depth, pos, ori)

        # 转回朝向目标的方向
        self.drone.set_yaw(base_yaw, duration=0.3)
        time.sleep(0.1)

        # 打印扫描后的地图状态
        stats = self.map_manager.get_map_stats()
        print(f"  [SCAN] After yaw scan: {stats['occupied_voxels']} occupied voxels")

    def get_trajectory(self) -> List[np.ndarray]:
        """获取已执行的轨迹"""
        return self.executed_trajectory.copy()

    def _detect_stuck(self, current_pos: np.ndarray, global_goal: np.ndarray) -> bool:
        """
        检测无人机是否卡住（原地打转）
        通过比较最近 N 次迭代的位置，如果移动距离很小则判定为卡住
        """
        self._position_history.append(current_pos.copy())

        # 保留最近 6 次位置
        if len(self._position_history) > 6:
            self._position_history = self._position_history[-6:]

        if len(self._position_history) < 4:
            return False

        # 计算最近 4 次迭代的总移动距离
        total_movement = 0.0
        for i in range(len(self._position_history) - 1):
            total_movement += np.linalg.norm(
                self._position_history[i+1] - self._position_history[i]
            )

        # 计算最远和最近位置之间的净位移
        net_displacement = np.linalg.norm(
            self._position_history[-1] - self._position_history[0]
        )

        # 如果移动了很多但净位移很小 → 来回摇摆
        if total_movement > 8.0 and net_displacement < 3.0:
            self._stuck_counter += 1
            return True

        # 如果几乎没移动
        if total_movement < 1.0:
            self._stuck_counter += 1
            return True

        self._stuck_counter = 0
        return False

    def _compute_forward_progress(self, current_pos: np.ndarray,
                                   global_goal: np.ndarray) -> float:
        """
        计算朝目标方向的前进进度（投影距离）
        返回值越大表示越接近目标
        """
        # 用起点到目标的方向作为参考轴
        # 简化：直接用 X 坐标作为前进进度（因为目标在 X=35）
        direction = global_goal - np.array([0.0, 0.0, global_goal[2]])
        norm = np.linalg.norm(direction[:2])
        if norm < 0.1:
            return 0.0
        direction = direction / norm

        # 当前位置在目标方向上的投影
        progress = np.dot(current_pos[:2], direction[:2])
        return progress
