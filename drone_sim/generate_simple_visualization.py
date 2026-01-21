"""
generate_simple_visualization.py - 生成简洁的3D动态可视化
展示无人机轨迹 + 实时障碍物点云 + 规划路径
"""

import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from planning.config import PlanningConfig
from mapping.incremental_map import IncrementalMapManager
from planning.rrt_star import RRTStar
from visualization.simple_3d_visualizer import Simple3DVisualizer


def create_synthetic_environment():
    """创建合成环境"""
    obstacles = []

    # 障碍物1: 墙壁 (X=8-10, Y=2-8)
    for x in np.arange(8, 10, 0.5):
        for y in np.arange(2, 8, 0.5):
            for z in np.arange(-5, -2, 0.5):
                obstacles.append([x, y, z])

    # 障碍物2: 柱子 (X=14, Y=6)
    for z in np.arange(-5, -2, 0.5):
        for dx in np.arange(-1.0, 1.5, 0.5):
            for dy in np.arange(-1.0, 1.5, 0.5):
                obstacles.append([14 + dx, 6 + dy, z])

    # 障碍物3: L形障碍 (X=18-20, Y=8-10)
    for x in np.arange(18, 20, 0.5):
        for y in np.arange(8, 10, 0.5):
            for z in np.arange(-5, -2, 0.5):
                obstacles.append([x, y, z])

    for x in np.arange(18, 19, 0.5):
        for y in np.arange(4, 8, 0.5):
            for z in np.arange(-5, -2, 0.5):
                obstacles.append([x, y, z])

    return np.array(obstacles)


def simulate_depth_observation(obstacles, drone_pos, fov=90, max_range=20):
    """模拟深度观测"""
    relative_pos = obstacles - drone_pos
    distances = np.linalg.norm(relative_pos, axis=1)

    # 距离过滤
    in_range = distances < max_range

    # 前方视场
    forward = relative_pos[:, 0] > 0

    # 视场角过滤
    angles = np.arctan2(relative_pos[:, 1], relative_pos[:, 0])
    fov_rad = np.radians(fov / 2)
    in_fov = np.abs(angles) < fov_rad

    visible = in_range & forward & in_fov
    return obstacles[visible]


def main():
    print("\n" + "=" * 70)
    print("  SIMPLE 3D VISUALIZATION GENERATOR")
    print("  Drone Trajectory + Real-time Obstacle Mapping")
    print("=" * 70)

    # === 1. 创建环境 ===
    print("\n[1] Creating synthetic environment...")
    all_obstacles = create_synthetic_environment()
    print(f"    Total obstacles: {len(all_obstacles)} points")

    # === 2. 初始化配置 ===
    print("\n[2] Initializing configuration...")
    start_pos = np.array([2.0, 5.0, -3.5])
    goal_pos = np.array([22.0, 5.0, -3.5])

    config = PlanningConfig(
        voxel_size=0.5,
        grid_size=(80, 80, 40),
        origin=(-5.0, -10.0, -10.0),
        max_depth=20.0,
        step_size=1.5,
        max_iterations=3000,
        goal_sample_rate=0.2,
        search_radius=4.0,
        safety_margin=0.8
    )

    print(f"    Start: {start_pos}")
    print(f"    Goal: {goal_pos}")

    # === 3. 初始化地图和可视化器 ===
    print("\n[3] Initializing map manager and visualizer...")
    map_manager = IncrementalMapManager(config)
    visualizer = Simple3DVisualizer(figsize=(12, 10), dpi=100)

    # === 4. 模拟滚动规划 ===
    print("\n[4] Simulating receding horizon planning...")

    current_pos = start_pos.copy()
    executed_trajectory = [current_pos.copy()]
    accumulated_obstacles = []  # 累积观测到的障碍物
    iteration = 0
    max_iterations = 20
    local_horizon = 6.0

    drone_ori = np.array([1.0, 0.0, 0.0, 0.0])

    while iteration < max_iterations:
        iteration += 1
        print(f"\n  Iteration {iteration}/{max_iterations}")

        # 检查是否到达目标
        dist_to_goal = np.linalg.norm(current_pos - goal_pos)
        print(f"    Distance to goal: {dist_to_goal:.2f}m")

        if dist_to_goal < 1.0:
            print("    [OK] Reached goal!")
            break

        # === 4.1 模拟深度观测 ===
        observed_obstacles = simulate_depth_observation(
            all_obstacles, current_pos
        )
        print(f"    Observed: {len(observed_obstacles)} new points")

        # 累积障碍物（去重）
        if len(observed_obstacles) > 0:
            if len(accumulated_obstacles) == 0:
                accumulated_obstacles = observed_obstacles.tolist()
            else:
                # 简单去重：检查距离
                for obs in observed_obstacles:
                    is_new = True
                    for existing in accumulated_obstacles:
                        if np.linalg.norm(obs - existing) < 0.3:
                            is_new = False
                            break
                    if is_new:
                        accumulated_obstacles.append(obs.tolist())

        accumulated_obstacles_arr = np.array(accumulated_obstacles) if len(accumulated_obstacles) > 0 else np.array([])
        print(f"    Total accumulated: {len(accumulated_obstacles)} points")

        # === 4.2 更新地图 ===
        for obs in observed_obstacles:
            idx = map_manager.voxel_grid.world_to_grid(obs)
            if map_manager.voxel_grid.is_valid_index(idx):
                map_manager.voxel_grid.grid[idx] = 1

        map_manager.esdf.compute()

        # === 4.3 选择局部目标 ===
        direction = goal_pos - current_pos
        distance = np.linalg.norm(direction)
        direction = direction / distance

        if distance < local_horizon:
            local_goal = goal_pos.copy()
        else:
            local_goal = current_pos + direction * local_horizon

        # 确保局部目标安全
        goal_dist_check = map_manager.esdf.get_distance(local_goal)
        if goal_dist_check < config.safety_margin:
            for dist_factor in [0.8, 0.6, 0.4]:
                test_goal = current_pos + direction * (local_horizon * dist_factor)
                test_dist = map_manager.esdf.get_distance(test_goal)
                if test_dist >= config.safety_margin:
                    local_goal = test_goal
                    break

        # === 4.4 规划路径 ===
        rrt = RRTStar(map_manager.voxel_grid, map_manager.esdf, config)
        path = rrt.plan(current_pos, local_goal)

        if path is None:
            print("    [X] Planning failed, trying alternatives...")
            # 尝试备选目标
            for angle in [-30, 30, -45, 45]:
                angle_rad = np.radians(angle)
                rotated_dir = np.array([
                    np.cos(angle_rad) * direction[0] - np.sin(angle_rad) * direction[1],
                    np.sin(angle_rad) * direction[0] + np.cos(angle_rad) * direction[1],
                    direction[2]
                ])
                alt_goal = current_pos + rotated_dir * (local_horizon * 0.7)
                alt_dist = map_manager.esdf.get_distance(alt_goal)

                if alt_dist > config.safety_margin:
                    path = rrt.plan(current_pos, alt_goal)
                    if path is not None:
                        local_goal = alt_goal
                        print(f"      [OK] Found alternative at {angle}deg")
                        break

            if path is None:
                print("    [X] All alternatives failed")
                break

        print(f"    Path planned: {len(path)} waypoints")

        # === 4.5 执行部分路径 ===
        execution_ratio = 0.5
        execution_length = max(1, int(len(path) * execution_ratio))
        waypoints_to_execute = path[:execution_length]

        # 移动到最后一个执行点
        if len(waypoints_to_execute) > 0:
            current_pos = waypoints_to_execute[-1].copy()
            executed_trajectory.extend(waypoints_to_execute)

        # === 4.6 记录帧 ===
        info = f"Obstacles: {len(accumulated_obstacles)} | Trajectory: {len(executed_trajectory)} points"

        frame_data = {
            'iteration': iteration,
            'obstacles': accumulated_obstacles_arr,
            'drone_pos': current_pos.copy(),
            'trajectory': executed_trajectory.copy(),
            'planned_path': path,
            'goal': goal_pos.copy(),
            'info': info
        }

        visualizer.add_frame(frame_data)
        print(f"    [OK] Frame {iteration} recorded")

    # === 5. 渲染所有帧 ===
    print("\n[5] Rendering frames...")
    output_dir = "simple_3d_frames"
    os.makedirs(output_dir, exist_ok=True)
    visualizer.save_all_frames(output_dir, prefix='frame')

    # === 6. 生成最终图像 ===
    print("\n[6] Generating final image...")
    if len(visualizer.frames) > 0:
        visualizer.render_frame(len(visualizer.frames) - 1, save_path='simple_3d_final.png')

    # === 7. 创建视频 ===
    print("\n[7] Creating video...")
    try:
        visualizer.create_video('simple_3d_visualization.mp4', fps=5)
    except Exception as e:
        print(f"    [X] Video creation failed: {e}")

    # === 8. 清理 ===
    print("\n[8] Cleaning up...")
    visualizer.close()

    # === 总结 ===
    print("\n" + "=" * 70)
    print("  VISUALIZATION COMPLETE!")
    print("=" * 70)
    print(f"\n  Output:")
    print(f"    - Frame sequence: {output_dir}/")
    print(f"    - Final image: simple_3d_final.png")
    print(f"    - Video: simple_3d_visualization.mp4")
    print(f"    - Total frames: {len(visualizer.frames)}")
    print(f"    - Total iterations: {iteration}")
    print(f"    - Final distance to goal: {np.linalg.norm(current_pos - goal_pos):.2f}m")
    print(f"    - Total obstacles observed: {len(accumulated_obstacles)}")
    print(f"    - Trajectory length: {len(executed_trajectory)} waypoints")
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
