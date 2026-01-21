"""
generate_paper_visualization.py - 生成论文级可视化
使用模拟数据展示增量式建图 + ESDF + RRT*的完整过程
"""

import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from planning.config import PlanningConfig
from mapping.voxel_grid import VoxelGrid
from mapping.esdf import ESDF
from mapping.incremental_map import IncrementalMapManager
from planning.rrt_star import RRTStar
from visualization.paper_quality_visualizer import PaperQualityVisualizer


def create_synthetic_environment():
    """
    创建合成环境用于演示
    返回障碍物点云
    """
    obstacles = []

    # 障碍物1: 墙壁 (X=8-10, Y=2-8, Z=-5到-2)
    for x in np.arange(8, 10, 0.5):
        for y in np.arange(2, 8, 0.5):
            for z in np.arange(-5, -2, 0.5):
                obstacles.append([x, y, z])

    # 障碍物2: 柱子 (X=14, Y=6, Z=-5到-2)
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


def simulate_depth_observation(obstacles, drone_pos, drone_ori, fov=90, max_range=20):
    """
    模拟从无人机位置观测到的障碍物点云

    Args:
        obstacles: 全局障碍物点云
        drone_pos: 无人机位置
        drone_ori: 无人机朝向（四元数）
        fov: 视场角（度）
        max_range: 最大观测距离

    Returns:
        观测到的障碍物点云（世界坐标系）
    """
    # 简化：假设无人机朝向X正方向
    # 计算每个障碍物相对于无人机的方向
    relative_pos = obstacles - drone_pos
    distances = np.linalg.norm(relative_pos, axis=1)

    # 距离过滤
    in_range = distances < max_range

    # 视场角过滤（简化：只考虑X方向）
    # 前方视场：X > 0 且在FOV范围内
    forward = relative_pos[:, 0] > 0

    # 计算水平角度
    angles = np.arctan2(relative_pos[:, 1], relative_pos[:, 0])
    fov_rad = np.radians(fov / 2)
    in_fov = np.abs(angles) < fov_rad

    # 组合过滤条件
    visible = in_range & forward & in_fov

    return obstacles[visible]


def main():
    print("\n" + "=" * 70)
    print("  PAPER-QUALITY VISUALIZATION GENERATOR")
    print("  Incremental Mapping + ESDF + RRT*")
    print("=" * 70)

    # === 1. 创建合成环境 ===
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

    print(f"    Voxel size: {config.voxel_size}m")
    print(f"    Grid size: {config.grid_size}")
    print(f"    Start: {start_pos}")
    print(f"    Goal: {goal_pos}")

    # === 3. 初始化地图管理器 ===
    print("\n[3] Initializing map manager...")
    map_manager = IncrementalMapManager(config)

    # === 4. 初始化可视化器 ===
    print("\n[4] Initializing visualizer...")
    visualizer = PaperQualityVisualizer(figsize=(20, 12), dpi=150)

    # === 5. 模拟滚动规划过程 ===
    print("\n[5] Simulating receding horizon planning...")

    current_pos = start_pos.copy()
    executed_trajectory = [current_pos.copy()]
    iteration = 0
    max_iterations = 15
    local_horizon = 6.0

    # 无人机朝向（简化：始终朝向X正方向）
    drone_ori = np.array([1.0, 0.0, 0.0, 0.0])  # w, x, y, z

    while iteration < max_iterations:
        iteration += 1
        print(f"\n  Iteration {iteration}/{max_iterations}")

        # 计算到目标的距离
        dist_to_goal = np.linalg.norm(current_pos - goal_pos)
        print(f"    Distance to goal: {dist_to_goal:.2f}m")

        if dist_to_goal < 1.0:
            print("    [OK] Reached goal!")
            break

        # === 5.1 模拟深度观测 ===
        observed_obstacles = simulate_depth_observation(
            all_obstacles, current_pos, drone_ori
        )
        print(f"    Observed obstacles: {len(observed_obstacles)} points")

        # === 5.2 更新地图 ===
        # 手动更新体素栅格（模拟增量式建图）
        new_occupied = 0
        for obs in observed_obstacles:
            idx = map_manager.voxel_grid.world_to_grid(obs)
            if map_manager.voxel_grid.is_valid_index(idx):
                if map_manager.voxel_grid.grid[idx] != 1:
                    map_manager.voxel_grid.grid[idx] = 1
                    new_occupied += 1

        # 更新ESDF
        map_manager.esdf.compute()

        map_stats = {
            'new_occupied': new_occupied,
            'total_occupied': np.sum(map_manager.voxel_grid.grid == 1),
            'free_voxels': np.sum(map_manager.voxel_grid.grid == -1),
            'unknown_voxels': np.sum(map_manager.voxel_grid.grid == 0)
        }

        print(f"    Map updated: +{new_occupied} voxels, "
              f"total {map_stats['total_occupied']} occupied")

        # === 5.3 选择局部目标 ===
        direction = goal_pos - current_pos
        distance = np.linalg.norm(direction)
        direction = direction / distance

        if distance < local_horizon:
            local_goal = goal_pos.copy()
        else:
            local_goal = current_pos + direction * local_horizon

        # 确保局部目标不在障碍物内
        goal_dist_check = map_manager.esdf.get_distance(local_goal)
        if goal_dist_check < config.safety_margin:
            print(f"    Local goal too close to obstacle ({goal_dist_check:.2f}m), adjusting...")
            # 尝试不同的距离
            for dist_factor in [0.8, 0.6, 0.4]:
                test_goal = current_pos + direction * (local_horizon * dist_factor)
                test_dist = map_manager.esdf.get_distance(test_goal)
                if test_dist >= config.safety_margin:
                    local_goal = test_goal
                    print(f"    Adjusted to {dist_factor * local_horizon:.1f}m ahead, distance: {test_dist:.2f}m")
                    break

        print(f"    Local goal: ({local_goal[0]:.2f}, {local_goal[1]:.2f}, {local_goal[2]:.2f})")

        # === 5.4 规划路径 ===
        rrt = RRTStar(map_manager.voxel_grid, map_manager.esdf, config)

        # 检查起点和终点的安全性
        start_dist = map_manager.esdf.get_distance(current_pos)
        goal_dist = map_manager.esdf.get_distance(local_goal)
        print(f"    Start distance to obstacle: {start_dist:.2f}m")
        print(f"    Goal distance to obstacle: {goal_dist:.2f}m")

        path = rrt.plan(current_pos, local_goal)

        # 获取RRT树结构
        rrt_tree = None
        if hasattr(rrt, 'nodes') and hasattr(rrt, 'parent'):
            nodes = rrt.nodes
            edges = []
            for i, node in enumerate(nodes):
                parent_idx = rrt.parent.get(i)
                if parent_idx is not None and parent_idx >= 0:
                    edges.append((nodes[parent_idx], node))

            rrt_tree = {
                'nodes': nodes,
                'edges': edges
            }

        if path is None:
            print("    [X] Planning failed!")
            # 尝试调整局部目标
            print("    Trying alternative goals...")
            for angle in [-30, 30, -45, 45]:
                angle_rad = np.radians(angle)
                rotated_dir = np.array([
                    np.cos(angle_rad) * direction[0] - np.sin(angle_rad) * direction[1],
                    np.sin(angle_rad) * direction[0] + np.cos(angle_rad) * direction[1],
                    direction[2]
                ])
                alt_goal = current_pos + rotated_dir * (local_horizon * 0.7)
                alt_dist = map_manager.esdf.get_distance(alt_goal)
                print(f"      Trying angle {angle}deg, distance to obstacle: {alt_dist:.2f}m")

                if alt_dist > config.safety_margin:
                    path = rrt.plan(current_pos, alt_goal)
                    if path is not None:
                        local_goal = alt_goal
                        print(f"      [OK] Found alternative path!")
                        break

            if path is None:
                print("    [X] All alternatives failed, stopping")
                break

        print(f"    Path planned: {len(path)} waypoints")

        # === 5.5 执行部分路径 ===
        execution_ratio = 0.4
        execution_length = max(1, int(len(path) * execution_ratio))
        waypoints_to_execute = path[:execution_length]

        print(f"    Executing {len(waypoints_to_execute)} waypoints...")

        # 移动到最后一个执行点
        if len(waypoints_to_execute) > 0:
            current_pos = waypoints_to_execute[-1].copy()
            executed_trajectory.extend(waypoints_to_execute)

        # === 5.6 记录帧数据 ===
        frame_data = {
            'iteration': iteration,
            'map_manager': map_manager,
            'current_pos': current_pos.copy(),
            'local_goal': local_goal.copy(),
            'global_goal': goal_pos.copy(),
            'current_path': path,
            'rrt_tree': rrt_tree,
            'executed_trajectory': executed_trajectory.copy(),
            'map_stats': map_stats
        }

        visualizer.add_frame(frame_data)
        print(f"    [OK] Frame {iteration} recorded")

    # === 6. 渲染所有帧 ===
    print("\n[6] Rendering frames...")

    # 创建输出目录
    output_dir = "paper_visualization_frames"
    os.makedirs(output_dir, exist_ok=True)

    # 保存所有帧
    visualizer.save_all_frames(output_dir, prefix='frame')

    # === 7. 生成最终高质量图像 ===
    print("\n[7] Generating final high-quality image...")
    if len(visualizer.frames) > 0:
        final_frame_idx = len(visualizer.frames) - 1
        visualizer.render_frame(final_frame_idx, save_path='paper_visualization_final.png')
    else:
        print("    [X] No frames to render")

    # === 8. 尝试生成视频 ===
    print("\n[8] Attempting to create video...")
    try:
        visualizer.create_video('paper_visualization.mp4', fps=2)
    except Exception as e:
        print(f"    [X] Video creation failed: {e}")
        print("    (FFmpeg may not be installed)")
        print("    You can create video from frames using:")
        print(f"    ffmpeg -framerate 2 -i {output_dir}/frame_%04d.png -c:v libx264 -pix_fmt yuv420p paper_visualization.mp4")

    # === 9. 清理 ===
    print("\n[9] Cleaning up...")
    visualizer.close()

    # === 总结 ===
    print("\n" + "=" * 70)
    print("  VISUALIZATION GENERATION COMPLETE!")
    print("=" * 70)
    print(f"\n  Output:")
    print(f"    - Frame sequence: {output_dir}/")
    print(f"    - Final image: paper_visualization_final.png")
    print(f"    - Total frames: {len(visualizer.frames)}")
    print(f"    - Total iterations: {iteration}")
    print(f"    - Final distance to goal: {np.linalg.norm(current_pos - goal_pos):.2f}m")
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
