
"""
fly_with_simple_visualization.py - 使用真实AirSim飞行 + 简洁3D可视化
"""

import numpy as np
import time
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from planning.config import PlanningConfig
from mapping.incremental_map import IncrementalMapManager
from planning.receding_horizon import RecedingHorizonPlanner
from control.drone_interface import DroneInterface
from visualization.simple_3d_visualizer import Simple3DVisualizer


def main():
    print("\n" + "=" * 70)
    print("  REAL AIRSIM FLIGHT WITH SIMPLE 3D VISUALIZATION")
    print("=" * 70)

    # === 1. 连接AirSim ===
    print("\n[1] Connecting to AirSim...")
    drone = DroneInterface()

    try:
        drone.connect()
        print("    [OK] Connected to AirSim")
    except Exception as e:
        print(f"    [X] Failed to connect: {e}")
        print("    Make sure AirSim is running!")
        return

    try:
        # === 2. 起飞 ===
        print("\n[2] Taking off...")
        drone.takeoff()
        time.sleep(1)

        initial_height = -3.0  # 3米高度
        print(f"    Flying to altitude: {-initial_height}m")
        drone.move_to_z(initial_height, velocity=2.0)
        time.sleep(1)

        # === 3. 获取初始位置 ===
        print("\n[3] Getting initial position...")
        initial_pos, initial_ori = drone.get_pose()
        print(f"    Position: ({initial_pos[0]:.2f}, {initial_pos[1]:.2f}, {initial_pos[2]:.2f})")

        # === 4. 初始化配置 ===
        print("\n[4] Initializing configuration...")
        config = PlanningConfig(
            voxel_size=0.5,
            grid_size=(80, 80, 40),
            origin=(initial_pos[0] - 20.0, initial_pos[1] - 20.0, initial_pos[2] - 10.0),
            max_depth=25.0,
            step_size=1.5,
            max_iterations=2000,
            goal_sample_rate=0.2,
            search_radius=4.0,
            safety_margin=0.8
        )

        # === 5. 初始化地图和可视化器 ===
        print("\n[5] Initializing map manager and visualizer...")
        map_manager = IncrementalMapManager(config)
        visualizer = Simple3DVisualizer(figsize=(12, 10), dpi=100)

        # === 6. 设置目标 ===
        print("\n[6] Setting goal...")
        global_goal = np.array([
            initial_pos[0] + 15.0,  # 前方15m
            initial_pos[1] + 8.0,   # 右侧8m
            initial_pos[2]          # 保持高度
        ])
        print(f"    Start: ({initial_pos[0]:.2f}, {initial_pos[1]:.2f}, {initial_pos[2]:.2f})")
        print(f"    Goal:  ({global_goal[0]:.2f}, {global_goal[1]:.2f}, {global_goal[2]:.2f})")

        # === 7. 滚动规划主循环 ===
        print("\n[7] Starting receding horizon planning...")
        print("    Press Ctrl+C to stop\n")

        current_pos = initial_pos.copy()
        executed_trajectory = [current_pos.copy()]
        accumulated_obstacles = []  # 累积观测到的障碍物
        iteration = 0
        max_iterations = 30
        local_horizon = 6.0

        while iteration < max_iterations:
            iteration += 1
            print(f"\n  === Iteration {iteration}/{max_iterations} ===")

            # 检查是否到达目标
            dist_to_goal = np.linalg.norm(current_pos - global_goal)
            print(f"    Distance to goal: {dist_to_goal:.2f}m")

            if dist_to_goal < 1.0:
                print("    [OK] Reached goal!")
                break

            # === 7.1 获取真实深度图和位置 ===
            print("    Getting depth image and pose...")
            depth_image = drone.get_depth_image()
            current_pos, current_ori = drone.get_pose()

            # === 7.2 更新地图 ===
            print("    Updating map...")
            map_stats = map_manager.update(depth_image, current_pos, current_ori)
            print(f"    Map: +{map_stats['new_occupied']} voxels, "
                  f"total {map_stats['total_occupied']} occupied")

            # 获取当前观测到的障碍物点云（用于可视化）
            occupied_voxels = map_manager.voxel_grid.get_occupied_voxels()
            if len(occupied_voxels) > 0:
                # 累积障碍物（简单合并）
                if len(accumulated_obstacles) == 0:
                    accumulated_obstacles = occupied_voxels.tolist()
                else:
                    # 添加新观测到的点
                    for voxel in occupied_voxels:
                        is_new = True
                        for existing in accumulated_obstacles:
                            if np.linalg.norm(voxel - existing) < 0.3:
                                is_new = False
                                break
                        if is_new:
                            accumulated_obstacles.append(voxel.tolist())

            print(f"    Total accumulated obstacles: {len(accumulated_obstacles)} points")

            # === 7.3 选择局部目标 ===
            direction = global_goal - current_pos
            distance = np.linalg.norm(direction)
            direction = direction / distance

            if distance < local_horizon:
                local_goal = global_goal.copy()
            else:
                local_goal = current_pos + direction * local_horizon

            # 确保局部目标安全
            goal_dist_check = map_manager.esdf.get_distance(local_goal)
            if goal_dist_check < config.safety_margin:
                print(f"    Local goal too close to obstacle ({goal_dist_check:.2f}m), adjusting...")
                for dist_factor in [0.8, 0.6, 0.4]:
                    test_goal = current_pos + direction * (local_horizon * dist_factor)
                    test_dist = map_manager.esdf.get_distance(test_goal)
                    if test_dist >= config.safety_margin:
                        local_goal = test_goal
                        break

            print(f"    Local goal: ({local_goal[0]:.2f}, {local_goal[1]:.2f}, {local_goal[2]:.2f})")

            # === 7.4 规划路径 ===
            print("    Planning path...")
            from planning.rrt_star import RRTStar
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
                    print("    [X] All alternatives failed, stopping")
                    break

            print(f"    Path planned: {len(path)} waypoints")

            # === 7.5 执行部分路径 ===
            execution_ratio = 0.4
            execution_length = max(1, int(len(path) * execution_ratio))
            waypoints_to_execute = path[:execution_length]

            print(f"    Executing {len(waypoints_to_execute)} waypoints...")
            for i, wp in enumerate(waypoints_to_execute):
                print(f"      -> Waypoint {i+1}/{len(waypoints_to_execute)}: "
                      f"({wp[0]:.2f}, {wp[1]:.2f}, {wp[2]:.2f})")
                drone.move_to_position(wp, velocity=2.5)
                executed_trajectory.append(wp.copy())

            # 更新当前位置
            current_pos, _ = drone.get_pose()

            # === 7.6 记录可视化帧 ===
            accumulated_obstacles_arr = np.array(accumulated_obstacles) if len(accumulated_obstacles) > 0 else np.array([])
            info = f"Obstacles: {len(accumulated_obstacles)} | Trajectory: {len(executed_trajectory)} points"

            frame_data = {
                'iteration': iteration,
                'obstacles': accumulated_obstacles_arr,
                'drone_pos': current_pos.copy(),
                'trajectory': executed_trajectory.copy(),
                'planned_path': path,
                'goal': global_goal.copy(),
                'info': info
            }

            visualizer.add_frame(frame_data)
            print(f"    [OK] Frame {iteration} recorded")

        # === 8. 悬停并降落 ===
        print("\n[8] Hovering for 3 seconds...")
        drone.hover()
        time.sleep(3)

        print("\n[9] Landing...")
        drone.land()

        # === 9. 保存可视化 ===
        print("\n[10] Saving visualization...")
        output_dir = "airsim_flight_frames"
        os.makedirs(output_dir, exist_ok=True)

        visualizer.save_all_frames(output_dir, prefix='frame')
        visualizer.render_frame(len(visualizer.frames) - 1, save_path='airsim_flight_final.png')

        try:
            visualizer.create_video('airsim_flight_visualization.mp4', fps=5)
        except Exception as e:
            print(f"    [X] Video creation failed: {e}")

        # === 总结 ===
        print("\n" + "=" * 70)
        print("  FLIGHT COMPLETE!")
        print("=" * 70)
        print(f"\n  Statistics:")
        print(f"    - Total iterations: {iteration}")
        print(f"    - Total frames: {len(visualizer.frames)}")
        print(f"    - Final distance to goal: {np.linalg.norm(current_pos - global_goal):.2f}m")
        print(f"    - Total obstacles observed: {len(accumulated_obstacles)}")
        print(f"    - Trajectory length: {len(executed_trajectory)} waypoints")
        print(f"\n  Output:")
        print(f"    - Frame sequence: {output_dir}/")
        print(f"    - Final image: airsim_flight_final.png")
        print(f"    - Video: airsim_flight_visualization.mp4")
        print("\n" + "=" * 70)

    except KeyboardInterrupt:
        print("\n\n[X] Interrupted by user!")
        print("  Emergency landing...")
        drone.land()

    except Exception as e:
        print(f"\n\n[X] Error occurred: {e}")
        import traceback
        traceback.print_exc()
        print("  Emergency landing...")
        try:
            drone.land()
        except:
            pass

    finally:
        # === 清理 ===
        print("\n[11] Cleaning up...")
        drone.disconnect()
        visualizer.close()


if __name__ == "__main__":
    main()
