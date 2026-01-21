"""
fly_planned_path.py - Receding Horizon Planning Flight Test
使用增量式建图 + 滚动规划实现动态避障

Phase 2: 解决单次感知的遮挡问题
"""

import numpy as np
import time
import sys
import os

# 添加模块路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from planning.config import PlanningConfig
from mapping.incremental_map import IncrementalMapManager
from planning.receding_horizon import RecedingHorizonPlanner
from control.drone_interface import DroneInterface


def main():
    print("\n" + "=" * 70)
    print("  RECEDING HORIZON PLANNING - Phase 2")
    print("  Incremental Mapping + Dynamic Replanning")
    print("=" * 70)

    # === 1. 初始化无人机接口 ===
    print("\n[1] Connecting to AirSim...")
    drone = DroneInterface()

    try:
        drone.connect()
    except Exception as e:
        print(f"✗ Failed to connect to AirSim: {e}")
        print("  Make sure AirSim is running!")
        return

    try:
        # === 2. 起飞 ===
        print("\n[2] Taking off...")
        drone.takeoff()
        time.sleep(1)

        # 飞到初始高度
        initial_height = -3.0  # NED coordinate, 3m altitude
        print(f"    Flying to altitude: {-initial_height}m")
        drone.move_to_z(initial_height, velocity=2.0)
        time.sleep(1)

        # === 3. 获取初始位置 ===
        print("\n[3] Getting initial position...")
        initial_pos, initial_ori = drone.get_pose()
        print(f"    Position: ({initial_pos[0]:.2f}, {initial_pos[1]:.2f}, {initial_pos[2]:.2f})")
        print(f"    Orientation: ({initial_ori[0]:.3f}, {initial_ori[1]:.3f}, "
              f"{initial_ori[2]:.3f}, {initial_ori[3]:.3f})")

        # === 4. 初始化规划配置 ===
        print("\n[4] Initializing planner configuration...")

        # 规划配置 - 使用固定origin覆盖整个飞行区域
        # Blocks场景中建筑物在X=23~53m范围，需要确保grid覆盖这个区域
        # 目标在Y=25，需要确保Y方向覆盖足够
        planning_config = PlanningConfig(
            voxel_size=0.5,
            grid_size=(120, 120, 40),  # 扩大Y方向覆盖60m
            origin=(-10.0, -30.0, -15.0),  # 固定origin，覆盖X=-10~50, Y=-30~30, Z=-15~5
            max_depth=25.0,
            step_size=1.5,
            max_iterations=3000,
            goal_sample_rate=0.2,
            search_radius=4.0,
            safety_margin=1.5  # 增大安全边距，防止撞墙
        )

        # 滚动规划配置 - 优化后的参数
        receding_config = {
            'local_horizon': 8.0,      # 增大局部目标距离到8m
            'execution_ratio': 0.5,    # 执行路径的前50%
            'replan_threshold': 1.5,   # 重规划阈值
            'goal_tolerance': 1.5,     # 放宽到达目标的判定阈值
            'max_iterations': 50,      # 最大循环次数
            'flight_velocity': 2.0,    # 飞行速度 m/s（稍微降低以提高稳定性）
            'visualize': True,         # 启用可视化
            'enhanced_viz': True,      # 使用增强版可视化器
            'save_video': True,        # 保存视频
            'video_fps': 10            # 视频帧率
        }

        print(f"    Voxel size: {planning_config.voxel_size}m")
        print(f"    Grid size: {planning_config.grid_size}")
        print(f"    Local horizon: {receding_config['local_horizon']}m")
        print(f"    Execution ratio: {receding_config['execution_ratio']}")

        # === 5. 初始化地图管理器和规划器 ===
        print("\n[5] Initializing map manager and planner...")
        map_manager = IncrementalMapManager(planning_config)
        planner = RecedingHorizonPlanner(map_manager, drone, receding_config)
        print("    [OK] Initialization complete")

        # === 6. 设置全局目标 ===
        print("\n[6] Setting global goal...")

        # 根据场景分析结果：
        # - Blocks场景中，最近的建筑物在 X=23~33m 处
        # - 建筑物高度覆盖 -1m ~ 14m（多层建筑）
        # - 建筑物Y范围约 -21.5 ~ 18.5m（很宽）
        #
        # 策略：设置目标在建筑物侧面，让无人机从侧面绕过
        # 建筑物边缘在Y=18.5，所以目标设在Y=25可以绕过
        global_goal = np.array([
            initial_pos[0] + 35.0,  # X: 前方 35m（超过建筑物）
            initial_pos[1] + 25.0,  # Y: 右侧 25m（绕过建筑物边缘）
            initial_pos[2]          # Z: 保持高度
        ])

        print(f"    Start:  ({initial_pos[0]:.2f}, {initial_pos[1]:.2f}, {initial_pos[2]:.2f})")
        print(f"    Goal:   ({global_goal[0]:.2f}, {global_goal[1]:.2f}, {global_goal[2]:.2f})")
        print(f"    Distance: {np.linalg.norm(global_goal - initial_pos):.2f}m")
        print("\n    *** OBSTACLE AVOIDANCE TEST ***")
        print("    Obstacle detected at X=8-10m (directly ahead)")
        print("    Goal is at X=15m (behind obstacle), Y=5m (lateral offset)")
        print("    The drone must navigate around the obstacle")

        # === 7. 执行滚动规划 ===
        print("\n[7] Starting receding horizon planning...")
        print("    Press Ctrl+C to stop\n")

        start_time = time.time()
        success = planner.plan_and_execute(global_goal)
        elapsed_time = time.time() - start_time

        # === 8. 结果报告 ===
        print("\n" + "=" * 70)
        if success:
            print("  [SUCCESS] MISSION SUCCESS!")
            print(f"  Reached goal in {elapsed_time:.1f} seconds")
        else:
            print("  [FAILED] MISSION FAILED")
            print(f"  Stopped after {elapsed_time:.1f} seconds")
        print("=" * 70)

        # 获取执行轨迹
        trajectory = planner.get_trajectory()
        print(f"\n  Trajectory statistics:")
        print(f"    Total waypoints executed: {len(trajectory)}")
        if len(trajectory) > 0:
            traj_arr = np.array(trajectory)
            total_distance = np.sum(np.linalg.norm(np.diff(traj_arr, axis=0), axis=1))
            print(f"    Total distance traveled: {total_distance:.2f}m")

        # 地图统计
        map_stats = map_manager.get_map_stats()
        print(f"\n  Map statistics:")
        print(f"    Total updates: {map_stats['total_updates']}")
        print(f"    Occupied voxels: {map_stats['occupied_voxels']}")
        print(f"    Free voxels: {map_stats['free_voxels']}")
        print(f"    Unknown voxels: {map_stats['unknown_voxels']}")

        # 保存可视化
        if planner.visualizer:
            print("\n[8] Saving visualization...")
            planner.visualizer.save_figure('receding_horizon_result.png')

        # === 9. 悬停并降落 ===
        print("\n[9] Hovering for 3 seconds...")
        drone.hover()
        time.sleep(3)

        print("\n[10] Landing...")
        drone.land()

    except KeyboardInterrupt:
        print("\n\n[INTERRUPTED] Interrupted by user!")
        print("  Emergency landing...")
        drone.land()

    except Exception as e:
        print(f"\n\n[ERROR] Error occurred: {e}")
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

        if planner.visualizer:
            planner.visualizer.close()

        print("\n" + "=" * 70)
        print("  Test completed!")
        print("=" * 70)


if __name__ == "__main__":
    main()
