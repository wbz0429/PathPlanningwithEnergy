"""
fly_planned_path.py - Receding Horizon Planning Flight Test
使用增量式建图 + 滚动规划实现动态避障

Phase 2: 解决单次感知的遮挡问题
Phase 3: 能量感知规划
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
from energy.physics_model import PhysicsEnergyModel


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
        # === 2. 重置无人机到起点 ===
        print("\n[2] Resetting drone to initial position...")
        drone.reset()
        time.sleep(2)

        # === 3. 起飞 ===
        print("\n[3] Taking off...")
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
            grid_size=(180, 120, 40),  # X方向覆盖90m(-10~80), Y方向覆盖60m(-30~30)
            origin=(-10.0, -30.0, -15.0),  # 固定origin，覆盖X=-10~80, Y=-30~30, Z=-15~5
            max_depth=25.0,
            step_size=1.5,
            max_iterations=3000,
            goal_sample_rate=0.2,
            search_radius=4.0,
            safety_margin=1.0,  # 安全边距：体素误差0.25m + 无人机半径0.3m + 缓冲0.45m
            # 能量感知规划参数
            energy_aware=True,
            weight_energy=0.6,
            weight_distance=0.3,
            weight_time=0.1
        )

        # 初始化能耗模型
        energy_model = PhysicsEnergyModel()
        print(f"    Energy model: BEMT physics model")
        print(f"    Hover power: {energy_model.compute_hover_power():.1f} W")

        # 滚动规划配置 - 优化后的参数
        receding_config = {
            'local_horizon': 8.0,      # 局部目标距离8m
            'execution_ratio': 0.3,    # 只执行30%，更频繁重规划以提高安全性
            'replan_threshold': 1.5,   # 重规划阈值
            'goal_tolerance': 2.0,     # 放宽到达目标的判定阈值
            'max_iterations': 80,      # 增大最大循环次数（因为每次执行更少）
            'flight_velocity': 1.5,    # 降低飞行速度，给感知更多反应时间
            'timeout_seconds': 300,    # 5分钟超时
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
        planner = RecedingHorizonPlanner(map_manager, drone, receding_config, energy_model=energy_model)
        print("    [OK] Initialization complete")
        print(f"    Energy-aware planning: {planning_config.energy_aware}")

        # === 6. 设置全局目标 ===
        print("\n[6] Setting global goal...")

        # 场景精确分析结果（通过 query_scene_objects.py 获取）：
        #
        # Row 1 (X=23.1~33.1): Y=[-21.5, 18.5] 全覆盖，无缝隙（实心墙）
        # Row 2 (X=33.1~43.1): Y=[-21.5,-11.5] 和 Y=[8.5,18.5]
        #                       中间有 20m 宽缝隙 Y=[-11.5, 8.5] 可穿过！
        # Row 3 (X=43.1~53.1): 同 Row 2，中间有 20m 缝隙
        # Row 4 (X=53.1~63.1): 实心墙
        # OrangeBall: X=34.2, Y=33.1, 半径5m -> 避开 Y=[28,38]
        #
        # 策略：绕过 R1 边缘，穿过 R2/R3 缝隙
        # 目标设在 R2 缝隙中间（X=38, Y=0），确保不在任何墙体内
        global_goal = np.array([
            70.0,                   # X: 穿过所有建筑排
            20.0,                   # Y: 目标Y=20
            -3.0                    # Z: 3m高度（NED）
        ])

        print(f"    Start:  ({initial_pos[0]:.2f}, {initial_pos[1]:.2f}, {initial_pos[2]:.2f})")
        print(f"    Goal:   ({global_goal[0]:.2f}, {global_goal[1]:.2f}, {global_goal[2]:.2f})")
        print(f"    Distance: {np.linalg.norm(global_goal - initial_pos):.2f}m")
        print("\n    *** OBSTACLE AVOIDANCE TEST ***")
        print("    Row 1 (X=23~33): solid wall, must go around edge (Y>18.5 or Y<-21.5)")
        print("    Row 2/3 (X=33~53): 20m gap at Y=[-11.5, 8.5], can fly through")
        print("    OrangeBall at Y=33, avoid Y=[28,38]")

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

        # 能耗统计
        energy_stats = planner.get_energy_stats()
        if energy_stats['total_energy_joules'] > 0:
            print(f"\n  Energy statistics:")
            print(f"    Total energy: {energy_stats['total_energy_joules']:.1f} J ({energy_stats['total_energy_wh']:.3f} Wh)")
            print(f"    Energy per meter: {energy_stats['energy_per_meter']:.1f} J/m")
            battery_wh = energy_model.params.battery_voltage * energy_model.params.battery_capacity / 1000
            battery_pct = energy_stats['total_energy_wh'] / battery_wh * 100
            print(f"    Battery used: {battery_pct:.1f}%")

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
