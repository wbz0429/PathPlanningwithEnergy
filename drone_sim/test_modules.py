"""
test_modules.py - 单元测试
测试各个模块的基本功能
"""

import numpy as np
import sys
import os

# 添加模块路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from planning.config import PlanningConfig
from mapping.voxel_grid import VoxelGrid
from mapping.esdf import ESDF
from mapping.incremental_map import IncrementalMapManager
from planning.rrt_star import RRTStar
from utils.transforms import quaternion_to_rotation_matrix, transform_camera_to_world, depth_image_to_camera_points
from utils.performance import PerformanceMonitor


def test_config():
    """测试配置模块"""
    print("\n" + "=" * 60)
    print("Test 1: PlanningConfig")
    print("=" * 60)

    config = PlanningConfig(
        voxel_size=0.5,
        grid_size=(80, 80, 40),
        origin=(-20.0, -20.0, 0.0)
    )

    print(f"[OK] Config created")
    print(f"  Voxel size: {config.voxel_size}m")
    print(f"  Grid size: {config.grid_size}")
    print(f"  Origin: {config.origin}")
    print(f"  Max depth: {config.max_depth}m")

    return True


def test_transforms():
    """测试坐标变换"""
    print("\n" + "=" * 60)
    print("Test 2: Coordinate Transforms")
    print("=" * 60)

    # 测试四元数转旋转矩阵
    q = np.array([1.0, 0.0, 0.0, 0.0])  # 单位四元数
    R = quaternion_to_rotation_matrix(q)

    print(f"[OK] Quaternion to rotation matrix")
    print(f"  Input quaternion: {q}")
    print(f"  Output rotation matrix:\n{R}")

    # 验证是单位矩阵
    identity = np.eye(3)
    if np.allclose(R, identity):
        print(f"  [OK] Correct: Identity matrix")
    else:
        print(f"  [FAIL] Error: Not identity matrix")
        return False

    # 测试深度图转点云
    depth = np.ones((144, 256)) * 5.0  # 5米深度
    points = depth_image_to_camera_points(depth, fov_deg=90.0, subsample=8)

    print(f"\n[OK] Depth image to camera points")
    print(f"  Depth image shape: {depth.shape}")
    print(f"  Point cloud shape: {points.shape}")
    print(f"  Sample points:\n{points[:3]}")

    # 测试相机到世界坐标变换
    drone_pos = np.array([0.0, 0.0, -3.0])
    drone_ori = np.array([1.0, 0.0, 0.0, 0.0])

    points_world = transform_camera_to_world(points, drone_pos, drone_ori)

    print(f"\n[OK] Camera to world transform")
    print(f"  Drone position: {drone_pos}")
    print(f"  World points shape: {points_world.shape}")
    print(f"  Sample world points:\n{points_world[:3]}")

    return True


def test_voxel_grid():
    """测试体素栅格"""
    print("\n" + "=" * 60)
    print("Test 3: VoxelGrid")
    print("=" * 60)

    config = PlanningConfig(
        voxel_size=0.5,
        grid_size=(40, 40, 20),
        origin=(-10.0, -10.0, -5.0)
    )

    voxel_grid = VoxelGrid(config)

    print(f"[OK] VoxelGrid created")
    print(f"  Grid shape: {voxel_grid.grid.shape}")
    print(f"  Voxel size: {voxel_grid.voxel_size}m")

    # 测试坐标转换
    world_point = np.array([0.0, 0.0, 0.0])
    grid_idx = voxel_grid.world_to_grid(world_point)
    world_back = voxel_grid.grid_to_world(grid_idx)

    print(f"\n[OK] Coordinate conversion")
    print(f"  World point: {world_point}")
    print(f"  Grid index: {grid_idx}")
    print(f"  World back: {world_back}")

    # 测试深度图更新
    depth = np.ones((144, 256)) * 5.0
    camera_pos = np.array([0.0, 0.0, -3.0])

    occupied_count = voxel_grid.update_from_depth_image(depth, camera_pos)

    print(f"\n[OK] Depth image update")
    print(f"  Occupied voxels: {occupied_count}")

    return True


def test_esdf():
    """测试 ESDF"""
    print("\n" + "=" * 60)
    print("Test 4: ESDF")
    print("=" * 60)

    config = PlanningConfig(
        voxel_size=0.5,
        grid_size=(40, 40, 20),
        origin=(-10.0, -10.0, -5.0)
    )

    voxel_grid = VoxelGrid(config)

    # 创建一个简单的障碍物
    for i in range(15, 25):
        for j in range(15, 25):
            for k in range(5, 15):
                voxel_grid.grid[i, j, k] = 1

    print(f"[OK] Created test obstacle")
    print(f"  Occupied voxels: {np.sum(voxel_grid.grid == 1)}")

    # 计算 ESDF
    esdf = ESDF(voxel_grid)
    esdf.compute()

    print(f"\n[OK] ESDF computed")
    print(f"  Distance field shape: {esdf.distance_field.shape}")

    # 测试距离查询
    test_point = np.array([0.0, 0.0, 0.0])
    distance = esdf.get_distance(test_point)

    print(f"\n[OK] Distance query")
    print(f"  Test point: {test_point}")
    print(f"  Distance to obstacle: {distance:.2f}m")

    # 测试安全性检查
    is_safe = esdf.is_safe(test_point, margin=1.0)
    print(f"  Is safe (margin=1.0m): {is_safe}")

    return True


def test_incremental_map():
    """测试增量式地图"""
    print("\n" + "=" * 60)
    print("Test 5: IncrementalMapManager")
    print("=" * 60)

    config = PlanningConfig(
        voxel_size=0.5,
        grid_size=(40, 40, 20),
        origin=(-10.0, -10.0, -5.0)
    )

    map_manager = IncrementalMapManager(config)

    print(f"[OK] IncrementalMapManager created")

    # 模拟多次更新
    for i in range(3):
        depth = np.ones((144, 256)) * (5.0 + i)
        drone_pos = np.array([i * 2.0, 0.0, -3.0])
        drone_ori = np.array([1.0, 0.0, 0.0, 0.0])

        stats = map_manager.update(depth, drone_pos, drone_ori)

        print(f"\n  Update {i+1}:")
        print(f"    New occupied: {stats['new_occupied']}")
        print(f"    Total occupied: {stats['total_occupied']}")
        print(f"    Update time: {stats['update_time_ms']:.1f}ms")

    # 获取地图统计
    map_stats = map_manager.get_map_stats()
    print(f"\n[OK] Map statistics:")
    print(f"  Total updates: {map_stats['total_updates']}")
    print(f"  Occupied voxels: {map_stats['occupied_voxels']}")
    print(f"  Free voxels: {map_stats['free_voxels']}")
    print(f"  Unknown voxels: {map_stats['unknown_voxels']}")

    return True


def test_rrt_star():
    """测试 RRT*"""
    print("\n" + "=" * 60)
    print("Test 6: RRT* Planner")
    print("=" * 60)

    config = PlanningConfig(
        voxel_size=0.5,
        grid_size=(40, 40, 20),
        origin=(-10.0, -10.0, -5.0),
        max_iterations=2000,
        step_size=1.0
    )

    voxel_grid = VoxelGrid(config)

    # 创建一个简单的障碍物墙，并将其余空间标记为已知空闲
    # 先将所有体素标记为空闲（模拟已探索环境）
    voxel_grid.grid[:] = -1

    # 再设置障碍物墙
    for i in range(18, 22):
        for j in range(10, 30):
            for k in range(5, 15):
                voxel_grid.grid[i, j, k] = 1

    print(f"[OK] Created test obstacle wall")

    esdf = ESDF(voxel_grid)
    esdf.compute()

    print(f"[OK] ESDF computed")

    rrt = RRTStar(voxel_grid, esdf, config)

    # 规划路径（绕过障碍物）
    start = np.array([-5.0, 0.0, 0.0])
    goal = np.array([5.0, 0.0, 0.0])

    print(f"\n[OK] Planning path...")
    print(f"  Start: {start}")
    print(f"  Goal: {goal}")

    import time
    start_time = time.time()
    path = rrt.plan(start, goal)
    planning_time = (time.time() - start_time) * 1000

    if path is not None:
        print(f"\n[OK] Path found!")
        print(f"  Waypoints: {len(path)}")
        print(f"  Planning time: {planning_time:.1f}ms")

        # 计算路径长度
        path_arr = np.array(path)
        path_length = np.sum(np.linalg.norm(np.diff(path_arr, axis=0), axis=1))
        print(f"  Path length: {path_length:.2f}m")

        return True
    else:
        print(f"\n[FAIL] Path planning failed")
        return False


def test_performance_monitor():
    """测试性能监控"""
    print("\n" + "=" * 60)
    print("Test 7: PerformanceMonitor")
    print("=" * 60)

    monitor = PerformanceMonitor()

    # 模拟一些操作
    import time

    with monitor.measure('operation1'):
        time.sleep(0.01)

    with monitor.measure('operation2'):
        time.sleep(0.02)

    with monitor.measure('operation1'):
        time.sleep(0.015)

    print(f"[OK] Performance monitoring")
    monitor.print_summary()

    avg1 = monitor.get_average('operation1')
    avg2 = monitor.get_average('operation2')

    print(f"\n  Average times:")
    print(f"    operation1: {avg1:.1f}ms")
    print(f"    operation2: {avg2:.1f}ms")

    return True


def main():
    """运行所有测试"""
    print("\n" + "=" * 70)
    print("  MODULE UNIT TESTS")
    print("=" * 70)

    tests = [
        ("Config", test_config),
        ("Transforms", test_transforms),
        ("VoxelGrid", test_voxel_grid),
        ("ESDF", test_esdf),
        ("IncrementalMap", test_incremental_map),
        ("RRT*", test_rrt_star),
        ("PerformanceMonitor", test_performance_monitor),
    ]

    results = []

    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, success))
        except Exception as e:
            print(f"\n[FAIL] Test failed with exception: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))

    # 打印总结
    print("\n" + "=" * 70)
    print("  TEST SUMMARY")
    print("=" * 70)

    passed = sum(1 for _, success in results if success)
    total = len(results)

    for name, success in results:
        status = "[PASS]" if success else "[FAIL]"
        print(f"  {status}: {name}")

    print(f"\n  Total: {passed}/{total} tests passed")

    if passed == total:
        print("\n  [OK] All tests passed!")
        return True
    else:
        print(f"\n  [FAIL] {total - passed} test(s) failed")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
