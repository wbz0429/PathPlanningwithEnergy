"""
test_mapping_and_planning.py - 分别测试建图和规划

不需要 AirSim，用已知障碍物位置直接构建地图，
然后测试 RRT* 能否规划出合理的避障路径。
"""

import numpy as np
import sys
import os
import time

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from planning.config import PlanningConfig
from mapping.voxel_grid import VoxelGrid
from mapping.esdf import ESDF
from planning.rrt_star import RRTStar


def build_ground_truth_map(voxel_grid):
    """
    用场景真实数据直接构建地图（不依赖深度图）

    Blocks 场景障碍物：
    Row 1 (center X=28.1): 4 columns of cubes, each 10x10x5m
      Y centers: -16.5, -6.5, 3.5, 13.5 (each ±5m)
      Z centers at flight level: -1.5 (each ±2.5m, so Z=-4 to 1)
      -> X=[23.1, 33.1], Y=[-21.5, 18.5], Z=[-4, 1] (solid wall)

    Row 2 (center X=38.1): only 2 columns at flight level
      Y=-16.5 and Y=13.5 -> Y=[-21.5,-11.5] and [8.5,18.5]
      GAP: Y=[-11.5, 8.5] is open!

    Row 3 (center X=48.1): same as Row 2

    OrangeBall: center (34.2, 33.1, -3.7), radius ~5m
    """
    count = 0

    # Row 1: solid wall
    for x in np.arange(23.1, 33.1, 0.5):
        for y in np.arange(-21.5, 18.5, 0.5):
            for z in np.arange(-4.0, 1.0, 0.5):
                idx = voxel_grid.world_to_grid(np.array([x, y, z]))
                if voxel_grid.is_valid_index(idx):
                    voxel_grid.grid[idx] = 1
                    count += 1

    # Row 2: two columns with gap
    for x in np.arange(33.1, 43.1, 0.5):
        for z in np.arange(-4.0, 1.0, 0.5):
            # Left column
            for y in np.arange(-21.5, -11.5, 0.5):
                idx = voxel_grid.world_to_grid(np.array([x, y, z]))
                if voxel_grid.is_valid_index(idx):
                    voxel_grid.grid[idx] = 1
                    count += 1
            # Right column
            for y in np.arange(8.5, 18.5, 0.5):
                idx = voxel_grid.world_to_grid(np.array([x, y, z]))
                if voxel_grid.is_valid_index(idx):
                    voxel_grid.grid[idx] = 1
                    count += 1

    # Row 3: same as Row 2
    for x in np.arange(43.1, 53.1, 0.5):
        for z in np.arange(-4.0, 1.0, 0.5):
            for y in np.arange(-21.5, -11.5, 0.5):
                idx = voxel_grid.world_to_grid(np.array([x, y, z]))
                if voxel_grid.is_valid_index(idx):
                    voxel_grid.grid[idx] = 1
                    count += 1
            for y in np.arange(8.5, 18.5, 0.5):
                idx = voxel_grid.world_to_grid(np.array([x, y, z]))
                if voxel_grid.is_valid_index(idx):
                    voxel_grid.grid[idx] = 1
                    count += 1

    return count


def test_esdf(esdf, test_points):
    """测试 ESDF 在关键位置的值"""
    print("\n=== ESDF Test ===")
    print(f"{'Point':<30} {'ESDF':>8} {'Status'}")
    print("-" * 55)

    for name, point in test_points:
        dist = esdf.get_distance(np.array(point))
        if dist < 0:
            status = "INSIDE OBSTACLE"
        elif dist < 1.0:
            status = "CLOSE"
        else:
            status = "safe"
        print(f"{name:<30} {dist:>7.2f}m  {status}")


def test_rrt_planning(voxel_grid, esdf, config, test_cases):
    """测试 RRT* 规划"""
    print("\n=== RRT* Planning Test ===")

    for name, start, goal in test_cases:
        print(f"\n--- {name} ---")
        print(f"  Start: ({start[0]:.1f}, {start[1]:.1f}, {start[2]:.1f})")
        print(f"  Goal:  ({goal[0]:.1f}, {goal[1]:.1f}, {goal[2]:.1f})")

        # 检查起点和终点的 ESDF
        start_dist = esdf.get_distance(np.array(start))
        goal_dist = esdf.get_distance(np.array(goal))
        print(f"  Start ESDF: {start_dist:.2f}m, Goal ESDF: {goal_dist:.2f}m")

        rrt = RRTStar(voxel_grid, esdf, config)

        t0 = time.time()
        path = rrt.plan(np.array(start), np.array(goal))
        dt = (time.time() - t0) * 1000

        if path is None:
            print(f"  RESULT: FAILED ({dt:.0f}ms)")
        else:
            total_dist = sum(np.linalg.norm(path[i+1] - path[i]) for i in range(len(path)-1))
            print(f"  RESULT: SUCCESS ({dt:.0f}ms)")
            print(f"  Path: {len(path)} waypoints, total distance: {total_dist:.1f}m")

            # 打印路径点
            for i, wp in enumerate(path):
                wp_esdf = esdf.get_distance(wp)
                safe = "OK" if wp_esdf > config.safety_margin else "UNSAFE!"
                print(f"    WP{i}: ({wp[0]:.1f}, {wp[1]:.1f}, {wp[2]:.1f}) ESDF={wp_esdf:.2f}m {safe}")

            # 检查路径是否穿过障碍物
            collision = False
            for i in range(len(path) - 1):
                direction = path[i+1] - path[i]
                dist = np.linalg.norm(direction)
                num_checks = max(3, int(dist / 0.2))
                for j in range(num_checks + 1):
                    t = j / num_checks
                    point = path[i] + t * direction
                    d = esdf.get_distance(point)
                    if d < 0:
                        collision = True
                        print(f"    COLLISION at ({point[0]:.1f}, {point[1]:.1f}, {point[2]:.1f}) ESDF={d:.2f}m")
                        break
                if collision:
                    break

            if not collision:
                print(f"    Path collision check: CLEAR")


def main():
    print("=" * 70)
    print("  MAPPING & PLANNING ISOLATED TEST")
    print("  Using ground truth obstacles (no AirSim needed)")
    print("=" * 70)

    # 配置
    config = PlanningConfig(
        voxel_size=0.5,
        grid_size=(140, 120, 40),
        origin=(-10.0, -30.0, -15.0),
        max_depth=25.0,
        step_size=1.5,
        max_iterations=3000,
        goal_sample_rate=0.2,
        search_radius=4.0,
        safety_margin=1.0,
    )

    print(f"\nGrid: {config.grid_size}, voxel: {config.voxel_size}m")
    print(f"World coverage: X=[{config.origin[0]}, {config.origin[0] + config.grid_size[0]*config.voxel_size}]")
    print(f"                Y=[{config.origin[1]}, {config.origin[1] + config.grid_size[1]*config.voxel_size}]")
    print(f"                Z=[{config.origin[2]}, {config.origin[2] + config.grid_size[2]*config.voxel_size}]")

    # 构建地图
    print("\n[1] Building ground truth map...")
    voxel_grid = VoxelGrid(config)
    count = build_ground_truth_map(voxel_grid)
    print(f"    Occupied voxels: {count}")

    # 计算 ESDF
    print("\n[2] Computing ESDF...")
    esdf = ESDF(voxel_grid)
    t0 = time.time()
    esdf.compute()
    dt = (time.time() - t0) * 1000
    print(f"    ESDF computed in {dt:.0f}ms")

    # 测试 ESDF
    test_points = [
        ("Drone start (0, 0, -3)",       [0.0, 0.0, -3.0]),
        ("Before wall (20, 0, -3)",       [20.0, 0.0, -3.0]),
        ("At wall (23, 0, -3)",           [23.0, 0.0, -3.0]),
        ("Inside wall (28, 0, -3)",       [28.0, 0.0, -3.0]),
        ("After wall (34, 0, -3)",        [34.0, 0.0, -3.0]),
        ("Row2 gap (38, 0, -3)",          [38.0, 0.0, -3.0]),
        ("Row2 wall (38, 15, -3)",        [38.0, 15.0, -3.0]),
        ("Wall edge Y=-12 (28, -12, -3)", [28.0, -12.0, -3.0]),
        ("Wall edge Y=19 (28, 19, -3)",   [28.0, 19.0, -3.0]),
        ("Goal (35, 0, -3)",              [35.0, 0.0, -3.0]),
    ]
    test_esdf(esdf, test_points)

    # 测试 RRT* 规划
    test_cases = [
        # Case 1: 简单直线（无障碍物）
        ("Simple straight line (no obstacles)",
         [0.0, 0.0, -3.0], [15.0, 0.0, -3.0]),

        # Case 2: 穿过 Row2/3 的缝隙
        ("Through Row2 gap (Y=0)",
         [34.0, 0.0, -3.0], [55.0, 0.0, -3.0]),

        # Case 3: 绕过 Row1（需要侧移到 Y>18.5 或 Y<-21.5）
        ("Around Row1 wall",
         [20.0, 0.0, -3.0], [34.0, 0.0, -3.0]),

        # Case 4: 从起点到终点（完整任务）
        ("Full mission: start to goal",
         [0.0, 0.0, -3.0], [35.0, 0.0, -3.0]),

        # Case 5: 沿墙侧移
        ("Lateral move along wall",
         [20.0, 0.0, -3.0], [20.0, -22.0, -3.0]),

        # Case 6: 绕墙后穿缝隙
        ("Around wall then through gap",
         [20.0, -22.0, -3.0], [35.0, 0.0, -3.0]),
    ]

    test_rrt_planning(voxel_grid, esdf, config, test_cases)

    print("\n" + "=" * 70)
    print("  Test complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
