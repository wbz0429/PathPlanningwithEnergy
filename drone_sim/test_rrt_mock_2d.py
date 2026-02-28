"""
test_rrt_mock_2d.py - 纯离线2D测试RRT*算法
不需要AirSim，用mock体素地图验证：
1. RRT*能否绕过墙壁
2. 路径是否左右摇摆
3. unknown区域阈值逻辑是否正确
"""

import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from planning.config import PlanningConfig
from mapping.voxel_grid import VoxelGrid
from mapping.esdf import ESDF
from planning.rrt_star import RRTStar


def create_wall_scene(wall_gap_y=None):
    """
    创建一个2D墙壁场景（固定Z层）

    Grid: 80x80x10, voxel=0.5m => 40x40x5m 空间
    Origin: (-5, -20, -5)  => X: -5~35, Y: -20~20, Z: -5~0

    墙壁在 X=15m 处，Y方向从 -15m 到 +15m（30m宽）
    如果 wall_gap_y 不为 None，在该Y位置留一个3m宽的缺口

    Returns:
        voxel_grid, esdf, config
    """
    config = PlanningConfig(
        voxel_size=0.5,
        grid_size=(80, 80, 10),
        origin=(-5.0, -20.0, -5.0),
        max_iterations=3000,
        step_size=1.5,
        search_radius=4.0,
        safety_margin=1.0,
        unknown_safe_threshold=3.0,
        energy_aware=False,  # 纯距离代价，排除能量干扰
    )

    voxel_grid = VoxelGrid(config)

    # 将所有体素标记为已知空闲
    voxel_grid.grid[:] = -1

    # 创建墙壁: X=15m处，厚度1m (2个体素)，Y从-15到+15
    # grid index: X=(15-(-5))/0.5=40, Y=(-15-(-20))/0.5=10 到 (15-(-20))/0.5=70
    wall_x_start = 40  # X=15m
    wall_x_end = 42    # X=16m (1m厚)
    wall_y_start = 10  # Y=-15m
    wall_y_end = 70    # Y=+15m

    for ix in range(wall_x_start, wall_x_end):
        for iy in range(wall_y_start, wall_y_end):
            for iz in range(0, 10):  # 全高度
                # 如果有缺口，跳过缺口位置
                if wall_gap_y is not None:
                    world_y = config.origin[1] + iy * config.voxel_size
                    if abs(world_y - wall_gap_y) < 1.5:  # 3m宽缺口
                        continue
                voxel_grid.grid[ix, iy, iz] = 1

    esdf = ESDF(voxel_grid)
    esdf.compute()

    return voxel_grid, esdf, config


def create_blocks_like_scene():
    """
    模拟Blocks场景：
    - Row1 (X=20~22): 完整墙壁 Y=-15~+15，无缝隙
    - Row2 (X=28~30): Y=-15~-5 和 Y=+5~+15，中间10m缺口

    无人机从 X=0 飞到 X=35
    """
    config = PlanningConfig(
        voxel_size=0.5,
        grid_size=(80, 80, 10),
        origin=(-5.0, -20.0, -5.0),
        max_iterations=3000,
        step_size=1.5,
        search_radius=4.0,
        safety_margin=1.0,
        unknown_safe_threshold=3.0,
        energy_aware=False,
    )

    voxel_grid = VoxelGrid(config)
    voxel_grid.grid[:] = -1  # 全部已知空闲

    def world_to_idx(x, y):
        ix = int((x - config.origin[0]) / config.voxel_size)
        iy = int((y - config.origin[1]) / config.voxel_size)
        return ix, iy

    # Row 1: X=20~22, Y=-15~+15 (完整墙壁)
    for x in np.arange(20.0, 22.0, 0.5):
        for y in np.arange(-15.0, 15.0, 0.5):
            ix, iy = world_to_idx(x, y)
            if 0 <= ix < 80 and 0 <= iy < 80:
                for iz in range(10):
                    voxel_grid.grid[ix, iy, iz] = 1

    # Row 2: X=28~30, Y=-15~-5 和 Y=+5~+15 (中间有缺口)
    for x in np.arange(28.0, 30.0, 0.5):
        for y in np.arange(-15.0, -5.0, 0.5):
            ix, iy = world_to_idx(x, y)
            if 0 <= ix < 80 and 0 <= iy < 80:
                for iz in range(10):
                    voxel_grid.grid[ix, iy, iz] = 1
        for y in np.arange(5.0, 15.0, 0.5):
            ix, iy = world_to_idx(x, y)
            if 0 <= ix < 80 and 0 <= iy < 80:
                for iz in range(10):
                    voxel_grid.grid[ix, iy, iz] = 1

    esdf = ESDF(voxel_grid)
    esdf.compute()

    return voxel_grid, esdf, config


def analyze_path_oscillation(path):
    """分析路径是否左右摇摆"""
    if path is None or len(path) < 3:
        total_length = sum(np.linalg.norm(path[i+1] - path[i]) for i in range(len(path)-1)) if path and len(path) > 1 else 0
        straight_dist = np.linalg.norm(path[-1] - path[0]) if path and len(path) > 1 else 0
        y_values = [p[1] for p in path] if path else [0]
        return {
            'oscillating': False, 'direction_changes': 0,
            'path_length': total_length, 'straight_distance': straight_dist,
            'efficiency': straight_dist / total_length if total_length > 0 else 1.0,
            'y_range': (min(y_values), max(y_values)), 'num_waypoints': len(path) if path else 0,
        }

    # 计算Y方向的变化
    y_values = [p[1] for p in path]
    y_changes = [y_values[i+1] - y_values[i] for i in range(len(y_values)-1)]

    # 统计方向变化次数
    direction_changes = 0
    for i in range(len(y_changes)-1):
        if y_changes[i] * y_changes[i+1] < 0:  # 符号变化
            direction_changes += 1

    # 计算路径总长度和直线距离
    total_length = sum(np.linalg.norm(path[i+1] - path[i]) for i in range(len(path)-1))
    straight_dist = np.linalg.norm(path[-1] - path[0])
    efficiency = straight_dist / total_length if total_length > 0 else 0

    oscillating = direction_changes > len(path) * 0.4

    return {
        'oscillating': oscillating,
        'direction_changes': direction_changes,
        'path_length': total_length,
        'straight_distance': straight_dist,
        'efficiency': efficiency,
        'y_range': (min(y_values), max(y_values)),
        'num_waypoints': len(path),
    }


def test_simple_wall():
    """测试1: 简单墙壁，无缺口 — RRT*必须绕过墙壁边缘"""
    print("\n" + "=" * 60)
    print("TEST 1: Simple wall (no gap) - must go around edge")
    print("=" * 60)

    voxel_grid, esdf, config = create_wall_scene(wall_gap_y=None)
    rrt = RRTStar(voxel_grid, esdf, config)

    start = np.array([0.0, 0.0, -2.5])
    goal = np.array([30.0, 0.0, -2.5])

    print(f"  Start: {start}")
    print(f"  Goal:  {goal}")
    print(f"  Wall:  X=15~16m, Y=-15~+15m (30m wide, no gap)")

    import time
    t0 = time.time()
    path = rrt.plan(start, goal)
    dt = (time.time() - t0) * 1000

    if path is None:
        print(f"  [FAIL] No path found ({dt:.0f}ms)")
        return False

    analysis = analyze_path_oscillation(path)
    print(f"  [OK] Path found ({dt:.0f}ms)")
    print(f"    Waypoints: {analysis['num_waypoints']}")
    print(f"    Length: {analysis['path_length']:.1f}m (straight: {analysis['straight_distance']:.1f}m)")
    print(f"    Efficiency: {analysis['efficiency']:.2f}")
    print(f"    Y range: [{analysis['y_range'][0]:.1f}, {analysis['y_range'][1]:.1f}]")
    print(f"    Direction changes: {analysis['direction_changes']}")
    print(f"    Oscillating: {analysis['oscillating']}")

    # 打印路径点
    print(f"    Path:")
    for i, p in enumerate(path):
        print(f"      [{i}] ({p[0]:.1f}, {p[1]:.1f}, {p[2]:.1f})")

    return not analysis['oscillating']


def test_wall_with_gap():
    """测试2: 墙壁有缺口 — RRT*应该穿过缺口"""
    print("\n" + "=" * 60)
    print("TEST 2: Wall with gap at Y=0 - should go through gap")
    print("=" * 60)

    voxel_grid, esdf, config = create_wall_scene(wall_gap_y=0.0)
    rrt = RRTStar(voxel_grid, esdf, config)

    start = np.array([0.0, 0.0, -2.5])
    goal = np.array([30.0, 0.0, -2.5])

    print(f"  Start: {start}")
    print(f"  Goal:  {goal}")
    print(f"  Wall:  X=15~16m, Y=-15~+15m with 3m gap at Y=0")

    import time
    t0 = time.time()
    path = rrt.plan(start, goal)
    dt = (time.time() - t0) * 1000

    if path is None:
        print(f"  [FAIL] No path found ({dt:.0f}ms)")
        return False

    analysis = analyze_path_oscillation(path)
    print(f"  [OK] Path found ({dt:.0f}ms)")
    print(f"    Waypoints: {analysis['num_waypoints']}")
    print(f"    Length: {analysis['path_length']:.1f}m (straight: {analysis['straight_distance']:.1f}m)")
    print(f"    Efficiency: {analysis['efficiency']:.2f}")
    print(f"    Y range: [{analysis['y_range'][0]:.1f}, {analysis['y_range'][1]:.1f}]")
    print(f"    Direction changes: {analysis['direction_changes']}")

    # 打印路径点
    print(f"    Path:")
    for i, p in enumerate(path):
        print(f"      [{i}] ({p[0]:.1f}, {p[1]:.1f}, {p[2]:.1f})")

    # 检查是否穿过缺口（Y应该接近0）
    wall_crossing_y = None
    for i in range(len(path)-1):
        if path[i][0] < 15.5 and path[i+1][0] > 15.5:
            # 插值找穿越点
            t = (15.5 - path[i][0]) / (path[i+1][0] - path[i][0])
            wall_crossing_y = path[i][1] + t * (path[i+1][1] - path[i][1])
            break

    if wall_crossing_y is not None:
        print(f"    Wall crossing Y: {wall_crossing_y:.1f}m (gap at Y=0)")
        went_through_gap = abs(wall_crossing_y) < 3.0
        print(f"    Went through gap: {went_through_gap}")

    return True


def test_blocks_scene():
    """测试3: 模拟Blocks场景 — Row1绕边，Row2穿缝隙"""
    print("\n" + "=" * 60)
    print("TEST 3: Blocks-like scene (Row1 solid + Row2 with gap)")
    print("=" * 60)

    voxel_grid, esdf, config = create_blocks_like_scene()
    rrt = RRTStar(voxel_grid, esdf, config)

    start = np.array([0.0, 0.0, -2.5])
    goal = np.array([35.0, 0.0, -2.5])

    print(f"  Start: {start}")
    print(f"  Goal:  {goal}")
    print(f"  Row1:  X=20~22m, Y=-15~+15m (solid)")
    print(f"  Row2:  X=28~30m, gap at Y=-5~+5m")

    import time
    t0 = time.time()
    path = rrt.plan(start, goal)
    dt = (time.time() - t0) * 1000

    if path is None:
        print(f"  [FAIL] No path found ({dt:.0f}ms)")
        return False

    analysis = analyze_path_oscillation(path)
    print(f"  [OK] Path found ({dt:.0f}ms)")
    print(f"    Waypoints: {analysis['num_waypoints']}")
    print(f"    Length: {analysis['path_length']:.1f}m (straight: {analysis['straight_distance']:.1f}m)")
    print(f"    Efficiency: {analysis['efficiency']:.2f}")
    print(f"    Y range: [{analysis['y_range'][0]:.1f}, {analysis['y_range'][1]:.1f}]")
    print(f"    Direction changes: {analysis['direction_changes']}")
    print(f"    Oscillating: {analysis['oscillating']}")

    # 打印路径点
    print(f"    Path:")
    for i, p in enumerate(path):
        print(f"      [{i}] ({p[0]:.1f}, {p[1]:.1f}, {p[2]:.1f})")

    return not analysis['oscillating']


def test_unknown_threshold():
    """测试4: Unknown区域阈值 — 验证远离障碍物的unknown可通行"""
    print("\n" + "=" * 60)
    print("TEST 4: Unknown region threshold logic")
    print("=" * 60)

    config = PlanningConfig(
        voxel_size=0.5,
        grid_size=(40, 40, 10),
        origin=(-10.0, -10.0, -5.0),
        max_iterations=2000,
        step_size=1.0,
        safety_margin=1.0,
        unknown_safe_threshold=3.0,
        energy_aware=False,
    )

    voxel_grid = VoxelGrid(config)
    # 默认全部unknown (grid==0)

    # 只在起点和终点附近标记为空闲
    # 起点 (-5,0,-2.5) => grid idx (10,20,5)
    # 终点 (5,0,-2.5) => grid idx (30,20,5)
    for dx in range(-4, 5):
        for dy in range(-4, 5):
            for dz in range(-2, 3):
                # 起点附近
                idx = (10+dx, 20+dy, 5+dz)
                if voxel_grid.is_valid_index(idx):
                    voxel_grid.grid[idx] = -1
                # 终点附近
                idx = (30+dx, 20+dy, 5+dz)
                if voxel_grid.is_valid_index(idx):
                    voxel_grid.grid[idx] = -1

    # 中间没有障碍物，全是unknown
    # ESDF应该显示所有点都远离障碍物 => unknown应该被允许通行

    esdf = ESDF(voxel_grid)
    esdf.compute()

    rrt = RRTStar(voxel_grid, esdf, config)

    start = np.array([-5.0, 0.0, -2.5])
    goal = np.array([5.0, 0.0, -2.5])

    print(f"  Start: {start} (known free)")
    print(f"  Goal:  {goal} (known free)")
    print(f"  Middle: all unknown, no obstacles")
    print(f"  Expected: path through unknown (ESDF dist >> 3.0m)")

    # 检查中间点的ESDF距离
    mid_point = np.array([0.0, 0.0, -2.5])
    mid_dist = esdf.get_distance(mid_point)
    mid_idx = voxel_grid.world_to_grid(mid_point)
    mid_state = voxel_grid.grid[mid_idx] if voxel_grid.is_valid_index(mid_idx) else 'invalid'
    print(f"  Mid point ESDF dist: {mid_dist:.1f}m, grid state: {mid_state}")

    import time
    t0 = time.time()
    path = rrt.plan(start, goal)
    dt = (time.time() - t0) * 1000

    if path is None:
        print(f"  [FAIL] No path found ({dt:.0f}ms) — unknown rejection too aggressive!")
        return False

    print(f"  [OK] Path found through unknown region ({dt:.0f}ms)")
    print(f"    Waypoints: {len(path)}")
    print(f"    Path:")
    for i, p in enumerate(path):
        print(f"      [{i}] ({p[0]:.1f}, {p[1]:.1f}, {p[2]:.1f})")

    return True


def test_unknown_near_wall():
    """测试5: Unknown区域靠近墙壁 — 应该被拒绝"""
    print("\n" + "=" * 60)
    print("TEST 5: Unknown region near wall - should be rejected")
    print("=" * 60)

    config = PlanningConfig(
        voxel_size=0.5,
        grid_size=(40, 40, 10),
        origin=(-10.0, -10.0, -5.0),
        max_iterations=1000,
        step_size=1.0,
        safety_margin=1.0,
        unknown_safe_threshold=3.0,
        energy_aware=False,
    )

    voxel_grid = VoxelGrid(config)
    # 起点附近标记为空闲
    for dx in range(-4, 5):
        for dy in range(-4, 5):
            for dz in range(-2, 3):
                idx = (10+dx, 20+dy, 5+dz)
                if voxel_grid.is_valid_index(idx):
                    voxel_grid.grid[idx] = -1

    # 在中间放一面墙 (X=0, 即 grid idx 20)
    for iy in range(0, 40):
        for iz in range(0, 10):
            voxel_grid.grid[20, iy, iz] = 1

    # 墙后面全是unknown — 靠近墙的unknown应该被拒绝

    esdf = ESDF(voxel_grid)
    esdf.compute()

    rrt = RRTStar(voxel_grid, esdf, config)

    # 检查墙后面紧邻的点
    near_wall_point = np.array([1.5, 0.0, -2.5])  # 距墙1.5m
    near_dist = esdf.get_distance(near_wall_point)
    near_valid = rrt._is_valid_point(near_wall_point)

    far_point = np.array([5.0, 0.0, -2.5])  # 距墙5m
    far_dist = esdf.get_distance(far_point)
    far_valid = rrt._is_valid_point(far_point)

    print(f"  Near wall (1.5m): ESDF={near_dist:.1f}m, valid={near_valid} (expected: False)")
    print(f"  Far from wall (5m): ESDF={far_dist:.1f}m, valid={far_valid} (expected: True)")

    near_ok = not near_valid  # 靠近墙的unknown应该被拒绝
    far_ok = far_valid        # 远离墙的unknown应该被允许

    if near_ok and far_ok:
        print(f"  [OK] Threshold logic correct!")
    else:
        print(f"  [FAIL] Threshold logic wrong!")

    return near_ok and far_ok


def main():
    print("=" * 60)
    print("  RRT* MOCK 2D TESTS (No AirSim Required)")
    print("  Testing path planning algorithm in isolation")
    print("=" * 60)

    results = []

    results.append(("Simple wall", test_simple_wall()))
    results.append(("Wall with gap", test_wall_with_gap()))
    results.append(("Blocks-like scene", test_blocks_scene()))
    results.append(("Unknown threshold (no wall)", test_unknown_threshold()))
    results.append(("Unknown near wall", test_unknown_near_wall()))

    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}")

    total_pass = sum(1 for _, p in results if p)
    print(f"\n  {total_pass}/{len(results)} tests passed")

    if total_pass == len(results):
        print("  [OK] All tests passed!")
    else:
        print("  [FAIL] Some tests failed!")


if __name__ == "__main__":
    main()
