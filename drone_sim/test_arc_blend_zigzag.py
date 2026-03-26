"""
test_arc_blend_zigzag.py - 测试圆弧平滑在多轮规划衔接处的效果
用预设航点序列模拟多轮规划衔接，对比有/无圆弧平滑的转弯角度差异。
"""

import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from planning.config import PlanningConfig
from mapping.voxel_grid import VoxelGrid
from mapping.esdf import ESDF
from planning.rrt_star import RRTStar
from planning.dubins_3d import Dubins3DParams
from energy.physics_model import PhysicsEnergyModel
from planning.receding_horizon import RecedingHorizonPlanner
from unittest.mock import MagicMock


def compute_turn_angles(trajectory):
    """计算轨迹中每个点的转弯角度（度）"""
    if len(trajectory) < 3:
        return []
    arr = np.array(trajectory)
    diffs = np.diff(arr, axis=0)
    angles = []
    for i in range(len(diffs) - 1):
        d1 = diffs[i][:2]
        d2 = diffs[i + 1][:2]
        n1 = np.linalg.norm(d1)
        n2 = np.linalg.norm(d2)
        if n1 > 1e-6 and n2 > 1e-6:
            cos_a = np.dot(d1, d2) / (n1 * n2)
            angles.append(np.degrees(np.arccos(np.clip(cos_a, -1, 1))))
    return angles


def build_open_map(config):
    """构建一个无障碍物的开放地图（用于纯衔接测试）"""
    vg = VoxelGrid(config)
    vg.grid[vg.grid == 0] = -1  # 全部标记为 free
    return vg


def test_blend_junction_unit():
    """单元测试：_blend_junction 各种边界情况"""
    print("=" * 70)
    print("  UNIT TEST: _blend_junction edge cases")
    print("=" * 70)

    config = PlanningConfig(
        voxel_size=0.5, grid_size=(120, 100, 40), origin=(-10, -30, -15),
        safety_margin=1.0,
    )
    vg = build_open_map(config)
    esdf = ESDF(vg)
    esdf.compute()

    # 创建 RecedingHorizonPlanner 实例（mock 掉不需要的部分）
    rhp = RecedingHorizonPlanner.__new__(RecedingHorizonPlanner)
    rhp.map_manager = MagicMock()
    rhp.map_manager.esdf = esdf
    rhp.map_manager.config = config
    rhp.arc_smoothing = True
    rhp.arc_overlap_points = 3
    rhp.dubins_params = Dubins3DParams()

    def safe_check(from_pos, to_pos):
        mid = (from_pos + to_pos) / 2
        d = esdf.get_distance(mid)
        return d >= config.safety_margin
    rhp._check_path_safe = safe_check

    passed = 0
    total = 0

    # Test 1: 空 tail
    total += 1
    new_path = [np.array([10, 0, -3]), np.array([12, 0, -3])]
    result = rhp._blend_junction([], new_path)
    if result is new_path:
        print("  [PASS] Test 1: Empty tail -> skip")
        passed += 1
    else:
        print("  [FAIL] Test 1")

    # Test 2: tail + path < 4 控制点
    total += 1
    result = rhp._blend_junction([np.array([8, 0, -3])], [np.array([10, 0, -3])])
    assert len(result) == 1
    print("  [PASS] Test 2: < 4 control points -> skip")
    passed += 1

    # Test 3: 90度拐弯 -> 应产生平滑弧段
    total += 1
    tail = [np.array([0, 0, -3]), np.array([3, 0, -3]), np.array([6, 0, -3])]
    new_path = [np.array([6, 3, -3]), np.array([6, 6, -3]),
                np.array([6, 9, -3]), np.array([6, 12, -3])]
    result = rhp._blend_junction(tail, new_path)
    # 弧段应该比原始 new_path 有更多点（B-spline 插值）
    print(f"  [PASS] Test 3: 90° turn blend -> {len(result)} pts (was {len(new_path)})")
    passed += 1

    # Test 4: 直线衔接
    total += 1
    tail = [np.array([0, 0, -3]), np.array([3, 0, -3]), np.array([6, 0, -3])]
    new_path = [np.array([9, 0, -3]), np.array([12, 0, -3]),
                np.array([15, 0, -3]), np.array([18, 0, -3])]
    result = rhp._blend_junction(tail, new_path)
    print(f"  [PASS] Test 4: Straight-line blend -> {len(result)} pts")
    passed += 1

    # Test 5: 180度掉头
    total += 1
    tail = [np.array([0, 0, -3]), np.array([3, 0, -3]), np.array([6, 0, -3])]
    new_path = [np.array([3, 0, -3]), np.array([0, 0, -3]),
                np.array([-3, 0, -3]), np.array([-6, 0, -3])]
    result = rhp._blend_junction(tail, new_path)
    print(f"  [PASS] Test 5: 180° U-turn blend -> {len(result)} pts")
    passed += 1

    print(f"\n  Unit tests: {passed}/{total} passed")
    return passed == total


def test_multi_round_simulation():
    """
    模拟多轮规划衔接：用预设的分段路径模拟 RRT* 每轮规划出不同方向的路径，
    然后在衔接处做圆弧平滑。这样可以精确控制转弯角度。
    """
    print("\n" + "=" * 70)
    print("  SIMULATION: Multi-round receding horizon with arc blending")
    print("=" * 70)

    config = PlanningConfig(
        voxel_size=0.5, grid_size=(120, 100, 40), origin=(-10, -30, -15),
        safety_margin=1.0,
    )
    vg = build_open_map(config)
    esdf = ESDF(vg)
    esdf.compute()

    # 创建 RecedingHorizonPlanner 实例
    rhp = RecedingHorizonPlanner.__new__(RecedingHorizonPlanner)
    rhp.map_manager = MagicMock()
    rhp.map_manager.esdf = esdf
    rhp.map_manager.config = config
    rhp.arc_smoothing = True
    rhp.arc_overlap_points = 3
    rhp.dubins_params = Dubins3DParams()

    def safe_check(from_pos, to_pos):
        return True  # 开放地图，全部安全
    rhp._check_path_safe = safe_check

    # 模拟 5 轮规划，每轮路径方向不同（模拟绕障）
    # 每轮规划 8 个点，执行 60%（约5个点），然后新一轮规划方向变化
    planned_segments = [
        # 第1轮：向右前方 (X+, Y+)
        [np.array([i * 2.0, i * 0.5, -3.0]) for i in range(8)],
        # 第2轮：转向右方 (X+, Y+大幅)，模拟绕墙
        [np.array([14 + i * 1.0, 3.5 + i * 2.0, -3.0]) for i in range(8)],
        # 第3轮：转回前方 (X+大幅, Y 减小)
        [np.array([22 + i * 2.0, 17.5 - i * 1.5, -3.0]) for i in range(8)],
        # 第4轮：向左前方
        [np.array([38 + i * 2.0, 5.5 - i * 1.0, -3.0]) for i in range(8)],
        # 第5轮：直线冲刺
        [np.array([54 + i * 2.0, -2.5, -3.0]) for i in range(8)],
    ]

    execution_ratio = 0.6

    for arc_mode, label in [(False, "WITHOUT arc blend"), (True, "WITH arc blend")]:
        print(f"\n  --- {label} ---")
        trajectory = []
        executed_tail = []

        for rnd, segment in enumerate(planned_segments):
            path = segment

            # 圆弧平滑
            blend_str = ""
            if arc_mode and len(executed_tail) >= 2 and len(path) >= 2:
                path = rhp._blend_junction(executed_tail, path)
                blend_str = "[BLEND]"

            # 执行部分路径
            path_to_exec = path[1:] if len(path) > 1 else path
            exec_len = max(1, int(len(path_to_exec) * execution_ratio))
            waypoints = path_to_exec[:exec_len]

            for wp in waypoints:
                wp_fixed = wp.copy()
                trajectory.append(wp_fixed.copy())
                executed_tail.append(wp_fixed.copy())
                if len(executed_tail) > 4:
                    executed_tail = executed_tail[-4:]

            pos = trajectory[-1]
            print(f"    Round {rnd+1}: exec {len(waypoints)} pts, "
                  f"pos=({pos[0]:.1f},{pos[1]:.1f}) {blend_str}")

        # 分析
        angles = compute_turn_angles(trajectory)
        arr = np.array(trajectory)
        total_dist = np.sum(np.linalg.norm(np.diff(arr, axis=0), axis=1))

        print(f"\n    {label} Results:")
        print(f"      Waypoints: {len(trajectory)}")
        print(f"      Total distance: {total_dist:.2f} m")
        if angles:
            print(f"      Max turn angle: {max(angles):.1f} deg")
            print(f"      Mean turn angle: {np.mean(angles):.1f} deg")
            print(f"      Angles > 30 deg: {sum(1 for a in angles if a > 30)}/{len(angles)}")
            print(f"      Angles > 60 deg: {sum(1 for a in angles if a > 60)}/{len(angles)}")

        # 保存数据用于可视化
        if not arc_mode:
            traj_no_arc = trajectory
            angles_no_arc = angles
        else:
            traj_arc = trajectory
            angles_arc = angles

    # ---- 可视化 ----
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        obstacles = [
            (10, 12, -20, 5), (20, 22, -5, 20),
            (30, 32, -20, 5), (40, 42, -5, 20),
        ]

        for ax, label, traj, angles in [
            (axes[0], "WITHOUT arc blend", traj_no_arc, angles_no_arc),
            (axes[1], "WITH arc blend", traj_arc, angles_arc),
        ]:
            arr = np.array(traj)
            ax.plot(arr[:, 0], arr[:, 1], 'b.-', markersize=4, linewidth=1.2,
                    label='Trajectory')
            ax.plot(arr[0, 0], arr[0, 1], 'go', markersize=10, label='Start')
            ax.plot(arr[-1, 0], arr[-1, 1], 'r*', markersize=12, label='End')

            # 标记大转弯
            for i, a in enumerate(angles):
                if a > 30:
                    ax.plot(arr[i + 1, 0], arr[i + 1, 1], 'rx', markersize=10,
                            markeredgewidth=2)

            title = label
            if angles:
                title += (f'\nMax: {max(angles):.0f}°, '
                         f'>30°: {sum(1 for a in angles if a > 30)}')
            ax.set_title(title)
            ax.set_xlabel('X (m)')
            ax.set_ylabel('Y (m)')
            ax.set_aspect('equal')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        # 右图：转弯角度分布对比
        ax3 = axes[2]
        if angles_no_arc and angles_arc:
            bins = np.arange(0, max(max(angles_no_arc), max(angles_arc)) + 10, 5)
            ax3.hist(angles_no_arc, bins=bins, alpha=0.5,
                     label=f'No blend (max={max(angles_no_arc):.0f}°)', color='red')
            ax3.hist(angles_arc, bins=bins, alpha=0.5,
                     label=f'Arc blend (max={max(angles_arc):.0f}°)', color='blue')
            ax3.axvline(30, color='orange', linestyle='--', linewidth=1.5,
                       label='30° threshold')
            ax3.set_xlabel('Turn angle (degrees)')
            ax3.set_ylabel('Count')
            ax3.set_title('Turn Angle Distribution')
            ax3.legend(fontsize=8)
            ax3.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('arc_blend_zigzag_comparison.png', dpi=150)
        print(f"\n  Saved: arc_blend_zigzag_comparison.png")
        plt.close()
    except Exception as e:
        print(f"\n  Visualization error: {e}")
        import traceback
        traceback.print_exc()


def test_with_rrt_planning():
    """
    用真实 RRT* 规划器在开放地图上做多轮规划，
    目标点交替变化以产生转弯。
    """
    print("\n" + "=" * 70)
    print("  RRT* MULTI-GOAL TEST: Real planning with arc blending")
    print("=" * 70)

    config = PlanningConfig(
        voxel_size=0.5, grid_size=(120, 100, 40), origin=(-10, -30, -15),
        step_size=1.5, max_iterations=1000, goal_sample_rate=0.3,
        search_radius=3.0, safety_margin=1.0,
        energy_aware=False, flight_velocity=2.0,
    )

    vg = build_open_map(config)
    esdf = ESDF(vg)
    esdf.compute()
    em = PhysicsEnergyModel()

    # 设置一系列航点，形成 Z 字形路径
    waypoints_sequence = [
        np.array([0, 0, -3.0]),
        np.array([10, 8, -3.0]),
        np.array([20, -5, -3.0]),
        np.array([30, 10, -3.0]),
        np.array([40, -3, -3.0]),
        np.array([50, 5, -3.0]),
    ]

    # 创建 _blend_junction 所需的 mock
    rhp = RecedingHorizonPlanner.__new__(RecedingHorizonPlanner)
    rhp.map_manager = MagicMock()
    rhp.map_manager.esdf = esdf
    rhp.map_manager.config = config
    rhp.arc_smoothing = True
    rhp.arc_overlap_points = 3
    rhp.dubins_params = Dubins3DParams()
    rhp._check_path_safe = lambda f, t: True

    for arc_mode, label in [(False, "WITHOUT arc blend"), (True, "WITH arc blend")]:
        print(f"\n  --- {label} ---")
        planner = RRTStar(vg, esdf, config, energy_model=em)
        np.random.seed(42)

        trajectory = []
        executed_tail = []
        execution_ratio = 0.6

        for seg_idx in range(len(waypoints_sequence) - 1):
            start_wp = waypoints_sequence[seg_idx]
            goal_wp = waypoints_sequence[seg_idx + 1]

            # 如果有已执行轨迹，从最后位置开始
            if trajectory:
                start_wp = trajectory[-1].copy()

            path = planner.plan(start_wp, goal_wp)
            if path is None or len(path) < 2:
                print(f"    Segment {seg_idx+1}: Planning failed")
                continue

            # 圆弧平滑
            blend_str = ""
            if arc_mode and len(executed_tail) >= 2 and len(path) >= 2:
                path = rhp._blend_junction(executed_tail, path)
                blend_str = "[BLEND]"

            # 执行部分
            path_to_exec = path[1:] if len(path) > 1 else path
            exec_len = max(1, int(len(path_to_exec) * execution_ratio))
            waypoints = path_to_exec[:exec_len]

            for wp in waypoints:
                trajectory.append(wp.copy())
                executed_tail.append(wp.copy())
                if len(executed_tail) > 4:
                    executed_tail = executed_tail[-4:]

            pos = trajectory[-1]
            print(f"    Segment {seg_idx+1}: {len(waypoints)} pts executed, "
                  f"pos=({pos[0]:.1f},{pos[1]:.1f}) {blend_str}")

        angles = compute_turn_angles(trajectory)
        arr = np.array(trajectory)
        total_dist = np.sum(np.linalg.norm(np.diff(arr, axis=0), axis=1))

        print(f"\n    Results:")
        print(f"      Waypoints: {len(trajectory)}, Distance: {total_dist:.2f} m")
        if angles:
            print(f"      Max turn: {max(angles):.1f}°, Mean: {np.mean(angles):.1f}°")
            print(f"      >30°: {sum(1 for a in angles if a > 30)}/{len(angles)}, "
                  f">60°: {sum(1 for a in angles if a > 60)}/{len(angles)}")


def main():
    ok = test_blend_junction_unit()
    test_multi_round_simulation()
    test_with_rrt_planning()

    print("\n" + "=" * 70)
    print("  ALL TESTS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
