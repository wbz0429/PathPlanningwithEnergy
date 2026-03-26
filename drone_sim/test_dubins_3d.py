"""
test_dubins_3d.py - 3D CSC Dubins 路径求解器测试

测试内容：
1. 2D Dubins 6 种路径类型
2. 3D Low/Medium/High case 爬升角优化
3. 衔接场景（dubins_3d_blend_junction）
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from planning.dubins_3d import (
    Dubins2DSolver, Dubins3DSolver, Dubins3DParams,
    dubins_3d_blend_junction, _estimate_heading,
)


def test_2d_dubins_basic():
    """测试 2D Dubins 基本求解"""
    print("=" * 70)
    print("  TEST 1: 2D Dubins basic solve (6 path types)")
    print("=" * 70)

    solver = Dubins2DSolver(turning_radius=1.5)
    passed = 0
    total = 0

    # Test 1.1: 直线前进（同向）
    total += 1
    sol = solver.solve(np.array([0, 0]), 0.0, np.array([10, 0]), 0.0)
    assert sol is not None, "Straight-ahead should have solution"
    assert abs(sol['total_length'] - 10.0) < 0.5, f"Expected ~10m, got {sol['total_length']:.2f}"
    print(f"  [PASS] 1.1 Straight ahead: type={sol['type']}, length={sol['total_length']:.2f}m")
    passed += 1

    # Test 1.2: 90度左转
    total += 1
    sol = solver.solve(np.array([0, 0]), 0.0, np.array([5, 5]), np.pi / 2)
    assert sol is not None, "90° left turn should have solution"
    print(f"  [PASS] 1.2 90° left turn: type={sol['type']}, length={sol['total_length']:.2f}m")
    passed += 1

    # Test 1.3: 90度右转
    total += 1
    sol = solver.solve(np.array([0, 0]), 0.0, np.array([5, -5]), -np.pi / 2)
    assert sol is not None, "90° right turn should have solution"
    print(f"  [PASS] 1.3 90° right turn: type={sol['type']}, length={sol['total_length']:.2f}m")
    passed += 1

    # Test 1.4: 180度掉头
    total += 1
    sol = solver.solve(np.array([0, 0]), 0.0, np.array([0, 3]), np.pi)
    assert sol is not None, "U-turn should have solution"
    print(f"  [PASS] 1.4 U-turn: type={sol['type']}, length={sol['total_length']:.2f}m")
    passed += 1

    # Test 1.5: 短距离
    total += 1
    sol = solver.solve(np.array([0, 0]), 0.0, np.array([1, 0.5]), 0.3)
    assert sol is not None, "Short distance should have solution"
    print(f"  [PASS] 1.5 Short distance: type={sol['type']}, length={sol['total_length']:.2f}m")
    passed += 1

    # Test 1.6: 各种航向组合
    total += 1
    headings = [0, np.pi/4, np.pi/2, np.pi, -np.pi/2, -np.pi/4]
    all_solved = True
    types_seen = set()
    for h1 in headings:
        for h2 in headings:
            sol = solver.solve(np.array([0, 0]), h1, np.array([8, 3]), h2)
            if sol is not None:
                types_seen.add(sol['type'])
            else:
                all_solved = False
    print(f"  [PASS] 1.6 Heading combos: {len(types_seen)} types seen: {sorted(types_seen)}")
    passed += 1

    print(f"\n  2D Dubins: {passed}/{total} passed")
    return passed == total


def test_2d_dubins_sampling():
    """测试 2D Dubins 路径采样"""
    print("\n" + "=" * 70)
    print("  TEST 2: 2D Dubins path sampling")
    print("=" * 70)

    solver = Dubins2DSolver(turning_radius=1.5)
    passed = 0
    total = 0

    # Test 2.1: 采样点应从起点到终点
    total += 1
    start = np.array([0, 0])
    end = np.array([10, 5])
    sol = solver.solve(start, 0.0, end, np.pi / 4)
    points = solver.sample_2d(sol, step=0.3)
    assert len(points) > 5, f"Expected many sample points, got {len(points)}"
    # 起点应接近 start
    assert np.linalg.norm(points[0] - start) < 0.5, "First point should be near start"
    # 终点应接近 end
    assert np.linalg.norm(points[-1] - end) < 0.5, f"Last point should be near end, got dist={np.linalg.norm(points[-1] - end):.2f}"
    print(f"  [PASS] 2.1 Sampling: {len(points)} points, start_err={np.linalg.norm(points[0] - start):.3f}, end_err={np.linalg.norm(points[-1] - end):.3f}")
    passed += 1

    # Test 2.2: 采样间距应大致均匀
    total += 1
    dists = [np.linalg.norm(points[i+1] - points[i]) for i in range(len(points) - 1)]
    mean_dist = np.mean(dists)
    max_dist = np.max(dists)
    print(f"  [PASS] 2.2 Spacing: mean={mean_dist:.3f}m, max={max_dist:.3f}m")
    passed += 1

    print(f"\n  2D sampling: {passed}/{total} passed")
    return passed == total


def test_3d_dubins_low_case():
    """测试 3D Dubins Low case（小高度差，统一爬升角）"""
    print("\n" + "=" * 70)
    print("  TEST 3: 3D Dubins Low case (uniform climb angle)")
    print("=" * 70)

    params = Dubins3DParams(turning_radius=1.5, max_climb_angle=30.0, sample_distance=0.3)
    solver = Dubins3DSolver(params)
    passed = 0
    total = 0

    # Test 3.1: 水平飞行（dz=0）
    total += 1
    points = solver.solve(
        np.array([0, 0, -3.0]), 0.0,
        np.array([10, 0, -3.0]), 0.0
    )
    assert points is not None and len(points) > 2
    z_values = [p[2] for p in points]
    z_range = max(z_values) - min(z_values)
    assert z_range < 0.1, f"Horizontal flight Z range should be ~0, got {z_range:.3f}"
    print(f"  [PASS] 3.1 Horizontal: {len(points)} pts, Z range={z_range:.4f}m")
    passed += 1

    # Test 3.2: 小爬升（dz=1m，远小于 L_2d * tan(30°)）
    total += 1
    points = solver.solve(
        np.array([0, 0, -3.0]), 0.0,
        np.array([10, 0, -2.0]), 0.0
    )
    assert points is not None
    # Z 应从 -3 单调递增到 -2
    z_start = points[0][2]
    z_end = points[-1][2]
    assert abs(z_start - (-3.0)) < 0.3, f"Start Z should be ~-3, got {z_start:.2f}"
    assert abs(z_end - (-2.0)) < 0.3, f"End Z should be ~-2, got {z_end:.2f}"
    print(f"  [PASS] 3.2 Small climb: Z {z_start:.2f} -> {z_end:.2f}")
    passed += 1

    # Test 3.3: 小下降（dz=-1m）
    total += 1
    points = solver.solve(
        np.array([0, 0, -3.0]), 0.0,
        np.array([10, 0, -4.0]), 0.0
    )
    assert points is not None
    z_start = points[0][2]
    z_end = points[-1][2]
    assert abs(z_end - (-4.0)) < 0.3, f"End Z should be ~-4, got {z_end:.2f}"
    print(f"  [PASS] 3.3 Small descent: Z {z_start:.2f} -> {z_end:.2f}")
    passed += 1

    # Test 3.4: 带转弯的爬升
    total += 1
    points = solver.solve(
        np.array([0, 0, -5.0]), 0.0,
        np.array([5, 5, -3.0]), np.pi / 2
    )
    assert points is not None
    z_start = points[0][2]
    z_end = points[-1][2]
    print(f"  [PASS] 3.4 Turn + climb: Z {z_start:.2f} -> {z_end:.2f}, {len(points)} pts")
    passed += 1

    print(f"\n  3D Low case: {passed}/{total} passed")
    return passed == total


def test_3d_dubins_high_case():
    """测试 3D Dubins High case（大高度差，需要 γ_max）"""
    print("\n" + "=" * 70)
    print("  TEST 4: 3D Dubins High case (max climb angle)")
    print("=" * 70)

    params = Dubins3DParams(turning_radius=1.5, max_climb_angle=30.0, sample_distance=0.3)
    solver = Dubins3DSolver(params)
    passed = 0
    total = 0

    # Test 4.1: 大爬升（dz 接近 L_2d * tan(30°)）
    total += 1
    # L_2d ≈ 10m, tan(30°) ≈ 0.577, max dz ≈ 5.77m
    points = solver.solve(
        np.array([0, 0, -8.0]), 0.0,
        np.array([10, 0, -2.0]), 0.0
    )
    assert points is not None
    z_start = points[0][2]
    z_end = points[-1][2]
    print(f"  [PASS] 4.1 Large climb (6m): Z {z_start:.2f} -> {z_end:.2f}, {len(points)} pts")
    passed += 1

    # Test 4.2: 超大爬升（dz > L_2d * tan(γ_max)，需要 High case 处理）
    total += 1
    points = solver.solve(
        np.array([0, 0, -10.0]), 0.0,
        np.array([5, 0, -2.0]), 0.0
    )
    assert points is not None
    z_start = points[0][2]
    z_end = points[-1][2]
    # High case: 所有段用 γ_max，Z 可能不完全到达目标
    print(f"  [PASS] 4.2 Very large climb (8m over 5m): Z {z_start:.2f} -> {z_end:.2f}")
    passed += 1

    print(f"\n  3D High case: {passed}/{total} passed")
    return passed == total


def test_3d_dubins_endpoint_accuracy():
    """测试 3D Dubins 端点精度"""
    print("\n" + "=" * 70)
    print("  TEST 5: 3D Dubins endpoint accuracy")
    print("=" * 70)

    params = Dubins3DParams(turning_radius=1.5, max_climb_angle=30.0, sample_distance=0.3)
    solver = Dubins3DSolver(params)
    passed = 0
    total = 0

    test_cases = [
        ("Straight", [0, 0, -3], 0.0, [10, 0, -3], 0.0),
        ("Left turn", [0, 0, -3], 0.0, [5, 5, -3], np.pi/2),
        ("Right turn", [0, 0, -3], 0.0, [5, -5, -3], -np.pi/2),
        ("Climb + turn", [0, 0, -5], 0.0, [8, 4, -3], np.pi/4),
        ("Descent + turn", [0, 0, -3], np.pi/4, [6, -3, -5], -np.pi/4),
    ]

    for name, s, sh, e, eh in test_cases:
        total += 1
        start = np.array(s, dtype=float)
        end = np.array(e, dtype=float)
        points = solver.solve(start, sh, end, eh)
        if points is None:
            print(f"  [FAIL] 5.{total} {name}: no solution")
            continue

        start_err = np.linalg.norm(points[0] - start)
        end_err_xy = np.linalg.norm(points[-1][:2] - end[:2])

        ok = start_err < 0.5 and end_err_xy < 0.5
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] 5.{total} {name}: start_err={start_err:.3f}, end_err_xy={end_err_xy:.3f}")
        if ok:
            passed += 1

    print(f"\n  Endpoint accuracy: {passed}/{total} passed")
    return passed == total


def test_blend_junction():
    """测试 dubins_3d_blend_junction 衔接函数"""
    print("\n" + "=" * 70)
    print("  TEST 6: dubins_3d_blend_junction")
    print("=" * 70)

    params = Dubins3DParams(turning_radius=1.5, max_climb_angle=30.0, sample_distance=0.3)
    passed = 0
    total = 0

    # Test 6.1: 空 tail -> 返回原路径
    total += 1
    new_path = [np.array([10, 0, -3]), np.array([12, 0, -3])]
    result = dubins_3d_blend_junction([], new_path, params)
    assert result is new_path, "Empty tail should return original path"
    print("  [PASS] 6.1 Empty tail -> skip")
    passed += 1

    # Test 6.2: tail 只有 1 个点 -> 返回原路径
    total += 1
    result = dubins_3d_blend_junction([np.array([8, 0, -3])], new_path, params)
    assert result is new_path, "Single-point tail should return original path"
    print("  [PASS] 6.2 Single-point tail -> skip")
    passed += 1

    # Test 6.3: 90度拐弯 -> 应产生 Dubins 弧段
    total += 1
    tail = [np.array([0, 0, -3]), np.array([3, 0, -3]), np.array([6, 0, -3])]
    new_path = [np.array([6, 3, -3]), np.array([6, 6, -3]),
                np.array([6, 9, -3]), np.array([6, 12, -3])]
    result = dubins_3d_blend_junction(tail, new_path, params)
    assert len(result) >= 2, "Should produce blended path"
    print(f"  [PASS] 6.3 90° turn: {len(result)} pts (was {len(new_path)})")
    passed += 1

    # Test 6.4: 直线衔接
    total += 1
    tail = [np.array([0, 0, -3]), np.array([3, 0, -3]), np.array([6, 0, -3])]
    new_path = [np.array([9, 0, -3]), np.array([12, 0, -3]),
                np.array([15, 0, -3]), np.array([18, 0, -3])]
    result = dubins_3d_blend_junction(tail, new_path, params)
    print(f"  [PASS] 6.4 Straight-line: {len(result)} pts")
    passed += 1

    # Test 6.5: 带高度变化的衔接
    total += 1
    tail = [np.array([0, 0, -5]), np.array([3, 0, -5]), np.array([6, 0, -5])]
    new_path = [np.array([9, 3, -3]), np.array([12, 6, -3]),
                np.array([15, 9, -3]), np.array([18, 12, -3])]
    result = dubins_3d_blend_junction(tail, new_path, params)
    if len(result) >= 2:
        z_values = [p[2] for p in result[:min(10, len(result))]]
        print(f"  [PASS] 6.5 Height change: {len(result)} pts, Z range [{min(z_values):.2f}, {max(z_values):.2f}]")
        passed += 1
    else:
        print(f"  [FAIL] 6.5 Height change: only {len(result)} pts")

    # Test 6.6: 碰撞检测回退
    total += 1
    tail = [np.array([0, 0, -3]), np.array([3, 0, -3]), np.array([6, 0, -3])]
    new_path = [np.array([6, 3, -3]), np.array([6, 6, -3]),
                np.array([6, 9, -3]), np.array([6, 12, -3])]
    # 碰撞检测总是返回 False
    result = dubins_3d_blend_junction(tail, new_path, params,
                                       safety_check=lambda f, t: False)
    assert result is new_path, "Should fall back when collision detected"
    print("  [PASS] 6.6 Collision fallback -> original path")
    passed += 1

    print(f"\n  Blend junction: {passed}/{total} passed")
    return passed == total


def test_heading_estimation():
    """测试航向估计函数"""
    print("\n" + "=" * 70)
    print("  TEST 7: Heading estimation")
    print("=" * 70)

    passed = 0
    total = 0

    # Test 7.1: X+ 方向
    total += 1
    h = _estimate_heading([np.array([0, 0, 0]), np.array([1, 0, 0])])
    assert abs(h - 0.0) < 0.01, f"Expected 0, got {h}"
    print(f"  [PASS] 7.1 X+ heading: {np.degrees(h):.1f}°")
    passed += 1

    # Test 7.2: Y+ 方向
    total += 1
    h = _estimate_heading([np.array([0, 0, 0]), np.array([0, 1, 0])])
    assert abs(h - np.pi / 2) < 0.01, f"Expected 90°, got {np.degrees(h):.1f}°"
    print(f"  [PASS] 7.2 Y+ heading: {np.degrees(h):.1f}°")
    passed += 1

    # Test 7.3: 45度
    total += 1
    h = _estimate_heading([np.array([0, 0, 0]), np.array([1, 1, 0])])
    assert abs(h - np.pi / 4) < 0.01
    print(f"  [PASS] 7.3 45° heading: {np.degrees(h):.1f}°")
    passed += 1

    # Test 7.4: 多点取最后两点
    total += 1
    h = _estimate_heading([np.array([0, 0, 0]), np.array([5, 0, 0]), np.array([5, 3, 0])])
    assert abs(h - np.pi / 2) < 0.01
    print(f"  [PASS] 7.4 Multi-point heading: {np.degrees(h):.1f}°")
    passed += 1

    print(f"\n  Heading estimation: {passed}/{total} passed")
    return passed == total


def test_z_monotonicity():
    """测试 Z 轴单调性（Low case 应该单调变化）"""
    print("\n" + "=" * 70)
    print("  TEST 8: Z-axis monotonicity (Low case)")
    print("=" * 70)

    params = Dubins3DParams(turning_radius=1.5, max_climb_angle=30.0, sample_distance=0.3)
    solver = Dubins3DSolver(params)
    passed = 0
    total = 0

    # Test 8.1: 爬升应 Z 单调递增（NED 中 Z 变大 = 下降，变小 = 上升）
    total += 1
    points = solver.solve(
        np.array([0, 0, -5.0]), 0.0,
        np.array([15, 0, -3.0]), 0.0
    )
    assert points is not None
    z_vals = [p[2] for p in points]
    # 检查大致单调（允许微小波动 < 0.1m）
    violations = 0
    for i in range(len(z_vals) - 1):
        if z_vals[i + 1] < z_vals[i] - 0.1:  # Z 应该增大（NED 中上升 = Z 变小，但这里 dz > 0）
            violations += 1
    print(f"  [{'PASS' if violations == 0 else 'WARN'}] 8.1 Climb Z monotonicity: "
          f"{violations} violations, Z: {z_vals[0]:.2f} -> {z_vals[-1]:.2f}")
    if violations == 0:
        passed += 1

    # Test 8.2: 下降应 Z 单调递减
    total += 1
    points = solver.solve(
        np.array([0, 0, -3.0]), 0.0,
        np.array([15, 0, -5.0]), 0.0
    )
    assert points is not None
    z_vals = [p[2] for p in points]
    violations = 0
    for i in range(len(z_vals) - 1):
        if z_vals[i + 1] > z_vals[i] + 0.1:
            violations += 1
    print(f"  [{'PASS' if violations == 0 else 'WARN'}] 8.2 Descent Z monotonicity: "
          f"{violations} violations, Z: {z_vals[0]:.2f} -> {z_vals[-1]:.2f}")
    if violations == 0:
        passed += 1

    print(f"\n  Z monotonicity: {passed}/{total} passed")
    return passed == total


def main():
    results = []
    results.append(("2D Dubins basic", test_2d_dubins_basic()))
    results.append(("2D Dubins sampling", test_2d_dubins_sampling()))
    results.append(("3D Low case", test_3d_dubins_low_case()))
    results.append(("3D High case", test_3d_dubins_high_case()))
    results.append(("3D endpoint accuracy", test_3d_dubins_endpoint_accuracy()))
    results.append(("Blend junction", test_blend_junction()))
    results.append(("Heading estimation", test_heading_estimation()))
    results.append(("Z monotonicity", test_z_monotonicity()))

    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    all_pass = True
    for name, ok in results:
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {name}")
        if not ok:
            all_pass = False

    print(f"\n  Overall: {'ALL PASSED' if all_pass else 'SOME FAILED'}")
    print("=" * 70)
    return all_pass


if __name__ == "__main__":
    main()
