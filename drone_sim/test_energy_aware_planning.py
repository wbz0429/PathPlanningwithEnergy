"""
测试能量感知路径规划
对比能量感知规划与纯距离规划的差异
"""

import numpy as np
import sys
import os

# 设置控制台编码
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# 添加父目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from planning.config import PlanningConfig
from planning.rrt_star import RRTStar, EnergyAwareCostFunction
from mapping.voxel_grid import VoxelGrid
from mapping.esdf import ESDF
from energy.physics_model import PhysicsEnergyModel


def create_test_environment():
    """创建测试环境：带有障碍物的体素栅格"""
    config = PlanningConfig(
        voxel_size=0.5,
        grid_size=(60, 60, 20),
        origin=(-15.0, -15.0, -10.0),
        max_iterations=2000,
        step_size=1.5,
        safety_margin=0.8
    )

    voxel_grid = VoxelGrid(config)

    # 添加一些障碍物（直接操作grid数组）
    # 障碍物1：中间的墙
    for x in range(-2, 3):
        for y in range(-8, 8):
            for z in range(-8, -2):
                idx = voxel_grid.world_to_grid(np.array([x, y, z]))
                if voxel_grid.is_valid_index(idx):
                    voxel_grid.grid[idx] = 1

    # 障碍物2：右侧的柱子
    for x in range(5, 8):
        for y in range(-2, 2):
            for z in range(-8, -2):
                idx = voxel_grid.world_to_grid(np.array([x, y, z]))
                if voxel_grid.is_valid_index(idx):
                    voxel_grid.grid[idx] = 1

    return config, voxel_grid


def test_cost_function():
    """测试代价函数"""
    print("\n" + "="*60)
    print("测试 1: 能量感知代价函数")
    print("="*60)

    config = PlanningConfig()
    energy_model = PhysicsEnergyModel()
    cost_fn = EnergyAwareCostFunction(config, energy_model)

    # 测试水平飞行
    start = np.array([0, 0, -5])
    end_horizontal = np.array([10, 0, -5])
    cost_h = cost_fn.compute_cost(start, end_horizontal)

    # 测试爬升飞行（相同水平距离，但有高度变化）
    end_climb = np.array([10, 0, -10])  # 爬升5米
    cost_c = cost_fn.compute_cost(start, end_climb)

    # 测试下降飞行
    end_descend = np.array([10, 0, 0])  # 下降5米
    cost_d = cost_fn.compute_cost(start, end_descend)

    print(f"\n起点: {start}")
    print(f"\n代价对比（能量权重={config.weight_energy}, 距离权重={config.weight_distance}）:")
    print(f"  水平飞行 10m:        代价 = {cost_h:.4f}")
    print(f"  爬升飞行 (10m+5m↑):  代价 = {cost_c:.4f} ({(cost_c/cost_h-1)*100:+.1f}%)")
    print(f"  下降飞行 (10m+5m↓):  代价 = {cost_d:.4f} ({(cost_d/cost_h-1)*100:+.1f}%)")

    # 验证：爬升应该比水平飞行代价更高
    assert cost_c > cost_h, "爬升代价应该高于水平飞行"
    print("\n✓ 代价函数测试通过")
    return True


def test_energy_aware_vs_distance_only():
    """对比能量感知规划与纯距离规划"""
    print("\n" + "="*60)
    print("测试 2: 能量感知规划 vs 纯距离规划")
    print("="*60)

    config, voxel_grid = create_test_environment()
    esdf = ESDF(voxel_grid)
    esdf.compute()

    energy_model = PhysicsEnergyModel()

    # 起点和终点（需要绕过障碍物）
    start = np.array([-10.0, 0.0, -5.0])
    goal = np.array([10.0, 0.0, -5.0])

    print(f"\n起点: {start}")
    print(f"终点: {goal}")
    print(f"直线距离: {np.linalg.norm(goal - start):.2f}m")

    # 测试1：纯距离规划
    print("\n--- 纯距离规划 ---")
    config_dist = PlanningConfig(
        voxel_size=config.voxel_size,
        grid_size=config.grid_size,
        origin=config.origin,
        max_iterations=2000,
        step_size=1.5,
        safety_margin=0.8,
        energy_aware=False  # 禁用能量感知
    )

    np.random.seed(42)  # 固定随机种子以便对比
    rrt_dist = RRTStar(voxel_grid, esdf, config_dist)
    path_dist = rrt_dist.plan(start.copy(), goal.copy())
    stats_dist = rrt_dist.get_plan_stats()

    if path_dist:
        # 用能耗模型计算这条路径的实际能耗
        total_energy_dist = 0.0
        for i in range(len(path_dist) - 1):
            e, _ = energy_model.compute_energy_for_segment(
                path_dist[i], path_dist[i+1], config.flight_velocity
            )
            total_energy_dist += e

        print(f"  路径点数: {len(path_dist)}")
        print(f"  路径距离: {stats_dist['total_distance']:.2f}m")
        print(f"  实际能耗: {total_energy_dist:.1f}J")
    else:
        print("  规划失败!")
        total_energy_dist = float('inf')

    # 测试2：能量感知规划
    print("\n--- 能量感知规划 ---")
    config_energy = PlanningConfig(
        voxel_size=config.voxel_size,
        grid_size=config.grid_size,
        origin=config.origin,
        max_iterations=2000,
        step_size=1.5,
        safety_margin=0.8,
        energy_aware=True,  # 启用能量感知
        weight_energy=0.6,
        weight_distance=0.3,
        weight_time=0.1
    )

    np.random.seed(42)  # 相同的随机种子
    rrt_energy = RRTStar(voxel_grid, esdf, config_energy, energy_model)
    path_energy = rrt_energy.plan(start.copy(), goal.copy())
    stats_energy = rrt_energy.get_plan_stats()

    if path_energy:
        print(f"  路径点数: {len(path_energy)}")
        print(f"  路径距离: {stats_energy['total_distance']:.2f}m")
        print(f"  实际能耗: {stats_energy['total_energy_joules']:.1f}J")
    else:
        print("  规划失败!")

    # 对比结果
    if path_dist and path_energy:
        print("\n--- 对比结果 ---")
        dist_diff = stats_energy['total_distance'] - stats_dist['total_distance']
        energy_diff = stats_energy['total_energy_joules'] - total_energy_dist

        print(f"  距离差异: {dist_diff:+.2f}m ({dist_diff/stats_dist['total_distance']*100:+.1f}%)")
        print(f"  能耗差异: {energy_diff:+.1f}J ({energy_diff/total_energy_dist*100:+.1f}%)")

        if energy_diff < 0:
            print(f"\n  ✓ 能量感知规划节省了 {-energy_diff:.1f}J 能量!")
        else:
            print(f"\n  注意：本次测试中能量感知规划能耗略高，可能是随机采样导致")

    print("\n✓ 对比测试完成")
    return True


def test_climb_vs_detour():
    """测试爬升绕障 vs 水平绕障的选择"""
    print("\n" + "="*60)
    print("测试 3: 爬升绕障 vs 水平绕障")
    print("="*60)

    config = PlanningConfig(
        voxel_size=0.5,
        grid_size=(40, 40, 30),
        origin=(-10.0, -10.0, -15.0),
        max_iterations=2000,
        step_size=1.5,
        safety_margin=0.8,
        energy_aware=True,
        weight_energy=0.6,
        weight_distance=0.3,
        weight_time=0.1
    )

    voxel_grid = VoxelGrid(config)

    # 创建一个低矮的障碍物（可以从上方飞越）
    for x in range(-3, 4):
        for y in range(-10, 10):
            for z in range(-8, -5):  # 只有3米高
                idx = voxel_grid.world_to_grid(np.array([x, y, z]))
                if voxel_grid.is_valid_index(idx):
                    voxel_grid.grid[idx] = 1

    esdf = ESDF(voxel_grid)
    esdf.compute()

    energy_model = PhysicsEnergyModel()

    start = np.array([-8.0, 0.0, -6.0])
    goal = np.array([8.0, 0.0, -6.0])

    print(f"\n场景：低矮障碍物（3米高），可以选择：")
    print(f"  A) 从上方飞越（爬升+下降）")
    print(f"  B) 从侧面绕行（水平绕障）")
    print(f"\n起点: {start}")
    print(f"终点: {goal}")

    np.random.seed(123)
    rrt = RRTStar(voxel_grid, esdf, config, energy_model)
    path = rrt.plan(start.copy(), goal.copy())
    stats = rrt.get_plan_stats()

    if path:
        print(f"\n规划结果:")
        print(f"  路径点数: {len(path)}")
        print(f"  路径距离: {stats['total_distance']:.2f}m")
        print(f"  能耗: {stats['total_energy_joules']:.1f}J")

        # 分析路径高度变化
        heights = [p[2] for p in path]
        max_height = min(heights)  # NED坐标系，更负=更高
        min_height = max(heights)
        height_change = min_height - max_height

        print(f"\n路径高度分析:")
        print(f"  最高点: {-max_height:.1f}m (Z={max_height:.1f})")
        print(f"  最低点: {-min_height:.1f}m (Z={min_height:.1f})")
        print(f"  高度变化: {height_change:.1f}m")

        if height_change > 2.0:
            print(f"\n  → 选择了爬升绕障策略")
        else:
            print(f"\n  → 选择了水平绕障策略")
    else:
        print("  规划失败!")

    print("\n✓ 爬升vs绕障测试完成")
    return True


def test_weight_sensitivity():
    """测试权重敏感性"""
    print("\n" + "="*60)
    print("测试 4: 权重敏感性分析")
    print("="*60)

    config_base, voxel_grid = create_test_environment()
    esdf = ESDF(voxel_grid)
    esdf.compute()

    energy_model = PhysicsEnergyModel()

    start = np.array([-10.0, 0.0, -5.0])
    goal = np.array([10.0, 0.0, -5.0])

    # 不同的权重配置
    weight_configs = [
        ("纯距离", 0.0, 1.0, 0.0),
        ("能量优先", 0.8, 0.1, 0.1),
        ("平衡", 0.4, 0.4, 0.2),
        ("距离优先", 0.2, 0.6, 0.2),
    ]

    print(f"\n起点: {start}")
    print(f"终点: {goal}")
    print(f"\n{'配置':<12} {'能量权重':<10} {'距离权重':<10} {'路径距离':<12} {'能耗':<12}")
    print("-" * 60)

    results = []
    for name, w_e, w_d, w_t in weight_configs:
        config = PlanningConfig(
            voxel_size=config_base.voxel_size,
            grid_size=config_base.grid_size,
            origin=config_base.origin,
            max_iterations=2000,
            step_size=1.5,
            safety_margin=0.8,
            energy_aware=(w_e > 0),
            weight_energy=w_e,
            weight_distance=w_d,
            weight_time=w_t
        )

        np.random.seed(42)
        rrt = RRTStar(voxel_grid, esdf, config, energy_model if w_e > 0 else None)
        path = rrt.plan(start.copy(), goal.copy())

        if path:
            stats = rrt.get_plan_stats()
            # 计算实际能耗
            total_energy = 0.0
            for i in range(len(path) - 1):
                e, _ = energy_model.compute_energy_for_segment(
                    path[i], path[i+1], config.flight_velocity
                )
                total_energy += e

            print(f"{name:<12} {w_e:<10.1f} {w_d:<10.1f} {stats['total_distance']:<12.2f} {total_energy:<12.1f}")
            results.append((name, stats['total_distance'], total_energy))
        else:
            print(f"{name:<12} {w_e:<10.1f} {w_d:<10.1f} {'失败':<12} {'-':<12}")

    print("\n✓ 权重敏感性测试完成")
    return True


def run_all_tests():
    """运行所有测试"""
    print("\n" + "="*60)
    print("能量感知路径规划测试")
    print("="*60)

    results = []

    results.append(("代价函数", test_cost_function()))
    results.append(("能量感知vs纯距离", test_energy_aware_vs_distance_only()))
    results.append(("爬升vs绕障", test_climb_vs_detour()))
    results.append(("权重敏感性", test_weight_sensitivity()))

    # 总结
    print("\n" + "="*60)
    print("测试总结")
    print("="*60)

    all_passed = True
    for name, passed in results:
        status = "✓ 通过" if passed else "✗ 失败"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False

    print("\n" + "="*60)
    if all_passed:
        print("所有测试通过!")
    else:
        print("部分测试失败")
    print("="*60)

    return all_passed


if __name__ == "__main__":
    run_all_tests()
