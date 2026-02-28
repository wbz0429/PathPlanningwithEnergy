"""
能量感知规划效果验证
对比不同场景下能量感知规划的优势
"""

import numpy as np
import sys
import os

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from planning.config import PlanningConfig
from planning.rrt_star import RRTStar, EnergyAwareCostFunction
from mapping.voxel_grid import VoxelGrid
from mapping.esdf import ESDF
from energy.physics_model import PhysicsEnergyModel


def analyze_path_energy(path, energy_model, velocity=2.0):
    """详细分析路径能耗"""
    if path is None or len(path) < 2:
        return None

    total_distance = 0.0
    total_energy = 0.0
    total_climb_energy = 0.0
    segments = []

    for i in range(len(path) - 1):
        start, end = path[i], path[i+1]
        dist = np.linalg.norm(end - start)
        energy, time = energy_model.compute_energy_for_segment(start, end, velocity)

        # 计算高度变化（NED坐标系，Z负为上）
        dz = end[2] - start[2]  # 负值表示爬升
        climb_power = energy_model.compute_climb_power(np.array([0, 0, dz/time])) if time > 0 else 0
        climb_energy = climb_power * time

        segments.append({
            'start': start,
            'end': end,
            'distance': dist,
            'dz': -dz,  # 转换为正值表示爬升
            'energy': energy,
            'climb_energy': climb_energy
        })

        total_distance += dist
        total_energy += energy
        total_climb_energy += abs(climb_energy)

    return {
        'total_distance': total_distance,
        'total_energy': total_energy,
        'total_climb_energy': total_climb_energy,
        'segments': segments,
        'path': path
    }


def test_climb_penalty():
    """测试爬升惩罚效果"""
    print("\n" + "="*70)
    print("测试: 爬升能耗惩罚效果")
    print("="*70)

    energy_model = PhysicsEnergyModel()

    # 对比两条等距离路径：水平 vs 爬升+下降
    print("\n场景：从A到B，两种路径选择")
    print("  路径1: 水平直飞 20m")
    print("  路径2: 爬升5m → 水平10m → 下降5m (总水平距离约18m)")

    # 路径1：水平直飞
    path1 = [
        np.array([0, 0, -5]),
        np.array([20, 0, -5])
    ]

    # 路径2：爬升-平飞-下降（总距离相近）
    path2 = [
        np.array([0, 0, -5]),
        np.array([5, 0, -10]),   # 爬升5m
        np.array([15, 0, -10]),  # 水平10m
        np.array([20, 0, -5])    # 下降5m
    ]

    analysis1 = analyze_path_energy(path1, energy_model)
    analysis2 = analyze_path_energy(path2, energy_model)

    print(f"\n路径1 (水平直飞):")
    print(f"  距离: {analysis1['total_distance']:.2f}m")
    print(f"  能耗: {analysis1['total_energy']:.1f}J")

    print(f"\n路径2 (爬升-下降):")
    print(f"  距离: {analysis2['total_distance']:.2f}m")
    print(f"  能耗: {analysis2['total_energy']:.1f}J")
    print(f"  爬升相关能耗: {analysis2['total_climb_energy']:.1f}J")

    energy_diff = analysis2['total_energy'] - analysis1['total_energy']
    print(f"\n能耗差异: {energy_diff:+.1f}J ({energy_diff/analysis1['total_energy']*100:+.1f}%)")

    if energy_diff > 0:
        print("✓ 爬升路径能耗更高，能量感知规划应该倾向选择水平路径")

    return True


def test_obstacle_scenarios():
    """测试不同障碍物场景"""
    print("\n" + "="*70)
    print("测试: 不同障碍物场景下的路径选择")
    print("="*70)

    energy_model = PhysicsEnergyModel()

    scenarios = []

    # 场景1：宽矮障碍物（适合从上方飞越）
    print("\n--- 场景1: 宽矮障碍物 (宽10m, 高2m) ---")
    config1 = PlanningConfig(
        voxel_size=0.5,
        grid_size=(60, 60, 30),
        origin=(-15.0, -15.0, -15.0),
        max_iterations=3000,
        step_size=1.0,
        safety_margin=0.8,
        energy_aware=True,
        weight_energy=0.6,
        weight_distance=0.3,
        weight_time=0.1
    )
    voxel1 = VoxelGrid(config1)
    # 宽矮障碍物
    for x in range(-5, 6):
        for y in range(-15, 15):
            for z in range(-7, -5):  # 只有2米高
                idx = voxel1.world_to_grid(np.array([float(x), float(y), float(z)]))
                if voxel1.is_valid_index(idx):
                    voxel1.grid[idx] = 1

    esdf1 = ESDF(voxel1)
    esdf1.compute()

    start1 = np.array([-10.0, 0.0, -6.0])
    goal1 = np.array([10.0, 0.0, -6.0])

    # 能量感知规划
    rrt1_energy = RRTStar(voxel1, esdf1, config1, energy_model)
    np.random.seed(100)
    path1_energy = rrt1_energy.plan(start1.copy(), goal1.copy())

    # 纯距离规划
    config1_dist = PlanningConfig(
        voxel_size=0.5,
        grid_size=(60, 60, 30),
        origin=(-15.0, -15.0, -15.0),
        max_iterations=3000,
        step_size=1.0,
        safety_margin=0.8,
        energy_aware=False
    )
    rrt1_dist = RRTStar(voxel1, esdf1, config1_dist)
    np.random.seed(100)
    path1_dist = rrt1_dist.plan(start1.copy(), goal1.copy())

    if path1_energy and path1_dist:
        analysis_energy = analyze_path_energy(path1_energy, energy_model)
        analysis_dist = analyze_path_energy(path1_dist, energy_model)

        print(f"  能量感知: 距离={analysis_energy['total_distance']:.1f}m, 能耗={analysis_energy['total_energy']:.0f}J")
        print(f"  纯距离:   距离={analysis_dist['total_distance']:.1f}m, 能耗={analysis_dist['total_energy']:.0f}J")

        scenarios.append({
            'name': '宽矮障碍物',
            'energy_aware': analysis_energy,
            'distance_only': analysis_dist
        })

    # 场景2：窄高障碍物（适合从侧面绕行）
    print("\n--- 场景2: 窄高障碍物 (宽2m, 高8m) ---")
    config2 = PlanningConfig(
        voxel_size=0.5,
        grid_size=(60, 60, 30),
        origin=(-15.0, -15.0, -15.0),
        max_iterations=3000,
        step_size=1.0,
        safety_margin=0.8,
        energy_aware=True,
        weight_energy=0.6,
        weight_distance=0.3,
        weight_time=0.1
    )
    voxel2 = VoxelGrid(config2)
    # 窄高障碍物
    for x in range(-1, 2):
        for y in range(-2, 3):
            for z in range(-12, -4):  # 8米高
                idx = voxel2.world_to_grid(np.array([float(x), float(y), float(z)]))
                if voxel2.is_valid_index(idx):
                    voxel2.grid[idx] = 1

    esdf2 = ESDF(voxel2)
    esdf2.compute()

    start2 = np.array([-8.0, 0.0, -8.0])
    goal2 = np.array([8.0, 0.0, -8.0])

    rrt2_energy = RRTStar(voxel2, esdf2, config2, energy_model)
    np.random.seed(200)
    path2_energy = rrt2_energy.plan(start2.copy(), goal2.copy())

    config2_dist = PlanningConfig(
        voxel_size=0.5,
        grid_size=(60, 60, 30),
        origin=(-15.0, -15.0, -15.0),
        max_iterations=3000,
        step_size=1.0,
        safety_margin=0.8,
        energy_aware=False
    )
    rrt2_dist = RRTStar(voxel2, esdf2, config2_dist)
    np.random.seed(200)
    path2_dist = rrt2_dist.plan(start2.copy(), goal2.copy())

    if path2_energy and path2_dist:
        analysis_energy = analyze_path_energy(path2_energy, energy_model)
        analysis_dist = analyze_path_energy(path2_dist, energy_model)

        print(f"  能量感知: 距离={analysis_energy['total_distance']:.1f}m, 能耗={analysis_energy['total_energy']:.0f}J")
        print(f"  纯距离:   距离={analysis_dist['total_distance']:.1f}m, 能耗={analysis_dist['total_energy']:.0f}J")

        scenarios.append({
            'name': '窄高障碍物',
            'energy_aware': analysis_energy,
            'distance_only': analysis_dist
        })

    # 总结
    print("\n" + "="*70)
    print("场景对比总结")
    print("="*70)
    print(f"{'场景':<15} {'能量感知能耗':<15} {'纯距离能耗':<15} {'节省':<15}")
    print("-"*60)

    for s in scenarios:
        e_energy = s['energy_aware']['total_energy']
        d_energy = s['distance_only']['total_energy']
        saving = d_energy - e_energy
        saving_pct = saving / d_energy * 100 if d_energy > 0 else 0
        print(f"{s['name']:<15} {e_energy:<15.0f} {d_energy:<15.0f} {saving:+.0f}J ({saving_pct:+.1f}%)")

    return True


def test_weight_effect():
    """测试权重对路径选择的影响"""
    print("\n" + "="*70)
    print("测试: 权重参数对路径选择的影响")
    print("="*70)

    energy_model = PhysicsEnergyModel()

    # 创建一个需要选择的场景
    base_config = PlanningConfig(
        voxel_size=0.5,
        grid_size=(60, 60, 30),
        origin=(-15.0, -15.0, -15.0),
        max_iterations=3000,
        step_size=1.0,
        safety_margin=0.8
    )

    voxel = VoxelGrid(base_config)
    # 中等大小障碍物
    for x in range(-3, 4):
        for y in range(-8, 8):
            for z in range(-10, -5):
                idx = voxel.world_to_grid(np.array([float(x), float(y), float(z)]))
                if voxel.is_valid_index(idx):
                    voxel.grid[idx] = 1

    esdf = ESDF(voxel)
    esdf.compute()

    start = np.array([-10.0, 0.0, -7.0])
    goal = np.array([10.0, 0.0, -7.0])

    # 不同权重配置
    weight_configs = [
        ("纯距离优化", 0.0, 1.0, 0.0, False),
        ("轻度能量感知", 0.3, 0.5, 0.2, True),
        ("平衡模式", 0.4, 0.4, 0.2, True),
        ("能量优先", 0.6, 0.3, 0.1, True),
        ("极端能量优先", 0.8, 0.1, 0.1, True),
    ]

    print(f"\n起点: {start}")
    print(f"终点: {goal}")
    print(f"\n{'配置':<18} {'w_e':<6} {'w_d':<6} {'距离(m)':<10} {'能耗(J)':<10} {'高度变化(m)':<12}")
    print("-"*70)

    results = []
    for name, w_e, w_d, w_t, energy_aware in weight_configs:
        config = PlanningConfig(
            voxel_size=0.5,
            grid_size=(60, 60, 30),
            origin=(-15.0, -15.0, -15.0),
            max_iterations=3000,
            step_size=1.0,
            safety_margin=0.8,
            energy_aware=energy_aware,
            weight_energy=w_e,
            weight_distance=w_d,
            weight_time=w_t
        )

        np.random.seed(42)
        rrt = RRTStar(voxel, esdf, config, energy_model if energy_aware else None)
        path = rrt.plan(start.copy(), goal.copy())

        if path:
            analysis = analyze_path_energy(path, energy_model)
            heights = [-p[2] for p in path]  # 转换为正高度
            height_change = max(heights) - min(heights)

            print(f"{name:<18} {w_e:<6.1f} {w_d:<6.1f} {analysis['total_distance']:<10.1f} {analysis['total_energy']:<10.0f} {height_change:<12.1f}")
            results.append({
                'name': name,
                'distance': analysis['total_distance'],
                'energy': analysis['total_energy'],
                'height_change': height_change
            })
        else:
            print(f"{name:<18} {w_e:<6.1f} {w_d:<6.1f} {'失败':<10} {'-':<10} {'-':<12}")

    return True


def main():
    """主函数"""
    print("\n" + "="*70)
    print("能量感知路径规划效果验证")
    print("="*70)

    test_climb_penalty()
    test_obstacle_scenarios()
    test_weight_effect()

    print("\n" + "="*70)
    print("验证完成!")
    print("="*70)


if __name__ == "__main__":
    main()
