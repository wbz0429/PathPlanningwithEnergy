"""
generate_energy_analysis_report.py - 生成能量感知规划分析报告

用于毕业设计答辩展示
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
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


def generate_power_model_analysis():
    """生成功率模型分析图"""
    print("Generating power model analysis...")

    model = PhysicsEnergyModel()

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. 功率随速度变化
    ax1 = axes[0, 0]
    speeds = np.linspace(0, 15, 50)
    powers = []
    induced = []
    profile = []
    parasite = []

    for v in speeds:
        vel = np.array([v, 0, 0])
        breakdown = model.get_power_breakdown(vel)
        powers.append(breakdown['electrical_total'])
        induced.append(breakdown['induced'])
        profile.append(breakdown['profile'])
        parasite.append(breakdown['parasite'])

    ax1.plot(speeds, powers, 'b-', linewidth=2, label='Total Electrical')
    ax1.plot(speeds, induced, 'r--', label='Induced')
    ax1.plot(speeds, profile, 'g--', label='Profile')
    ax1.plot(speeds, parasite, 'm--', label='Parasite')
    ax1.set_xlabel('Forward Speed (m/s)')
    ax1.set_ylabel('Power (W)')
    ax1.set_title('Power vs Forward Speed')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 15)

    # 2. 爬升功率
    ax2 = axes[0, 1]
    climb_rates = np.linspace(-3, 3, 50)  # 负值=爬升(NED)
    climb_powers = []

    for vz in climb_rates:
        vel = np.array([2, 0, vz])  # 2m/s前飞 + 垂直速度
        power = model.compute_electrical_power(vel)
        climb_powers.append(power)

    ax2.plot(-climb_rates, climb_powers, 'b-', linewidth=2)  # 转换为正值=爬升
    ax2.axhline(y=model.compute_electrical_power(np.array([2, 0, 0])),
                color='gray', linestyle='--', label='Level flight')
    ax2.axvline(x=0, color='gray', linestyle=':', alpha=0.5)
    ax2.set_xlabel('Climb Rate (m/s)')
    ax2.set_ylabel('Power (W)')
    ax2.set_title('Power vs Climb Rate (at 2m/s forward)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. 能耗对比：水平 vs 爬升
    ax3 = axes[1, 0]
    distances = [5, 10, 15, 20]
    energy_horizontal = []
    energy_climb = []

    for d in distances:
        # 水平飞行
        p1 = np.array([0, 0, -5])
        p2_h = np.array([d, 0, -5])
        e_h, _ = model.compute_energy_for_segment(p1, p2_h, 2.0)
        energy_horizontal.append(e_h)

        # 爬升飞行（相同水平距离，但爬升5m）
        p2_c = np.array([d, 0, -10])
        e_c, _ = model.compute_energy_for_segment(p1, p2_c, 2.0)
        energy_climb.append(e_c)

    x = np.arange(len(distances))
    width = 0.35
    ax3.bar(x - width/2, energy_horizontal, width, label='Horizontal', color='#3498db')
    ax3.bar(x + width/2, energy_climb, width, label='Climb 5m', color='#e74c3c')
    ax3.set_xlabel('Horizontal Distance (m)')
    ax3.set_ylabel('Energy (J)')
    ax3.set_title('Energy: Horizontal vs Climb Flight')
    ax3.set_xticks(x)
    ax3.set_xticklabels(distances)
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')

    # 添加百分比标注
    for i, (eh, ec) in enumerate(zip(energy_horizontal, energy_climb)):
        pct = (ec/eh - 1) * 100
        ax3.annotate(f'+{pct:.0f}%', xy=(i + width/2, ec),
                    ha='center', va='bottom', fontsize=9)

    # 4. 续航估算
    ax4 = axes[1, 1]
    battery_wh = model.params.battery_voltage * model.params.battery_capacity / 1000
    battery_j = battery_wh * 3600

    speeds_range = [0, 2, 5, 8, 10]
    endurance_time = []
    endurance_range = []

    for v in speeds_range:
        vel = np.array([v, 0, 0])
        power = model.compute_electrical_power(vel)
        time_s = battery_j / power
        endurance_time.append(time_s / 60)  # minutes
        endurance_range.append(v * time_s / 1000)  # km

    ax4_twin = ax4.twinx()

    line1 = ax4.bar(np.arange(len(speeds_range)) - 0.2, endurance_time, 0.4,
                    label='Endurance (min)', color='#2ecc71')
    line2 = ax4_twin.bar(np.arange(len(speeds_range)) + 0.2, endurance_range, 0.4,
                         label='Range (km)', color='#9b59b6')

    ax4.set_xlabel('Speed (m/s)')
    ax4.set_ylabel('Endurance (min)', color='#2ecc71')
    ax4_twin.set_ylabel('Range (km)', color='#9b59b6')
    ax4.set_title('Endurance and Range vs Speed')
    ax4.set_xticks(np.arange(len(speeds_range)))
    ax4.set_xticklabels(speeds_range)
    ax4.legend(loc='upper left')
    ax4_twin.legend(loc='upper right')
    ax4.grid(True, alpha=0.3, axis='y')

    plt.suptitle('BEMT Energy Model Analysis\n(1.5kg Quadrotor, 4S 5000mAh)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('energy_model_analysis.png', dpi=150, bbox_inches='tight')
    print("  Saved: energy_model_analysis.png")
    plt.close()


def generate_cost_function_analysis():
    """生成代价函数分析图"""
    print("Generating cost function analysis...")

    energy_model = PhysicsEnergyModel()

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 1. 代价函数组成
    ax1 = axes[0]

    config = PlanningConfig(energy_aware=True)
    cost_fn = EnergyAwareCostFunction(config, energy_model)

    # 计算10m水平飞行的代价分解
    p1 = np.array([0, 0, -5])
    p2 = np.array([10, 0, -5])

    energy, time = energy_model.compute_energy_for_segment(p1, p2, 2.0)
    distance = 10.0

    cost_e = config.weight_energy * (energy / config.energy_ref)
    cost_d = config.weight_distance * (distance / config.distance_ref)
    cost_t = config.weight_time * (time / config.time_ref)

    components = [cost_e, cost_d, cost_t]
    labels = [f'Energy\n({config.weight_energy}×E/{config.energy_ref}J)',
              f'Distance\n({config.weight_distance}×D/{config.distance_ref}m)',
              f'Time\n({config.weight_time}×T/{config.time_ref}s)']
    colors = ['#e74c3c', '#3498db', '#2ecc71']

    bars = ax1.bar(labels, components, color=colors)
    ax1.set_ylabel('Cost Component')
    ax1.set_title('Cost Function Breakdown\n(10m horizontal flight)')
    ax1.grid(True, alpha=0.3, axis='y')

    for bar, val in zip(bars, components):
        ax1.annotate(f'{val:.3f}', xy=(bar.get_x() + bar.get_width()/2, val),
                    ha='center', va='bottom')

    # 2. 水平 vs 爬升代价对比
    ax2 = axes[1]

    scenarios = ['Horizontal\n10m', 'Climb 3m\n(10m path)', 'Climb 5m\n(11m path)', 'Climb 7m\n(12m path)']
    paths = [
        (np.array([0, 0, -5]), np.array([10, 0, -5])),
        (np.array([0, 0, -5]), np.array([10, 0, -8])),
        (np.array([0, 0, -5]), np.array([10, 0, -10])),
        (np.array([0, 0, -5]), np.array([10, 0, -12])),
    ]

    costs_energy_mode = []
    costs_dist_mode = []

    config_e = PlanningConfig(energy_aware=True)
    config_d = PlanningConfig(energy_aware=False)
    cost_fn_e = EnergyAwareCostFunction(config_e, energy_model)
    cost_fn_d = EnergyAwareCostFunction(config_d, None)

    for p1, p2 in paths:
        costs_energy_mode.append(cost_fn_e.compute_cost(p1, p2))
        costs_dist_mode.append(cost_fn_d.compute_cost(p1, p2))

    x = np.arange(len(scenarios))
    width = 0.35

    bars1 = ax2.bar(x - width/2, costs_dist_mode, width, label='Distance-only', color='#3498db')
    bars2 = ax2.bar(x + width/2, costs_energy_mode, width, label='Energy-aware', color='#e74c3c')

    ax2.set_ylabel('Total Cost')
    ax2.set_title('Cost Comparison: Distance vs Energy Mode')
    ax2.set_xticks(x)
    ax2.set_xticklabels(scenarios)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')

    # 3. 权重敏感性分析
    ax3 = axes[2]

    weight_configs = [
        (0.2, 0.6, 0.2, 'Distance\nPriority'),
        (0.4, 0.4, 0.2, 'Balanced'),
        (0.6, 0.3, 0.1, 'Energy\nPriority'),
        (0.8, 0.1, 0.1, 'Strong\nEnergy'),
    ]

    # 计算水平和爬升路径在不同权重下的代价差异
    p1 = np.array([0, 0, -5])
    p2_h = np.array([10, 0, -5])
    p2_c = np.array([10, 0, -10])  # 5m climb

    cost_diffs = []
    labels_w = []

    for w_e, w_d, w_t, label in weight_configs:
        config = PlanningConfig(
            energy_aware=True,
            weight_energy=w_e,
            weight_distance=w_d,
            weight_time=w_t
        )
        cost_fn = EnergyAwareCostFunction(config, energy_model)

        cost_h = cost_fn.compute_cost(p1, p2_h)
        cost_c = cost_fn.compute_cost(p1, p2_c)

        diff_pct = (cost_c / cost_h - 1) * 100
        cost_diffs.append(diff_pct)
        labels_w.append(label)

    bars = ax3.bar(labels_w, cost_diffs, color='#9b59b6')
    ax3.set_ylabel('Climb Cost Penalty (%)')
    ax3.set_title('Weight Sensitivity\n(Climb 5m vs Horizontal)')
    ax3.grid(True, alpha=0.3, axis='y')

    for bar, val in zip(bars, cost_diffs):
        ax3.annotate(f'+{val:.1f}%', xy=(bar.get_x() + bar.get_width()/2, val),
                    ha='center', va='bottom')

    plt.suptitle('Energy-Aware Cost Function Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('cost_function_analysis.png', dpi=150, bbox_inches='tight')
    print("  Saved: cost_function_analysis.png")
    plt.close()


def generate_planning_comparison():
    """生成规划对比分析图"""
    print("Generating planning comparison...")

    # 创建测试环境
    config_base = PlanningConfig(
        voxel_size=0.5,
        grid_size=(60, 60, 30),
        origin=(-15.0, -15.0, -15.0),
        max_iterations=3000,
        step_size=1.0,
        safety_margin=0.8
    )

    voxel = VoxelGrid(config_base)

    # 添加障碍物
    for x in range(-2, 3):
        for y in range(-8, 8):
            for z in range(-9, -5):
                idx = voxel.world_to_grid(np.array([float(x), float(y), float(z)]))
                if voxel.is_valid_index(idx):
                    voxel.grid[idx] = 1

    esdf = ESDF(voxel)
    esdf.compute()

    energy_model = PhysicsEnergyModel()

    start = np.array([-10.0, 0.0, -7.0])
    goal = np.array([10.0, 0.0, -7.0])

    # 运行多次规划
    n_trials = 30

    results_energy = {'dist': [], 'energy': [], 'paths': []}
    results_dist = {'dist': [], 'energy': [], 'paths': []}

    print(f"  Running {n_trials} trials for each mode...")

    for trial in range(n_trials):
        # Energy-aware
        config_e = PlanningConfig(
            voxel_size=0.5, grid_size=(60, 60, 30), origin=(-15.0, -15.0, -15.0),
            max_iterations=3000, step_size=1.0, safety_margin=0.8,
            energy_aware=True, weight_energy=0.6, weight_distance=0.3, weight_time=0.1
        )
        np.random.seed(trial * 100 + 1)
        rrt_e = RRTStar(voxel, esdf, config_e, energy_model)
        path_e = rrt_e.plan(start.copy(), goal.copy())

        if path_e:
            stats = rrt_e.get_plan_stats()
            results_energy['dist'].append(stats['total_distance'])
            results_energy['energy'].append(stats['total_energy_joules'])
            if trial < 3:
                results_energy['paths'].append(path_e)

        # Distance-only
        config_d = PlanningConfig(
            voxel_size=0.5, grid_size=(60, 60, 30), origin=(-15.0, -15.0, -15.0),
            max_iterations=3000, step_size=1.0, safety_margin=0.8,
            energy_aware=False
        )
        np.random.seed(trial * 100 + 50)
        rrt_d = RRTStar(voxel, esdf, config_d)
        path_d = rrt_d.plan(start.copy(), goal.copy())

        if path_d:
            total_e = sum(energy_model.compute_energy_for_segment(path_d[i], path_d[i+1], 2.0)[0]
                         for i in range(len(path_d)-1))
            total_d = sum(np.linalg.norm(path_d[i+1] - path_d[i]) for i in range(len(path_d)-1))
            results_dist['dist'].append(total_d)
            results_dist['energy'].append(total_e)
            if trial < 3:
                results_dist['paths'].append(path_d)

    # 绘图
    fig = plt.figure(figsize=(15, 10))
    gs = GridSpec(2, 3, figure=fig)

    # 1. 能耗分布对比
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(results_energy['energy'], bins=15, alpha=0.7, label='Energy-aware', color='#e74c3c')
    ax1.hist(results_dist['energy'], bins=15, alpha=0.7, label='Distance-only', color='#3498db')
    ax1.axvline(np.mean(results_energy['energy']), color='#c0392b', linestyle='--', linewidth=2)
    ax1.axvline(np.mean(results_dist['energy']), color='#2980b9', linestyle='--', linewidth=2)
    ax1.set_xlabel('Energy (J)')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Energy Distribution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. 距离分布对比
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.hist(results_energy['dist'], bins=15, alpha=0.7, label='Energy-aware', color='#e74c3c')
    ax2.hist(results_dist['dist'], bins=15, alpha=0.7, label='Distance-only', color='#3498db')
    ax2.axvline(np.mean(results_energy['dist']), color='#c0392b', linestyle='--', linewidth=2)
    ax2.axvline(np.mean(results_dist['dist']), color='#2980b9', linestyle='--', linewidth=2)
    ax2.set_xlabel('Distance (m)')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Distance Distribution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. 统计摘要
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.axis('off')

    avg_e_energy = np.mean(results_energy['energy'])
    avg_e_dist = np.mean(results_energy['dist'])
    avg_d_energy = np.mean(results_dist['energy'])
    avg_d_dist = np.mean(results_dist['dist'])

    energy_saving = avg_d_energy - avg_e_energy
    energy_pct = energy_saving / avg_d_energy * 100

    summary = f"""
    Planning Comparison Results
    (n={n_trials} trials each)

    Energy-Aware Mode:
      Avg Energy: {avg_e_energy:.0f} J
      Avg Distance: {avg_e_dist:.1f} m
      Std Energy: {np.std(results_energy['energy']):.0f} J

    Distance-Only Mode:
      Avg Energy: {avg_d_energy:.0f} J
      Avg Distance: {avg_d_dist:.1f} m
      Std Energy: {np.std(results_dist['energy']):.0f} J

    Comparison:
      Energy Difference: {energy_saving:.0f} J
      Percentage: {energy_pct:+.1f}%
    """

    ax3.text(0.1, 0.9, summary, transform=ax3.transAxes, fontsize=11,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # 4. 示例路径 - 俯视图
    ax4 = fig.add_subplot(gs[1, 0])

    # 绘制障碍物
    obstacle_x = [-2, 3, 3, -2, -2]
    obstacle_y = [-8, -8, 8, 8, -8]
    ax4.fill(obstacle_x, obstacle_y, color='gray', alpha=0.5, label='Obstacle')

    # 绘制路径
    for i, path in enumerate(results_energy['paths'][:3]):
        path_arr = np.array(path)
        ax4.plot(path_arr[:, 0], path_arr[:, 1], 'r-', alpha=0.5, linewidth=1)
    for i, path in enumerate(results_dist['paths'][:3]):
        path_arr = np.array(path)
        ax4.plot(path_arr[:, 0], path_arr[:, 1], 'b-', alpha=0.5, linewidth=1)

    ax4.plot(start[0], start[1], 'go', markersize=10, label='Start')
    ax4.plot(goal[0], goal[1], 'r*', markersize=15, label='Goal')
    ax4.set_xlabel('X (m)')
    ax4.set_ylabel('Y (m)')
    ax4.set_title('Sample Paths (Top View)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_aspect('equal')

    # 5. 能耗 vs 距离散点图
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.scatter(results_energy['dist'], results_energy['energy'],
               alpha=0.6, label='Energy-aware', color='#e74c3c', s=50)
    ax5.scatter(results_dist['dist'], results_dist['energy'],
               alpha=0.6, label='Distance-only', color='#3498db', s=50)
    ax5.set_xlabel('Distance (m)')
    ax5.set_ylabel('Energy (J)')
    ax5.set_title('Energy vs Distance')
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # 6. 箱线图对比
    ax6 = fig.add_subplot(gs[1, 2])
    data = [results_energy['energy'], results_dist['energy']]
    bp = ax6.boxplot(data, labels=['Energy-aware', 'Distance-only'], patch_artist=True)
    bp['boxes'][0].set_facecolor('#e74c3c')
    bp['boxes'][1].set_facecolor('#3498db')
    ax6.set_ylabel('Energy (J)')
    ax6.set_title('Energy Distribution Comparison')
    ax6.grid(True, alpha=0.3, axis='y')

    plt.suptitle('Energy-Aware vs Distance-Only Planning Comparison',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('planning_comparison.png', dpi=150, bbox_inches='tight')
    print("  Saved: planning_comparison.png")
    plt.close()


def main():
    print("="*70)
    print("Generating Energy-Aware Planning Analysis Report")
    print("="*70)
    print()

    generate_power_model_analysis()
    generate_cost_function_analysis()
    generate_planning_comparison()

    print()
    print("="*70)
    print("Report generation complete!")
    print("="*70)
    print()
    print("Generated files:")
    print("  1. energy_model_analysis.png - Power model characteristics")
    print("  2. cost_function_analysis.png - Cost function breakdown")
    print("  3. planning_comparison.png - Planning mode comparison")


if __name__ == "__main__":
    main()
