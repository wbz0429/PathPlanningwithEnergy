"""
benchmark_planning.py - 路径规划算法评测
对比 A* 全局最优 baseline 与 RRT* 在不同权重配置下的表现

评测指标：
1. 路径长度 (m)
2. 能耗 (J)
3. 飞行时间 (s)
4. Path Ratio (实际路径 / A*最优路径)
5. 代价函数值
"""

import numpy as np
import heapq
import time
import sys
import os
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from planning.config import PlanningConfig
from mapping.voxel_grid import VoxelGrid
from mapping.esdf import ESDF
from planning.rrt_star import RRTStar, EnergyAwareCostFunction
from energy.physics_model import PhysicsEnergyModel


# ============================================================
# Blocks 场景障碍物定义
# ============================================================
BLOCKS_OBSTACLES = [
    # Row 1 (X=23.1~33.1): 实心墙, Y=[-21.5, 18.5], Z=0~-10 (NED)
    {'x_range': (23.1, 33.1), 'y_range': (-21.5, 18.5), 'z_range': (-10.0, 0.0)},
    # Row 2 (X=33.1~43.1): 两段墙，中间有缝隙 Y=[-11.5, 8.5]
    {'x_range': (33.1, 43.1), 'y_range': (-21.5, -11.5), 'z_range': (-10.0, 0.0)},
    {'x_range': (33.1, 43.1), 'y_range': (8.5, 18.5), 'z_range': (-10.0, 0.0)},
    # Row 3 (X=43.1~53.1): 同 Row 2
    {'x_range': (43.1, 53.1), 'y_range': (-21.5, -11.5), 'z_range': (-10.0, 0.0)},
    {'x_range': (43.1, 53.1), 'y_range': (8.5, 18.5), 'z_range': (-10.0, 0.0)},
    # Row 4 (X=53.1~63.1): 实心墙
    {'x_range': (53.1, 63.1), 'y_range': (-21.5, 18.5), 'z_range': (-10.0, 0.0)},
]


def build_known_map(config: PlanningConfig) -> VoxelGrid:
    """用已知障碍物信息构建完整地图"""
    voxel_grid = VoxelGrid(config)
    vs = config.voxel_size

    for obs in BLOCKS_OBSTACLES:
        x_min, x_max = obs['x_range']
        y_min, y_max = obs['y_range']
        z_min, z_max = obs['z_range']

        # 遍历障碍物范围内的所有体素
        x = x_min
        while x < x_max:
            y = y_min
            while y < y_max:
                z = z_min
                while z < z_max:
                    idx = voxel_grid.world_to_grid(np.array([x, y, z]))
                    if voxel_grid.is_valid_index(idx):
                        voxel_grid.grid[idx[0], idx[1], idx[2]] = 1
                    z += vs
                y += vs
            x += vs

    # 将非障碍物区域标记为 free
    voxel_grid.grid[voxel_grid.grid == 0] = -1

    occupied = np.sum(voxel_grid.grid == 1)
    free = np.sum(voxel_grid.grid == -1)
    print(f"  Known map built: {occupied} occupied, {free} free voxels")
    return voxel_grid


# ============================================================
# A* 全局最优路径规划
# ============================================================
class AStarPlanner:
    """3D A* 路径规划器，作为最优 baseline"""

    def __init__(self, voxel_grid: VoxelGrid, esdf: ESDF, config: PlanningConfig):
        self.voxel_grid = voxel_grid
        self.esdf = esdf
        self.config = config
        self.safety_margin = config.safety_margin

    def plan(self, start: np.ndarray, goal: np.ndarray) -> Optional[List[np.ndarray]]:
        """A* 搜索，返回世界坐标路径"""
        start_idx = self.voxel_grid.world_to_grid(start)
        goal_idx = self.voxel_grid.world_to_grid(goal)

        if not self.voxel_grid.is_valid_index(start_idx):
            print(f"  [A*] Start {start} -> grid {start_idx} out of bounds")
            return None
        if not self.voxel_grid.is_valid_index(goal_idx):
            print(f"  [A*] Goal {goal} -> grid {goal_idx} out of bounds")
            return None

        # 26-connected neighbors (包括对角线)
        neighbors = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                for dz in [-1, 0, 1]:
                    if dx == 0 and dy == 0 and dz == 0:
                        continue
                    dist = np.sqrt(dx*dx + dy*dy + dz*dz) * self.config.voxel_size
                    neighbors.append((dx, dy, dz, dist))

        open_set = []
        start_tuple = tuple(start_idx)
        goal_tuple = tuple(goal_idx)

        g_score = {start_tuple: 0.0}
        h = np.linalg.norm(np.array(goal_idx) - np.array(start_idx)) * self.config.voxel_size
        heapq.heappush(open_set, (h, start_tuple))
        came_from = {}
        closed_set = set()

        iterations = 0
        max_iterations = 500000

        while open_set and iterations < max_iterations:
            iterations += 1
            f_current, current = heapq.heappop(open_set)

            if current == goal_tuple:
                # 重建路径
                path = []
                node = current
                while node in came_from:
                    world_pos = self.voxel_grid.grid_to_world(node)
                    path.append(world_pos)
                    node = came_from[node]
                world_pos = self.voxel_grid.grid_to_world(node)
                path.append(world_pos)
                path.reverse()
                print(f"  [A*] Found path: {len(path)} waypoints, {iterations} iterations")
                return path

            if current in closed_set:
                continue
            closed_set.add(current)

            for dx, dy, dz, step_dist in neighbors:
                nx = current[0] + dx
                ny = current[1] + dy
                nz = current[2] + dz
                neighbor = (nx, ny, nz)

                if not self.voxel_grid.is_valid_index(neighbor):
                    continue
                if neighbor in closed_set:
                    continue

                # 碰撞检测：检查体素是否被占据
                if self.voxel_grid.grid[nx, ny, nz] == 1:
                    continue

                # 安全边距检查（用 ESDF）
                world_pos = self.voxel_grid.grid_to_world(neighbor)
                dist_to_obs = self.esdf.get_distance(world_pos)
                if dist_to_obs < self.safety_margin:
                    continue

                tentative_g = g_score[current] + step_dist

                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    g_score[neighbor] = tentative_g
                    came_from[neighbor] = current
                    h = np.linalg.norm(np.array(goal_idx) - np.array(neighbor)) * self.config.voxel_size
                    heapq.heappush(open_set, (tentative_g + h, neighbor))

        print(f"  [A*] Failed after {iterations} iterations")
        return None


def smooth_path(path: List[np.ndarray], esdf: ESDF, safety_margin: float,
                max_iterations: int = 50) -> List[np.ndarray]:
    """路径平滑：尝试跳过中间点"""
    if len(path) <= 2:
        return path

    smoothed = [path[0]]
    i = 0
    while i < len(path) - 1:
        # 尝试跳到尽可能远的点
        best_j = i + 1
        for j in range(len(path) - 1, i + 1, -1):
            if _line_safe(smoothed[-1], path[j], esdf, safety_margin):
                best_j = j
                break
        smoothed.append(path[best_j])
        i = best_j

    return smoothed


def _line_safe(p1: np.ndarray, p2: np.ndarray, esdf: ESDF, margin: float,
               step: float = 0.3) -> bool:
    """检查两点之间的直线是否安全"""
    dist = np.linalg.norm(p2 - p1)
    if dist < 1e-6:
        return True
    n_steps = max(int(dist / step), 1)
    for i in range(n_steps + 1):
        t = i / n_steps
        point = p1 + t * (p2 - p1)
        if esdf.get_distance(point) < margin:
            return False
    return True


def compute_path_length(path: List[np.ndarray]) -> float:
    """计算路径总长度"""
    total = 0.0
    for i in range(len(path) - 1):
        total += np.linalg.norm(path[i+1] - path[i])
    return total


# ============================================================
# 评测主流程
# ============================================================
def run_benchmark():
    print("=" * 70)
    print("  PATH PLANNING BENCHMARK")
    print("  A* Baseline vs RRT* with Different Cost Weights")
    print("=" * 70)

    # --- 配置 ---
    config = PlanningConfig(
        voxel_size=0.5,
        grid_size=(180, 120, 40),
        origin=(-10.0, -30.0, -15.0),
        max_depth=25.0,
        step_size=1.5,
        max_iterations=3000,
        goal_sample_rate=0.2,
        search_radius=4.0,
        safety_margin=1.0,
    )

    energy_model = PhysicsEnergyModel()
    flight_velocity = 2.0

    # --- 构建已知地图 ---
    print("\n[1] Building known map from Blocks scene layout...")
    voxel_grid = build_known_map(config)
    esdf = ESDF(voxel_grid)
    print("  Computing ESDF...")
    t0 = time.time()
    esdf.compute()
    print(f"  ESDF computed in {(time.time()-t0)*1000:.0f}ms")

    # --- 测试场景 ---
    scenarios = [
        {
            'name': 'Scenario A: Straight through (Y=0)',
            'start': np.array([0.0, 0.0, -3.0]),
            'goal': np.array([70.0, 0.0, -3.0]),
            'desc': 'Must navigate through Row2/3 gap or around walls'
        },
        {
            'name': 'Scenario B: Diagonal (Y=20)',
            'start': np.array([0.0, 0.0, -3.0]),
            'goal': np.array([70.0, 20.0, -3.0]),
            'desc': 'Must go around Row1 top edge, target near wall boundary'
        },
        {
            'name': 'Scenario C: Opposite side (Y=-25)',
            'start': np.array([0.0, 0.0, -3.0]),
            'goal': np.array([70.0, -25.0, -3.0]),
            'desc': 'Must go around Row1 bottom edge'
        },
    ]

    # --- RRT* 权重配置（消融实验） ---
    weight_configs = [
        {'name': 'Distance-only', 'energy_aware': False,
         'weight_energy': 0.0, 'weight_distance': 1.0, 'weight_time': 0.0},
        {'name': 'Energy-only', 'energy_aware': True,
         'weight_energy': 1.0, 'weight_distance': 0.0, 'weight_time': 0.0},
        {'name': 'Balanced', 'energy_aware': True,
         'weight_energy': 0.6, 'weight_distance': 0.3, 'weight_time': 0.1},
        {'name': 'Time-priority', 'energy_aware': True,
         'weight_energy': 0.1, 'weight_distance': 0.3, 'weight_time': 0.6},
    ]

    rrt_runs_per_config = 10  # 每种配置跑10次取统计

    all_results = {}

    for scenario in scenarios:
        print(f"\n{'='*70}")
        print(f"  {scenario['name']}")
        print(f"  {scenario['desc']}")
        print(f"  Start: {scenario['start']}  Goal: {scenario['goal']}")
        straight_dist = np.linalg.norm(scenario['goal'] - scenario['start'])
        print(f"  Straight-line distance: {straight_dist:.2f}m")
        print(f"{'='*70}")

        scenario_results = {}

        # --- A* Baseline ---
        print("\n  [A*] Computing optimal baseline...")
        astar = AStarPlanner(voxel_grid, esdf, config)
        t0 = time.time()
        astar_path = astar.plan(scenario['start'], scenario['goal'])
        astar_time = time.time() - t0

        if astar_path:
            # 平滑 A* 路径
            astar_path = smooth_path(astar_path, esdf, config.safety_margin)
            astar_length = compute_path_length(astar_path)
            astar_energy, astar_flight_time = energy_model.compute_energy_for_path(
                astar_path, velocity=flight_velocity)
            print(f"  [A*] Length: {astar_length:.2f}m, Energy: {astar_energy:.1f}J, "
                  f"Time: {astar_flight_time:.1f}s, Compute: {astar_time*1000:.0f}ms")
            scenario_results['A*'] = {
                'path_length': astar_length,
                'energy': astar_energy,
                'flight_time': astar_flight_time,
                'compute_time': astar_time,
                'path_ratio': astar_length / straight_dist,
                'path': astar_path,
                'success_rate': 1.0,
            }
        else:
            print("  [A*] No path found!")
            scenario_results['A*'] = None

        # --- RRT* 消融实验 ---
        for wc in weight_configs:
            print(f"\n  [RRT*] Config: {wc['name']} "
                  f"(E={wc['weight_energy']}, D={wc['weight_distance']}, T={wc['weight_time']})")

            cfg = PlanningConfig(
                voxel_size=config.voxel_size,
                grid_size=config.grid_size,
                origin=config.origin,
                max_depth=config.max_depth,
                step_size=config.step_size,
                max_iterations=config.max_iterations,
                goal_sample_rate=config.goal_sample_rate,
                search_radius=config.search_radius,
                safety_margin=config.safety_margin,
                energy_aware=wc['energy_aware'],
                weight_energy=wc['weight_energy'],
                weight_distance=wc['weight_distance'],
                weight_time=wc['weight_time'],
                flight_velocity=flight_velocity,
            )

            lengths = []
            energies = []
            flight_times = []
            compute_times = []
            costs = []
            successes = 0
            best_path = None
            best_length = float('inf')

            for run in range(rrt_runs_per_config):
                planner = RRTStar(voxel_grid, esdf, cfg, energy_model=energy_model)
                t0 = time.time()
                path = planner.plan(scenario['start'], scenario['goal'])
                ct = time.time() - t0

                if path and len(path) >= 2:
                    successes += 1
                    length = compute_path_length(path)
                    energy, ft = energy_model.compute_energy_for_path(path, velocity=flight_velocity)
                    cost = planner.get_plan_stats().get('total_cost', 0)

                    lengths.append(length)
                    energies.append(energy)
                    flight_times.append(ft)
                    compute_times.append(ct)
                    costs.append(cost)

                    if length < best_length:
                        best_length = length
                        best_path = path

            success_rate = successes / rrt_runs_per_config
            if successes > 0:
                result = {
                    'path_length': np.mean(lengths),
                    'path_length_std': np.std(lengths),
                    'energy': np.mean(energies),
                    'energy_std': np.std(energies),
                    'flight_time': np.mean(flight_times),
                    'flight_time_std': np.std(flight_times),
                    'compute_time': np.mean(compute_times),
                    'cost': np.mean(costs),
                    'cost_std': np.std(costs),
                    'path_ratio': np.mean(lengths) / straight_dist,
                    'success_rate': success_rate,
                    'best_path': best_path,
                    'all_lengths': lengths,
                    'all_energies': energies,
                }
                if astar_path:
                    result['vs_astar'] = np.mean(lengths) / astar_length

                print(f"    Success: {successes}/{rrt_runs_per_config}")
                print(f"    Length: {np.mean(lengths):.2f} ± {np.std(lengths):.2f}m")
                print(f"    Energy: {np.mean(energies):.1f} ± {np.std(energies):.1f}J")
                print(f"    Time:   {np.mean(flight_times):.1f} ± {np.std(flight_times):.1f}s")
                print(f"    Compute: {np.mean(compute_times)*1000:.0f}ms avg")
                if astar_path:
                    print(f"    vs A*:  {result['vs_astar']:.3f}x")
            else:
                result = {'success_rate': 0.0}
                print(f"    All {rrt_runs_per_config} runs failed!")

            scenario_results[wc['name']] = result

        all_results[scenario['name']] = scenario_results

    # --- 输出汇总表格 ---
    print_summary_table(all_results, scenarios)

    # --- 生成可视化 ---
    generate_benchmark_figures(all_results, scenarios, voxel_grid, config)

    print("\n  Benchmark complete!")


def print_summary_table(all_results: Dict, scenarios: List[Dict]):
    """打印汇总表格"""
    print("\n" + "=" * 100)
    print("  BENCHMARK SUMMARY")
    print("=" * 100)

    for scenario in scenarios:
        sname = scenario['name']
        results = all_results[sname]
        straight_dist = np.linalg.norm(scenario['goal'] - scenario['start'])

        print(f"\n  {sname}")
        print(f"  Straight-line: {straight_dist:.2f}m")
        print(f"  {'Method':<18} {'Length(m)':<14} {'Energy(J)':<14} {'Time(s)':<12} "
              f"{'PathRatio':<12} {'vs A*':<10} {'Success':<8}")
        print(f"  {'-'*88}")

        astar = results.get('A*')
        if astar:
            print(f"  {'A* (baseline)':<18} {astar['path_length']:<14.2f} "
                  f"{astar['energy']:<14.1f} {astar['flight_time']:<12.1f} "
                  f"{astar['path_ratio']:<12.3f} {'1.000':<10} {'100%':<8}")

        for method_name in ['Distance-only', 'Energy-only', 'Balanced', 'Time-priority']:
            r = results.get(method_name)
            if r and r['success_rate'] > 0:
                vs_astar = f"{r.get('vs_astar', 0):.3f}" if 'vs_astar' in r else "N/A"
                length_str = f"{r['path_length']:.2f}±{r['path_length_std']:.1f}"
                energy_str = f"{r['energy']:.0f}±{r['energy_std']:.0f}"
                time_str = f"{r['flight_time']:.1f}±{r['flight_time_std']:.1f}"
                print(f"  {method_name:<18} {length_str:<14} {energy_str:<14} "
                      f"{time_str:<12} {r['path_ratio']:<12.3f} {vs_astar:<10} "
                      f"{r['success_rate']*100:.0f}%")
            elif r:
                print(f"  {method_name:<18} {'FAILED':<14} {'':<14} {'':<12} "
                      f"{'':<12} {'':<10} {'0%':<8}")


def generate_benchmark_figures(all_results: Dict, scenarios: List[Dict],
                                voxel_grid: VoxelGrid, config: PlanningConfig):
    """生成评测可视化图"""
    print("\n[Visualization] Generating benchmark figures...")

    fig = plt.figure(figsize=(20, 5 * len(scenarios)))

    for s_idx, scenario in enumerate(scenarios):
        sname = scenario['name']
        results = all_results[sname]

        # --- 子图1: 路径对比 (XY 俯视图) ---
        ax1 = fig.add_subplot(len(scenarios), 3, s_idx * 3 + 1)

        # 画障碍物
        for obs in BLOCKS_OBSTACLES:
            rect = plt.Rectangle(
                (obs['x_range'][0], obs['y_range'][0]),
                obs['x_range'][1] - obs['x_range'][0],
                obs['y_range'][1] - obs['y_range'][0],
                color='gray', alpha=0.5
            )
            ax1.add_patch(rect)

        # 画路径
        colors = {'A*': 'black', 'Distance-only': 'blue',
                  'Energy-only': 'red', 'Balanced': 'green', 'Time-priority': 'orange'}
        linestyles = {'A*': '-', 'Distance-only': '--',
                      'Energy-only': '--', 'Balanced': '-', 'Time-priority': '-.'}

        for method_name, color in colors.items():
            r = results.get(method_name)
            if r and r.get('success_rate', 0) > 0:
                path = r.get('path') or r.get('best_path')
                if path:
                    path_arr = np.array(path)
                    ax1.plot(path_arr[:, 0], path_arr[:, 1],
                             color=color, linestyle=linestyles[method_name],
                             linewidth=1.5, label=method_name, alpha=0.8)

        # 起点终点
        ax1.plot(*scenario['start'][:2], 'g^', markersize=10, label='Start')
        ax1.plot(*scenario['goal'][:2], 'r*', markersize=12, label='Goal')

        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')
        ax1.set_title(f'{sname.split(":")[0]}: Path Comparison')
        ax1.legend(fontsize=7, loc='upper left')
        ax1.set_aspect('equal')
        ax1.grid(True, alpha=0.3)

        # --- 子图2: 指标柱状图 ---
        ax2 = fig.add_subplot(len(scenarios), 3, s_idx * 3 + 2)

        methods = []
        lengths_mean = []
        lengths_std = []
        energies_mean = []
        energies_std = []

        for method_name in ['A*', 'Distance-only', 'Energy-only', 'Balanced', 'Time-priority']:
            r = results.get(method_name)
            if r and r.get('success_rate', 0) > 0:
                methods.append(method_name)
                lengths_mean.append(r['path_length'])
                lengths_std.append(r.get('path_length_std', 0))
                energies_mean.append(r['energy'])
                energies_std.append(r.get('energy_std', 0))

        if methods:
            x = np.arange(len(methods))
            width = 0.35
            bars1 = ax2.bar(x - width/2, lengths_mean, width, yerr=lengths_std,
                           label='Path Length (m)', color='steelblue', alpha=0.8, capsize=3)
            ax2_twin = ax2.twinx()
            bars2 = ax2_twin.bar(x + width/2, energies_mean, width, yerr=energies_std,
                                label='Energy (J)', color='coral', alpha=0.8, capsize=3)

            ax2.set_xticks(x)
            ax2.set_xticklabels(methods, rotation=30, ha='right', fontsize=8)
            ax2.set_ylabel('Path Length (m)', color='steelblue')
            ax2_twin.set_ylabel('Energy (J)', color='coral')
            ax2.set_title(f'{sname.split(":")[0]}: Length vs Energy')

            lines1, labels1 = ax2.get_legend_handles_labels()
            lines2, labels2 = ax2_twin.get_legend_handles_labels()
            ax2.legend(lines1 + lines2, labels1 + labels2, fontsize=7)

        # --- 子图3: RRT* 路径长度分布 (箱线图) ---
        ax3 = fig.add_subplot(len(scenarios), 3, s_idx * 3 + 3)

        box_data = []
        box_labels = []
        for method_name in ['Distance-only', 'Energy-only', 'Balanced', 'Time-priority']:
            r = results.get(method_name)
            if r and r.get('success_rate', 0) > 0 and 'all_lengths' in r:
                box_data.append(r['all_lengths'])
                box_labels.append(method_name)

        if box_data:
            bp = ax3.boxplot(box_data, labels=box_labels, patch_artist=True)
            box_colors = ['steelblue', 'coral', 'seagreen', 'orange']
            for patch, color in zip(bp['boxes'], box_colors[:len(box_data)]):
                patch.set_facecolor(color)
                patch.set_alpha(0.6)

            # A* baseline 水平线
            astar = results.get('A*')
            if astar and astar.get('success_rate', 0) > 0:
                ax3.axhline(y=astar['path_length'], color='black',
                           linestyle='--', linewidth=1.5, label=f"A* optimal ({astar['path_length']:.1f}m)")
                ax3.legend(fontsize=7)

            ax3.set_ylabel('Path Length (m)')
            ax3.set_title(f'{sname.split(":")[0]}: RRT* Variance ({rrt_runs_per_config}x runs)')
            ax3.tick_params(axis='x', rotation=30)

    plt.tight_layout()
    output_path = 'benchmark_results.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()


# 全局变量供 print 使用
rrt_runs_per_config = 10

if __name__ == "__main__":
    run_benchmark()
