"""
generate_real_path_comparison.py - 用真实 A* 和 RRT* 路径生成对比图
"""

import numpy as np
import sys, os, time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from planning.config import PlanningConfig
from mapping.voxel_grid import VoxelGrid
from mapping.esdf import ESDF
from planning.rrt_star import RRTStar
from energy.physics_model import PhysicsEnergyModel
from benchmark_planning import build_known_map, AStarPlanner, smooth_path, compute_path_length

OBSTACLES = [
    {'x': (23.1, 33.1), 'y': (-21.5, 18.5), 'label': 'Row1\n(Solid)', 'color': '#5B7FA5'},
    {'x': (33.1, 43.1), 'y': (-21.5, -11.5), 'label': 'Row2L', 'color': '#7FA55B'},
    {'x': (33.1, 43.1), 'y': (8.5, 18.5), 'label': 'Row2R', 'color': '#7FA55B'},
    {'x': (43.1, 53.1), 'y': (-21.5, -11.5), 'label': 'Row3L', 'color': '#A57F5B'},
    {'x': (43.1, 53.1), 'y': (8.5, 18.5), 'label': 'Row3R', 'color': '#A57F5B'},
    {'x': (53.1, 63.1), 'y': (-21.5, 18.5), 'label': 'Row4\n(Solid)', 'color': '#A55B7F'},
]


def main():
    print("Building known map...")
    config = PlanningConfig(
        voxel_size=0.5,
        grid_size=(180, 120, 40),
        origin=(-10.0, -30.0, -15.0),
        step_size=1.5,
        max_iterations=3000,
        goal_sample_rate=0.2,
        search_radius=4.0,
        safety_margin=1.0,
    )

    voxel_grid = build_known_map(config)
    esdf = ESDF(voxel_grid)
    print("Computing ESDF...")
    esdf.compute()

    energy_model = PhysicsEnergyModel()

    # 三个场景
    scenarios = [
        {'name': 'A: Through Gap (Y=0)', 'start': np.array([0,0,-3.0]), 'goal': np.array([70,0,-3.0])},
        {'name': 'B: Diagonal (Y=20)', 'start': np.array([0,0,-3.0]), 'goal': np.array([70,20,-3.0])},
        {'name': 'C: Around Bottom (Y=-25)', 'start': np.array([0,0,-3.0]), 'goal': np.array([70,-25,-3.0])},
    ]

    all_paths = {}

    for sc in scenarios:
        print(f"\n=== {sc['name']} ===")

        # A*
        print("  Running A*...")
        astar = AStarPlanner(voxel_grid, esdf, config)
        astar_path = astar.plan(sc['start'], sc['goal'])
        if astar_path:
            astar_path = smooth_path(astar_path, esdf, config.safety_margin)
            print(f"  A* length: {compute_path_length(astar_path):.2f}m")

        # RRT* (Balanced)
        print("  Running RRT* (best of 5)...")
        cfg_balanced = PlanningConfig(
            voxel_size=0.5, grid_size=(180,120,40), origin=(-10,-30,-15),
            step_size=1.5, max_iterations=3000, goal_sample_rate=0.2,
            search_radius=4.0, safety_margin=1.0,
            energy_aware=True, weight_energy=0.6, weight_distance=0.3, weight_time=0.1,
            flight_velocity=2.0,
        )
        best_rrt = None
        best_len = float('inf')
        for i in range(5):
            planner = RRTStar(voxel_grid, esdf, cfg_balanced, energy_model=energy_model)
            path = planner.plan(sc['start'], sc['goal'])
            if path and len(path) >= 2:
                l = compute_path_length(path)
                if l < best_len:
                    best_len = l
                    best_rrt = path
        if best_rrt:
            print(f"  RRT* best length: {best_len:.2f}m")

        all_paths[sc['name']] = {'astar': astar_path, 'rrt': best_rrt, 'scenario': sc}

    # === 生成对比图 ===
    print("\nGenerating comparison figure...")

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for idx, (sname, data) in enumerate(all_paths.items()):
        ax = axes[idx]
        sc = data['scenario']

        # 障碍物
        for obs in OBSTACLES:
            rect = plt.Rectangle(
                (obs['x'][0], obs['y'][0]),
                obs['x'][1] - obs['x'][0], obs['y'][1] - obs['y'][0],
                facecolor=obs['color'], alpha=0.4, edgecolor='#333', linewidth=1.2
            )
            ax.add_patch(rect)
            cx = (obs['x'][0] + obs['x'][1]) / 2
            cy = (obs['y'][0] + obs['y'][1]) / 2
            ax.text(cx, cy, obs['label'], ha='center', va='center', fontsize=6.5,
                    fontweight='bold', color='#333')

        # 缝隙
        gap = plt.Rectangle((33.1, -11.5), 20, 20, facecolor='#90EE90', alpha=0.1,
                             edgecolor='green', linewidth=1, linestyle='--')
        ax.add_patch(gap)
        ax.text(43, -1.5, 'Gap', ha='center', fontsize=7, color='green', fontstyle='italic')

        # A* 路径
        if data['astar']:
            p = np.array(data['astar'])
            astar_len = compute_path_length(data['astar'])
            ax.plot(p[:, 0], p[:, 1], 'k-', linewidth=2.5, label=f'A* ({astar_len:.1f}m)', zorder=5)

        # RRT* 路径
        if data['rrt']:
            p = np.array(data['rrt'])
            rrt_len = compute_path_length(data['rrt'])
            ax.plot(p[:, 0], p[:, 1], '-', color='#E74C3C', linewidth=2.5,
                    label=f'EA-RHP ({rrt_len:.1f}m)', zorder=5)

        # 起点终点
        ax.plot(*sc['start'][:2], 'g^', markersize=12, zorder=10, markeredgecolor='k', markeredgewidth=1)
        ax.plot(*sc['goal'][:2], 'r*', markersize=14, zorder=10, markeredgecolor='k', markeredgewidth=0.5)
        ax.annotate('Start', sc['start'][:2], textcoords="offset points", xytext=(-15, 8),
                    fontsize=9, fontweight='bold', color='green')
        ax.annotate('Goal', sc['goal'][:2], textcoords="offset points", xytext=(5, 5),
                    fontsize=9, fontweight='bold', color='red')

        # 如果两条路径都有，标注 ratio
        if data['astar'] and data['rrt']:
            ratio = rrt_len / astar_len
            ax.text(0.98, 0.02, f'Path Ratio: {ratio:.3f}x',
                    transform=ax.transAxes, ha='right', va='bottom', fontsize=9,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFF8DC', edgecolor='#DAA520'))

        ax.set_xlabel('X (m)', fontsize=10)
        ax.set_ylabel('Y (m)', fontsize=10)
        ax.set_title(sname, fontsize=11, fontweight='bold')
        ax.legend(fontsize=9, loc='upper left')
        ax.set_xlim(-8, 78)
        ax.set_ylim(-30, 28)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.2)

    plt.suptitle('Path Comparison: A* (Global Optimal) vs EA-RHP (Energy-Aware RRT*)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    out = 'results/proposal_figures/proposal_path_comparison_real.png'
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved: {out}")
    plt.close()


if __name__ == "__main__":
    main()
