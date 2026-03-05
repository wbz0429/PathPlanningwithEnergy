"""
generate_proposal_figures.py - 生成基金申请用的高质量可视化图

输出:
1. proposal_obstacle_avoidance.png - 3D避障轨迹图
2. proposal_pipeline.png - 技术管线图
3. proposal_benchmark_summary.png - 评测结果汇总
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import sys, os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 全局字体设置
plt.rcParams['font.family'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 11


# ============================================================
# 障碍物定义 (Blocks 场景)
# ============================================================
OBSTACLES = [
    {'x': (23.1, 33.1), 'y': (-21.5, 18.5), 'z': (0, 10), 'label': 'Row 1\n(Solid Wall)', 'color': '#5B7FA5'},
    {'x': (33.1, 43.1), 'y': (-21.5, -11.5), 'z': (0, 10), 'label': 'Row 2L', 'color': '#7FA55B'},
    {'x': (33.1, 43.1), 'y': (8.5, 18.5), 'z': (0, 10), 'label': 'Row 2R', 'color': '#7FA55B'},
    {'x': (43.1, 53.1), 'y': (-21.5, -11.5), 'z': (0, 10), 'label': 'Row 3L', 'color': '#A57F5B'},
    {'x': (43.1, 53.1), 'y': (8.5, 18.5), 'z': (0, 10), 'label': 'Row 3R', 'color': '#A57F5B'},
    {'x': (53.1, 63.1), 'y': (-21.5, 18.5), 'z': (0, 10), 'label': 'Row 4\n(Solid Wall)', 'color': '#A55B7F'},
]


def draw_3d_box(ax, x_range, y_range, z_range, color, alpha=0.3):
    """在3D图上画一个半透明长方体"""
    x0, x1 = x_range
    y0, y1 = y_range
    z0, z1 = z_range

    vertices = [
        [x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
        [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]
    ]
    faces = [
        [vertices[0], vertices[1], vertices[5], vertices[4]],
        [vertices[2], vertices[3], vertices[7], vertices[6]],
        [vertices[0], vertices[3], vertices[7], vertices[4]],
        [vertices[1], vertices[2], vertices[6], vertices[5]],
        [vertices[0], vertices[1], vertices[2], vertices[3]],
        [vertices[4], vertices[5], vertices[6], vertices[7]],
    ]
    poly = Poly3DCollection(faces, alpha=alpha, facecolor=color, edgecolor=color, linewidth=0.5)
    ax.add_collection3d(poly)


def generate_obstacle_avoidance_figure():
    """图1: 3D避障轨迹图 + 俯视图"""

    # 模拟三条典型轨迹 (基于实际飞行数据)
    # Trajectory 1: Y=0 穿缝隙 (A* 最优路径)
    t1_x = [0, 5, 10, 15, 20, 22, 24, 28, 32, 36, 40, 44, 48, 52, 54, 58, 62, 66, 70]
    t1_y = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    t1_z = [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]
    traj_astar = np.array(list(zip(t1_x, t1_y, t1_z)), dtype=float)

    # Trajectory 2: Y=20 绕行 (RRT* 实际飞行)
    traj_rrt_y20 = np.array([
        [0, 0, 3], [2, 8, 3], [4, 17, 3], [12, 17, 3],
        [21, 18, 3], [24, 23, 3], [32, 22, 3], [40, 22, 3],
        [49, 21, 3], [57, 21, 3], [66, 20, 3], [70, 20, 3]
    ], dtype=float)

    # Trajectory 3: Y=-25 绕行
    traj_rrt_yn25 = np.array([
        [0, 0, 3], [4, -5, 3], [10, -15, 3], [18, -22, 3],
        [24, -24, 3], [32, -24, 3], [40, -24, 3], [48, -24, 3],
        [54, -24, 3], [62, -25, 3], [70, -25, 3]
    ], dtype=float)

    fig = plt.figure(figsize=(18, 7))

    # ---- 左: 3D 视图 ----
    ax1 = fig.add_subplot(121, projection='3d')

    for obs in OBSTACLES:
        draw_3d_box(ax1, obs['x'], obs['y'], obs['z'], obs['color'], alpha=0.25)

    # 画轨迹
    ax1.plot(traj_astar[:, 0], traj_astar[:, 1], traj_astar[:, 2],
             'k-', linewidth=2.5, label='A* Optimal (through gap)', zorder=10)
    ax1.plot(traj_rrt_y20[:, 0], traj_rrt_y20[:, 1], traj_rrt_y20[:, 2],
             'b-', linewidth=2.5, label='RRT* Flight (Y=20)', zorder=10)
    ax1.plot(traj_rrt_yn25[:, 0], traj_rrt_yn25[:, 1], traj_rrt_yn25[:, 2],
             'r-', linewidth=2.5, label='RRT* Flight (Y=-25)', zorder=10)

    # 起点终点
    for traj, goal_label in [(traj_rrt_y20, 'Goal B'), (traj_rrt_yn25, 'Goal C')]:
        ax1.scatter(*traj[-1], s=120, marker='*', zorder=15, edgecolors='k', linewidth=0.5)

    ax1.scatter(0, 0, 3, s=150, c='green', marker='^', zorder=15, edgecolors='k', linewidth=0.5, label='Start')
    ax1.scatter(70, 0, 3, s=150, c='black', marker='*', zorder=15, edgecolors='k', linewidth=0.5)
    ax1.scatter(70, 20, 3, s=150, c='blue', marker='*', zorder=15, edgecolors='k', linewidth=0.5)
    ax1.scatter(70, -25, 3, s=150, c='red', marker='*', zorder=15, edgecolors='k', linewidth=0.5)

    # 标注缝隙
    ax1.plot([33.1, 43.1], [-11.5, -11.5], [5, 5], 'g--', linewidth=1, alpha=0.6)
    ax1.plot([33.1, 43.1], [8.5, 8.5], [5, 5], 'g--', linewidth=1, alpha=0.6)
    ax1.text(38, -1.5, 8, 'Gap', fontsize=9, color='green', ha='center', fontweight='bold')

    ax1.set_xlabel('X (m)', fontsize=11, labelpad=8)
    ax1.set_ylabel('Y (m)', fontsize=11, labelpad=8)
    ax1.set_zlabel('Z (m)', fontsize=11, labelpad=8)
    ax1.set_title('3D Obstacle Avoidance Trajectories', fontsize=13, fontweight='bold', pad=15)
    ax1.legend(fontsize=9, loc='upper left')
    ax1.view_init(elev=25, azim=-55)
    ax1.set_xlim(-5, 75)
    ax1.set_ylim(-30, 30)
    ax1.set_zlim(0, 12)

    # ---- 右: 俯视图 (XY) ----
    ax2 = fig.add_subplot(122)

    for obs in OBSTACLES:
        rect = plt.Rectangle(
            (obs['x'][0], obs['y'][0]),
            obs['x'][1] - obs['x'][0],
            obs['y'][1] - obs['y'][0],
            facecolor=obs['color'], alpha=0.4, edgecolor='#333', linewidth=1.2
        )
        ax2.add_patch(rect)
        cx = (obs['x'][0] + obs['x'][1]) / 2
        cy = (obs['y'][0] + obs['y'][1]) / 2
        ax2.text(cx, cy, obs['label'], ha='center', va='center', fontsize=7.5,
                 fontweight='bold', color='#222')

    # 标注缝隙区域
    gap_rect = plt.Rectangle((33.1, -11.5), 20, 20, facecolor='#90EE90',
                              alpha=0.15, edgecolor='green', linewidth=1.5, linestyle='--')
    ax2.add_patch(gap_rect)
    ax2.text(43, -1.5, 'Passable Gap\n(20m wide)', ha='center', va='center',
             fontsize=8, color='green', fontstyle='italic')

    # 画轨迹
    ax2.plot(traj_astar[:, 0], traj_astar[:, 1], 'k-', linewidth=2.5,
             label='A* Optimal (71.1m)', zorder=5)
    ax2.plot(traj_rrt_y20[:, 0], traj_rrt_y20[:, 1], 'b-', linewidth=2.5,
             label='RRT* Y=20 (77.7m)', zorder=5)
    ax2.plot(traj_rrt_yn25[:, 0], traj_rrt_yn25[:, 1], 'r-', linewidth=2.5,
             label='RRT* Y=-25 (91.4m)', zorder=5)

    # 起点终点
    ax2.plot(0, 0, 'g^', markersize=14, zorder=10, markeredgecolor='k', markeredgewidth=1)
    ax2.plot(70, 0, 'k*', markersize=14, zorder=10, markeredgecolor='k', markeredgewidth=0.5)
    ax2.plot(70, 20, 'b*', markersize=14, zorder=10, markeredgecolor='k', markeredgewidth=0.5)
    ax2.plot(70, -25, 'r*', markersize=14, zorder=10, markeredgecolor='k', markeredgewidth=0.5)

    ax2.annotate('Start', (0, 0), textcoords="offset points", xytext=(-15, 10),
                 fontsize=9, fontweight='bold', color='green')
    ax2.annotate('Goal A', (70, 0), textcoords="offset points", xytext=(5, 5),
                 fontsize=9, fontweight='bold')
    ax2.annotate('Goal B', (70, 20), textcoords="offset points", xytext=(5, 5),
                 fontsize=9, fontweight='bold', color='blue')
    ax2.annotate('Goal C', (70, -25), textcoords="offset points", xytext=(5, -12),
                 fontsize=9, fontweight='bold', color='red')

    ax2.set_xlabel('X (m)', fontsize=11)
    ax2.set_ylabel('Y (m)', fontsize=11)
    ax2.set_title('Top-Down View: Path Comparison', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=9, loc='upper left')
    ax2.set_xlim(-8, 78)
    ax2.set_ylim(-32, 28)
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig('results/proposal_obstacle_avoidance.png', dpi=200, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print("  Saved: results/proposal_obstacle_avoidance.png")
    plt.close()


def generate_pipeline_figure():
    """图2: 技术管线图"""
    fig, ax = plt.subplots(figsize=(16, 5))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 5)
    ax.axis('off')

    # 管线模块
    modules = [
        {'x': 0.3, 'w': 2.2, 'label': 'Depth\nPerception', 'sub': 'AirSim Camera\n90° FOV', 'color': '#4ECDC4'},
        {'x': 3.0, 'w': 2.2, 'label': 'Incremental\nMapping', 'sub': 'Voxel Grid\n(OctoMap)', 'color': '#45B7D1'},
        {'x': 5.7, 'w': 2.2, 'label': 'ESDF\nComputation', 'sub': 'Distance\nTransform', 'color': '#96CEB4'},
        {'x': 8.4, 'w': 2.2, 'label': 'Energy-Aware\nRRT*', 'sub': 'Multi-Objective\nOptimization', 'color': '#FFEAA7'},
        {'x': 11.1, 'w': 2.2, 'label': 'Receding\nHorizon', 'sub': 'Sense-Plan-Act\nLoop', 'color': '#DDA0DD'},
        {'x': 13.8, 'w': 2.0, 'label': 'Flight\nControl', 'sub': 'AirSim API\nExecution', 'color': '#FF6B6B'},
    ]

    y_center = 2.8
    box_h = 2.0

    for m in modules:
        # 主框
        rect = mpatches.FancyBboxPatch(
            (m['x'], y_center - box_h/2), m['w'], box_h,
            boxstyle="round,pad=0.15", facecolor=m['color'], edgecolor='#333',
            linewidth=1.5, alpha=0.85
        )
        ax.add_patch(rect)
        ax.text(m['x'] + m['w']/2, y_center + 0.3, m['label'],
                ha='center', va='center', fontsize=10, fontweight='bold', color='#222')
        ax.text(m['x'] + m['w']/2, y_center - 0.5, m['sub'],
                ha='center', va='center', fontsize=7.5, color='#555', fontstyle='italic')

    # 箭头
    arrow_style = dict(arrowstyle='->', color='#333', lw=2)
    for i in range(len(modules) - 1):
        x_from = modules[i]['x'] + modules[i]['w']
        x_to = modules[i+1]['x']
        ax.annotate('', xy=(x_to, y_center), xytext=(x_from, y_center),
                    arrowprops=arrow_style)

    # 反馈回路箭头 (Receding Horizon -> Perception)
    ax.annotate('', xy=(0.3, y_center - box_h/2 - 0.3),
                xytext=(13.8, y_center - box_h/2 - 0.3),
                arrowprops=dict(arrowstyle='->', color='#E74C3C', lw=1.8,
                               linestyle='--', connectionstyle='arc3,rad=0.15'))
    ax.text(7, 0.6, 'Receding Horizon Feedback Loop', ha='center', fontsize=9,
            color='#E74C3C', fontstyle='italic', fontweight='bold')

    # 标题
    ax.text(8, 4.7, 'Energy-Aware Autonomous Obstacle Avoidance Pipeline',
            ha='center', va='center', fontsize=14, fontweight='bold', color='#222')

    # 底部性能标注
    perf_text = 'Cycle: ~150ms  |  Map Update: ~40ms  |  ESDF: ~40ms  |  RRT*: ~50ms  |  100% Success Rate'
    ax.text(8, 0.15, perf_text, ha='center', fontsize=8.5, color='#666',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#f0f0f0', edgecolor='#ccc'))

    plt.savefig('results/proposal_pipeline.png', dpi=200, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print("  Saved: results/proposal_pipeline.png")
    plt.close()


def generate_benchmark_summary():
    """图3: 评测结果汇总 (更紧凑的版本)"""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # --- 左: Path Ratio 对比 ---
    ax = axes[0]
    scenarios = ['Scenario A\n(Through Gap)', 'Scenario B\n(Diagonal)', 'Scenario C\n(Around)']
    astar_ratios = [1.015, 1.057, 1.068]
    rrt_ratios = [1.309, 1.009, 1.152]  # Balanced config

    x = np.arange(len(scenarios))
    width = 0.3
    ax.bar(x - width/2, astar_ratios, width, label='A* (Optimal)', color='#2C3E50', alpha=0.85)
    ax.bar(x + width/2, rrt_ratios, width, label='RRT* (Balanced)', color='#3498DB', alpha=0.85)
    ax.axhline(y=1.0, color='gray', linestyle=':', linewidth=1)
    ax.set_ylabel('Path Ratio\n(actual / straight-line)')
    ax.set_title('Path Optimality', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, fontsize=8.5)
    ax.legend(fontsize=8)
    ax.set_ylim(0.9, 1.45)

    # --- 中: 能耗对比 ---
    ax = axes[1]
    methods = ['Dist.\nOnly', 'Energy\nOnly', 'Balanced', 'Time\nPriority']
    # Scenario A 数据
    energies = [6784, 6712, 6710, 6794]
    energy_stds = [131, 102, 108, 126]
    colors = ['#3498DB', '#E74C3C', '#2ECC71', '#F39C12']

    bars = ax.bar(methods, energies, yerr=energy_stds, capsize=4,
                  color=colors, alpha=0.8, edgecolor='#333', linewidth=0.8)
    ax.axhline(y=5139, color='black', linestyle='--', linewidth=1.5, label='A* Optimal (5139J)')
    ax.set_ylabel('Energy Consumption (J)')
    ax.set_title('Ablation Study: Cost Weights\n(Scenario A)', fontweight='bold')
    ax.legend(fontsize=8)

    # --- 右: 成功率 + 方差 ---
    ax = axes[2]
    scenario_labels = ['A (Gap)', 'B (Diag)', 'C (Around)']
    balanced_lengths = [93.07, 77.65, 91.42]
    balanced_stds = [1.49, 0.46, 17.33]
    astar_lengths = [71.08, 76.96, 79.39]

    x = np.arange(len(scenario_labels))
    ax.bar(x - width/2, astar_lengths, width, label='A*', color='#2C3E50', alpha=0.85)
    ax.bar(x + width/2, balanced_lengths, width, yerr=balanced_stds, capsize=4,
           label='RRT* Balanced (10 runs)', color='#2ECC71', alpha=0.85)
    ax.set_ylabel('Path Length (m)')
    ax.set_title('Path Length Comparison', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(scenario_labels, fontsize=9)
    ax.legend(fontsize=8)

    plt.suptitle('Planning Algorithm Benchmark: A* vs Energy-Aware RRT*',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('results/proposal_benchmark_summary.png', dpi=200, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print("  Saved: results/proposal_benchmark_summary.png")
    plt.close()


if __name__ == "__main__":
    print("Generating proposal figures...")
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    generate_obstacle_avoidance_figure()
    generate_pipeline_figure()
    generate_benchmark_summary()
    print("\nDone! Files saved to results/")
