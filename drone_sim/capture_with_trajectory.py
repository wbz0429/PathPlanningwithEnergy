"""
capture_with_trajectory.py - 带轨迹的避障截图
飞行过程中截取 RGB 图，最后生成带轨迹标注的汇总图
"""

import airsim
import numpy as np
import cv2
import time
import os
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from control.drone_interface import DroneInterface

OBSTACLES = [
    {'x': (23.1, 33.1), 'y': (-21.5, 18.5), 'label': 'Row1 (Wall)', 'color': '#5B7FA5'},
    {'x': (33.1, 43.1), 'y': (-21.5, -11.5), 'label': 'Row2L', 'color': '#7FA55B'},
    {'x': (33.1, 43.1), 'y': (8.5, 18.5), 'label': 'Row2R', 'color': '#7FA55B'},
    {'x': (43.1, 53.1), 'y': (-21.5, -11.5), 'label': 'Row3L', 'color': '#A57F5B'},
    {'x': (43.1, 53.1), 'y': (8.5, 18.5), 'label': 'Row3R', 'color': '#A57F5B'},
    {'x': (53.1, 63.1), 'y': (-21.5, 18.5), 'label': 'Row4 (Wall)', 'color': '#A55B7F'},
]

save_dir = 'results/proposal_figures/obstacle_avoidance_trajectory'


def capture_rgb(client, label):
    """截取 RGB 图"""
    responses = client.simGetImages([
        airsim.ImageRequest("0", airsim.ImageType.Scene, False, False),
    ])
    img = np.frombuffer(responses[0].image_data_uint8, dtype=np.uint8)
    img = img.reshape(responses[0].height, responses[0].width, 3)
    path = os.path.join(save_dir, f'{label}.png')
    cv2.imwrite(path, img)
    return img


def capture_depth_float(client):
    """截取浮点深度图用于伪彩色"""
    responses = client.simGetImages([
        airsim.ImageRequest("0", airsim.ImageType.DepthPlanar, True),
    ])
    depth = np.array(responses[0].image_data_float, dtype=np.float32)
    depth = depth.reshape(responses[0].height, responses[0].width)
    return depth


def depth_to_colormap(depth, max_dist=30.0):
    """深度图转伪彩色"""
    depth_clipped = np.clip(depth, 0, max_dist)
    depth_norm = (depth_clipped / max_dist * 255).astype(np.uint8)
    colored = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)
    return colored


def main():
    os.makedirs(save_dir, exist_ok=True)

    print("=" * 60)
    print("  Obstacle Avoidance with Trajectory Capture")
    print("=" * 60)

    drone = DroneInterface()
    drone.connect()
    client = drone.client

    try:
        drone.reset()
        time.sleep(2)
        drone.takeoff()
        time.sleep(1)
        drone.move_to_z(-3.0, velocity=2.0)
        time.sleep(2)

        # 沿实际避障路径飞行的航点
        waypoints = [
            (0, 0, -3),
            (5, 4, -3),
            (10, 10, -3),
            (15, 15, -3),
            (18, 17, -3),
            (21, 19, -3),      # 接近 Row1
            (24, 23, -3),      # 绕行 Row1 上方
            (28, 22.5, -3),
            (32, 22, -3),      # 过 Row1
            (36, 22, -3),
            (40, 22, -3),      # Row2/Row3 之间
            (45, 21.5, -3),
            (50, 21, -3),
            (55, 21, -3),
            (60, 20.5, -3),
            (65, 20, -3),
            (70, 20, -3),      # 目标
        ]

        trajectory = []
        key_captures = {}  # 关键帧截图

        # 定义关键帧
        key_indices = {
            0: 'A_start',
            5: 'B_detect',
            6: 'C_avoid',
            8: 'D_pass_row1',
            10: 'E_clear',
            16: 'F_goal',
        }

        for i, (x, y, z) in enumerate(waypoints):
            print(f"  [{i+1}/{len(waypoints)}] -> ({x}, {y}, {z})")
            client.moveToPositionAsync(x, y, z, 2.5).join()
            time.sleep(0.8)

            # 获取实际位置
            state = client.getMultirotorState()
            pos = state.kinematics_estimated.position
            trajectory.append((pos.x_val, pos.y_val, pos.z_val))

            # 关键帧截图
            if i in key_indices:
                label = key_indices[i]
                # 朝向下一个航点
                if i < len(waypoints) - 1:
                    nx, ny, nz = waypoints[i + 1]
                    yaw = np.arctan2(ny - y, nx - x) * 180 / np.pi
                    client.rotateToYawAsync(yaw).join()
                    time.sleep(0.3)

                rgb = capture_rgb(client, f'{label}_rgb')
                depth = capture_depth_float(client)
                depth_color = depth_to_colormap(depth)
                cv2.imwrite(os.path.join(save_dir, f'{label}_depth.png'), depth_color)
                key_captures[label] = {'rgb': rgb, 'pos': (pos.x_val, pos.y_val)}
                print(f"    Captured: {label}")

        print("\nFlight complete. Generating figures...")

        # === 图1: 俯视轨迹图 + 关键帧标注 ===
        generate_trajectory_figure(trajectory, key_captures)

        # === 图2: 关键帧序列 (RGB + Depth) ===
        generate_keyframe_sequence(key_captures)

        # === 图3: 综合大图 (轨迹 + 关键帧) ===
        generate_combined_figure(trajectory, key_captures)

        drone.land()

    except KeyboardInterrupt:
        print("\nInterrupted!")
        drone.land()
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        drone.land()
    finally:
        drone.disconnect()

    print(f"\nAll images saved to: {save_dir}/")


def generate_trajectory_figure(trajectory, key_captures):
    """俯视轨迹图"""
    fig, ax = plt.subplots(figsize=(12, 6))

    # 障碍物
    for obs in OBSTACLES:
        rect = plt.Rectangle(
            (obs['x'][0], obs['y'][0]),
            obs['x'][1] - obs['x'][0], obs['y'][1] - obs['y'][0],
            facecolor=obs['color'], alpha=0.4, edgecolor='#333', linewidth=1.5
        )
        ax.add_patch(rect)
        cx = (obs['x'][0] + obs['x'][1]) / 2
        cy = (obs['y'][0] + obs['y'][1]) / 2
        ax.text(cx, cy, obs['label'], ha='center', va='center', fontsize=7, fontweight='bold', color='#333')

    # 缝隙标注
    gap = plt.Rectangle((33.1, -11.5), 20, 20, facecolor='#90EE90', alpha=0.1,
                         edgecolor='green', linewidth=1.2, linestyle='--')
    ax.add_patch(gap)
    ax.text(43, -1.5, 'Gap (20m)', ha='center', fontsize=7, color='green', fontstyle='italic')

    # 轨迹
    traj = np.array(trajectory)
    # 用颜色渐变表示时间
    for i in range(len(traj) - 1):
        t = i / (len(traj) - 1)
        color = plt.cm.cool(t)
        ax.plot(traj[i:i+2, 0], traj[i:i+2, 1], '-', color=color, linewidth=3, solid_capstyle='round')

    # 关键帧标注
    markers = {'A_start': ('Start', 'green', '^'),
               'B_detect': ('Detect', '#E67E22', 'D'),
               'C_avoid': ('Avoid', 'red', 'o'),
               'D_pass_row1': ('Pass R1', '#3498DB', 's'),
               'E_clear': ('Clear', '#9B59B6', 'p'),
               'F_goal': ('Goal', 'red', '*')}

    for label, (text, color, marker) in markers.items():
        if label in key_captures:
            px, py = key_captures[label]['pos']
            ax.plot(px, py, marker=marker, color=color, markersize=12,
                    markeredgecolor='black', markeredgewidth=1, zorder=10)
            ax.annotate(text, (px, py), textcoords="offset points",
                       xytext=(8, 8), fontsize=9, fontweight='bold', color=color,
                       bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8, edgecolor=color))

    ax.set_xlabel('X (m)', fontsize=11)
    ax.set_ylabel('Y (m)', fontsize=11)
    ax.set_title('UAV Obstacle Avoidance Trajectory (Top-Down View)', fontsize=13, fontweight='bold')
    ax.set_xlim(-5, 78)
    ax.set_ylim(-28, 30)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.2)

    # colorbar 表示时间
    sm = plt.cm.ScalarMappable(cmap='cool', norm=plt.Normalize(0, 1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.6, pad=0.02)
    cbar.set_label('Flight Progress', fontsize=9)
    cbar.set_ticks([0, 0.5, 1])
    cbar.set_ticklabels(['Start', 'Mid', 'Goal'])

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'trajectory_topdown.png'), dpi=150,
                bbox_inches='tight', facecolor='white')
    print("  Saved: trajectory_topdown.png")
    plt.close()


def generate_keyframe_sequence(key_captures):
    """关键帧序列图"""
    labels_order = ['A_start', 'B_detect', 'C_avoid', 'D_pass_row1', 'E_clear', 'F_goal']
    titles = ['1. Start', '2. Obstacle\nDetected', '3. Avoiding\nWall', '4. Passed\nRow1', '5. Clear\nPath', '6. Goal\nReached']

    fig, axes = plt.subplots(2, 6, figsize=(16, 5.5))

    for i, (label, title) in enumerate(zip(labels_order, titles)):
        # RGB
        rgb_path = os.path.join(save_dir, f'{label}_rgb.png')
        if os.path.exists(rgb_path):
            img = cv2.imread(rgb_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            axes[0, i].imshow(img)
        axes[0, i].set_title(title, fontsize=9, fontweight='bold')
        axes[0, i].axis('off')

        # Depth
        depth_path = os.path.join(save_dir, f'{label}_depth.png')
        if os.path.exists(depth_path):
            img = cv2.imread(depth_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            axes[1, i].imshow(img)
        axes[1, i].axis('off')

    axes[0, 0].set_ylabel('RGB', fontsize=10, fontweight='bold')
    axes[1, 0].set_ylabel('Depth', fontsize=10, fontweight='bold')

    plt.suptitle('Obstacle Avoidance Sequence: First-Person View (RGB + Depth)',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'keyframe_sequence.png'), dpi=150,
                bbox_inches='tight', facecolor='white')
    print("  Saved: keyframe_sequence.png")
    plt.close()


def generate_combined_figure(trajectory, key_captures):
    """综合大图: 左边轨迹，右边关键帧"""
    labels_order = ['A_start', 'B_detect', 'C_avoid', 'D_pass_row1', 'E_clear', 'F_goal']
    titles = ['Start', 'Detect', 'Avoid', 'Pass R1', 'Clear', 'Goal']

    fig = plt.figure(figsize=(16, 8))

    # 左: 轨迹图 (占 2/3)
    ax_traj = fig.add_axes([0.02, 0.05, 0.55, 0.88])

    for obs in OBSTACLES:
        rect = plt.Rectangle(
            (obs['x'][0], obs['y'][0]),
            obs['x'][1] - obs['x'][0], obs['y'][1] - obs['y'][0],
            facecolor=obs['color'], alpha=0.35, edgecolor='#333', linewidth=1.2
        )
        ax_traj.add_patch(rect)

    traj = np.array(trajectory)
    for i in range(len(traj) - 1):
        t = i / (len(traj) - 1)
        ax_traj.plot(traj[i:i+2, 0], traj[i:i+2, 1], '-', color=plt.cm.cool(t),
                     linewidth=3.5, solid_capstyle='round')

    # 标注关键帧位置并画连线到右侧
    marker_colors = ['green', '#E67E22', 'red', '#3498DB', '#9B59B6', 'red']
    marker_shapes = ['^', 'D', 'o', 's', 'p', '*']
    marker_sizes = [12, 10, 10, 10, 10, 14]

    for i, (label, title) in enumerate(zip(labels_order, titles)):
        if label in key_captures:
            px, py = key_captures[label]['pos']
            ax_traj.plot(px, py, marker=marker_shapes[i], color=marker_colors[i],
                        markersize=marker_sizes[i], markeredgecolor='black',
                        markeredgewidth=1, zorder=10)
            ax_traj.annotate(f'{i+1}', (px, py), textcoords="offset points",
                           xytext=(-5, 8), fontsize=8, fontweight='bold',
                           color='white',
                           bbox=dict(boxstyle='circle,pad=0.2', facecolor=marker_colors[i],
                                    edgecolor='black', alpha=0.9))

    ax_traj.set_xlabel('X (m)', fontsize=11)
    ax_traj.set_ylabel('Y (m)', fontsize=11)
    ax_traj.set_title('Flight Trajectory with Obstacle Avoidance', fontsize=12, fontweight='bold')
    ax_traj.set_xlim(-5, 78)
    ax_traj.set_ylim(-28, 30)
    ax_traj.set_aspect('equal')
    ax_traj.grid(True, alpha=0.2)

    # 右: 关键帧 RGB (3行2列)
    for i, (label, title) in enumerate(zip(labels_order, titles)):
        row = i // 2
        col = i % 2
        ax = fig.add_axes([0.60 + col * 0.20, 0.68 - row * 0.32, 0.18, 0.28])

        rgb_path = os.path.join(save_dir, f'{label}_rgb.png')
        if os.path.exists(rgb_path):
            img = cv2.imread(rgb_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            ax.imshow(img)

        ax.set_title(f'{i+1}. {title}', fontsize=9, fontweight='bold', color=marker_colors[i])
        ax.axis('off')
        # 边框颜色
        for spine in ax.spines.values():
            spine.set_edgecolor(marker_colors[i])
            spine.set_linewidth(2)
            spine.set_visible(True)

    plt.savefig(os.path.join(save_dir, 'combined_trajectory_views.png'), dpi=150,
                bbox_inches='tight', facecolor='white')
    print("  Saved: combined_trajectory_views.png")
    plt.close()


if __name__ == "__main__":
    main()
