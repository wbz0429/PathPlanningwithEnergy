"""
capture_obstacle_avoidance.py - 避障过程截图
在关键位置截取 RGB 场景图 + 深度图，展示避障过程
"""

import airsim
import numpy as np
import cv2
import time
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from control.drone_interface import DroneInterface


def capture_images(client, label, save_dir):
    """截取 RGB + 深度图"""
    responses = client.simGetImages([
        airsim.ImageRequest("0", airsim.ImageType.Scene, False, False),  # RGB
        airsim.ImageRequest("0", airsim.ImageType.DepthVis, False, False),  # 深度可视化
    ])

    # RGB
    img_rgb = np.frombuffer(responses[0].image_data_uint8, dtype=np.uint8)
    img_rgb = img_rgb.reshape(responses[0].height, responses[0].width, 3)
    cv2.imwrite(os.path.join(save_dir, f'{label}_rgb.png'), img_rgb)

    # Depth visualization
    img_depth = np.frombuffer(responses[1].image_data_uint8, dtype=np.uint8)
    img_depth = img_depth.reshape(responses[1].height, responses[1].width, 3)
    cv2.imwrite(os.path.join(save_dir, f'{label}_depth.png'), img_depth)

    print(f"  Captured: {label}")
    return img_rgb, img_depth


def main():
    save_dir = 'results/proposal_figures/obstacle_avoidance_sequence'
    os.makedirs(save_dir, exist_ok=True)

    print("=" * 60)
    print("  Obstacle Avoidance Image Capture")
    print("=" * 60)

    drone = DroneInterface()
    drone.connect()
    client = drone.client

    try:
        # 重置
        drone.reset()
        time.sleep(2)

        # 起飞
        drone.takeoff()
        time.sleep(1)
        drone.move_to_z(-3.0, velocity=2.0)
        time.sleep(2)

        # === 关键位置截图序列 ===
        waypoints = [
            {'pos': (0, 0, -3), 'label': '01_start', 'desc': 'Starting position - clear view ahead'},
            {'pos': (10, 5, -3), 'label': '02_approach', 'desc': 'Approaching obstacle zone'},
            {'pos': (18, 15, -3), 'label': '03_detect', 'desc': 'Obstacle detected - Row1 wall ahead'},
            {'pos': (22, 19, -3), 'label': '04_wall_close', 'desc': 'Close to Row1 wall - planning avoidance'},
            {'pos': (24, 23, -3), 'label': '05_avoid', 'desc': 'Avoiding Row1 - moving around top edge'},
            {'pos': (32, 22, -3), 'label': '06_pass_row1', 'desc': 'Passed Row1 wall'},
            {'pos': (40, 22, -3), 'label': '07_between_rows', 'desc': 'Between Row2/Row3 - clear path'},
            {'pos': (55, 21, -3), 'label': '08_approach_row4', 'desc': 'Approaching Row4'},
            {'pos': (64, 20, -3), 'label': '09_pass_row4', 'desc': 'Passed all obstacles'},
            {'pos': (70, 20, -3), 'label': '10_goal', 'desc': 'Goal reached'},
        ]

        # 记录所有位置用于轨迹
        positions = []

        for i, wp in enumerate(waypoints):
            x, y, z = wp['pos']
            print(f"\n[{i+1}/{len(waypoints)}] {wp['desc']}")
            print(f"  Flying to ({x}, {y}, {z})...")

            # 飞到目标位置
            client.moveToPositionAsync(x, y, z, 2.0).join()
            time.sleep(1.5)  # 等稳定

            # 调整朝向：面向前进方向
            if i < len(waypoints) - 1:
                nx, ny, nz = waypoints[i+1]['pos']
                yaw = np.arctan2(ny - y, nx - x) * 180 / np.pi
            else:
                yaw = 0  # 最后一个点朝前

            client.rotateToYawAsync(yaw).join()
            time.sleep(0.5)

            # 获取实际位置
            state = client.getMultirotorState()
            pos = state.kinematics_estimated.position
            positions.append((pos.x_val, pos.y_val, pos.z_val))
            print(f"  Actual pos: ({pos.x_val:.1f}, {pos.y_val:.1f}, {pos.z_val:.1f})")

            # 截图
            capture_images(client, wp['label'], save_dir)

        # === 生成汇总拼图 ===
        print("\n\nGenerating summary montage...")
        generate_montage(save_dir, waypoints, positions)

        # 降落
        print("\nLanding...")
        drone.land()

    except KeyboardInterrupt:
        print("\nInterrupted! Landing...")
        drone.land()
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        drone.land()
    finally:
        drone.disconnect()

    print(f"\nImages saved to: {save_dir}/")


def generate_montage(save_dir, waypoints, positions):
    """生成避障过程汇总拼图"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # 选取关键帧: 起飞、检测障碍、避障中、通过、到达
    key_frames = ['01_start', '03_detect', '05_avoid', '07_between_rows', '10_goal']
    key_labels = ['Start', 'Obstacle\nDetected', 'Avoiding\nWall', 'Through\nGap', 'Goal\nReached']

    fig, axes = plt.subplots(2, 5, figsize=(18, 7))

    for i, (frame, label) in enumerate(zip(key_frames, key_labels)):
        # RGB
        rgb_path = os.path.join(save_dir, f'{frame}_rgb.png')
        if os.path.exists(rgb_path):
            img = cv2.imread(rgb_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            axes[0, i].imshow(img)
        axes[0, i].set_title(label, fontsize=11, fontweight='bold')
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_ylabel('RGB View', fontsize=11, fontweight='bold')

        # Depth
        depth_path = os.path.join(save_dir, f'{frame}_depth.png')
        if os.path.exists(depth_path):
            img = cv2.imread(depth_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            axes[1, i].imshow(img)
        axes[1, i].axis('off')
        if i == 0:
            axes[1, i].set_ylabel('Depth View', fontsize=11, fontweight='bold')

    plt.suptitle('Obstacle Avoidance Sequence: RGB + Depth Perception',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'avoidance_montage.png'),
                dpi=150, bbox_inches='tight', facecolor='white')
    print(f"  Saved: avoidance_montage.png")
    plt.close()


if __name__ == "__main__":
    main()
