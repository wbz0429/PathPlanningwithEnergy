"""
scan_scene.py - 扫描AirSim场景，显示障碍物位置

自动飞行扫描场景，生成障碍物地图
"""

import airsim
import numpy as np
import time
import sys

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')


def scan_direction(client, x, y, z):
    """飞到指定位置并扫描前方"""
    client.moveToPositionAsync(x, y, z, 3).join()
    time.sleep(0.5)

    # 获取深度图
    responses = client.simGetImages([
        airsim.ImageRequest('0', airsim.ImageType.DepthPlanar, True)
    ])

    depth = np.array(responses[0].image_data_float, dtype=np.float32)
    depth = depth.reshape(responses[0].height, responses[0].width)

    # 获取中心深度
    h, w = depth.shape
    center_depth = depth[h//2, w//2]
    min_depth = depth[depth > 0.5].min() if (depth > 0.5).any() else 100

    return center_depth, min_depth


def main():
    print("="*70)
    print("AirSim Scene Scanner")
    print("="*70)

    client = airsim.MultirotorClient()
    client.confirmConnection()
    client.enableApiControl(True)
    client.armDisarm(True)

    print("Connected! Taking off...")
    client.takeoffAsync().join()

    flight_height = -5  # 5米高度
    client.moveToZAsync(flight_height, 2).join()

    print(f"Flying at altitude: {-flight_height}m")
    print()

    # 扫描结果
    obstacles = []
    free_areas = []

    # 扫描前方不同距离
    print("="*70)
    print("Scanning forward (X direction)...")
    print("="*70)
    print(f"{'X':>6} {'Y':>6} {'Center Depth':>14} {'Min Depth':>12} {'Status'}")
    print("-"*60)

    for x in range(0, 35, 5):
        center_d, min_d = scan_direction(client, x, 0, flight_height)

        if min_d < 5:
            status = "OBSTACLE!"
            obstacles.append((x + min_d, 0, flight_height))
        else:
            status = "clear"
            free_areas.append((x, 0, flight_height))

        print(f"{x:>6} {0:>6} {center_d:>14.1f} {min_d:>12.1f} {status}")

        if min_d < 3:
            print(f"  -> Obstacle detected at X≈{x + min_d:.1f}m, stopping forward scan")
            break

    # 扫描右侧 (Y+)
    print()
    print("="*70)
    print("Scanning right side (Y+ direction)...")
    print("="*70)

    # 先回到起点
    client.moveToPositionAsync(0, 0, flight_height, 3).join()

    print(f"{'X':>6} {'Y':>6} {'Center Depth':>14} {'Min Depth':>12} {'Status'}")
    print("-"*60)

    for y in range(0, 30, 5):
        # 飞到侧面位置，朝向X+
        client.moveToPositionAsync(5, y, flight_height, 3).join()
        center_d, min_d = scan_direction(client, 10, y, flight_height)

        if min_d < 5:
            status = "OBSTACLE"
        else:
            status = "clear"
            free_areas.append((15, y, flight_height))

        print(f"{10:>6} {y:>6} {center_d:>14.1f} {min_d:>12.1f} {status}")

    # 扫描左侧 (Y-)
    print()
    print("="*70)
    print("Scanning left side (Y- direction)...")
    print("="*70)

    client.moveToPositionAsync(0, 0, flight_height, 3).join()

    print(f"{'X':>6} {'Y':>6} {'Center Depth':>14} {'Min Depth':>12} {'Status'}")
    print("-"*60)

    for y in range(0, -25, -5):
        client.moveToPositionAsync(5, y, flight_height, 3).join()
        center_d, min_d = scan_direction(client, 10, y, flight_height)

        if min_d < 5:
            status = "OBSTACLE"
        else:
            status = "clear"
            free_areas.append((15, y, flight_height))

        print(f"{10:>6} {y:>6} {center_d:>14.1f} {min_d:>12.1f} {status}")

    # 返回起点并降落
    print()
    print("Returning to start and landing...")
    client.moveToPositionAsync(0, 0, flight_height, 3).join()
    client.landAsync().join()
    client.armDisarm(False)
    client.enableApiControl(False)

    # 打印结果
    print()
    print("="*70)
    print("SCAN RESULTS")
    print("="*70)

    print()
    print("Detected obstacles near:")
    for obs in obstacles:
        print(f"  X≈{obs[0]:.1f}, Y≈{obs[1]:.1f}")

    print()
    print("Safe areas (potential goals):")
    for area in free_areas[-5:]:  # 最后5个
        print(f"  ({area[0]:.1f}, {area[1]:.1f}, {area[2]:.1f})")

    print()
    print("="*70)
    print("RECOMMENDED GOAL POINTS")
    print("="*70)
    print()

    # 根据扫描结果推荐目标点
    if free_areas:
        # 找一个需要绕障的目标
        for area in free_areas:
            if area[0] > 10 and abs(area[1]) > 5:
                print(f"Goal requiring obstacle avoidance:")
                print(f"  global_goal = np.array([{area[0]:.1f}, {area[1]:.1f}, {area[2]:.1f}])")
                break

    print()
    print("Simple goals (no obstacles):")
    print("  global_goal = np.array([10.0, 0.0, -5.0])")
    print("  global_goal = np.array([5.0, 10.0, -5.0])")


if __name__ == "__main__":
    main()
