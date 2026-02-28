"""
diagnose_perception.py - 诊断感知和避障问题

让无人机起飞，朝前方飞行，同时记录：
1. 深度图的实际覆盖范围（FOV）
2. 点云转换后的世界坐标分布
3. 地图中障碍物的实际位置
4. ESDF 在前方路径上的距离值

用于定位"感知+规划了但还是撞"的根本原因
"""

import numpy as np
import time
import sys
import os
import airsim

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from planning.config import PlanningConfig
from mapping.incremental_map import IncrementalMapManager
from control.drone_interface import DroneInterface
from utils.transforms import depth_image_to_camera_points, transform_camera_to_world


def analyze_depth_image(depth_image):
    """分析深度图的基本统计"""
    valid = depth_image[(depth_image > 0.1) & (depth_image < 100)]
    if len(valid) == 0:
        return {'valid_pixels': 0}

    h, w = depth_image.shape
    # 中心区域
    center_h, center_w = h // 2, w // 2
    center_region = depth_image[center_h-20:center_h+20, center_w-20:center_w+20]
    center_valid = center_region[(center_region > 0.1) & (center_region < 100)]

    return {
        'image_size': f'{w}x{h}',
        'valid_pixels': len(valid),
        'total_pixels': h * w,
        'valid_ratio': f'{len(valid) / (h*w) * 100:.1f}%',
        'min_depth': f'{valid.min():.2f}m',
        'max_depth': f'{valid.max():.2f}m',
        'mean_depth': f'{valid.mean():.2f}m',
        'center_min': f'{center_valid.min():.2f}m' if len(center_valid) > 0 else 'N/A',
        'center_mean': f'{center_valid.mean():.2f}m' if len(center_valid) > 0 else 'N/A',
    }


def analyze_point_cloud(points_world, drone_pos):
    """分析世界坐标系点云的分布"""
    if len(points_world) == 0:
        return {'num_points': 0}

    relative = points_world - drone_pos

    return {
        'num_points': len(points_world),
        'X_range': f'[{relative[:, 0].min():.1f}, {relative[:, 0].max():.1f}]m (forward)',
        'Y_range': f'[{relative[:, 1].min():.1f}, {relative[:, 1].max():.1f}]m (right)',
        'Z_range': f'[{relative[:, 2].min():.1f}, {relative[:, 2].max():.1f}]m (down)',
        'X_mean': f'{relative[:, 0].mean():.1f}m',
        'Y_mean': f'{relative[:, 1].mean():.1f}m',
        'Z_mean': f'{relative[:, 2].mean():.1f}m',
    }


def check_esdf_ahead(esdf, drone_pos, direction, max_dist=15.0, step=0.5):
    """检查前方路径上的 ESDF 距离值"""
    results = []
    for d in np.arange(0.5, max_dist, step):
        point = drone_pos + direction * d
        dist = esdf.get_distance(point)
        results.append((d, dist))
        if dist < 0:
            break
    return results


def main():
    print("=" * 70)
    print("  PERCEPTION DIAGNOSIS")
    print("  Analyzing depth FOV, point cloud, and obstacle detection")
    print("=" * 70)

    drone = DroneInterface()
    try:
        drone.connect()
    except Exception as e:
        print(f"Failed to connect: {e}")
        return

    try:
        # 起飞
        print("\n[1] Taking off...")
        drone.takeoff()
        time.sleep(1)

        flight_height = -3.0
        drone.move_to_z(flight_height, velocity=2.0)
        time.sleep(1)

        initial_pos, initial_ori = drone.get_pose()
        print(f"    Position: ({initial_pos[0]:.2f}, {initial_pos[1]:.2f}, {initial_pos[2]:.2f})")

        # 初始化地图
        config = PlanningConfig(
            voxel_size=0.5,
            grid_size=(120, 120, 40),
            origin=(-10.0, -30.0, -15.0),
            max_depth=25.0,
            safety_margin=2.0,
        )
        map_manager = IncrementalMapManager(config)

        # === 诊断循环：朝前方飞，每步分析感知 ===
        print("\n[2] Starting diagnostic flight (forward)...")
        print("    Will fly forward in small steps, analyzing perception at each step\n")

        num_steps = 15
        step_distance = 2.0  # 每步前进2m

        for step in range(num_steps):
            print(f"\n{'='*70}")
            print(f"  STEP {step + 1}/{num_steps}")
            print(f"{'='*70}")

            # 获取当前状态
            pos, ori = drone.get_pose()
            print(f"\n  Drone position: ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})")

            # 获取深度图
            depth = drone.get_depth_image()

            # 1. 分析深度图
            depth_stats = analyze_depth_image(depth)
            print(f"\n  [Depth Image]")
            for k, v in depth_stats.items():
                print(f"    {k}: {v}")

            # 2. 转换为点云并分析
            points_cam = depth_image_to_camera_points(depth, fov_deg=90.0, subsample=4, max_depth=25.0)
            points_world = transform_camera_to_world(points_cam, pos, ori)

            cloud_stats = analyze_point_cloud(points_world, pos)
            print(f"\n  [Point Cloud (world frame, relative to drone)]")
            for k, v in cloud_stats.items():
                print(f"    {k}: {v}")

            # 3. 分析点云中有多少被地面过滤掉
            if len(points_world) > 0:
                ground_threshold = -0.5
                ground_points = points_world[points_world[:, 2] > ground_threshold]
                obstacle_points = points_world[points_world[:, 2] <= ground_threshold]

                # 进一步分析：保护半径过滤
                if len(obstacle_points) > 0:
                    dists_to_drone = np.linalg.norm(obstacle_points - pos, axis=1)
                    within_protection = np.sum(dists_to_drone < 0.8)
                    within_old_protection = np.sum(dists_to_drone < 2.0)

                    print(f"\n  [Filtering Analysis]")
                    print(f"    Total points: {len(points_world)}")
                    print(f"    Ground filtered (Z > {ground_threshold}): {len(ground_points)} ({len(ground_points)/len(points_world)*100:.1f}%)")
                    print(f"    Obstacle candidates (Z <= {ground_threshold}): {len(obstacle_points)} ({len(obstacle_points)/len(points_world)*100:.1f}%)")
                    print(f"    Within NEW protection radius (0.8m): {within_protection}")
                    print(f"    Within OLD protection radius (2.0m): {within_old_protection} <-- these were being DISCARDED before!")

            # 4. 更新地图
            map_stats = map_manager.update(depth, pos, ori)
            print(f"\n  [Map Update]")
            print(f"    New occupied voxels: {map_stats['new_occupied']}")
            print(f"    Total occupied: {map_stats['total_occupied']}")
            print(f"    Update time: {map_stats['update_time_ms']:.1f}ms")

            # 5. 检查前方 ESDF
            forward_dir = np.array([1.0, 0.0, 0.0])  # 假设朝X+方向
            esdf_ahead = check_esdf_ahead(map_manager.esdf, pos, forward_dir)
            print(f"\n  [ESDF ahead (X+ direction)]")
            print(f"    {'Dist':>6}  {'ESDF':>8}  Status")
            print(f"    {'-'*30}")
            for d, esdf_val in esdf_ahead:
                if esdf_val < 0:
                    status = "INSIDE OBSTACLE!"
                elif esdf_val < config.safety_margin:
                    status = f"UNSAFE (margin={config.safety_margin}m)"
                else:
                    status = "safe"
                print(f"    {d:>5.1f}m  {esdf_val:>7.2f}m  {status}")

            # 6. 检查左右方向的 ESDF（看 FOV 覆盖）
            right_dir = np.array([0.0, 1.0, 0.0])
            left_dir = np.array([0.0, -1.0, 0.0])

            esdf_right = check_esdf_ahead(map_manager.esdf, pos, right_dir, max_dist=10.0)
            esdf_left = check_esdf_ahead(map_manager.esdf, pos, left_dir, max_dist=10.0)

            print(f"\n  [ESDF lateral coverage]")
            if esdf_right:
                min_right = min(v for _, v in esdf_right)
                print(f"    Right (Y+): min ESDF = {min_right:.2f}m at distances up to {esdf_right[-1][0]:.1f}m")
            if esdf_left:
                min_left = min(v for _, v in esdf_left)
                print(f"    Left  (Y-): min ESDF = {min_left:.2f}m at distances up to {esdf_left[-1][0]:.1f}m")

            # 7. 检查无人机朝向（yaw）
            from utils.transforms import quaternion_to_rotation_matrix
            R = quaternion_to_rotation_matrix(ori)
            forward_body = R @ np.array([1, 0, 0])  # 机体X轴在世界坐标系中的方向
            yaw_deg = np.degrees(np.arctan2(forward_body[1], forward_body[0]))
            print(f"\n  [Drone Orientation]")
            print(f"    Yaw: {yaw_deg:.1f} deg (0=North/X+, 90=East/Y+)")
            print(f"    Forward direction in world: ({forward_body[0]:.2f}, {forward_body[1]:.2f}, {forward_body[2]:.2f})")

            # 也检查实际朝向方向的 ESDF
            forward_body_2d = forward_body.copy()
            forward_body_2d[2] = 0
            norm = np.linalg.norm(forward_body_2d)
            if norm > 0.01:
                forward_body_2d /= norm
                esdf_facing = check_esdf_ahead(map_manager.esdf, pos, forward_body_2d)
                print(f"\n  [ESDF in ACTUAL facing direction]")
                print(f"    {'Dist':>6}  {'ESDF':>8}  Status")
                print(f"    {'-'*30}")
                for d, esdf_val in esdf_facing:
                    if esdf_val < 0:
                        status = "INSIDE OBSTACLE!"
                    elif esdf_val < config.safety_margin:
                        status = f"UNSAFE"
                    else:
                        status = "safe"
                    print(f"    {d:>5.1f}m  {esdf_val:>7.2f}m  {status}")

            # 前进一步
            if step < num_steps - 1:
                next_pos = pos.copy()
                next_pos[0] += step_distance
                next_pos[2] = flight_height

                # 检查下一步是否安全
                next_esdf = map_manager.esdf.get_distance(next_pos)
                if next_esdf < 1.0:
                    print(f"\n  [STOP] Next position unsafe (ESDF={next_esdf:.2f}m), stopping forward flight")
                    print(f"  This is where the drone would collide without avoidance!")
                    break

                print(f"\n  Moving forward {step_distance}m...")
                drone.move_to_position(next_pos, velocity=1.5, timeout=5.0)
                time.sleep(0.5)

        # 最终地图统计
        print(f"\n\n{'='*70}")
        print("  FINAL MAP STATISTICS")
        print(f"{'='*70}")
        stats = map_manager.get_map_stats()
        for k, v in stats.items():
            print(f"  {k}: {v}")

        # 获取所有占据体素的位置分布
        occupied = map_manager.voxel_grid.get_occupied_voxels()
        if len(occupied) > 0:
            print(f"\n  Occupied voxels world coordinate ranges:")
            print(f"    X: [{occupied[:, 0].min():.1f}, {occupied[:, 0].max():.1f}]m")
            print(f"    Y: [{occupied[:, 1].min():.1f}, {occupied[:, 1].max():.1f}]m")
            print(f"    Z: [{occupied[:, 2].min():.1f}, {occupied[:, 2].max():.1f}]m")

        # 悬停并降落
        print("\n[3] Landing...")
        drone.hover()
        time.sleep(2)
        drone.land()

    except KeyboardInterrupt:
        print("\n\nInterrupted! Landing...")
        drone.land()
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        try:
            drone.land()
        except:
            pass
    finally:
        drone.disconnect()
        print("\nDiagnosis complete!")


if __name__ == "__main__":
    main()
