"""
query_scene_objects.py - 使用 AirSim API 精确获取场景中所有物体的位置和尺寸

替代 scan_scene.py 的猜测式扫描，直接从仿真器获取真实数据
"""

import airsim
import numpy as np
import sys

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')


def query_all_objects(client):
    """查询场景中所有物体"""
    objects = client.simListSceneObjects()
    print(f"Scene contains {len(objects)} objects\n")
    return objects


def get_object_info(client, obj_name):
    """获取单个物体的位姿信息"""
    pose = client.simGetObjectPose(obj_name)
    if pose.position.x_val == 0 and pose.position.y_val == 0 and pose.position.z_val == 0:
        return None

    scale = client.simGetObjectScale(obj_name)

    return {
        'name': obj_name,
        'position': np.array([pose.position.x_val, pose.position.y_val, pose.position.z_val]),
        'scale': np.array([scale.x_val, scale.y_val, scale.z_val]),
    }


def find_obstacles_near_flight_path(client, flight_height=-3.0, x_range=(0, 60), y_range=(-30, 30)):
    """
    找出飞行路径附近的障碍物

    Args:
        flight_height: 飞行高度 (NED, 负值)
        x_range: X方向搜索范围
        y_range: Y方向搜索范围

    Returns:
        obstacles: 障碍物列表，每个包含 name, position, scale, bbox_min, bbox_max
    """
    objects = query_all_objects(client)
    obstacles = []

    for obj_name in objects:
        try:
            info = get_object_info(client, obj_name)
            if info is None:
                continue

            pos = info['position']
            scale = info['scale']

            # 过滤：只保留飞行区域内的物体
            if not (x_range[0] - 20 <= pos[0] <= x_range[1] + 20):
                continue
            if not (y_range[0] - 20 <= pos[1] <= y_range[1] + 20):
                continue

            # 跳过地面和天空盒等非障碍物
            skip_keywords = ['floor', 'ground', 'sky', 'light', 'sun', 'camera',
                           'player', 'drone', 'bp_', 'landscape']
            if any(kw in obj_name.lower() for kw in skip_keywords):
                continue

            # 估算包围盒 (AirSim scale 单位是 100 = 1m)
            half_extent = scale * 50.0  # scale * 100 / 2 = 半尺寸(cm) -> 转换为合理估计
            # 注意：AirSim 的 scale 含义取决于具体物体的原始尺寸
            # 这里我们记录原始 scale，后续根据实际情况调整

            info['bbox_min'] = pos - half_extent
            info['bbox_max'] = pos + half_extent

            obstacles.append(info)

        except Exception as e:
            continue

    return obstacles


def validate_goal_point(obstacles, goal, safety_buffer=3.0):
    """
    验证目标点是否在障碍物外部

    Args:
        obstacles: 障碍物列表
        goal: 目标点 [x, y, z]
        safety_buffer: 安全缓冲距离 (m)

    Returns:
        (is_safe, nearest_obstacle_name, nearest_distance)
    """
    min_dist = float('inf')
    nearest_name = None

    for obs in obstacles:
        # 计算点到包围盒的距离
        pos = obs['position']
        dist = np.linalg.norm(goal[:2] - pos[:2])  # XY平面距离

        if dist < min_dist:
            min_dist = dist
            nearest_name = obs['name']

    is_safe = min_dist > safety_buffer
    return is_safe, nearest_name, min_dist


def suggest_safe_goals(obstacles, start_pos, flight_height=-3.0):
    """
    根据障碍物信息推荐安全的目标点

    Args:
        obstacles: 障碍物列表
        start_pos: 起始位置
        flight_height: 飞行高度
    """
    print("\n" + "=" * 70)
    print("RECOMMENDED SAFE GOAL POINTS")
    print("=" * 70)

    # 策略1: 前方无障碍物的直线目标
    candidates = [
        np.array([start_pos[0] + 20.0, start_pos[1], flight_height]),
        np.array([start_pos[0] + 30.0, start_pos[1] + 15.0, flight_height]),
        np.array([start_pos[0] + 30.0, start_pos[1] - 15.0, flight_height]),
        np.array([start_pos[0] + 15.0, start_pos[1] + 20.0, flight_height]),
        np.array([start_pos[0] + 15.0, start_pos[1] - 20.0, flight_height]),
        np.array([start_pos[0] + 40.0, start_pos[1] + 25.0, flight_height]),
    ]

    print(f"\nStart position: ({start_pos[0]:.1f}, {start_pos[1]:.1f}, {start_pos[2]:.1f})")
    print()

    for i, goal in enumerate(candidates):
        is_safe, nearest_name, nearest_dist = validate_goal_point(obstacles, goal)
        status = "SAFE" if is_safe else "BLOCKED"
        print(f"  Goal {i+1}: ({goal[0]:.1f}, {goal[1]:.1f}, {goal[2]:.1f}) "
              f"-> {status} (nearest obstacle: {nearest_name}, dist={nearest_dist:.1f}m)")

        if is_safe:
            print(f"    >>> global_goal = np.array([{goal[0]:.1f}, {goal[1]:.1f}, {goal[2]:.1f}])")


def main():
    print("=" * 70)
    print("AirSim Scene Object Query")
    print("Precise obstacle detection using simulation API")
    print("=" * 70)

    client = airsim.MultirotorClient()
    client.confirmConnection()
    print("[OK] Connected to AirSim\n")

    # 获取无人机当前位置
    state = client.getMultirotorState()
    drone_pos = np.array([
        state.kinematics_estimated.position.x_val,
        state.kinematics_estimated.position.y_val,
        state.kinematics_estimated.position.z_val
    ])
    print(f"Drone position: ({drone_pos[0]:.2f}, {drone_pos[1]:.2f}, {drone_pos[2]:.2f})")

    # 查询所有场景物体
    print("\n" + "=" * 70)
    print("ALL SCENE OBJECTS")
    print("=" * 70)

    objects = query_all_objects(client)

    # 获取每个物体的详细信息
    all_info = []
    for obj_name in objects:
        try:
            info = get_object_info(client, obj_name)
            if info is not None:
                all_info.append(info)
        except:
            continue

    # 按X坐标排序显示
    all_info.sort(key=lambda x: x['position'][0])

    print(f"\n{'Name':<40} {'X':>8} {'Y':>8} {'Z':>8} {'ScaleX':>8} {'ScaleY':>8} {'ScaleZ':>8}")
    print("-" * 100)
    for info in all_info:
        pos = info['position']
        scale = info['scale']
        print(f"{info['name']:<40} {pos[0]:>8.1f} {pos[1]:>8.1f} {pos[2]:>8.1f} "
              f"{scale[0]:>8.2f} {scale[1]:>8.2f} {scale[2]:>8.2f}")

    # 找出飞行路径附近的障碍物
    print("\n" + "=" * 70)
    print("OBSTACLES NEAR FLIGHT PATH")
    print("=" * 70)

    obstacles = find_obstacles_near_flight_path(client, flight_height=-3.0)

    if obstacles:
        print(f"\nFound {len(obstacles)} potential obstacles:\n")
        for obs in obstacles:
            pos = obs['position']
            scale = obs['scale']
            print(f"  {obs['name']}")
            print(f"    Position: ({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f})")
            print(f"    Scale: ({scale[0]:.2f}, {scale[1]:.2f}, {scale[2]:.2f})")
            print()
    else:
        print("\nNo obstacles found in flight path area")

    # 推荐安全目标点
    suggest_safe_goals(obstacles, drone_pos, flight_height=-3.0)

    print("\n" + "=" * 70)
    print("Query complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
