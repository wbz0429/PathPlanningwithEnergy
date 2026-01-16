"""
test_path_planning.py - 路径规划可行性验证测试
从 AirSim 获取深度图，构建地图，规划路径
"""

import airsim
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time

from path_planning import (
    PathPlanner, PlanningConfig,
    visualize_planning_result
)


def get_depth_and_position(client):
    """从 AirSim 获取深度图和无人机位置"""
    # 获取深度图
    responses = client.simGetImages([
        airsim.ImageRequest("0", airsim.ImageType.Scene, False, False),
        airsim.ImageRequest("0", airsim.ImageType.DepthPlanar, True)
    ])

    # 解析 RGB
    rgb = None
    if responses[0].width > 0:
        img1d = np.frombuffer(responses[0].image_data_uint8, dtype=np.uint8)
        rgb = img1d.reshape(responses[0].height, responses[0].width, 3)

    # 解析深度图
    depth = np.array(responses[1].image_data_float, dtype=np.float32)
    depth = depth.reshape(responses[1].height, responses[1].width)

    # 获取无人机状态
    state = client.getMultirotorState()
    pos = state.kinematics_estimated.position
    drone_pos = np.array([pos.x_val, pos.y_val, pos.z_val])

    return rgb, depth, drone_pos


def main():
    print("=" * 60)
    print("路径规划可行性验证测试")
    print("=" * 60)

    # 连接 AirSim
    print("\n[1] 连接 AirSim...")
    client = airsim.MultirotorClient()
    client.confirmConnection()
    print("连接成功!")

    # 获取深度图和位置
    print("\n[2] 获取深度图和无人机位置...")
    rgb, depth, drone_pos = get_depth_and_position(client)
    print(f"    深度图尺寸: {depth.shape}")
    print(f"    深度范围: {depth.min():.2f}m - {depth.max():.2f}m")
    print(f"    无人机位置: ({drone_pos[0]:.2f}, {drone_pos[1]:.2f}, {drone_pos[2]:.2f})")

    # 配置规划器 - 根据无人机位置动态设置原点
    print("\n[3] 初始化路径规划器...")
    # 栅格原点设置为以无人机为中心的区域
    grid_size = (80, 80, 40)  # 40m x 40m x 20m 空间
    voxel_size = 0.5
    origin = (
        drone_pos[0] - 10.0,  # X: 无人机后方 10m
        drone_pos[1] - 20.0,  # Y: 无人机左侧 20m
        drone_pos[2] - 10.0   # Z: 无人机下方 10m
    )
    config = PlanningConfig(
        voxel_size=voxel_size,
        grid_size=grid_size,
        origin=origin,
        max_depth=25.0,
        step_size=1.0,
        max_iterations=5000,
        goal_sample_rate=0.15,
        search_radius=3.0,
        safety_margin=0.8  # 稍微减小安全边距
    )
    planner = PathPlanner(config)
    print(f"    体素大小: {config.voxel_size}m")
    print(f"    栅格尺寸: {config.grid_size}")
    print(f"    规划空间: {config.grid_size[0]*config.voxel_size}m x {config.grid_size[1]*config.voxel_size}m x {config.grid_size[2]*config.voxel_size}m")

    # 更新地图
    print("\n[4] 从深度图构建 3D 栅格地图...")
    t0 = time.time()
    occupied_count = planner.update_map(depth, drone_pos)
    t1 = time.time()
    print(f"    占据体素数量: {occupied_count}")
    print(f"    地图构建耗时: {(t1-t0)*1000:.1f}ms")

    # 计算 ESDF
    print("\n[5] 计算 ESDF 距离场...")
    t0 = time.time()
    planner.esdf.compute()
    t1 = time.time()
    print(f"    ESDF 计算耗时: {(t1-t0)*1000:.1f}ms")

    # 设置起点和终点
    # 起点：无人机当前位置（注意 AirSim 使用 NED 坐标系，Z 向下为正）
    # 为了避免起点在障碍物内，稍微调整起点位置
    start = drone_pos.copy()
    start[0] -= 1.0  # 稍微后退，确保不在障碍物内

    # 终点：绕过障碍物到达前方
    # 根据场景调整，这里假设障碍物在正前方，目标在右前方
    goal = np.array([drone_pos[0] + 15.0, drone_pos[1] + 10.0, drone_pos[2]])

    print(f"\n[6] 路径规划...")
    print(f"    起点: ({start[0]:.2f}, {start[1]:.2f}, {start[2]:.2f})")
    print(f"    终点: ({goal[0]:.2f}, {goal[1]:.2f}, {goal[2]:.2f})")

    # 检查起点和终点的安全性
    start_dist = planner.get_distance_to_obstacle(start)
    goal_dist = planner.get_distance_to_obstacle(goal)
    print(f"    起点到障碍物距离: {start_dist:.2f}m")
    print(f"    终点到障碍物距离: {goal_dist:.2f}m")

    # 规划路径
    t0 = time.time()
    path = planner.plan_path(start, goal)
    t1 = time.time()

    if path is not None:
        print(f"    规划成功!")
        print(f"    路径点数: {len(path)}")
        print(f"    规划耗时: {(t1-t0)*1000:.1f}ms")

        # 计算路径长度
        path_length = sum(np.linalg.norm(path[i+1] - path[i]) for i in range(len(path)-1))
        print(f"    路径长度: {path_length:.2f}m")

        # 打印路径点
        print("\n    路径点:")
        for i, p in enumerate(path):
            dist = planner.get_distance_to_obstacle(p)
            print(f"      [{i}] ({p[0]:.2f}, {p[1]:.2f}, {p[2]:.2f}) - 距障碍物: {dist:.2f}m")
    else:
        print(f"    规划失败!")
        print(f"    耗时: {(t1-t0)*1000:.1f}ms")

    # 可视化
    print("\n[7] 生成可视化结果...")

    # 创建综合可视化
    fig = plt.figure(figsize=(18, 10))

    # 1. RGB 图像
    ax1 = fig.add_subplot(231)
    if rgb is not None:
        import cv2
        ax1.imshow(cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB))
    ax1.set_title("RGB Image")
    ax1.axis('off')

    # 2. 深度图
    ax2 = fig.add_subplot(232)
    im = ax2.imshow(depth, cmap='turbo', vmin=0, vmax=25)
    ax2.set_title("Depth Map")
    plt.colorbar(im, ax=ax2, label='Distance (m)')
    ax2.axis('off')

    # 3. 3D 占据栅格和路径
    ax3 = fig.add_subplot(233, projection='3d')
    occupied = planner.voxel_grid.get_occupied_voxels()
    if len(occupied) > 0:
        # 下采样显示
        if len(occupied) > 2000:
            indices = np.random.choice(len(occupied), 2000, replace=False)
            occupied_show = occupied[indices]
        else:
            occupied_show = occupied
        ax3.scatter(occupied_show[:, 0], occupied_show[:, 1], occupied_show[:, 2],
                   c='red', s=2, alpha=0.3, label='Obstacles')

    if path is not None:
        path_arr = np.array(path)
        ax3.plot(path_arr[:, 0], path_arr[:, 1], path_arr[:, 2],
                'b-', linewidth=2, label='Path')
        ax3.scatter(path_arr[:, 0], path_arr[:, 1], path_arr[:, 2],
                   c='blue', s=30)

    ax3.scatter(*start, c='green', s=100, marker='o', label='Start')
    ax3.scatter(*goal, c='orange', s=100, marker='*', label='Goal')
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_zlabel('Z')
    ax3.set_title('3D Voxel Grid & Path')
    ax3.legend()

    # 4. ESDF XY 切片
    ax4 = fig.add_subplot(234)
    if planner.esdf.distance_field is not None:
        # 找到无人机所在的 Z 层
        z_idx = int((drone_pos[2] - config.origin[2]) / config.voxel_size)
        z_idx = np.clip(z_idx, 0, config.grid_size[2] - 1)
        slice_xy = planner.esdf.distance_field[:, :, z_idx].T
        im = ax4.imshow(slice_xy, cmap='RdYlGn', origin='lower',
                       extent=[config.origin[0],
                              config.origin[0] + config.grid_size[0] * config.voxel_size,
                              config.origin[1],
                              config.origin[1] + config.grid_size[1] * config.voxel_size],
                       vmin=-5, vmax=10)
        plt.colorbar(im, ax=ax4, label='Distance (m)')

        if path is not None:
            path_arr = np.array(path)
            ax4.plot(path_arr[:, 0], path_arr[:, 1], 'b-', linewidth=2, label='Path')
        ax4.scatter(start[0], start[1], c='green', s=100, marker='o', label='Start')
        ax4.scatter(goal[0], goal[1], c='orange', s=100, marker='*', label='Goal')
        ax4.legend()

    ax4.set_xlabel('X (m)')
    ax4.set_ylabel('Y (m)')
    ax4.set_title(f'ESDF XY Slice (Z={z_idx})')

    # 5. ESDF XZ 切片
    ax5 = fig.add_subplot(235)
    if planner.esdf.distance_field is not None:
        y_idx = int((drone_pos[1] - config.origin[1]) / config.voxel_size)
        y_idx = np.clip(y_idx, 0, config.grid_size[1] - 1)
        slice_xz = planner.esdf.distance_field[:, y_idx, :].T
        im = ax5.imshow(slice_xz, cmap='RdYlGn', origin='lower',
                       extent=[config.origin[0],
                              config.origin[0] + config.grid_size[0] * config.voxel_size,
                              config.origin[2],
                              config.origin[2] + config.grid_size[2] * config.voxel_size],
                       vmin=-5, vmax=10)
        plt.colorbar(im, ax=ax5, label='Distance (m)')

        if path is not None:
            path_arr = np.array(path)
            ax5.plot(path_arr[:, 0], path_arr[:, 2], 'b-', linewidth=2)
        ax5.scatter(start[0], start[2], c='green', s=100, marker='o')
        ax5.scatter(goal[0], goal[2], c='orange', s=100, marker='*')

    ax5.set_xlabel('X (m)')
    ax5.set_ylabel('Z (m)')
    ax5.set_title(f'ESDF XZ Slice (Y={y_idx})')

    # 6. 路径俯视图
    ax6 = fig.add_subplot(236)
    if len(occupied) > 0:
        ax6.scatter(occupied[:, 0], occupied[:, 1], c='red', s=1, alpha=0.1, label='Obstacles')
    if path is not None:
        path_arr = np.array(path)
        ax6.plot(path_arr[:, 0], path_arr[:, 1], 'b-', linewidth=2, label='Path')
        ax6.scatter(path_arr[:, 0], path_arr[:, 1], c='blue', s=20)
    ax6.scatter(start[0], start[1], c='green', s=100, marker='o', label='Start')
    ax6.scatter(goal[0], goal[1], c='orange', s=100, marker='*', label='Goal')
    ax6.set_xlabel('X (m)')
    ax6.set_ylabel('Y (m)')
    ax6.set_title('Top View (XY Plane)')
    ax6.legend()
    ax6.set_aspect('equal')
    ax6.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("path_planning_result.png", dpi=150)
    print("    已保存: path_planning_result.png")

    # plt.show()  # 注释掉以避免阻塞

    # 总结
    print("\n" + "=" * 60)
    print("可行性验证总结")
    print("=" * 60)
    print(f"1. 3D 栅格地图构建: {'成功' if occupied_count > 0 else '失败'}")
    print(f"2. ESDF 距离场计算: 成功")
    print(f"3. RRT* 路径规划: {'成功' if path is not None else '失败'}")
    if path is not None:
        print(f"4. 路径长度: {path_length:.2f}m")
        print(f"5. 路径点数: {len(path)}")

    print("\n技术路线可行性: ", end="")
    if path is not None and occupied_count > 0:
        print("验证通过!")
    else:
        print("需要进一步调试")


if __name__ == "__main__":
    main()
