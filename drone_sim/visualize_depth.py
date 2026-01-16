"""
visualize_depth.py - 深度图可视化工具
功能：实时显示深度图的 2D 热力图和 3D 点云
"""

import airsim
import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def get_images(client):
    """获取 RGB 和深度图"""
    responses = client.simGetImages([
        airsim.ImageRequest("0", airsim.ImageType.Scene, False, False),
        airsim.ImageRequest("0", airsim.ImageType.DepthPlanar, True)
    ])

    # RGB 图像
    rgb = None
    if responses[0].width > 0:
        img1d = np.frombuffer(responses[0].image_data_uint8, dtype=np.uint8)
        rgb = img1d.reshape(responses[0].height, responses[0].width, 3)

    # 深度图
    depth = None
    if responses[1].width > 0:
        depth = np.array(responses[1].image_data_float, dtype=np.float32)
        depth = depth.reshape(responses[1].height, responses[1].width)

    return rgb, depth


def depth_to_colormap(depth, max_depth=50.0):
    """将深度图转换为彩色热力图"""
    # 裁剪到最大深度
    depth_clipped = np.clip(depth, 0, max_depth)
    # 归一化到 0-255
    depth_normalized = (depth_clipped / max_depth * 255).astype(np.uint8)
    # 应用颜色映射 (TURBO 更直观)
    colormap = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_TURBO)
    return colormap


def create_point_cloud(depth, fov_deg=90, max_depth=30.0, subsample=4):
    """
    从深度图创建 3D 点云

    Args:
        depth: 深度图
        fov_deg: 相机视场角
        max_depth: 最大深度（过滤远处点）
        subsample: 下采样因子（加速显示）
    """
    h, w = depth.shape

    # 下采样
    depth_sub = depth[::subsample, ::subsample]
    h_sub, w_sub = depth_sub.shape

    # 计算相机内参
    fov_rad = np.radians(fov_deg)
    fx = w / (2 * np.tan(fov_rad / 2))
    fy = fx  # 假设正方形像素
    cx, cy = w / 2, h / 2

    # 生成像素坐标网格
    u = np.arange(0, w, subsample)
    v = np.arange(0, h, subsample)
    u, v = np.meshgrid(u, v)

    # 深度值
    z = depth_sub.flatten()

    # 过滤无效点
    valid = (z > 0) & (z < max_depth)
    z = z[valid]
    u = u.flatten()[valid]
    v = v.flatten()[valid]

    # 反投影到 3D
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy

    return x, y, z


def visualize_realtime():
    """实时可视化深度图"""
    print("正在连接 AirSim...")
    client = airsim.MultirotorClient()
    client.confirmConnection()
    print("连接成功！")

    print("\n按 'q' 退出，按 's' 保存当前帧")

    cv2.namedWindow("RGB", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Depth Colormap", cv2.WINDOW_NORMAL)

    frame_count = 0

    while True:
        rgb, depth = get_images(client)

        if rgb is None or depth is None:
            continue

        # 深度热力图
        depth_color = depth_to_colormap(depth, max_depth=30.0)

        # 添加距离标注
        h, w = depth.shape
        center_depth = depth[h//2, w//2]
        cv2.putText(depth_color, f"Center: {center_depth:.2f}m", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # 显示最小/最大深度
        valid_depth = depth[(depth > 0) & (depth < 100)]
        if len(valid_depth) > 0:
            min_d, max_d = valid_depth.min(), valid_depth.max()
            cv2.putText(depth_color, f"Min: {min_d:.2f}m  Max: {max_d:.2f}m",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # 显示图像
        cv2.imshow("RGB", rgb)
        cv2.imshow("Depth Colormap", depth_color)

        key = cv2.waitKey(100) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            # 保存当前帧
            cv2.imwrite(f"rgb_{frame_count}.png", rgb)
            cv2.imwrite(f"depth_{frame_count}.png", depth_color)
            np.save(f"depth_raw_{frame_count}.npy", depth)
            print(f"已保存帧 {frame_count}")
            frame_count += 1

    cv2.destroyAllWindows()


def visualize_3d_pointcloud():
    """显示 3D 点云（单帧）"""
    print("正在连接 AirSim...")
    client = airsim.MultirotorClient()
    client.confirmConnection()
    print("连接成功！")

    # 获取一帧
    rgb, depth = get_images(client)

    if depth is None:
        print("无法获取深度图")
        return

    print("正在生成 3D 点云...")
    x, y, z = create_point_cloud(depth, max_depth=20.0, subsample=4)

    print(f"点云包含 {len(x)} 个点")

    # 创建 3D 图
    fig = plt.figure(figsize=(14, 5))

    # 子图1: RGB
    ax1 = fig.add_subplot(131)
    if rgb is not None:
        ax1.imshow(cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB))
    ax1.set_title("RGB Image")
    ax1.axis('off')

    # 子图2: 深度热力图
    ax2 = fig.add_subplot(132)
    im = ax2.imshow(depth, cmap='turbo', vmin=0, vmax=30)
    ax2.set_title("Depth Map (meters)")
    plt.colorbar(im, ax=ax2, label='Distance (m)')
    ax2.axis('off')

    # 子图3: 3D 点云
    ax3 = fig.add_subplot(133, projection='3d')

    # 用深度值着色
    scatter = ax3.scatter(x, z, -y, c=z, cmap='turbo', s=1, alpha=0.6)

    ax3.set_xlabel('X (left-right)')
    ax3.set_ylabel('Z (depth)')
    ax3.set_zlabel('Y (up-down)')
    ax3.set_title("3D Point Cloud")

    # 设置视角
    ax3.view_init(elev=20, azim=-60)

    plt.tight_layout()
    plt.savefig("depth_visualization.png", dpi=150)
    print("已保存: depth_visualization.png")
    plt.show()


if __name__ == "__main__":
    print("=" * 50)
    print("深度图可视化工具")
    print("=" * 50)
    print("\n选择模式:")
    print("1. 实时 2D 可视化 (按 q 退出)")
    print("2. 3D 点云可视化 (单帧)")

    choice = input("\n请输入选项 (1/2): ").strip()

    if choice == "1":
        visualize_realtime()
    elif choice == "2":
        visualize_3d_pointcloud()
    else:
        print("无效选项，默认运行实时可视化")
        visualize_realtime()
