"""
Coordinate transformation utilities
"""

import numpy as np


def quaternion_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    """
    四元数转旋转矩阵

    Args:
        q: 四元数 [w, x, y, z] (AirSim 格式)

    Returns:
        3x3 旋转矩阵
    """
    w, x, y, z = q

    R = np.array([
        [1 - 2*(y**2 + z**2), 2*(x*y - w*z), 2*(x*z + w*y)],
        [2*(x*y + w*z), 1 - 2*(x**2 + z**2), 2*(y*z - w*x)],
        [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x**2 + y**2)]
    ])

    return R


def transform_camera_to_world(points_camera: np.ndarray,
                              drone_position: np.ndarray,
                              drone_orientation: np.ndarray) -> np.ndarray:
    """
    相机坐标系 → 世界坐标系

    AirSim 相机坐标系: Z前, X右, Y下
    AirSim 机体坐标系: X前, Y右, Z下
    世界坐标系: NED (X北, Y东, Z下)

    Args:
        points_camera: Nx3 点云数组（相机坐标系）
        drone_position: 无人机位置 [x, y, z]
        drone_orientation: 无人机姿态四元数 [w, x, y, z]

    Returns:
        Nx3 点云数组（世界坐标系）
    """
    # 1. 相机 → 机体坐标系
    # 相机: Z前, X右, Y下
    # 机体: X前, Y右, Z下
    R_body_camera = np.array([
        [0, 0, 1],   # 机体X = 相机Z
        [1, 0, 0],   # 机体Y = 相机X
        [0, 1, 0]    # 机体Z = 相机Y
    ])

    points_body = (R_body_camera @ points_camera.T).T

    # 2. 机体 → 世界坐标系
    R_world_body = quaternion_to_rotation_matrix(drone_orientation)
    points_world = (R_world_body @ points_body.T).T + drone_position

    return points_world


def depth_image_to_camera_points(depth_image: np.ndarray,
                                 fov_deg: float = 90.0,
                                 subsample: int = 4,
                                 max_depth: float = 25.0) -> np.ndarray:
    """
    深度图转相机坐标系点云

    Args:
        depth_image: HxW 深度图（米）
        fov_deg: 视场角（度）
        subsample: 下采样倍数
        max_depth: 最大有效深度

    Returns:
        Nx3 点云数组（相机坐标系）
    """
    h, w = depth_image.shape

    # 计算相机内参
    fov_rad = np.radians(fov_deg)
    fx = w / (2 * np.tan(fov_rad / 2))
    fy = fx
    cx, cy = w / 2, h / 2

    # 下采样
    depth_sub = depth_image[::subsample, ::subsample]

    # 生成像素坐标网格
    u = np.arange(0, w, subsample)
    v = np.arange(0, h, subsample)
    u, v = np.meshgrid(u, v)

    # 展平
    z = depth_sub.flatten()
    u = u.flatten()
    v = v.flatten()

    # 过滤有效深度
    valid = (z > 0) & (z < max_depth)
    z = z[valid]
    u = u[valid]
    v = v[valid]

    if len(z) == 0:
        return np.array([]).reshape(0, 3)

    # 计算相机坐标系下的 3D 点
    x_cam = (u - cx) * z / fx
    y_cam = (v - cy) * z / fy
    z_cam = z

    # 相机坐标系: Z前, X右, Y下
    points_camera = np.stack([x_cam, y_cam, z_cam], axis=1)

    return points_camera
