"""
Drone Interface - AirSim 接口封装
"""

import airsim
import numpy as np
from typing import Tuple


class DroneInterface:
    """
    无人机接口封装
    简化 AirSim API 调用
    """

    def __init__(self):
        self.client = None
        self.is_connected = False

    def connect(self):
        """连接到 AirSim"""
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        self.is_connected = True
        print("[OK] Connected to AirSim")

    def disconnect(self):
        """断开连接"""
        if self.client and self.is_connected:
            self.client.armDisarm(False)
            self.client.enableApiControl(False)
            self.is_connected = False
            print("[OK] Disconnected from AirSim")

    def takeoff(self):
        """起飞"""
        if not self.is_connected:
            raise RuntimeError("Not connected to AirSim")
        self.client.takeoffAsync().join()
        print("[OK] Takeoff complete")

    def land(self):
        """降落"""
        if not self.is_connected:
            raise RuntimeError("Not connected to AirSim")
        self.client.landAsync().join()
        print("[OK] Landing complete")

    def get_position(self) -> np.ndarray:
        """
        获取无人机位置

        Returns:
            位置 [x, y, z] (NED坐标系)
        """
        if not self.is_connected:
            raise RuntimeError("Not connected to AirSim")

        state = self.client.getMultirotorState()
        pos = state.kinematics_estimated.position
        return np.array([pos.x_val, pos.y_val, pos.z_val])

    def get_orientation(self) -> np.ndarray:
        """
        获取无人机姿态（四元数）

        Returns:
            四元数 [w, x, y, z]
        """
        if not self.is_connected:
            raise RuntimeError("Not connected to AirSim")

        state = self.client.getMultirotorState()
        q = state.kinematics_estimated.orientation
        return np.array([q.w_val, q.x_val, q.y_val, q.z_val])

    def get_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        获取无人机位姿

        Returns:
            (position, orientation)
        """
        if not self.is_connected:
            raise RuntimeError("Not connected to AirSim")

        state = self.client.getMultirotorState()
        pos = state.kinematics_estimated.position
        q = state.kinematics_estimated.orientation

        position = np.array([pos.x_val, pos.y_val, pos.z_val])
        orientation = np.array([q.w_val, q.x_val, q.y_val, q.z_val])

        return position, orientation

    def get_depth_image(self) -> np.ndarray:
        """
        获取深度图

        Returns:
            HxW 深度图（米）
        """
        if not self.is_connected:
            raise RuntimeError("Not connected to AirSim")

        responses = self.client.simGetImages([
            airsim.ImageRequest("0", airsim.ImageType.DepthPlanar, True)
        ])

        depth = np.array(responses[0].image_data_float, dtype=np.float32)
        depth = depth.reshape(responses[0].height, responses[0].width)

        return depth

    def move_to_position(self, target: np.ndarray, velocity: float = 2.0, timeout: float = 30.0):
        """
        移动到目标位置

        Args:
            target: 目标位置 [x, y, z]
            velocity: 飞行速度 (m/s)
            timeout: 超时时间 (秒)
        """
        if not self.is_connected:
            raise RuntimeError("Not connected to AirSim")

        self.client.moveToPositionAsync(
            target[0], target[1], target[2],
            velocity,
            timeout_sec=timeout,
            drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
            yaw_mode=airsim.YawMode(is_rate=False, yaw_or_rate=0)
        ).join()

    def move_to_z(self, z: float, velocity: float = 2.0):
        """
        移动到指定高度

        Args:
            z: 目标高度 (NED坐标系，负值表示向上)
            velocity: 飞行速度 (m/s)
        """
        if not self.is_connected:
            raise RuntimeError("Not connected to AirSim")

        self.client.moveToZAsync(z, velocity).join()

    def hover(self):
        """悬停"""
        if not self.is_connected:
            raise RuntimeError("Not connected to AirSim")

        self.client.hoverAsync().join()
