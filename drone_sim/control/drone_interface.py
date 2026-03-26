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

    def move_to_position(self, target: np.ndarray, velocity: float = 2.0, timeout: float = 10.0):
        """
        移动到目标位置

        Args:
            target: 目标位置 [x, y, z]
            velocity: 飞行速度 (m/s)
            timeout: 超时时间 (秒) - 缩短默认值，避免卡在不可达航点
        """
        if not self.is_connected:
            raise RuntimeError("Not connected to AirSim")

        # Z 补偿：AirSim 水平飞行时 Z 会下沉（NED 坐标 Z 变更负）
        # 检测下沉并预先往上偏移抵消
        actual_z = self.client.getMultirotorState().kinematics_estimated.position.z_val
        z_compensated = target[2]
        if actual_z < target[2] - 0.1:  # 无人机比目标低 0.1m 以上
            correction = (target[2] - actual_z) * 0.8
            correction = min(correction, 0.3)  # 最多往上补偿 0.3m，防止过冲
            z_compensated = target[2] + correction

        self.client.moveToPositionAsync(
            target[0], target[1], z_compensated,
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

    def set_yaw(self, yaw_deg: float, duration: float = 0.5):
        """
        原地转向指定 yaw 角度（不移动位置）

        Args:
            yaw_deg: 目标 yaw 角度（度），0=North/X+, 90=East/Y+
            duration: 转向持续时间（秒）
        """
        if not self.is_connected:
            raise RuntimeError("Not connected to AirSim")

        self.client.moveByVelocityAsync(
            0, 0, 0, duration,
            drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
            yaw_mode=airsim.YawMode(is_rate=False, yaw_or_rate=yaw_deg)
        ).join()

    def hover(self):
        """悬停"""
        if not self.is_connected:
            raise RuntimeError("Not connected to AirSim")

        self.client.hoverAsync().join()

    def reset(self):
        """
        重置无人机到初始位置

        注意：reset后需要重新enableApiControl和armDisarm
        """
        if not self.client:
            raise RuntimeError("Not connected to AirSim")

        self.client.reset()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        print("[OK] Reset to initial position")
