"""
logger.py - 飞行数据记录模块
功能：记录无人机飞行状态数据到 CSV 文件
"""

import csv
import os
import time
from datetime import datetime
from typing import Optional


class FlightLogger:
    """
    飞行数据记录器
    记录无人机的位置、速度、加速度等状态数据
    """

    def __init__(self, log_dir: str = "logs", filename: Optional[str] = None):
        """
        初始化记录器

        Args:
            log_dir: 日志文件保存目录
            filename: 日志文件名，默认使用时间戳命名
        """
        self.log_dir = log_dir

        # 确保目录存在
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        # 生成文件名
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"flight_log_{timestamp}.csv"

        self.filepath = os.path.join(log_dir, filename)
        self.file = None
        self.writer = None
        self.start_time = None

        # CSV 列名
        self.fieldnames = [
            "timestamp",      # 时间戳（秒）
            "elapsed_time",   # 相对起始时间（秒）
            "px", "py", "pz", # 位置 (米)
            "vx", "vy", "vz", # 速度 (米/秒)
            "ax", "ay", "az", # 加速度 (米/秒^2)
            "roll", "pitch", "yaw",  # 姿态角 (弧度)
            "obstacle_detected",     # 是否检测到障碍物
            "obstacle_distance"      # 障碍物距离 (米)
        ]

    def start(self):
        """开始记录"""
        self.file = open(self.filepath, 'w', newline='', encoding='utf-8')
        self.writer = csv.DictWriter(self.file, fieldnames=self.fieldnames)
        self.writer.writeheader()
        self.start_time = time.time()
        print(f"开始记录飞行数据: {self.filepath}")

    def log(self,
            position: tuple,
            velocity: tuple,
            acceleration: tuple,
            orientation: tuple = (0, 0, 0),
            obstacle_detected: bool = False,
            obstacle_distance: float = -1):
        """
        记录一条飞行数据

        Args:
            position: (px, py, pz) 位置坐标
            velocity: (vx, vy, vz) 速度
            acceleration: (ax, ay, az) 加速度
            orientation: (roll, pitch, yaw) 姿态角
            obstacle_detected: 是否检测到障碍物
            obstacle_distance: 障碍物距离
        """
        if self.writer is None:
            raise RuntimeError("Logger 未启动，请先调用 start()")

        current_time = time.time()
        elapsed = current_time - self.start_time

        row = {
            "timestamp": current_time,
            "elapsed_time": round(elapsed, 4),
            "px": round(position[0], 4),
            "py": round(position[1], 4),
            "pz": round(position[2], 4),
            "vx": round(velocity[0], 4),
            "vy": round(velocity[1], 4),
            "vz": round(velocity[2], 4),
            "ax": round(acceleration[0], 4),
            "ay": round(acceleration[1], 4),
            "az": round(acceleration[2], 4),
            "roll": round(orientation[0], 4),
            "pitch": round(orientation[1], 4),
            "yaw": round(orientation[2], 4),
            "obstacle_detected": int(obstacle_detected),
            "obstacle_distance": round(obstacle_distance, 4)
        }

        self.writer.writerow(row)
        self.file.flush()  # 实时写入

    def stop(self):
        """停止记录"""
        if self.file:
            self.file.close()
            self.file = None
            self.writer = None
            print(f"飞行数据已保存: {self.filepath}")

    def __enter__(self):
        """支持 with 语句"""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """支持 with 语句"""
        self.stop()
        return False


def extract_state_from_airsim(state) -> dict:
    """
    从 AirSim 状态对象中提取数据

    Args:
        state: airsim.MultirotorState 对象

    Returns:
        包含位置、速度、加速度、姿态的字典
    """
    kin = state.kinematics_estimated

    position = (
        kin.position.x_val,
        kin.position.y_val,
        kin.position.z_val
    )

    velocity = (
        kin.linear_velocity.x_val,
        kin.linear_velocity.y_val,
        kin.linear_velocity.z_val
    )

    acceleration = (
        kin.linear_acceleration.x_val,
        kin.linear_acceleration.y_val,
        kin.linear_acceleration.z_val
    )

    # 从四元数转换为欧拉角
    q = kin.orientation
    import math

    # 四元数转欧拉角
    sinr_cosp = 2 * (q.w_val * q.x_val + q.y_val * q.z_val)
    cosr_cosp = 1 - 2 * (q.x_val * q.x_val + q.y_val * q.y_val)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2 * (q.w_val * q.y_val - q.z_val * q.x_val)
    if abs(sinp) >= 1:
        pitch = math.copysign(math.pi / 2, sinp)
    else:
        pitch = math.asin(sinp)

    siny_cosp = 2 * (q.w_val * q.z_val + q.x_val * q.y_val)
    cosy_cosp = 1 - 2 * (q.y_val * q.y_val + q.z_val * q.z_val)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    orientation = (roll, pitch, yaw)

    return {
        "position": position,
        "velocity": velocity,
        "acceleration": acceleration,
        "orientation": orientation
    }
