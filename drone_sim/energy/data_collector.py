"""
Energy Data Collector for AirSim

从 AirSim 采集无人机飞行数据，用于训练神经网络残差模型
"""

import numpy as np
import time
import os
import csv
from datetime import datetime
from typing import Optional, List, Tuple

try:
    import airsim
    AIRSIM_AVAILABLE = True
except ImportError:
    AIRSIM_AVAILABLE = False
    print("Warning: airsim not available, data collection will not work")


class EnergyDataCollector:
    """
    能耗数据采集器

    采集的数据包括：
    - 位置 (x, y, z)
    - 速度 (vx, vy, vz)
    - 加速度 (ax, ay, az)
    - 姿态角 (roll, pitch, yaw)
    - 角速度 (p, q, r)
    - 电机转速 (rpm1, rpm2, rpm3, rpm4)
    - 推力 (thrust)
    - 功率估算 (power)
    """

    def __init__(self, save_dir: str = "energy_data"):
        """
        初始化数据采集器

        Args:
            save_dir: 数据保存目录
        """
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

        self.client = None
        self.is_connected = False

        # 数据缓存
        self.data_buffer = []

        # 上一帧数据（用于计算加速度）
        self.prev_velocity = None
        self.prev_time = None

        # 采集参数
        self.sample_rate = 50  # Hz

    def connect(self) -> bool:
        """连接到 AirSim"""
        if not AIRSIM_AVAILABLE:
            print("AirSim not available")
            return False

        try:
            self.client = airsim.MultirotorClient()
            self.client.confirmConnection()
            self.client.enableApiControl(True)
            self.is_connected = True
            print("Connected to AirSim")
            return True
        except Exception as e:
            print(f"Failed to connect to AirSim: {e}")
            return False

    def disconnect(self):
        """断开连接"""
        if self.client:
            self.client.enableApiControl(False)
            self.is_connected = False
            print("Disconnected from AirSim")

    def get_state(self) -> Optional[dict]:
        """
        获取当前无人机状态

        Returns:
            状态字典
        """
        if not self.is_connected:
            return None

        try:
            # 获取运动学状态
            state = self.client.getMultirotorState()
            kinematics = state.kinematics_estimated

            # 位置
            position = np.array([
                kinematics.position.x_val,
                kinematics.position.y_val,
                kinematics.position.z_val
            ])

            # 速度
            velocity = np.array([
                kinematics.linear_velocity.x_val,
                kinematics.linear_velocity.y_val,
                kinematics.linear_velocity.z_val
            ])

            # 角速度
            angular_velocity = np.array([
                kinematics.angular_velocity.x_val,
                kinematics.angular_velocity.y_val,
                kinematics.angular_velocity.z_val
            ])

            # 姿态（四元数转欧拉角）
            q = kinematics.orientation
            euler = self._quaternion_to_euler(q.w_val, q.x_val, q.y_val, q.z_val)

            # 计算加速度
            current_time = time.time()
            if self.prev_velocity is not None and self.prev_time is not None:
                dt = current_time - self.prev_time
                if dt > 0:
                    acceleration = (velocity - self.prev_velocity) / dt
                else:
                    acceleration = np.zeros(3)
            else:
                acceleration = np.zeros(3)

            self.prev_velocity = velocity.copy()
            self.prev_time = current_time

            # 获取电机信息（如果可用）
            rotor_states = None
            try:
                rotor_states = self.client.getRotorStates()
            except:
                pass

            # 构建状态字典
            state_dict = {
                'timestamp': current_time,
                'position': position,
                'velocity': velocity,
                'acceleration': acceleration,
                'euler_angles': euler,
                'angular_velocity': angular_velocity,
                'rotor_states': rotor_states
            }

            return state_dict

        except Exception as e:
            print(f"Error getting state: {e}")
            return None

    def _quaternion_to_euler(self, w: float, x: float, y: float, z: float) -> np.ndarray:
        """四元数转欧拉角 (roll, pitch, yaw)"""
        # Roll (x-axis rotation)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)

        # Pitch (y-axis rotation)
        sinp = 2 * (w * y - z * x)
        if abs(sinp) >= 1:
            pitch = np.copysign(np.pi / 2, sinp)
        else:
            pitch = np.arcsin(sinp)

        # Yaw (z-axis rotation)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)

        return np.array([roll, pitch, yaw])

    def estimate_power_from_state(self, state: dict) -> float:
        """
        从状态估算功率（简化模型）

        实际应用中应该从电池电压/电流传感器获取

        Args:
            state: 状态字典

        Returns:
            估算功率 (W)
        """
        # 使用物理模型估算
        from .physics_model import PhysicsEnergyModel

        model = PhysicsEnergyModel()
        power = model.compute_electrical_power(
            state['velocity'],
            state['acceleration']
        )

        # 添加一些模拟的非线性效应和噪声
        # 这模拟了真实传感器的测量值
        euler = state['euler_angles']
        v_mag = np.linalg.norm(state['velocity'])
        a_mag = np.linalg.norm(state['acceleration'])

        # 模拟气动干扰
        aero_effect = 5.0 * v_mag ** 1.5 * np.abs(np.sin(euler[1]))  # 俯仰影响
        coupling_effect = 2.0 * a_mag * v_mag  # 加速度耦合
        roll_effect = 3.0 * np.abs(euler[0]) * v_mag  # 横滚影响

        # 添加测量噪声
        noise = np.random.normal(0, 3)

        measured_power = power + aero_effect + coupling_effect + roll_effect + noise

        return max(measured_power, 0)

    def collect_sample(self) -> Optional[dict]:
        """
        采集一个数据样本

        Returns:
            数据样本字典
        """
        state = self.get_state()
        if state is None:
            return None

        # 估算功率
        power = self.estimate_power_from_state(state)

        sample = {
            'timestamp': state['timestamp'],
            'px': state['position'][0],
            'py': state['position'][1],
            'pz': state['position'][2],
            'vx': state['velocity'][0],
            'vy': state['velocity'][1],
            'vz': state['velocity'][2],
            'ax': state['acceleration'][0],
            'ay': state['acceleration'][1],
            'az': state['acceleration'][2],
            'roll': state['euler_angles'][0],
            'pitch': state['euler_angles'][1],
            'yaw': state['euler_angles'][2],
            'wx': state['angular_velocity'][0],
            'wy': state['angular_velocity'][1],
            'wz': state['angular_velocity'][2],
            'power': power
        }

        return sample

    def run_collection_maneuver(self, maneuver_type: str = "random",
                                 duration: float = 60.0) -> List[dict]:
        """
        执行数据采集机动

        Args:
            maneuver_type: 机动类型 ("random", "hover", "forward", "climb", "mixed")
            duration: 采集时长 (s)

        Returns:
            采集的数据列表
        """
        if not self.is_connected:
            print("Not connected to AirSim")
            return []

        print(f"Starting {maneuver_type} maneuver for {duration}s...")

        # 起飞
        self.client.takeoffAsync().join()
        time.sleep(2)

        # 移动到初始高度
        self.client.moveToZAsync(-5, 2).join()
        time.sleep(1)

        data = []
        start_time = time.time()
        sample_interval = 1.0 / self.sample_rate

        while time.time() - start_time < duration:
            loop_start = time.time()

            # 根据机动类型执行动作
            if maneuver_type == "random":
                self._execute_random_maneuver()
            elif maneuver_type == "hover":
                pass  # 保持悬停
            elif maneuver_type == "forward":
                self._execute_forward_maneuver()
            elif maneuver_type == "climb":
                self._execute_climb_maneuver()
            elif maneuver_type == "mixed":
                self._execute_mixed_maneuver(time.time() - start_time)

            # 采集数据
            sample = self.collect_sample()
            if sample:
                data.append(sample)

            # 控制采样率
            elapsed = time.time() - loop_start
            if elapsed < sample_interval:
                time.sleep(sample_interval - elapsed)

        print(f"Collected {len(data)} samples")

        # 降落
        self.client.landAsync().join()

        return data

    def _execute_random_maneuver(self):
        """执行随机机动"""
        # 随机速度指令
        vx = np.random.uniform(-3, 3)
        vy = np.random.uniform(-3, 3)
        vz = np.random.uniform(-1, 1)

        self.client.moveByVelocityAsync(vx, vy, vz, 0.5)

    def _execute_forward_maneuver(self):
        """执行前飞机动"""
        speed = np.random.uniform(1, 5)
        self.client.moveByVelocityAsync(speed, 0, 0, 0.5)

    def _execute_climb_maneuver(self):
        """执行爬升/下降机动"""
        vz = np.random.uniform(-2, 2)
        self.client.moveByVelocityAsync(0, 0, vz, 0.5)

    def _execute_mixed_maneuver(self, elapsed_time: float):
        """执行混合机动"""
        phase = int(elapsed_time / 10) % 4

        if phase == 0:  # 前飞
            self.client.moveByVelocityAsync(3, 0, 0, 0.5)
        elif phase == 1:  # 爬升
            self.client.moveByVelocityAsync(0, 0, -2, 0.5)
        elif phase == 2:  # 侧飞
            self.client.moveByVelocityAsync(0, 3, 0, 0.5)
        else:  # 下降
            self.client.moveByVelocityAsync(0, 0, 1, 0.5)

    def save_data(self, data: List[dict], filename: Optional[str] = None):
        """
        保存采集的数据

        Args:
            data: 数据列表
            filename: 文件名（不含扩展名）
        """
        if not data:
            print("No data to save")
            return

        if filename is None:
            filename = f"energy_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        filepath = os.path.join(self.save_dir, f"{filename}.csv")

        # 获取字段名
        fieldnames = list(data[0].keys())

        with open(filepath, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data)

        print(f"Data saved to {filepath}")

    def load_data(self, filename: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        加载数据用于训练

        Args:
            filename: 文件名

        Returns:
            (X, y): 特征和目标值
        """
        filepath = os.path.join(self.save_dir, filename)

        data = []
        with open(filepath, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                data.append(row)

        # 构建特征矩阵
        X = []
        y = []

        for row in data:
            features = [
                float(row['vx']), float(row['vy']), float(row['vz']),
                float(row['ax']), float(row['ay']), float(row['az']),
                float(row['roll']), float(row['pitch']), float(row['yaw'])
            ]
            X.append(features)
            y.append(float(row['power']))

        return np.array(X), np.array(y)


def collect_training_data(duration: float = 120.0, save_dir: str = "energy_data"):
    """
    便捷函数：采集训练数据

    Args:
        duration: 总采集时长 (s)
        save_dir: 保存目录
    """
    collector = EnergyDataCollector(save_dir)

    if not collector.connect():
        print("Failed to connect to AirSim")
        return

    try:
        # 采集不同类型的机动数据
        all_data = []

        maneuvers = ["hover", "forward", "climb", "random", "mixed"]
        time_per_maneuver = duration / len(maneuvers)

        for maneuver in maneuvers:
            print(f"\n=== Collecting {maneuver} data ===")
            data = collector.run_collection_maneuver(maneuver, time_per_maneuver)
            all_data.extend(data)
            time.sleep(3)  # 休息一下

        # 保存所有数据
        collector.save_data(all_data, "training_data_full")

        print(f"\nTotal samples collected: {len(all_data)}")

    finally:
        collector.disconnect()


if __name__ == "__main__":
    # 运行数据采集
    collect_training_data(duration=120.0)
