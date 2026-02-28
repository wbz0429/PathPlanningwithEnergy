"""
Hybrid Energy Model for Quadrotor UAV

混合能耗模型：物理基准 + 神经网络残差补偿
参考：NeuroBEM (Bauersfeld et al., 2021)
"""

import numpy as np
from typing import Optional, Tuple, List
import os

from .physics_model import PhysicsEnergyModel, QuadrotorParams
from .neural_model import NeuralResidualModel


class HybridEnergyModel:
    """
    混合能耗模型

    P_total = P_physics + ΔP_neural

    其中：
    - P_physics: 基于 BEMT 理论的物理模型预测
    - ΔP_neural: 神经网络残差补偿
    """

    def __init__(self, params: Optional[QuadrotorParams] = None,
                 neural_model_path: Optional[str] = None):
        """
        初始化混合模型

        Args:
            params: 无人机物理参数
            neural_model_path: 预训练神经网络模型路径
        """
        # 物理模型
        self.physics_model = PhysicsEnergyModel(params)

        # 神经网络残差模型
        self.neural_model = NeuralResidualModel(
            input_dim=9,  # 3速度 + 3加速度 + 3姿态角
            hidden_dims=[32, 16],
            output_dim=1
        )

        # 加载预训练模型
        if neural_model_path and os.path.exists(neural_model_path):
            self.neural_model.load(neural_model_path)

        # 是否启用神经网络补偿
        self.use_neural_compensation = self.neural_model.is_trained

    def compute_power(self, velocity: np.ndarray,
                      acceleration: Optional[np.ndarray] = None,
                      euler_angles: Optional[np.ndarray] = None) -> float:
        """
        计算总功率

        Args:
            velocity: 速度向量 [vx, vy, vz] (m/s)
            acceleration: 加速度向量 [ax, ay, az] (m/s^2)
            euler_angles: 姿态角 [roll, pitch, yaw] (rad)

        Returns:
            总功率 (W)
        """
        # 物理模型预测
        P_physics = self.physics_model.compute_electrical_power(velocity, acceleration)

        # 神经网络残差补偿
        P_residual = 0.0
        if self.use_neural_compensation and acceleration is not None and euler_angles is not None:
            P_residual = self.neural_model.predict(velocity, acceleration, euler_angles)

        # 总功率
        P_total = P_physics + P_residual

        return max(P_total, 0)  # 功率不能为负

    def compute_energy_for_segment(self, start_pos: np.ndarray, end_pos: np.ndarray,
                                    velocity_magnitude: float = 2.0,
                                    euler_angles: Optional[np.ndarray] = None) -> Tuple[float, float]:
        """
        计算路径段的能耗

        Args:
            start_pos: 起点位置 [x, y, z]
            end_pos: 终点位置 [x, y, z]
            velocity_magnitude: 飞行速度大小 (m/s)
            euler_angles: 姿态角，可选

        Returns:
            (能耗 (J), 飞行时间 (s))
        """
        # 计算方向和距离
        direction = end_pos - start_pos
        distance = np.linalg.norm(direction)

        if distance < 1e-6:
            return 0.0, 0.0

        direction = direction / distance

        # 速度向量
        velocity = direction * velocity_magnitude

        # 飞行时间
        flight_time = distance / velocity_magnitude

        # 估算加速度（假设匀速飞行，加速度为0）
        acceleration = np.zeros(3)

        # 估算姿态角（根据速度方向）
        if euler_angles is None:
            # 简化：根据速度方向估算俯仰角
            pitch = np.arctan2(-velocity[2], np.sqrt(velocity[0]**2 + velocity[1]**2))
            yaw = np.arctan2(velocity[1], velocity[0])
            roll = 0.0
            euler_angles = np.array([roll, pitch, yaw])

        # 计算功率
        power = self.compute_power(velocity, acceleration, euler_angles)

        # 能耗 = 功率 × 时间
        energy = power * flight_time

        return energy, flight_time

    def compute_energy_for_path(self, path: List[np.ndarray],
                                 velocity: float = 2.0) -> Tuple[float, float, List[float]]:
        """
        计算整条路径的能耗

        Args:
            path: 路径点列表
            velocity: 飞行速度 (m/s)

        Returns:
            (总能耗 (J), 总飞行时间 (s), 各段能耗列表)
        """
        total_energy = 0.0
        total_time = 0.0
        segment_energies = []

        for i in range(len(path) - 1):
            start = np.array(path[i])
            end = np.array(path[i + 1])

            energy, time = self.compute_energy_for_segment(start, end, velocity)
            total_energy += energy
            total_time += time
            segment_energies.append(energy)

        return total_energy, total_time, segment_energies

    def compute_energy_cost(self, start_pos: np.ndarray, end_pos: np.ndarray,
                            velocity: float = 2.0) -> float:
        """
        计算两点之间的能耗代价（用于路径规划）

        Args:
            start_pos: 起点
            end_pos: 终点
            velocity: 飞行速度

        Returns:
            能耗代价 (J)
        """
        energy, _ = self.compute_energy_for_segment(start_pos, end_pos, velocity)
        return energy

    def train_neural_model(self, X: np.ndarray, y_true: np.ndarray,
                           epochs: int = 1000, verbose: bool = True) -> dict:
        """
        训练神经网络残差模型

        Args:
            X: 输入特征 (n_samples, 9) - [velocity, acceleration, euler_angles]
            y_true: 真实功率测量值 (n_samples,)
            epochs: 训练轮数
            verbose: 是否打印训练信息

        Returns:
            训练历史
        """
        # 计算物理模型预测
        y_physics = np.zeros(len(X))
        for i in range(len(X)):
            velocity = X[i, :3]
            acceleration = X[i, 3:6]
            y_physics[i] = self.physics_model.compute_electrical_power(velocity, acceleration)

        # 计算残差
        y_residual = y_true - y_physics

        print(f"Training neural residual model...")
        print(f"  Physics model mean power: {np.mean(y_physics):.2f} W")
        print(f"  True mean power: {np.mean(y_true):.2f} W")
        print(f"  Mean residual: {np.mean(y_residual):.2f} W")
        print(f"  Residual std: {np.std(y_residual):.2f} W")

        # 训练神经网络
        history = self.neural_model.train(X, y_residual, epochs=epochs, verbose=verbose)

        self.use_neural_compensation = True

        return history

    def save_neural_model(self, filepath: str):
        """保存神经网络模型"""
        self.neural_model.save(filepath)

    def load_neural_model(self, filepath: str):
        """加载神经网络模型"""
        self.neural_model.load(filepath)
        self.use_neural_compensation = True

    def get_power_breakdown(self, velocity: np.ndarray,
                            acceleration: Optional[np.ndarray] = None,
                            euler_angles: Optional[np.ndarray] = None) -> dict:
        """
        获取功率分解详情

        Returns:
            各项功率的字典
        """
        breakdown = self.physics_model.get_power_breakdown(velocity, acceleration)

        if self.use_neural_compensation and acceleration is not None and euler_angles is not None:
            breakdown['neural_residual'] = self.neural_model.predict(velocity, acceleration, euler_angles)
        else:
            breakdown['neural_residual'] = 0.0

        breakdown['hybrid_total'] = breakdown['electrical_total'] + breakdown['neural_residual']

        return breakdown

    def estimate_flight_range(self, velocity: float = 2.0) -> dict:
        """
        估算续航能力

        Args:
            velocity: 巡航速度 (m/s)

        Returns:
            续航信息字典
        """
        params = self.physics_model.params

        # 计算巡航功率
        vel_vector = np.array([velocity, 0, 0])
        cruise_power = self.compute_power(vel_vector, np.zeros(3), np.zeros(3))

        # 电池能量 (J)
        battery_energy = params.battery_voltage * params.battery_capacity / 1000 * 3600  # mAh to J

        # 续航时间 (s)
        flight_time = battery_energy / cruise_power

        # 续航距离 (m)
        flight_range = velocity * flight_time

        return {
            'cruise_power_watts': cruise_power,
            'hover_power_watts': self.physics_model.compute_hover_power(),
            'battery_energy_joules': battery_energy,
            'battery_energy_wh': battery_energy / 3600,
            'flight_time_seconds': flight_time,
            'flight_time_minutes': flight_time / 60,
            'flight_range_meters': flight_range,
            'flight_range_km': flight_range / 1000
        }


class EnergyCostFunction:
    """
    能耗代价函数（用于路径规划集成）

    可以作为 RRT* 的代价函数使用
    """

    def __init__(self, hybrid_model: Optional[HybridEnergyModel] = None,
                 velocity: float = 2.0,
                 weight_energy: float = 1.0,
                 weight_distance: float = 0.0):
        """
        初始化代价函数

        Args:
            hybrid_model: 混合能耗模型
            velocity: 飞行速度
            weight_energy: 能耗权重
            weight_distance: 距离权重
        """
        self.model = hybrid_model or HybridEnergyModel()
        self.velocity = velocity
        self.weight_energy = weight_energy
        self.weight_distance = weight_distance

    def compute_cost(self, start: np.ndarray, end: np.ndarray) -> float:
        """
        计算两点之间的代价

        Args:
            start: 起点
            end: 终点

        Returns:
            代价值
        """
        # 能耗代价
        energy_cost = self.model.compute_energy_cost(start, end, self.velocity)

        # 距离代价
        distance_cost = np.linalg.norm(end - start)

        # 加权总代价
        total_cost = self.weight_energy * energy_cost + self.weight_distance * distance_cost

        return total_cost

    def compute_path_cost(self, path: List[np.ndarray]) -> float:
        """
        计算路径总代价

        Args:
            path: 路径点列表

        Returns:
            总代价
        """
        total_cost = 0.0
        for i in range(len(path) - 1):
            total_cost += self.compute_cost(np.array(path[i]), np.array(path[i + 1]))
        return total_cost
