"""
Physics-based Energy Model for Quadrotor UAV

基于动量叶素理论（BEMT）的物理能耗模型
参考文献：
- Michael N, et al. Control of quadrotors using the blade element momentum theory. ICRA, 2010.
- Bauersfeld L, et al. NeuroBEM: Hybrid Aerodynamic Quadrotor Model. IEEE RA-L, 2021.
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional


@dataclass
class QuadrotorParams:
    """四旋翼无人机物理参数"""
    # 质量与惯性
    mass: float = 1.5  # 总质量 (kg)
    g: float = 9.81  # 重力加速度 (m/s^2)

    # 旋翼参数
    num_rotors: int = 4  # 旋翼数量
    rotor_radius: float = 0.12  # 旋翼半径 (m)
    blade_chord: float = 0.02  # 叶片弦长 (m)
    num_blades: int = 2  # 每个旋翼的叶片数

    # 空气动力学参数
    air_density: float = 1.225  # 空气密度 (kg/m^3)
    thrust_coefficient: float = 0.012  # 推力系数 Ct
    power_coefficient: float = 0.0015  # 功率系数 Cp
    induced_power_factor: float = 1.15  # 诱导功率因子 κ
    profile_drag_coefficient: float = 0.02  # 叶片剖面阻力系数

    # 机身阻力参数
    drag_coefficient: float = 0.5  # 机身阻力系数
    frontal_area: float = 0.02  # 迎风面积 (m^2)

    # 电机效率
    motor_efficiency: float = 0.85  # 电机效率
    esc_efficiency: float = 0.95  # 电调效率

    # 电池参数
    battery_voltage: float = 14.8  # 电池标称电压 (V)
    battery_capacity: float = 5000  # 电池容量 (mAh)


class PhysicsEnergyModel:
    """
    基于 BEMT 理论的物理能耗模型

    计算四旋翼在不同飞行状态下的功率消耗：
    - 悬停功率 (Hover Power)
    - 诱导功率 (Induced Power)
    - 剖面功率 (Profile Power)
    - 寄生功率 (Parasite Power)
    - 爬升功率 (Climb Power)
    """

    def __init__(self, params: Optional[QuadrotorParams] = None):
        self.params = params or QuadrotorParams()
        self._precompute_constants()

    def _precompute_constants(self):
        """预计算常用常量"""
        p = self.params

        # 旋翼盘面积
        self.rotor_disk_area = np.pi * p.rotor_radius ** 2
        self.total_disk_area = p.num_rotors * self.rotor_disk_area

        # 悬停时的诱导速度 (m/s)
        # v_h = sqrt(T / (2 * rho * A)) = sqrt(mg / (2 * rho * A))
        self.hover_induced_velocity = np.sqrt(
            (p.mass * p.g) / (2 * p.air_density * self.total_disk_area)
        )

        # 悬停功率 (W)
        # P_hover = T * v_h = mg * v_h
        self.hover_power = p.mass * p.g * self.hover_induced_velocity * p.induced_power_factor

        # 叶片实度 (solidity)
        self.solidity = (p.num_blades * p.blade_chord) / (np.pi * p.rotor_radius)

        # 叶尖速度（假设悬停时的转速）
        # 从推力系数反推：T = Ct * rho * A * (Omega * R)^2
        thrust_per_rotor = (p.mass * p.g) / p.num_rotors
        self.hover_tip_speed = np.sqrt(
            thrust_per_rotor / (p.thrust_coefficient * p.air_density * self.rotor_disk_area)
        )

    def compute_hover_power(self) -> float:
        """
        计算悬停功率 (W)

        Returns:
            悬停状态下的机械功率
        """
        return self.hover_power

    def compute_induced_power(self, velocity: np.ndarray, thrust: Optional[float] = None) -> float:
        """
        计算诱导功率 (W)

        诱导功率是克服旋翼下洗流所需的功率

        Args:
            velocity: 飞行速度向量 [vx, vy, vz] (m/s)，NED坐标系
            thrust: 推力 (N)，默认为悬停推力

        Returns:
            诱导功率 (W)
        """
        p = self.params

        if thrust is None:
            thrust = p.mass * p.g

        # 水平速度和垂直速度
        v_horizontal = np.sqrt(velocity[0]**2 + velocity[1]**2)
        v_vertical = -velocity[2]  # NED坐标系，向上为负

        # 悬停诱导速度
        v_h = self.hover_induced_velocity

        # 归一化速度
        mu = v_horizontal / (self.hover_tip_speed + 1e-6)  # 前进比
        lambda_c = v_vertical / (self.hover_tip_speed + 1e-6)  # 爬升比

        # 使用 Glauert 公式计算诱导速度
        # 对于前飞状态，诱导速度会减小
        if v_horizontal < 0.1:  # 近似悬停
            v_induced = v_h
        else:
            # 前飞时的诱导速度近似
            v_induced = thrust / (2 * p.air_density * self.total_disk_area *
                                  np.sqrt(v_horizontal**2 + (v_vertical + v_h)**2))

        # 诱导功率 = κ * T * v_i
        P_induced = p.induced_power_factor * thrust * v_induced

        return P_induced

    def compute_profile_power(self, velocity: np.ndarray) -> float:
        """
        计算剖面功率 (W)

        剖面功率是克服叶片剖面阻力所需的功率

        Args:
            velocity: 飞行速度向量 [vx, vy, vz] (m/s)

        Returns:
            剖面功率 (W)
        """
        p = self.params

        # 水平速度
        v_horizontal = np.sqrt(velocity[0]**2 + velocity[1]**2)

        # 前进比
        mu = v_horizontal / (self.hover_tip_speed + 1e-6)

        # 剖面功率公式
        # P_profile = (sigma * Cd0 / 8) * rho * A * (Omega * R)^3 * (1 + 4.65 * mu^2)
        P_profile_hover = (self.solidity * p.profile_drag_coefficient / 8) * \
                          p.air_density * self.total_disk_area * self.hover_tip_speed**3

        # 前飞时剖面功率增加
        P_profile = P_profile_hover * (1 + 4.65 * mu**2)

        return P_profile

    def compute_parasite_power(self, velocity: np.ndarray) -> float:
        """
        计算寄生功率/机身阻力功率 (W)

        Args:
            velocity: 飞行速度向量 [vx, vy, vz] (m/s)

        Returns:
            寄生功率 (W)
        """
        p = self.params

        # 总速度
        v_total = np.linalg.norm(velocity)

        # 寄生功率 = 0.5 * rho * Cd * A * V^3
        P_parasite = 0.5 * p.air_density * p.drag_coefficient * p.frontal_area * v_total**3

        return P_parasite

    def compute_climb_power(self, velocity: np.ndarray) -> float:
        """
        计算爬升功率 (W)

        Args:
            velocity: 飞行速度向量 [vx, vy, vz] (m/s)，NED坐标系

        Returns:
            爬升功率 (W)，下降时为负
        """
        p = self.params

        # 垂直速度（NED坐标系，向上为负）
        v_climb = -velocity[2]

        # 爬升功率 = m * g * v_climb
        P_climb = p.mass * p.g * v_climb

        return P_climb

    def compute_acceleration_power(self, velocity: np.ndarray, acceleration: np.ndarray) -> float:
        """
        计算加速功率 (W)

        Args:
            velocity: 飞行速度向量 [vx, vy, vz] (m/s)
            acceleration: 加速度向量 [ax, ay, az] (m/s^2)

        Returns:
            加速功率 (W)
        """
        p = self.params

        # 加速功率 = m * a · v
        P_accel = p.mass * np.dot(acceleration, velocity)

        return P_accel

    def compute_total_mechanical_power(self, velocity: np.ndarray,
                                        acceleration: Optional[np.ndarray] = None) -> float:
        """
        计算总机械功率 (W)

        Args:
            velocity: 飞行速度向量 [vx, vy, vz] (m/s)
            acceleration: 加速度向量 [ax, ay, az] (m/s^2)，可选

        Returns:
            总机械功率 (W)
        """
        P_induced = self.compute_induced_power(velocity)
        P_profile = self.compute_profile_power(velocity)
        P_parasite = self.compute_parasite_power(velocity)
        P_climb = self.compute_climb_power(velocity)

        P_total = P_induced + P_profile + P_parasite + P_climb

        if acceleration is not None:
            P_accel = self.compute_acceleration_power(velocity, acceleration)
            P_total += P_accel

        return max(P_total, 0)  # 功率不能为负

    def compute_electrical_power(self, velocity: np.ndarray,
                                  acceleration: Optional[np.ndarray] = None) -> float:
        """
        计算电气功率 (W)

        考虑电机和电调效率

        Args:
            velocity: 飞行速度向量 [vx, vy, vz] (m/s)
            acceleration: 加速度向量 [ax, ay, az] (m/s^2)，可选

        Returns:
            电气功率 (W)
        """
        p = self.params

        P_mechanical = self.compute_total_mechanical_power(velocity, acceleration)
        P_electrical = P_mechanical / (p.motor_efficiency * p.esc_efficiency)

        return P_electrical

    def compute_energy_for_segment(self, start_pos: np.ndarray, end_pos: np.ndarray,
                                    velocity: float, dt: float = 0.1) -> Tuple[float, float]:
        """
        计算路径段的能耗

        Args:
            start_pos: 起点位置 [x, y, z]
            end_pos: 终点位置 [x, y, z]
            velocity: 飞行速度 (m/s)
            dt: 时间步长 (s)

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
        vel_vector = direction * velocity

        # 飞行时间
        flight_time = distance / velocity

        # 计算功率
        power = self.compute_electrical_power(vel_vector)

        # 能耗 = 功率 × 时间
        energy = power * flight_time

        return energy, flight_time

    def compute_energy_for_path(self, path: list, velocity: float = 2.0) -> Tuple[float, float]:
        """
        计算整条路径的能耗

        Args:
            path: 路径点列表 [np.array([x,y,z]), ...]
            velocity: 飞行速度 (m/s)

        Returns:
            (总能耗 (J), 总飞行时间 (s))
        """
        total_energy = 0.0
        total_time = 0.0

        for i in range(len(path) - 1):
            energy, time = self.compute_energy_for_segment(
                np.array(path[i]), np.array(path[i+1]), velocity
            )
            total_energy += energy
            total_time += time

        return total_energy, total_time

    def get_power_breakdown(self, velocity: np.ndarray,
                            acceleration: Optional[np.ndarray] = None) -> dict:
        """
        获取功率分解详情

        Args:
            velocity: 飞行速度向量 [vx, vy, vz] (m/s)
            acceleration: 加速度向量，可选

        Returns:
            各项功率的字典
        """
        breakdown = {
            'induced': self.compute_induced_power(velocity),
            'profile': self.compute_profile_power(velocity),
            'parasite': self.compute_parasite_power(velocity),
            'climb': self.compute_climb_power(velocity),
            'acceleration': 0.0
        }

        if acceleration is not None:
            breakdown['acceleration'] = self.compute_acceleration_power(velocity, acceleration)

        breakdown['mechanical_total'] = sum(breakdown.values())
        breakdown['electrical_total'] = self.compute_electrical_power(velocity, acceleration)

        return breakdown


# 便捷函数
def estimate_flight_energy(path: list, velocity: float = 2.0,
                           params: Optional[QuadrotorParams] = None) -> dict:
    """
    估算飞行能耗的便捷函数

    Args:
        path: 路径点列表
        velocity: 飞行速度 (m/s)
        params: 无人机参数，可选

    Returns:
        包含能耗信息的字典
    """
    model = PhysicsEnergyModel(params)
    energy, time = model.compute_energy_for_path(path, velocity)

    return {
        'total_energy_joules': energy,
        'total_energy_wh': energy / 3600,
        'flight_time_seconds': time,
        'average_power_watts': energy / time if time > 0 else 0,
        'hover_power_watts': model.compute_hover_power()
    }
