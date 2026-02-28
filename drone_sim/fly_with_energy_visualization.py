"""
fly_with_energy_visualization.py - 能量感知飞行与能耗可视化

在AirSim中执行飞行任务，实时记录并可视化能耗数据
"""

import numpy as np
import time
import sys
import os
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# 设置编码
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# 添加模块路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from planning.config import PlanningConfig
from mapping.incremental_map import IncrementalMapManager
from planning.receding_horizon import RecedingHorizonPlanner
from control.drone_interface import DroneInterface
from energy.physics_model import PhysicsEnergyModel


class EnergyVisualizer:
    """能耗可视化器"""

    def __init__(self):
        self.trajectory = []
        self.energy_data = []
        self.power_data = []
        self.time_data = []
        self.cumulative_energy = []
        self.segment_types = []  # 'horizontal', 'climb', 'descend'

        self.start_time = None
        self.energy_model = PhysicsEnergyModel()

        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False

    def start(self):
        """开始记录"""
        self.start_time = time.time()

    def record_segment(self, from_pos: np.ndarray, to_pos: np.ndarray, velocity: float):
        """记录一个飞行段的能耗"""
        if self.start_time is None:
            self.start()

        # 计算能耗
        energy, flight_time = self.energy_model.compute_energy_for_segment(
            from_pos, to_pos, velocity
        )

        # 计算功率
        power = energy / flight_time if flight_time > 0 else 0

        # 计算高度变化
        dz = to_pos[2] - from_pos[2]  # NED: 负值表示爬升
        if dz < -0.5:
            seg_type = 'climb'
        elif dz > 0.5:
            seg_type = 'descend'
        else:
            seg_type = 'horizontal'

        # 记录数据
        current_time = time.time() - self.start_time
        self.trajectory.append(to_pos.copy())
        self.energy_data.append(energy)
        self.power_data.append(power)
        self.time_data.append(current_time)
        self.segment_types.append(seg_type)

        # 累积能耗
        total = sum(self.energy_data)
        self.cumulative_energy.append(total)

        return energy, power

    def get_summary(self) -> dict:
        """获取能耗摘要"""
        if not self.energy_data:
            return {}

        total_energy = sum(self.energy_data)
        total_distance = 0
        if len(self.trajectory) > 1:
            for i in range(len(self.trajectory) - 1):
                total_distance += np.linalg.norm(
                    self.trajectory[i+1] - self.trajectory[i]
                )

        # 按类型统计
        climb_energy = sum(e for e, t in zip(self.energy_data, self.segment_types) if t == 'climb')
        descend_energy = sum(e for e, t in zip(self.energy_data, self.segment_types) if t == 'descend')
        horizontal_energy = sum(e for e, t in zip(self.energy_data, self.segment_types) if t == 'horizontal')

        return {
            'total_energy_j': total_energy,
            'total_energy_wh': total_energy / 3600,
            'total_distance_m': total_distance,
            'energy_per_meter': total_energy / total_distance if total_distance > 0 else 0,
            'avg_power_w': np.mean(self.power_data) if self.power_data else 0,
            'max_power_w': max(self.power_data) if self.power_data else 0,
            'min_power_w': min(self.power_data) if self.power_data else 0,
            'climb_energy_j': climb_energy,
            'descend_energy_j': descend_energy,
            'horizontal_energy_j': horizontal_energy,
            'num_segments': len(self.energy_data),
            'flight_time_s': self.time_data[-1] if self.time_data else 0
        }

    def plot_results(self, save_path: str = 'energy_visualization.png'):
        """绘制能耗可视化图"""
        if not self.energy_data:
            print("No data to plot!")
            return

        fig = plt.figure(figsize=(16, 12))
        gs = GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.25)

        summary = self.get_summary()

        # 1. 3D轨迹图（带能耗着色）
        ax1 = fig.add_subplot(gs[0, 0], projection='3d')
        if len(self.trajectory) > 1:
            traj = np.array(self.trajectory)
            # 使用累积能耗着色
            colors = plt.cm.hot(np.array(self.cumulative_energy) / max(self.cumulative_energy))
            for i in range(len(traj) - 1):
                ax1.plot3D(traj[i:i+2, 0], traj[i:i+2, 1], -traj[i:i+2, 2],
                          color=colors[i], linewidth=2)
            ax1.scatter(traj[0, 0], traj[0, 1], -traj[0, 2], c='green', s=100, marker='o', label='Start')
            ax1.scatter(traj[-1, 0], traj[-1, 1], -traj[-1, 2], c='red', s=100, marker='*', label='End')
        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')
        ax1.set_zlabel('Altitude (m)')
        ax1.set_title('3D Trajectory (colored by cumulative energy)')
        ax1.legend()

        # 2. 累积能耗曲线
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(self.time_data, self.cumulative_energy, 'b-', linewidth=2)
        ax2.fill_between(self.time_data, self.cumulative_energy, alpha=0.3)
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Cumulative Energy (J)')
        ax2.set_title(f'Cumulative Energy Consumption\nTotal: {summary["total_energy_j"]:.1f} J ({summary["total_energy_wh"]:.3f} Wh)')
        ax2.grid(True, alpha=0.3)

        # 3. 功率曲线
        ax3 = fig.add_subplot(gs[1, 0])
        colors = {'horizontal': 'blue', 'climb': 'red', 'descend': 'green'}
        for i, (t, p, seg_type) in enumerate(zip(self.time_data, self.power_data, self.segment_types)):
            ax3.bar(t, p, width=0.5, color=colors[seg_type], alpha=0.7)
        ax3.axhline(y=summary['avg_power_w'], color='orange', linestyle='--',
                   label=f'Avg: {summary["avg_power_w"]:.1f} W')
        ax3.axhline(y=self.energy_model.compute_hover_power() /
                   (self.energy_model.params.motor_efficiency * self.energy_model.params.esc_efficiency),
                   color='gray', linestyle=':', label='Hover power')
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Power (W)')
        ax3.set_title('Power Consumption per Segment')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # 4. 能耗分布饼图
        ax4 = fig.add_subplot(gs[1, 1])
        energy_by_type = [
            summary['horizontal_energy_j'],
            summary['climb_energy_j'],
            summary['descend_energy_j']
        ]
        labels = ['Horizontal', 'Climb', 'Descend']
        colors_pie = ['#3498db', '#e74c3c', '#2ecc71']
        non_zero = [(l, e, c) for l, e, c in zip(labels, energy_by_type, colors_pie) if e > 0]
        if non_zero:
            labels_nz, energy_nz, colors_nz = zip(*non_zero)
            wedges, texts, autotexts = ax4.pie(energy_nz, labels=labels_nz, colors=colors_nz,
                                               autopct='%1.1f%%', startangle=90)
            ax4.set_title('Energy Distribution by Flight Type')

        # 5. 高度-能耗关系
        ax5 = fig.add_subplot(gs[2, 0])
        if len(self.trajectory) > 1 and len(self.energy_data) > 0:
            # trajectory比energy_data多一个起点，需要对齐
            altitudes = [-p[2] for p in self.trajectory[1:]]  # 跳过起点
            if len(altitudes) == len(self.energy_data):
                ax5.scatter(altitudes, self.energy_data, c=range(len(altitudes)),
                           cmap='viridis', s=50, alpha=0.7)
                ax5.set_xlabel('Altitude (m)')
                ax5.set_ylabel('Segment Energy (J)')
                ax5.set_title('Energy vs Altitude')
                ax5.grid(True, alpha=0.3)

        # 6. 统计摘要
        ax6 = fig.add_subplot(gs[2, 1])
        ax6.axis('off')

        stats_text = f"""
        Flight Statistics Summary
        ========================

        Total Energy:     {summary['total_energy_j']:.1f} J ({summary['total_energy_wh']:.3f} Wh)
        Total Distance:   {summary['total_distance_m']:.1f} m
        Flight Time:      {summary['flight_time_s']:.1f} s

        Energy Efficiency: {summary['energy_per_meter']:.1f} J/m

        Power Statistics:
          Average:  {summary['avg_power_w']:.1f} W
          Maximum:  {summary['max_power_w']:.1f} W
          Minimum:  {summary['min_power_w']:.1f} W

        Energy by Type:
          Horizontal: {summary['horizontal_energy_j']:.1f} J
          Climb:      {summary['climb_energy_j']:.1f} J
          Descend:    {summary['descend_energy_j']:.1f} J

        Battery Usage:    {summary['total_energy_wh'] / 74.0 * 100:.1f}%
        (74 Wh battery)
        """
        ax6.text(0.1, 0.9, stats_text, transform=ax6.transAxes, fontsize=11,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.suptitle('Energy-Aware Flight Analysis', fontsize=14, fontweight='bold')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")
        plt.close()


def main():
    print("\n" + "=" * 70)
    print("  ENERGY-AWARE FLIGHT WITH VISUALIZATION")
    print("=" * 70)

    # 初始化能耗可视化器
    energy_viz = EnergyVisualizer()

    # === 1. 连接AirSim ===
    print("\n[1] Connecting to AirSim...")
    drone = DroneInterface()

    try:
        drone.connect()
    except Exception as e:
        print(f"Failed to connect to AirSim: {e}")
        print("Make sure AirSim is running!")
        return

    try:
        # === 2. 重置到初始位置 ===
        print("\n[2] Resetting to initial position...")
        drone.reset()
        time.sleep(1)

        # === 3. 起飞 ===
        print("\n[3] Taking off...")
        drone.takeoff()
        time.sleep(1)

        initial_height = -5.0  # 5m altitude
        print(f"    Flying to altitude: {-initial_height}m")
        drone.move_to_z(initial_height, velocity=2.0)
        time.sleep(1)

        # === 4. 获取初始位置 ===
        print("\n[4] Getting initial position...")
        initial_pos, initial_ori = drone.get_pose()
        print(f"    Position: ({initial_pos[0]:.2f}, {initial_pos[1]:.2f}, {initial_pos[2]:.2f})")

        # 记录起点
        energy_viz.start()
        energy_viz.trajectory.append(initial_pos.copy())

        # === 5. 初始化规划器 ===
        print("\n[5] Initializing planner...")

        planning_config = PlanningConfig(
            voxel_size=0.5,
            grid_size=(120, 120, 40),  # 扩大地图覆盖范围
            origin=(-10.0, -30.0, -15.0),  # 覆盖更大区域
            max_depth=25.0,
            step_size=1.0,  # 减小步长
            max_iterations=3000,
            safety_margin=3.0,  # 增大安全边距到3米
            energy_aware=True,
            weight_energy=0.6,
            weight_distance=0.3,
            weight_time=0.1
        )

        energy_model = PhysicsEnergyModel()
        print(f"    Energy model: BEMT physics")
        print(f"    Hover power: {energy_model.compute_hover_power():.1f} W")

        receding_config = {
            'local_horizon': 6.0,  # 减小局部目标距离
            'execution_ratio': 0.3,  # 只执行30%的路径，更频繁重规划
            'goal_tolerance': 2.0,
            'max_iterations': 60,
            'flight_velocity': 1.5,  # 降低速度
            'visualize': False
        }

        map_manager = IncrementalMapManager(planning_config)
        planner = RecedingHorizonPlanner(map_manager, drone, receding_config, energy_model=energy_model)

        # === 6. 设置目标 ===
        print("\n[6] Setting goal...")

        # 根据场景扫描结果：
        # - 障碍物在 X≈22m 处
        # - 需要从侧面绕过
        # 目标设在障碍物后方右侧，迫使无人机绕行
        global_goal = np.array([30.0, 20.0, initial_pos[2]])

        print(f"    Start: ({initial_pos[0]:.2f}, {initial_pos[1]:.2f}, {initial_pos[2]:.2f})")
        print(f"    Goal:  ({global_goal[0]:.2f}, {global_goal[1]:.2f}, {global_goal[2]:.2f})")
        print(f"    Distance: {np.linalg.norm(global_goal - initial_pos):.2f}m")
        print(f"    Note: Obstacle at X~22m, drone must navigate around")

        # === 7. 执行飞行（手动控制以记录能耗） ===
        print("\n[7] Starting flight with energy tracking...")

        iteration = 0
        max_iterations = receding_config['max_iterations']
        current_pos = initial_pos.copy()
        flight_velocity = receding_config['flight_velocity']

        while iteration < max_iterations:
            iteration += 1

            # 检查是否到达目标
            dist_to_goal = np.linalg.norm(current_pos - global_goal)
            if dist_to_goal < receding_config['goal_tolerance']:
                print(f"\n[SUCCESS] Reached goal!")
                break

            # 获取当前状态
            current_pos, current_ori = drone.get_pose()
            depth_image = drone.get_depth_image()

            print(f"\n--- Iteration {iteration} ---")
            print(f"Position: ({current_pos[0]:.2f}, {current_pos[1]:.2f}, {current_pos[2]:.2f})")
            print(f"Distance to goal: {dist_to_goal:.2f}m")

            # 更新地图
            map_stats = map_manager.update(depth_image, current_pos, current_ori)

            # 选择局部目标
            direction = global_goal - current_pos
            distance = np.linalg.norm(direction)
            if distance > receding_config['local_horizon']:
                direction = direction / distance
                local_goal = current_pos + direction * receding_config['local_horizon']
            else:
                local_goal = global_goal.copy()
            local_goal[2] = current_pos[2]  # 保持高度

            # 规划路径
            from planning.rrt_star import RRTStar
            rrt = RRTStar(map_manager.voxel_grid, map_manager.esdf, planning_config, energy_model)
            path = rrt.plan(current_pos, local_goal)

            if path is None or len(path) < 2:
                print("  Planning failed, trying direct approach...")
                # 直接朝目标移动一小步
                step = direction * min(2.0, distance)
                next_pos = current_pos + step
                next_pos[2] = current_pos[2]
                path = [current_pos, next_pos]

            # 执行路径并记录能耗
            execution_length = max(1, int(len(path) * receding_config['execution_ratio']))
            waypoints = path[1:execution_length+1]

            prev_pos = current_pos.copy()
            for wp in waypoints:
                wp_fixed = wp.copy()
                wp_fixed[2] = global_goal[2]

                # 记录能耗
                energy, power = energy_viz.record_segment(prev_pos, wp_fixed, flight_velocity)
                print(f"  -> Moving to ({wp_fixed[0]:.2f}, {wp_fixed[1]:.2f}, {wp_fixed[2]:.2f})")
                print(f"     Energy: {energy:.1f}J, Power: {power:.1f}W")

                # 执行移动
                drone.move_to_position(wp_fixed, velocity=flight_velocity)
                prev_pos = wp_fixed.copy()

            current_pos = prev_pos

        # === 7. 生成可视化 ===
        print("\n[7] Generating energy visualization...")
        energy_viz.plot_results('energy_flight_visualization.png')

        # 打印摘要
        summary = energy_viz.get_summary()
        print("\n" + "=" * 50)
        print("ENERGY CONSUMPTION SUMMARY")
        print("=" * 50)
        print(f"Total Energy:     {summary['total_energy_j']:.1f} J ({summary['total_energy_wh']:.3f} Wh)")
        print(f"Total Distance:   {summary['total_distance_m']:.1f} m")
        print(f"Energy/meter:     {summary['energy_per_meter']:.1f} J/m")
        print(f"Average Power:    {summary['avg_power_w']:.1f} W")
        print(f"Battery Used:     {summary['total_energy_wh']/74.0*100:.1f}%")
        print("=" * 50)

        # === 8. 降落 ===
        print("\n[8] Landing...")
        drone.hover()
        time.sleep(2)
        drone.land()

    except KeyboardInterrupt:
        print("\n\nInterrupted! Emergency landing...")
        drone.land()

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        drone.land()

    finally:
        drone.disconnect()
        print("\nFlight completed!")


if __name__ == "__main__":
    main()
