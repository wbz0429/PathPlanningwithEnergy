"""
main_control.py - 主控制程序
功能：串联所有模块，实现无人机自动飞行、障碍物检测和数据采集
"""

import airsim
import numpy as np
import cv2
import time
import os
import sys

# 添加当前目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from perception import ObstacleDetector, DetectionResult
from logger import FlightLogger, extract_state_from_airsim


class DroneController:
    """无人机控制器"""

    def __init__(self,
                 forward_speed: float = 2.0,
                 safety_threshold: float = 3.0,
                 loop_frequency: float = 10.0,
                 flight_duration: float = 60.0,
                 flight_height: float = -8.0):
        """
        初始化控制器

        Args:
            forward_speed: 前进速度 (m/s)
            safety_threshold: 安全距离阈值 (m)
            loop_frequency: 主循环频率 (Hz)
            flight_duration: 飞行时长 (s)
            flight_height: 飞行高度 (m)，NED坐标系，负值表示向上
        """
        self.forward_speed = forward_speed
        self.safety_threshold = safety_threshold
        self.loop_period = 1.0 / loop_frequency
        self.flight_duration = flight_duration
        self.flight_height = flight_height

        # 初始化 AirSim 客户端
        self.client = None

        # 初始化障碍物检测器
        self.detector = ObstacleDetector(safety_threshold=safety_threshold)

        # 初始化数据记录器
        log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
        self.logger = FlightLogger(log_dir=log_dir)

        # 状态标志
        self.is_hovering = False
        self.current_yaw = 0  # 当前航向角

    def connect(self):
        """连接仿真器"""
        print("正在连接 AirSim 仿真器...")
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        print("连接成功！")

    def setup(self):
        """初始化无人机"""
        # 获取控制权
        self.client.enableApiControl(True)
        print("已获取 API 控制权")

        # 解锁电机
        self.client.armDisarm(True)
        print("电机已解锁")

    def takeoff(self):
        """起飞并升高到安全高度"""
        print("正在起飞...")
        self.client.takeoffAsync().join()
        print("起飞完成！")

        # 升高到指定高度，避开地面障碍物
        print(f"正在升高到 {abs(self.flight_height)} 米...")
        self.client.moveToZAsync(self.flight_height, 2).join()
        print("已到达飞行高度！")

        # 稳定一下
        time.sleep(1)

    def get_depth_image(self) -> np.ndarray:
        """
        获取深度图

        Returns:
            深度图 (浮点数矩阵，单位米)
        """
        responses = self.client.simGetImages([
            airsim.ImageRequest("0", airsim.ImageType.DepthPlanar, True)
        ])

        if responses and responses[0].width > 0:
            # 解析深度数据
            depth = np.array(responses[0].image_data_float, dtype=np.float32)
            depth = depth.reshape(responses[0].height, responses[0].width)
            return depth
        else:
            return np.zeros((144, 256), dtype=np.float32)

    def get_r/gb_image(self) -> np.ndarray:
        """
        获取 RGB 图像

        Returns:
            RGB 图像
        """
        responses = self.client.simGetImages([
            airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)
        ])

        if responses and responses[0].width > 0:
            # 解析图像数据
            img1d = np.frombuffer(responses[0].image_data_uint8, dtype=np.uint8)
            img_rgb = img1d.reshape(responses[0].height, responses[0].width, 3)
            return img_rgb
        else:
            return np.zeros((144, 256, 3), dtype=np.uint8)

    def move_forward(self):
        """向前飞行"""
        if not self.is_hovering:
            # 使用 NED 坐标系，x 为前进方向
            self.client.moveByVelocityAsync(
                self.forward_speed, 0, 0,
                duration=self.loop_period * 2
            )

    def turn_right(self, angle_deg: float = 45):
        """向右转向避障"""
        print(f"\n[避障] 向右转向 {angle_deg} 度...")
        self.current_yaw += angle_deg
        # 转换为弧度
        yaw_rad = np.radians(self.current_yaw)
        self.client.rotateToYawAsync(self.current_yaw, timeout_sec=3).join()

    def turn_left(self, angle_deg: float = 45):
        """向左转向避障"""
        print(f"\n[避障] 向左转向 {angle_deg} 度...")
        self.current_yaw -= angle_deg
        self.client.rotateToYawAsync(self.current_yaw, timeout_sec=3).join()

    def hover(self):
        """悬停"""
        self.client.hoverAsync()
        self.is_hovering = True

    def resume_flight(self):
        """恢复飞行"""
        self.is_hovering = False

    def land(self):
        """降落"""
        print("正在降落...")
        self.client.landAsync().join()
        print("降落完成！")

    def cleanup(self):
        """清理资源"""
        self.client.armDisarm(False)
        self.client.enableApiControl(False)
        print("已释放控制权")

    def run(self):
        """主控制循环"""
        # 连接并初始化
        self.connect()
        self.setup()
        self.takeoff()

        # 开始记录数据
        self.logger.start()

        # 创建显示窗口
        cv2.namedWindow("Drone View", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Drone View", 800, 600)

        print(f"开始自动飞行，持续 {self.flight_duration} 秒...")
        print("按 'q' 键可提前退出")

        start_time = time.time()
        last_obstacle_time = 0

        try:
            while True:
                loop_start = time.time()

                # 检查是否超时
                elapsed = loop_start - start_time
                if elapsed >= self.flight_duration:
                    print(f"\n飞行时间已达 {self.flight_duration} 秒，准备降落")
                    break

                # 获取图像
                rgb_image = self.get_rgb_image()
                depth_image = self.get_depth_image()

                # 障碍物检测
                detection = self.detector.detect(depth_image)

                # 获取无人机状态
                state = self.client.getMultirotorState()
                state_data = extract_state_from_airsim(state)

                # 记录数据
                self.logger.log(
                    position=state_data["position"],
                    velocity=state_data["velocity"],
                    acceleration=state_data["acceleration"],
                    orientation=state_data["orientation"],
                    obstacle_detected=detection.has_obstacle,
                    obstacle_distance=detection.min_distance
                )

                # 控制逻辑
                if detection.has_obstacle:
                    if not self.is_hovering:
                        print(f"\n[警告] 检测到障碍物！距离: {detection.min_distance:.2f}m, "
                              f"位置: {detection.obstacle_position}")
                        self.hover()
                        last_obstacle_time = loop_start

                        # 尝试转向避障
                        time.sleep(0.5)
                        self.turn_right(45)
                        self.is_hovering = False  # 转向后继续飞行
                else:
                    # 没有障碍物，正常前进
                    self.is_hovering = False
                    self.move_forward()

                # 可视化
                vis_image = self.detector.visualize(rgb_image, detection)

                # 添加状态信息
                pos = state_data["position"]
                vel = state_data["velocity"]
                speed = np.sqrt(vel[0]**2 + vel[1]**2 + vel[2]**2)

                info_text = f"Pos: ({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f}) | Speed: {speed:.2f} m/s"
                cv2.putText(vis_image, info_text, (10, vis_image.shape[0] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                time_text = f"Time: {elapsed:.1f}s / {self.flight_duration}s"
                cv2.putText(vis_image, time_text, (vis_image.shape[1] - 200, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

                # 显示图像
                cv2.imshow("Drone View", vis_image)

                # 检查按键
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\n用户请求退出")
                    break

                # 控制循环频率
                loop_time = time.time() - loop_start
                if loop_time < self.loop_period:
                    time.sleep(self.loop_period - loop_time)

        except KeyboardInterrupt:
            print("\n检测到 Ctrl+C，正在退出...")

        finally:
            # 清理
            cv2.destroyAllWindows()
            self.logger.stop()
            self.land()
            self.cleanup()


def main():
    """主函数"""
    controller = DroneController(
        forward_speed=3.0,       # 前进速度 3 m/s
        safety_threshold=3.0,    # 安全距离 3 米
        loop_frequency=10.0,     # 10 Hz
        flight_duration=60.0,    # 飞行 60 秒
        flight_height=-8.0       # 飞行高度 8 米（负值表示向上）
    )

    controller.run()


if __name__ == "__main__":
    main()
