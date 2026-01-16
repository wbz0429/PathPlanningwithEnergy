"""
connect_test.py - AirSim/Colosseum 连接测试脚本
功能：测试与仿真器的连接，控制无人机起飞并悬停
"""

import airsim
import time


def main():
    # 连接到 AirSim 仿真器
    print("正在连接 AirSim 仿真器...")
    client = airsim.MultirotorClient()
    client.confirmConnection()
    print("连接成功！")

    # 获取无人机控制权
    client.enableApiControl(True)
    print("已获取 API 控制权")

    # 解锁电机
    client.armDisarm(True)
    print("电机已解锁")

    # 起飞
    print("正在起飞...")
    client.takeoffAsync().join()
    print("起飞完成！")

    # 悬停 5 秒
    print("悬停中...")
    time.sleep(5)

    # 获取当前状态
    state = client.getMultirotorState()
    pos = state.kinematics_estimated.position
    print(f"当前位置: x={pos.x_val:.2f}, y={pos.y_val:.2f}, z={pos.z_val:.2f}")

    # 降落
    print("正在降落...")
    client.landAsync().join()
    print("降落完成！")

    # 锁定电机并释放控制权
    client.armDisarm(False)
    client.enableApiControl(False)
    print("测试完成！")


if __name__ == "__main__":
    main()
