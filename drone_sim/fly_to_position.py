"""
fly_to_position.py - 飞到指定位置并显示坐标

用法:
  python fly_to_position.py          # 交互式输入坐标
  python fly_to_position.py 10 5 -5  # 直接飞到 (10, 5, -5)
"""

import airsim
import sys
import time

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')


def get_current_position(client):
    """获取当前位置"""
    state = client.getMultirotorState()
    pos = state.kinematics_estimated.position
    return pos.x_val, pos.y_val, pos.z_val


def main():
    print("="*60)
    print("AirSim Fly To Position Tool")
    print("="*60)

    # 连接AirSim
    client = airsim.MultirotorClient()
    client.confirmConnection()
    client.enableApiControl(True)
    client.armDisarm(True)
    print("Connected!")

    # 获取初始位置
    x, y, z = get_current_position(client)
    print(f"Current position: ({x:.2f}, {y:.2f}, {z:.2f})")

    # 起飞
    print("\nTaking off...")
    client.takeoffAsync().join()
    time.sleep(1)

    x, y, z = get_current_position(client)
    print(f"After takeoff: ({x:.2f}, {y:.2f}, {z:.2f})")

    # 如果命令行有参数，直接飞到指定位置
    if len(sys.argv) == 4:
        target_x = float(sys.argv[1])
        target_y = float(sys.argv[2])
        target_z = float(sys.argv[3])
        print(f"\nFlying to ({target_x}, {target_y}, {target_z})...")
        client.moveToPositionAsync(target_x, target_y, target_z, 3).join()
        x, y, z = get_current_position(client)
        print(f"Arrived at: ({x:.2f}, {y:.2f}, {z:.2f})")

    else:
        # 交互式模式
        print("\n" + "="*60)
        print("Interactive Mode")
        print("="*60)
        print("Commands:")
        print("  x y z    - Fly to position (e.g., '10 5 -5')")
        print("  f x      - Fly forward x meters (e.g., 'f 10')")
        print("  r y      - Fly right y meters (e.g., 'r 5')")
        print("  u z      - Fly up z meters (e.g., 'u 3')")
        print("  p        - Print current position")
        print("  land     - Land the drone")
        print("  q        - Quit")
        print()

        while True:
            x, y, z = get_current_position(client)
            cmd = input(f"[{x:.1f}, {y:.1f}, {z:.1f}] > ").strip().lower()

            if not cmd:
                continue

            if cmd == 'q':
                break

            elif cmd == 'p':
                print(f"  Position: ({x:.2f}, {y:.2f}, {z:.2f})")
                print(f"  Code: np.array([{x:.2f}, {y:.2f}, {z:.2f}])")

            elif cmd == 'land':
                print("  Landing...")
                client.landAsync().join()
                print("  Landed!")

            elif cmd.startswith('f '):
                try:
                    dist = float(cmd[2:])
                    target_x = x + dist
                    print(f"  Flying forward {dist}m to X={target_x:.1f}...")
                    client.moveToPositionAsync(target_x, y, z, 3).join()
                except ValueError:
                    print("  Invalid distance")

            elif cmd.startswith('r '):
                try:
                    dist = float(cmd[2:])
                    target_y = y + dist
                    print(f"  Flying right {dist}m to Y={target_y:.1f}...")
                    client.moveToPositionAsync(x, target_y, z, 3).join()
                except ValueError:
                    print("  Invalid distance")

            elif cmd.startswith('u '):
                try:
                    dist = float(cmd[2:])
                    target_z = z - dist  # NED: 负值是上
                    print(f"  Flying up {dist}m to Z={target_z:.1f}...")
                    client.moveToPositionAsync(x, y, target_z, 2).join()
                except ValueError:
                    print("  Invalid distance")

            else:
                # 尝试解析为坐标
                try:
                    parts = cmd.split()
                    if len(parts) == 3:
                        target_x = float(parts[0])
                        target_y = float(parts[1])
                        target_z = float(parts[2])
                        print(f"  Flying to ({target_x}, {target_y}, {target_z})...")
                        client.moveToPositionAsync(target_x, target_y, target_z, 3).join()
                    else:
                        print("  Unknown command. Use 'x y z' or 'f/r/u distance'")
                except ValueError:
                    print("  Invalid input")

    # 清理
    print("\nDisconnecting...")
    client.armDisarm(False)
    client.enableApiControl(False)
    print("Done!")


if __name__ == "__main__":
    main()
