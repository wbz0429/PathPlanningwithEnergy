"""
get_target_position.py - 获取目标点坐标工具

手动控制无人机飞到目标位置，按Enter记录坐标
"""

import airsim
import time
import sys

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

def main():
    print("="*60)
    print("AirSim Target Position Tool")
    print("="*60)
    print()
    print("Instructions:")
    print("  1. Use AirSim window to manually fly the drone")
    print("  2. Press Enter here to record current position")
    print("  3. Type 'q' to quit")
    print()

    client = airsim.MultirotorClient()
    client.confirmConnection()
    print("Connected to AirSim!")
    print()

    recorded_positions = []

    while True:
        # 获取当前位置
        state = client.getMultirotorState()
        pos = state.kinematics_estimated.position

        print(f"\rCurrent position: X={pos.x_val:7.2f}, Y={pos.y_val:7.2f}, Z={pos.z_val:7.2f}  ", end="")

        # 检查用户输入 (非阻塞)
        import msvcrt
        if msvcrt.kbhit():
            key = msvcrt.getch().decode('utf-8', errors='ignore')

            if key == '\r' or key == '\n':  # Enter
                pos_tuple = (pos.x_val, pos.y_val, pos.z_val)
                recorded_positions.append(pos_tuple)
                print(f"\n\n[RECORDED] Position {len(recorded_positions)}: ({pos.x_val:.2f}, {pos.y_val:.2f}, {pos.z_val:.2f})")
                print()

            elif key == 'q' or key == 'Q':
                break

        time.sleep(0.1)

    # 打印所有记录的位置
    print("\n")
    print("="*60)
    print("Recorded Positions:")
    print("="*60)

    for i, pos in enumerate(recorded_positions):
        print(f"  {i+1}. np.array([{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}])")

    if recorded_positions:
        print()
        print("Copy-paste for goal:")
        last = recorded_positions[-1]
        print(f"  global_goal = np.array([{last[0]:.2f}, {last[1]:.2f}, {last[2]:.2f}])")


if __name__ == "__main__":
    main()
