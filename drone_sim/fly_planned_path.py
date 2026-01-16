"""
fly_planned_path.py - Fly along planned path test
1. Get depth image and build map
2. Plan path
3. Control drone to fly along path
"""

import airsim
import numpy as np
import time
import cv2

from path_planning import PathPlanner, PlanningConfig


def get_depth_image(client):
    """Get depth image"""
    responses = client.simGetImages([
        airsim.ImageRequest("0", airsim.ImageType.DepthPlanar, True)
    ])
    depth = np.array(responses[0].image_data_float, dtype=np.float32)
    depth = depth.reshape(responses[0].height, responses[0].width)
    return depth


def get_drone_position(client):
    """Get drone position"""
    state = client.getMultirotorState()
    pos = state.kinematics_estimated.position
    return np.array([pos.x_val, pos.y_val, pos.z_val])


def fly_to_point(client, target, velocity=2.0):
    """
    Fly to target point

    Args:
        client: AirSim client
        target: Target position [x, y, z]
        velocity: Flight speed m/s
    """
    client.moveToPositionAsync(
        target[0], target[1], target[2],
        velocity,
        timeout_sec=30,
        drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
        yaw_mode=airsim.YawMode(is_rate=False, yaw_or_rate=0)
    ).join()


def main():
    print("=" * 60)
    print("Path Planning Flight Test")
    print("=" * 60)

    # Connect to AirSim
    print("\n[1] Connecting to AirSim...")
    client = airsim.MultirotorClient()
    client.confirmConnection()
    client.enableApiControl(True)
    client.armDisarm(True)
    print("Connected!")

    # Takeoff
    print("\n[2] Taking off...")
    client.takeoffAsync().join()
    time.sleep(1)

    # Fly to initial height (low altitude to test obstacle avoidance)
    initial_height = -2.0  # NED coordinate, negative = up, 2m height
    print(f"    Flying to height: {-initial_height}m")
    client.moveToZAsync(initial_height, 2).join()
    time.sleep(1)

    # Get current position and depth image
    print("\n[3] Getting environment info...")
    drone_pos = get_drone_position(client)
    depth = get_depth_image(client)
    print(f"    Drone position: ({drone_pos[0]:.2f}, {drone_pos[1]:.2f}, {drone_pos[2]:.2f})")
    print(f"    Depth image size: {depth.shape}")

    # Initialize planner
    print("\n[4] Initializing path planner...")
    config = PlanningConfig(
        voxel_size=0.5,
        grid_size=(80, 80, 40),
        origin=(drone_pos[0] - 10.0, drone_pos[1] - 20.0, drone_pos[2] - 10.0),
        max_depth=25.0,
        step_size=1.5,
        max_iterations=5000,
        goal_sample_rate=0.15,
        search_radius=3.0,
        safety_margin=1.0
    )
    planner = PathPlanner(config)

    # Update map
    print("\n[5] Building 3D map...")
    occupied_count = planner.update_map(depth, drone_pos)
    print(f"    Occupied voxels: {occupied_count}")

    # Set start and goal
    start = drone_pos.copy()
    start[0] -= 1.0  # Move back slightly to ensure safe start

    # Goal: bypass obstacle to front-right
    goal = np.array([drone_pos[0] + 15.0, drone_pos[1] + 10.0, drone_pos[2]])

    print(f"\n[6] Planning path...")
    print(f"    Start: ({start[0]:.2f}, {start[1]:.2f}, {start[2]:.2f})")
    print(f"    Goal:  ({goal[0]:.2f}, {goal[1]:.2f}, {goal[2]:.2f})")

    # Plan path
    path = planner.plan_path(start, goal)

    if path is None:
        print("    Path planning FAILED!")
        client.landAsync().join()
        client.armDisarm(False)
        client.enableApiControl(False)
        return

    print(f"    Planning SUCCESS! Waypoints: {len(path)}")
    for i, p in enumerate(path):
        print(f"      [{i}] ({p[0]:.2f}, {p[1]:.2f}, {p[2]:.2f})")

    # Fly along path
    print("\n[7] Flying along path...")
    flight_speed = 3.0  # m/s

    for i, waypoint in enumerate(path):
        print(f"\n    Flying to waypoint [{i}]: ({waypoint[0]:.2f}, {waypoint[1]:.2f}, {waypoint[2]:.2f})")

        # Fly to target
        fly_to_point(client, waypoint, velocity=flight_speed)

        # Get current position
        current_pos = get_drone_position(client)
        distance = np.linalg.norm(current_pos - waypoint)
        print(f"    Arrived! Position: ({current_pos[0]:.2f}, {current_pos[1]:.2f}, {current_pos[2]:.2f})")
        print(f"    Distance to target: {distance:.2f}m")

        time.sleep(0.5)

    print("\n[8] Path flight completed!")

    # Hover for a moment
    print("    Hovering for 3 seconds...")
    time.sleep(3)

    # Land
    print("\n[9] Landing...")
    client.landAsync().join()

    # Release control
    client.armDisarm(False)
    client.enableApiControl(False)

    print("\n" + "=" * 60)
    print("Test completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
