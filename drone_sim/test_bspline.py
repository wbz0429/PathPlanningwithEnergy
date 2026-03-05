"""test_bspline.py - 测试 B-spline 平滑效果"""
import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from planning.config import PlanningConfig
from mapping.voxel_grid import VoxelGrid
from mapping.esdf import ESDF
from planning.rrt_star import RRTStar
from energy.physics_model import PhysicsEnergyModel
from benchmark_planning import build_known_map, compute_path_length

config = PlanningConfig(
    voxel_size=0.5, grid_size=(180,120,40), origin=(-10,-30,-15),
    step_size=1.5, max_iterations=3000, goal_sample_rate=0.2,
    search_radius=4.0, safety_margin=1.0,
    energy_aware=True, weight_energy=0.6, weight_distance=0.3, weight_time=0.1,
    flight_velocity=2.0,
)

print("Building map...")
vg = build_known_map(config)
esdf = ESDF(vg)
esdf.compute()
em = PhysicsEnergyModel()

start = np.array([0, 0, -3.0])
goal = np.array([70, 20, -3.0])

print(f"\nPlanning {start} -> {goal}...")
planner = RRTStar(vg, esdf, config, energy_model=em)
path = planner.plan(start, goal)

if path:
    arr = np.array(path)
    length = compute_path_length(path)
    print(f"Path points: {len(path)}, Length: {length:.2f}m")

    # 计算转弯角度
    diffs = np.diff(arr, axis=0)
    angles = []
    for i in range(len(diffs) - 1):
        d1 = diffs[i][:2]
        d2 = diffs[i+1][:2]
        n1 = np.linalg.norm(d1)
        n2 = np.linalg.norm(d2)
        if n1 > 1e-6 and n2 > 1e-6:
            cos_a = np.dot(d1, d2) / (n1 * n2)
            angles.append(np.degrees(np.arccos(np.clip(cos_a, -1, 1))))

    if angles:
        print(f"Max turn angle: {max(angles):.1f} deg")
        print(f"Mean turn angle: {np.mean(angles):.1f} deg")
        print(f"Angles > 30 deg: {sum(1 for a in angles if a > 30)}/{len(angles)}")
    print(f"Straight-line: {np.linalg.norm(goal - start):.2f}m, Ratio: {length / np.linalg.norm(goal - start):.3f}")
else:
    print("Planning failed!")
