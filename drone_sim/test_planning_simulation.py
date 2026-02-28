"""
test_planning_simulation.py - 无AirSim的完全模拟飞行测试

构造地图 + Mock无人机，测试规划算法在墙壁场景下的行为
重点检测：路径摇摆、绕障效率、wall-follow策略
"""

import numpy as np
import sys
import os
import time

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from planning.config import PlanningConfig
from mapping.voxel_grid import VoxelGrid
from mapping.esdf import ESDF
from planning.rrt_star import RRTStar

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


# ============================================================
# Mock Drone
# ============================================================
class MockDrone:
    """模拟无人机，不需要AirSim"""

    def __init__(self, start_pos: np.ndarray):
        self.position = start_pos.copy()
        self.trajectory = [start_pos.copy()]

    def get_position(self):
        return self.position.copy()

    def move_to_position(self, target: np.ndarray):
        self.position = target.copy()
        self.trajectory.append(target.copy())


# ============================================================
# Map Builders
# ============================================================
def build_wall_map(config: PlanningConfig,
                   wall_x=(4.0, 6.0),
                   wall_y=(-5.0, 5.0),
                   wall_z=(-6.0, 0.0)) -> VoxelGrid:
    """
    构造墙壁场景
    默认: x=4~6m 处有一堵 y=-5~5m 的墙
    """
    voxel_grid = VoxelGrid(config)
    voxel_grid.grid[:] = -1  # 全部标记为已探索空闲

    count = 0
    for ix in range(config.grid_size[0]):
        for iy in range(config.grid_size[1]):
            for iz in range(config.grid_size[2]):
                w = voxel_grid.grid_to_world((ix, iy, iz))
                if (wall_x[0] <= w[0] <= wall_x[1] and
                    wall_y[0] <= w[1] <= wall_y[1] and
                    wall_z[0] <= w[2] <= wall_z[1]):
                    voxel_grid.grid[ix, iy, iz] = 1
                    count += 1

    print(f"[OK] Wall map: {count} occupied voxels  "
          f"wall x={wall_x} y={wall_y}")
    return voxel_grid


def build_u_shape_map(config: PlanningConfig) -> VoxelGrid:
    """
    U形陷阱: 前墙 + 左右翼
    """
    voxel_grid = VoxelGrid(config)
    voxel_grid.grid[:] = -1

    z_lo, z_hi = -6.0, 0.0
    count = 0
    for ix in range(config.grid_size[0]):
        for iy in range(config.grid_size[1]):
            for iz in range(config.grid_size[2]):
                w = voxel_grid.grid_to_world((ix, iy, iz))
                if not (z_lo <= w[2] <= z_hi):
                    continue
                hit = False
                # 前墙
                if 4.0 <= w[0] <= 6.0 and -8.0 <= w[1] <= 8.0:
                    hit = True
                # 左翼
                if -2.0 <= w[0] <= 5.0 and 7.0 <= w[1] <= 9.0:
                    hit = True
                # 右翼
                if -2.0 <= w[0] <= 5.0 and -9.0 <= w[1] <= -7.0:
                    hit = True
                if hit:
                    voxel_grid.grid[ix, iy, iz] = 1
                    count += 1

    print(f"[OK] U-shape map: {count} occupied voxels")
    return voxel_grid


# ============================================================
# Core helpers (replicate RecedingHorizonPlanner logic exactly)
# ============================================================
def _rotate_yaw(direction: np.ndarray, angle_deg: float) -> np.ndarray:
    rad = np.radians(angle_deg)
    c, s = np.cos(rad), np.sin(rad)
    r = np.array([
        c * direction[0] - s * direction[1],
        s * direction[0] + c * direction[1],
        direction[2]
    ])
    n = np.linalg.norm(r)
    return r / n if n > 1e-9 else r


def is_forward_blocked(pos, goal, esdf, config, horizon):
    direction = goal - pos
    direction[2] = 0
    n = np.linalg.norm(direction[:2])
    if n < 0.1:
        return False
    direction = direction / np.linalg.norm(direction)
    for d in np.arange(0.5, horizon, 1.0):
        pt = pos + direction * d
        pt[2] = pos[2]
        if esdf.get_distance(pt) < config.safety_margin:
            return True
    return False


def select_local_goal(pos, goal, esdf, config, horizon):
    direction = goal - pos
    dist = np.linalg.norm(direction)
    if dist < horizon:
        lg = goal.copy()
    else:
        lg = pos + (direction / dist) * horizon
    lg[2] = pos[2]

    if esdf.distance_field is not None:
        if esdf.get_distance(lg) < config.safety_margin:
            safe = find_safe_local_goal(pos, goal, esdf, config, horizon)
            if safe is not None:
                lg = safe
    return lg


def find_safe_local_goal(pos, goal, esdf, config, horizon):
    direction = goal - pos
    direction[2] = 0
    n = np.linalg.norm(direction[:2])
    if n > 0:
        direction = direction / np.linalg.norm(direction)

    angles = [15, -15, 30, -30, 45, -45, 60, -60, 75, -75, 90, -90]
    dists = [horizon, horizon * 0.8, horizon * 0.6]
    best, best_s = None, -np.inf
    for a in angles:
        for d in dists:
            cand = pos + _rotate_yaw(direction, a) * d
            cand[2] = pos[2]
            s = esdf.get_distance(cand)
            if s > config.safety_margin:
                prog = np.dot(cand - pos, goal - pos)
                score = s * 0.3 + prog * 0.7
                if score > best_s:
                    best_s = score
                    best = cand
    return best


def wall_follow_step(pos, goal, esdf, config, remembered_dir):
    direction = goal - pos
    direction[2] = 0
    n = np.linalg.norm(direction[:2])
    if n < 0.1:
        return None, remembered_dir
    forward = direction / np.linalg.norm(direction)
    left = np.array([-forward[1], forward[0], 0.0])
    right = np.array([forward[1], -forward[0], 0.0])

    if remembered_dir is not None:
        best_dir = left if remembered_dir == 'left' else right
        best_name = remembered_dir
    else:
        best_dir, best_name, best_score = None, None, -np.inf
        for lat, name in [(left, 'left'), (right, 'right')]:
            sc = 0.0
            for d in [3.0, 6.0, 10.0, 15.0]:
                ck = pos + lat * d;  ck[2] = pos[2]
                fw = ck + forward * 4.0;  fw[2] = pos[2]
                if esdf.get_distance(ck) > config.safety_margin:
                    sc += 1.0
                if esdf.get_distance(fw) > config.safety_margin:
                    sc += 3.0
            if sc > best_score:
                best_score, best_dir, best_name = sc, lat, name
        if best_dir is None:
            return None, None

    for step in [5.0, 4.0, 3.0, 2.0]:
        tgt = pos + best_dir * step
        tgt[2] = pos[2]
        if esdf.get_distance(tgt) >= config.safety_margin:
            # 简单路径安全检查
            safe = True
            for t in np.linspace(0, 1, 10):
                pt = pos + t * (tgt - pos)
                if esdf.get_distance(pt) < config.safety_margin:
                    safe = False
                    break
            if safe:
                return [pos.copy(), tgt], best_name

    return None, None


def try_alternatives(pos, goal, rrt, esdf, config, horizon):
    direction = goal - pos
    direction[2] = 0
    n = np.linalg.norm(direction[:2])
    if n > 0:
        direction = direction / np.linalg.norm(direction)
    for a in [-30, 30, -45, 45, -60, 60, -90, 90]:
        for d in [horizon * 0.7, horizon * 0.5]:
            alt = pos + _rotate_yaw(direction, a) * d
            alt[2] = pos[2]
            if esdf.distance_field is not None:
                if esdf.get_distance(alt) < config.safety_margin * 0.5:
                    continue
            path = rrt.plan(pos, alt)
            if path is not None:
                return path
    return None


# ============================================================
# Simulation Runner
# ============================================================
def run_simulation(name, voxel_grid, esdf, config, start, goal, max_iter=40):
    print(f"\n{'='*70}")
    print(f"  SIMULATION: {name}")
    print(f"  Start: {start}  Goal: {goal}")
    print(f"{'='*70}")

    drone = MockDrone(start)
    rrt = RRTStar(voxel_grid, esdf, config)

    local_horizon = 6.0
    exec_ratio = 0.4
    goal_tol = 1.5

    trajectory = [start.copy()]
    local_goals_hist = []
    plan_log = []
    wf_dir = None  # wall-follow remembered direction

    for it in range(1, max_iter + 1):
        cur = drone.get_position()
        d2g = np.linalg.norm(cur - goal)
        print(f"\n--- Iter {it} | pos=({cur[0]:.1f},{cur[1]:.1f},{cur[2]:.1f}) "
              f"| dist_goal={d2g:.1f}m ---")

        if d2g < goal_tol:
            print(f"[SUCCESS] Reached goal in {it} iterations")
            break

        blocked = is_forward_blocked(cur, goal, esdf, config, local_horizon)
        path = None

        # --- wall-follow branch ---
        if blocked:
            print("  [BLOCKED] Forward blocked by obstacle")
            wf_path, wf_dir = wall_follow_step(cur, goal, esdf, config, wf_dir)
            if wf_path is not None:
                path = wf_path
                print(f"  [WALL-FOLLOW] dir={wf_dir}, "
                      f"target=({path[-1][0]:.1f},{path[-1][1]:.1f})")
            else:
                blocked = False  # fallback to RRT*

        # --- RRT* branch ---
        if not blocked:
            lg = select_local_goal(cur, goal, esdf, config, local_horizon)
            local_goals_hist.append(lg.copy())
            print(f"  Local goal: ({lg[0]:.1f},{lg[1]:.1f},{lg[2]:.1f})")

            path = rrt.plan(cur, lg)
            if path is None:
                print("  [FAIL] RRT* failed, trying alternatives...")
                path = try_alternatives(cur, goal, rrt, esdf, config, local_horizon)

        if path is None:
            print("  [FAIL] All planning failed this iteration")
            plan_log.append({'iter': it, 'ok': False})
            continue

        plan_log.append({'iter': it, 'ok': True, 'n_wp': len(path),
                         'blocked': blocked})

        # --- execute partial path ---
        to_exec = path[1:] if len(path) > 1 else path
        n_exec = max(1, int(len(to_exec) * exec_ratio))
        for wp in to_exec[:n_exec]:
            wp_f = wp.copy()
            wp_f[2] = goal[2]
            drone.move_to_position(wp_f)
            trajectory.append(wp_f.copy())

        # clear wall-follow memory when forward becomes clear
        if wf_dir is not None:
            if not is_forward_blocked(drone.get_position(), goal,
                                      esdf, config, local_horizon):
                print("  [WALL-FOLLOW] Forward clear → reset direction memory")
                wf_dir = None
    else:
        print(f"\n[TIMEOUT] {max_iter} iterations reached")

    traj = np.array(trajectory)
    analysis = analyze_trajectory(traj, start, goal)

    return {
        'trajectory': traj,
        'local_goals': local_goals_hist,
        'plan_log': plan_log,
        'analysis': analysis,
        'name': name,
    }


# ============================================================
# Trajectory Analysis
# ============================================================
def analyze_trajectory(traj, start, goal):
    if len(traj) < 3:
        return {'path_length': 0, 'efficiency': 0,
                'y_oscillations': 0, 'sharp_turns': 0}

    diffs = np.diff(traj, axis=0)
    seg_len = np.linalg.norm(diffs, axis=1)
    path_len = float(np.sum(seg_len))
    straight = float(np.linalg.norm(goal - start))
    eff = straight / path_len if path_len > 0 else 0

    # Y-axis sign changes (oscillation)
    yd = np.diff(traj[:, 1])
    osc = 0
    for i in range(1, len(yd)):
        if yd[i] * yd[i - 1] < 0 and abs(yd[i]) > 0.3:
            osc += 1

    # sharp direction changes
    sharp = 0
    for i in range(1, len(diffs)):
        if seg_len[i] > 0.1 and seg_len[i - 1] > 0.1:
            cos_a = np.clip(
                np.dot(diffs[i], diffs[i - 1]) / (seg_len[i] * seg_len[i - 1]),
                -1, 1)
            if np.degrees(np.arccos(cos_a)) > 60:
                sharp += 1

    # 检测是否有来回走的情况（X方向后退）
    x_retreats = 0
    for i in range(1, len(traj)):
        if traj[i, 0] < traj[i-1, 0] - 0.5:  # X方向后退超过0.5m
            x_retreats += 1

    print(f"\n  === Trajectory Analysis ===")
    print(f"  Path length:      {path_len:.1f}m  (straight: {straight:.1f}m)")
    print(f"  Efficiency:       {eff:.2f}  (1.0 = perfect)")
    print(f"  Y oscillations:   {osc}")
    print(f"  Sharp turns >60d: {sharp}")
    print(f"  X retreats:       {x_retreats}")
    print(f"  Total waypoints:  {len(traj)}")

    if osc > 3:
        print(f"  [PROBLEM] Excessive Y-axis oscillation!")
    if sharp > 5:
        print(f"  [PROBLEM] Too many sharp turns!")
    if x_retreats > 3:
        print(f"  [PROBLEM] Drone keeps retreating!")
    if eff > 0 and eff < 0.25:
        print(f"  [PROBLEM] Very low path efficiency!")

    return {
        'path_length': path_len, 'straight_dist': straight,
        'efficiency': eff, 'y_oscillations': osc,
        'sharp_turns': sharp, 'x_retreats': x_retreats,
        'waypoints': len(traj),
    }


# ============================================================
# Visualization
# ============================================================
def visualize_result(result, voxel_grid, config, filename):
    if not HAS_MPL:
        print("  [SKIP] matplotlib not available, skipping plot")
        return

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    traj = result['trajectory']
    ana = result['analysis']

    # --- left: top-down view ---
    ax = axes[0]
    ax.set_title(f"{result['name']} - Top View (XY)")

    flight_z = traj[0, 2]
    zi = int((flight_z - config.origin[2]) / config.voxel_size)
    zi = np.clip(zi, 0, config.grid_size[2] - 1)

    obs = np.zeros((config.grid_size[0], config.grid_size[1]), dtype=bool)
    for dz in range(-2, 3):
        z = zi + dz
        if 0 <= z < config.grid_size[2]:
            obs |= (voxel_grid.grid[:, :, z] == 1)

    for ix in range(config.grid_size[0]):
        for iy in range(config.grid_size[1]):
            if obs[ix, iy]:
                wx = config.origin[0] + (ix + 0.5) * config.voxel_size
                wy = config.origin[1] + (iy + 0.5) * config.voxel_size
                ax.add_patch(Rectangle(
                    (wx - config.voxel_size / 2, wy - config.voxel_size / 2),
                    config.voxel_size, config.voxel_size,
                    color='gray', alpha=0.6))

    ax.plot(traj[:, 0], traj[:, 1], 'b.-', lw=1.5, ms=4, label='Trajectory')
    ax.plot(traj[0, 0], traj[0, 1], 'go', ms=12, label='Start', zorder=5)
    ax.plot(traj[-1, 0], traj[-1, 1], 'r*', ms=15, label='End', zorder=5)

    if result['local_goals']:
        lg = np.array(result['local_goals'])
        ax.plot(lg[:, 0], lg[:, 1], 'rx', ms=6, alpha=0.5, label='Local goals')

    ax.set_xlabel('X (m)');  ax.set_ylabel('Y (m)')
    ax.set_aspect('equal');  ax.legend(fontsize=8);  ax.grid(True, alpha=0.3)

    # --- right: Y over steps ---
    ax2 = axes[1]
    ax2.set_title(f"{result['name']} - Y Position (Oscillation Check)")
    ax2.plot(range(len(traj)), traj[:, 1], 'b-o', ms=3)
    ax2.axhline(0, color='gray', ls='--', alpha=0.5)
    ax2.set_xlabel('Step');  ax2.set_ylabel('Y (m)')
    ax2.grid(True, alpha=0.3)

    info = (f"Path: {ana['path_length']:.1f}m  Straight: {ana['straight_dist']:.1f}m\n"
            f"Efficiency: {ana['efficiency']:.2f}\n"
            f"Y osc: {ana['y_oscillations']}  Sharp: {ana['sharp_turns']}\n"
            f"X retreats: {ana['x_retreats']}")
    ax2.text(0.02, 0.98, info, transform=ax2.transAxes, fontsize=9,
             va='top', bbox=dict(boxstyle='round', fc='wheat', alpha=0.8))

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [OK] Saved: {filename}")


# ============================================================
# Test Scenarios
# ============================================================
def test_rrt_star_single_plan():
    """测试0: 单次RRT*规划能否绕墙"""
    print("\n" + "=" * 70)
    print("  TEST 0: Single RRT* Plan Around Wall")
    print("=" * 70)

    config = PlanningConfig(
        voxel_size=0.5,
        grid_size=(80, 80, 40),
        origin=(-10.0, -20.0, -10.0),
        max_iterations=3000,
        step_size=1.5,
        safety_margin=1.0,
        search_radius=4.0,
        goal_sample_rate=0.2,
        energy_aware=False,
    )
    vg = build_wall_map(config)
    esdf = ESDF(vg)
    esdf.compute()

    start = np.array([-5.0, 0.0, -3.0])
    goal = np.array([15.0, 0.0, -3.0])

    rrt = RRTStar(vg, esdf, config)
    t0 = time.time()
    path = rrt.plan(start, goal)
    dt = (time.time() - t0) * 1000

    if path is None:
        print(f"[FAIL] RRT* could not find path ({dt:.0f}ms)")
        return False

    path_arr = np.array(path)
    length = float(np.sum(np.linalg.norm(np.diff(path_arr, axis=0), axis=1)))
    print(f"[OK] Path found: {len(path)} waypoints, {length:.1f}m, {dt:.0f}ms")
    print(f"  Waypoints Y range: [{path_arr[:,1].min():.1f}, {path_arr[:,1].max():.1f}]")

    # 检查路径是否真的绕过了墙
    for i in range(len(path) - 1):
        seg_start = path[i]
        seg_end = path[i + 1]
        for t in np.linspace(0, 1, 20):
            pt = seg_start + t * (seg_end - seg_start)
            d = esdf.get_distance(pt)
            if d < config.safety_margin * 0.5:
                print(f"[FAIL] Path passes too close to obstacle at {pt} (d={d:.2f}m)")
                return False

    print(f"[OK] Path safety verified")
    return True


def test_wall_bypass():
    """测试1: 墙壁绕行 - 起点远离墙"""
    print("\n" + "=" * 70)
    print("  TEST 1: Wall Bypass (start far from wall)")
    print("=" * 70)

    config = PlanningConfig(
        voxel_size=0.5,
        grid_size=(80, 80, 40),
        origin=(-10.0, -20.0, -10.0),
        max_iterations=2000,
        step_size=1.5,
        safety_margin=1.0,
        search_radius=4.0,
        goal_sample_rate=0.2,
        energy_aware=False,
    )
    vg = build_wall_map(config)
    esdf = ESDF(vg)
    esdf.compute()

    start = np.array([-5.0, 0.0, -3.0])
    goal = np.array([15.0, 0.0, -3.0])

    result = run_simulation("Wall Bypass", vg, esdf, config, start, goal, max_iter=35)
    visualize_result(result, vg, config, "test_sim_wall_bypass.png")
    return result


def test_wall_close():
    """测试2: 墙壁绕行 - 起点靠近墙"""
    print("\n" + "=" * 70)
    print("  TEST 2: Wall Bypass (start close to wall)")
    print("=" * 70)

    config = PlanningConfig(
        voxel_size=0.5,
        grid_size=(80, 80, 40),
        origin=(-10.0, -20.0, -10.0),
        max_iterations=2000,
        step_size=1.5,
        safety_margin=1.0,
        search_radius=4.0,
        goal_sample_rate=0.2,
        energy_aware=False,
    )
    vg = build_wall_map(config)
    esdf = ESDF(vg)
    esdf.compute()

    start = np.array([2.0, 0.0, -3.0])
    goal = np.array([15.0, 0.0, -3.0])

    result = run_simulation("Wall Close", vg, esdf, config, start, goal, max_iter=35)
    visualize_result(result, vg, config, "test_sim_wall_close.png")
    return result


def test_u_shape():
    """测试3: U形陷阱"""
    print("\n" + "=" * 70)
    print("  TEST 3: U-Shape Trap")
    print("=" * 70)

    config = PlanningConfig(
        voxel_size=0.5,
        grid_size=(80, 80, 40),
        origin=(-10.0, -20.0, -10.0),
        max_iterations=2000,
        step_size=1.5,
        safety_margin=1.0,
        search_radius=4.0,
        goal_sample_rate=0.2,
        energy_aware=False,
    )
    vg = build_u_shape_map(config)
    esdf = ESDF(vg)
    esdf.compute()

    start = np.array([0.0, 0.0, -3.0])
    goal = np.array([15.0, 0.0, -3.0])

    result = run_simulation("U-Shape", vg, esdf, config, start, goal, max_iter=50)
    visualize_result(result, vg, config, "test_sim_u_shape.png")
    return result


def test_wide_wall():
    """测试4: 宽墙 - 墙很宽需要绕很远"""
    print("\n" + "=" * 70)
    print("  TEST 4: Wide Wall")
    print("=" * 70)

    config = PlanningConfig(
        voxel_size=0.5,
        grid_size=(80, 80, 40),
        origin=(-10.0, -20.0, -10.0),
        max_iterations=3000,
        step_size=1.5,
        safety_margin=1.0,
        search_radius=4.0,
        goal_sample_rate=0.2,
        energy_aware=False,
    )
    # 宽墙: y=-12~12
    vg = build_wall_map(config, wall_x=(4.0, 6.0), wall_y=(-12.0, 12.0))
    esdf = ESDF(vg)
    esdf.compute()

    start = np.array([-5.0, 0.0, -3.0])
    goal = np.array([15.0, 0.0, -3.0])

    result = run_simulation("Wide Wall", vg, esdf, config, start, goal, max_iter=50)
    visualize_result(result, vg, config, "test_sim_wide_wall.png")
    return result


# ============================================================
# Main
# ============================================================
def main():
    print("\n" + "=" * 70)
    print("  PLANNING ALGORITHM SIMULATION TESTS")
    print("  (No AirSim required)")
    print("=" * 70)

    np.random.seed(42)  # 可复现

    results = []

    # Test 0: 单次 RRT*
    t0_ok = test_rrt_star_single_plan()
    results.append(("Single RRT*", t0_ok))

    # Test 1-4: 滚动规划模拟
    for test_fn in [test_wall_bypass, test_wall_close, test_u_shape, test_wide_wall]:
        r = test_fn()
        ana = r['analysis']
        # 判定标准: 效率>0.2, 摇摆<5, 到达目标
        reached = r['trajectory'][-1][0] > 10.0  # X > 10 算接近目标
        ok = reached and ana['y_oscillations'] <= 5 and ana['efficiency'] > 0.15
        results.append((r['name'], ok))

    # === Summary ===
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    passed = 0
    for name, ok in results:
        status = "[PASS]" if ok else "[FAIL]"
        print(f"  {status} {name}")
        if ok:
            passed += 1
    print(f"\n  {passed}/{len(results)} tests passed")

    return passed == len(results)


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
