"""sim_flight.py — 动力学闭环:规划路径让 M100 尺度的四旋翼(RotorPy 刚体动力学 + SE3 控制器)真飞一遍。
回答:'绕行省能'的结论过了动力学(加减速/转弯/平滑/跟踪误差)还成立吗?

链条:A* 规划(距离代价→翻越 / M100代价→绕行)→ RDP 抽稀航点 → MinSnap 平滑轨迹
     → RotorPy 100Hz 动力学跟踪 → 飞出的 v_h(t),v_z(t),a_z(t) 喂真机 M100 功率模型 → 能量积分。

诚实边界:RotorPy 是通用四旋翼刚体动力学(质量/惯量/桨按 M100 尺度配:2.65kg、650mm、悬停550rad/s),
不是 M100 气动标定;它的作用是生成"可飞的"轨迹和真实速度剖面,能量仍由真数据锚定的模型评。
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import physics_eval as pe
from planning_experiment import M100Em, DistanceEm, score_path
from planning_connect import PowerModel
from wall_experiment import clean_wall_map

HERE = os.path.dirname(os.path.abspath(__file__))


def m100_scale_params():
    from rotorpy.vehicles.crazyflie_params import quad_params
    p = dict(quad_params)
    p['mass'] = 2.65                                   # 2.4 kg M100 + 250 g 载荷
    p['Ixx'] = p['Iyy'] = 0.045; p['Izz'] = 0.08
    p['Ixy'] = p['Ixz'] = p['Iyz'] = 0.0
    L = 0.325 / np.sqrt(2)                             # 650mm 轴距 X 型
    p['rotor_pos'] = {'r1': np.array([L, L, 0]), 'r2': np.array([L, -L, 0]),
                      'r3': np.array([-L, -L, 0]), 'r4': np.array([-L, L, 0])}
    w_h = 550.0                                        # 13寸桨悬停转速量级
    p['k_eta'] = p['mass'] * 9.81 / 4 / w_h ** 2
    p['k_m'] = p['k_eta'] * 0.022
    p['rotor_speed_min'], p['rotor_speed_max'] = 0.0, 1100.0
    p['tau_m'] = 0.02; p['motor_noise_std'] = 0.0
    p['k_d'] = p['k_z'] = p['k_h'] = p['k_flap'] = 0.0  # 通用动力学,不假装 M100 气动
    p['c_Dx'] = p['c_Dy'] = p['c_Dz'] = 0.5
    return p, w_h


def rdp(points, eps=1.0):
    """Douglas-Peucker 抽稀(A* 锯齿 → 角点)。"""
    P = np.asarray(points, float)
    if len(P) < 3:
        return P
    d = P[-1] - P[0]; n = np.linalg.norm(d)
    if n < 1e-9:
        dist = np.linalg.norm(P - P[0], axis=1)
    else:
        dist = np.linalg.norm(np.cross(P - P[0], d / n), axis=1)
    i = int(np.argmax(dist))
    if dist[i] > eps:
        a = rdp(P[:i + 1], eps); b = rdp(P[i:], eps)
        return np.vstack([a[:-1], b])
    return np.array([P[0], P[-1]])


def resample(P, step):
    """沿弧长均匀重采样(段长均匀 → MinSnap QP 数值稳定)。"""
    d = np.r_[0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
    n = max(4, int(np.ceil(d[-1] / step)) + 1)
    si = np.linspace(0, d[-1], n)
    return np.column_stack([np.interp(si, d, P[:, k]) for k in range(3)])


def make_traj(P_ned, v_avg):
    """规划路径(NED)→ 可行 MinSnap 轨迹(ENU)。cvxopt QP 挑剔,梯度降级重采样直到成功。"""
    from rotorpy.trajectories.minsnap import MinSnap
    for step in (10., 12., 15., 18., 22.):
        W = resample(P_ned, step)
        W = np.column_stack([W[:, 0], W[:, 1], -W[:, 2]])       # NED→ENU
        try:
            traj = MinSnap(points=W, v_avg=v_avg, verbose=False)
            return traj, W, step
        except Exception:
            continue
    raise RuntimeError("所有重采样档位 MinSnap 均失败")


def fly(traj, start_enu):
    """SE3 跟踪 MinSnap 轨迹,返回 (t, x, v) 100Hz 序列。"""
    from rotorpy.vehicles.multirotor import Multirotor
    from rotorpy.controllers.quadrotor_control import SE3Control
    from rotorpy.environments import Environment
    p, w_h = m100_scale_params()
    env = Environment(vehicle=Multirotor(p), controller=SE3Control(p), trajectory=traj, sim_rate=100)
    env.vehicle.initial_state = {'x': np.asarray(start_enu, float), 'v': np.zeros(3),
                                 'q': np.array([0, 0, 0, 1.]), 'w': np.zeros(3),
                                 'wind': np.zeros(3), 'rotor_speeds': np.ones(4) * w_h}
    T = traj.t_keyframes[-1] + 1.0
    res = env.run(t_final=T, use_mocap=False, terminate=False, plot=False, verbose=False)
    return res['time'], res['state']['x'], res['state']['v']


def flown_energy(t, v, model, payload=250.0):
    """飞出的状态序列 → M100 功率 → 能量积分(J)。a_z 用有限差分(比规划时的 a=0 更真实)。"""
    dt = np.diff(t)
    vh = np.hypot(v[:, 0], v[:, 1])
    vz = v[:, 2]                                        # ENU: 正=上升,与模型约定一致
    az = np.gradient(v[:, 2], t)
    st = dict(v_h=vh, v_z=vz, a_h=np.hypot(*np.gradient(v[:, :2], t, axis=0).T[:2]),
              a_z=az, omega=np.zeros_like(vh), payload=np.full_like(vh, payload),
              wind=np.zeros_like(vh), speed=np.hypot(vh, vz))
    P = np.maximum(0.0, model.predict(st))
    return float(np.sum(0.5 * (P[1:] + P[:-1]) * dt)), float(t[-1]), float(P.mean()), P


def min_dist_to_wall(x_enu, half_y=16., height=12.):
    """飞行轨迹到墙盒(x∈[39,42],y∈±half_y,z∈[0,height])的最小距离(m)。"""
    box_lo = np.array([39., -half_y, 0.]); box_hi = np.array([42., half_y, height])
    d = np.maximum(box_lo - x_enu, 0) + np.maximum(x_enu - box_hi, 0)
    return float(np.linalg.norm(d, axis=1).min())


def main():
    src = open(os.path.join(HERE, "state", "best_featurize.py")).read()
    m100 = M100Em(src, payload=250.); dist = DistanceEm()
    power = PowerModel(src, "M100")
    vg, esdf, bemt = clean_wall_map(16., 12.)
    s = np.array([10., 0., -3.]); g = np.array([72., 0., -3.])
    V = 8.0

    print("=== ① 规划(同 wall_experiment 中墙场景)===")
    plans = {}
    for nm, em in [("翻越(距离代价)", dist), ("绕行(M100代价)", m100)]:
        # 裕度 2.0m(比 wall_experiment 的 0.6 大):MinSnap 平滑会切角,给平滑留碰撞余量
        p, _ = pe.energy_astar(vg, esdf, em, s, g, velocity=V, safety_margin=2.0, max_expand=1500000)
        P = np.array([np.asarray(w, float).reshape(-1)[:3] for w in p])
        plans[nm] = P
        print(f"  {nm}: {len(P)}航点 planned能量(恒速折线)={score_path(p, m100, V):.0f}J")

    print("\n=== ② 动力学飞行(RotorPy, M100尺度, MinSnap v_avg=6)===")
    out = {}
    trajs = {}
    for nm, P in plans.items():
        traj, wps_enu, step = make_traj(P, v_avg=6.0)
        t, x, v = fly(traj, wps_enu[0])
        E, T, Pm, Pser = flown_energy(t, v, power)
        margin = min_dist_to_wall(x)
        planned_E = score_path([w for w in P], m100, V)
        trajs[nm + "_t"] = np.asarray(t); trajs[nm + "_x"] = np.asarray(x); trajs[nm + "_P"] = Pser
        out[nm] = dict(waypoints=len(wps_enu), flown_E=round(E, 0), flight_time=round(T, 1),
                       mean_power=round(Pm, 0), planned_E=round(planned_E, 0),
                       wall_margin=round(margin, 2), max_alt=round(float(x[:, 2].max()), 1))
        print(f"  {nm}: {len(wps_enu)}航点 飞行{T:.1f}s 均功率{Pm:.0f}W 飞行能量={E:.0f}J "
              f"(规划折线预测{planned_E:.0f}J) 距墙最近{margin:.2f}m 最高{x[:,2].max():.1f}m")

    ko = out["翻越(距离代价)"]; ka = out["绕行(M100代价)"]
    sv_flown = 100 * (ko["flown_E"] - ka["flown_E"]) / ko["flown_E"]
    sv_plan = 100 * (ko["planned_E"] - ka["planned_E"]) / ko["planned_E"]
    print(f"\n=== ③ 结论 ===")
    print(f"  规划折线预测省能: {sv_plan:+.1f}%   动力学飞行后省能: {sv_flown:+.1f}%")
    print(f"  → {'绕行优势过了动力学仍成立' if sv_flown > 0 else '动力学推翻了折线结论(诚实报告)'}")
    out["saving_planned%"] = round(sv_plan, 1); out["saving_flown%"] = round(sv_flown, 1)
    json.dump(out, open(os.path.join(HERE, "experiments", "sim_flight_result.json"), "w"),
              ensure_ascii=False, indent=2)
    np.savez(os.path.join(HERE, "experiments", "sim_flight_trajs.npz"), **trajs)
    _make_fig(trajs, out)
    print("落盘 experiments/sim_flight_result.json + sim_flight_trajs.npz")


def _make_fig(trajs, out):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "Heiti TC"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(16, 4.2))
    colors = {"翻越(距离代价)": "tab:red", "绕行(M100代价)": "tab:green"}
    a1.add_patch(Rectangle((39, 0), 3, 12, color="gray", alpha=.6, label="墙"))
    a2.add_patch(Rectangle((39, -16), 3, 32, color="gray", alpha=.6))
    for nm in trajs:
        if nm.endswith("_x"):
            base = nm[:-2]; X = trajs[nm]; c = colors[base]
            a1.plot(X[:, 0], X[:, 2], color=c, lw=2, label=base)
            a2.plot(X[:, 0], X[:, 1], color=c, lw=2)
            t = trajs[base + "_t"]; P = trajs[base + "_P"]
            a3.plot(t, P, color=c, lw=1.5, label=f"{base} 均{P.mean():.0f}W")
    a1.set_xlabel("x (m)"); a1.set_ylabel("高度 (m)"); a1.set_title("动力学飞行·侧视"); a1.legend(fontsize=8)
    a2.set_xlabel("x (m)"); a2.set_ylabel("y (m)"); a2.set_title("动力学飞行·俯视")
    a3.set_xlabel("t (s)"); a3.set_ylabel("M100 功率 (W)"); a3.set_title("飞行中瞬时功率(真机模型)"); a3.legend(fontsize=8)
    plt.suptitle(f"动力学闭环(RotorPy·M100尺度·SE3跟踪):绕行省能过动力学仍成立 "
                 f"{out['saving_flown%']}%(折线预测 {out['saving_planned%']}%)")
    plt.tight_layout()
    p = os.path.join(HERE, "experiments", "sim_flight.png")
    plt.savefig(p, dpi=125); print(f"图存 {p}")


if __name__ == "__main__":
    main()
