"""pl_iter2_wind_robust.py — 规划层 iter2:H2 风扰动鲁棒性(预注册实验)

═══ 预注册(先写判据,后跑实验)═══════════════════════════════════════════
假设 H2:墙场景"绕行(M100代价) vs 翻越(距离代价)"的省能结论(无风动力学 6.1%),
  在 RotorPy 动力学加常值风场后仍成立——风只扰动动力学(倾转补偿/跟踪误差/加速度修正),
  能量尺不变(仍用冻结 M100 模型评飞出的地面系运动学)。
判据(预注册):
  (1) 风格子 = {逆风, 侧风} × {3, 6} m/s 共 4 格(逆风=-x,侧风=+y,ENU;航线主向+x);
      每格两条路线同风对比;
  (2) SUPPORTED 当且仅当:4 格中省能符号全部不翻(E_绕行 < E_翻越),且两条路线
      都保持无碰(wall_margin > 0)、跟踪不发散(终点误差 < 5m);
  (3) 任一格符号翻转 → REFUTED(报翻转边界);任一格发散 → 该格单独报告,
      若因发散无法比较能量则 INCONCLUSIVE(写明缺什么);
  (4) 额外报告(不进判据):无风基线复现 + 顺风 6 m/s(全范围诚实报告)。
诚实边界(先声明):能量尺以地面系速度喂模型,不含风气动能量项(iter5 已证伪该项,
  死路勿碰)——H2 只测"结论对风扰动后飞行运动学的鲁棒性",不是风感知能量核算;
  SE3 跟踪较强,预期扰动可能很小,若省能变化 <1% 将如实报"动力学级风鲁棒性平凡成立"。
═══════════════════════════════════════════════════════════════════════

用法:.venv/bin/python experiments/pl_iter2_wind_robust.py
输出:experiments/pl_iter2_wind_robust.json
"""
import os, sys, json, time

HERE = os.path.dirname(os.path.abspath(__file__))
EM100 = os.path.dirname(HERE)
AR = os.path.dirname(EM100)
sys.path.insert(0, AR)
sys.path.insert(0, EM100)
import numpy as np
import physics_eval as pe
from planning_experiment import M100Em, DistanceEm, score_path
from planning_connect import PowerModel
from wall_experiment import clean_wall_map
from sim_flight import m100_scale_params, make_traj, flown_energy, min_dist_to_wall

V_PLAN = 8.0
WINDS = [("nowind", (0., 0., 0.)),          # 基线复现(不进判据)
         ("head3", (-3., 0., 0.)), ("head6", (-6., 0., 0.)),
         ("cross3", (0., 3., 0.)), ("cross6", (0., 6., 0.)),
         ("tail6", (6., 0., 0.))]           # 额外报告(不进判据)
JUDGED = {"head3", "head6", "cross3", "cross6"}


def fly_wind(traj, start_enu, wind_vec):
    """同 sim_flight.fly,但注入常值风场(动力学扰动;能量尺不变)。"""
    from rotorpy.vehicles.multirotor import Multirotor
    from rotorpy.controllers.quadrotor_control import SE3Control
    from rotorpy.environments import Environment
    from rotorpy.wind.default_winds import ConstantWind
    p, w_h = m100_scale_params()
    env = Environment(vehicle=Multirotor(p), controller=SE3Control(p), trajectory=traj,
                      wind_profile=ConstantWind(*wind_vec), sim_rate=100)
    env.vehicle.initial_state = {'x': np.asarray(start_enu, float), 'v': np.zeros(3),
                                 'q': np.array([0, 0, 0, 1.]), 'w': np.zeros(3),
                                 'wind': np.asarray(wind_vec, float),
                                 'rotor_speeds': np.ones(4) * w_h}
    T = traj.t_keyframes[-1] + 1.0
    res = env.run(t_final=T, use_mocap=False, terminate=False, plot=False, verbose=False)
    return res['time'], res['state']['x'], res['state']['v']


def main():
    src = open(os.path.join(EM100, "state", "best_featurize.py")).read()
    m100 = M100Em(src, payload=250.); dist = DistanceEm()
    power = PowerModel(src, "M100")
    vg, esdf, _ = clean_wall_map(16., 12.)
    s = np.array([10., 0., -3.]); g = np.array([72., 0., -3.])

    print("① 规划两条路线(同 sim_flight:裕度 2.0m)...")
    plans = {}
    for nm, em in [("climb_dist", dist), ("detour_m100", m100)]:
        p, _ = pe.energy_astar(vg, esdf, em, s, g, velocity=V_PLAN,
                               safety_margin=2.0, max_expand=1500000)
        plans[nm] = np.array([np.asarray(w, float).reshape(-1)[:3] for w in p])
        print(f"  {nm}: {len(plans[nm])}航点 planned={score_path(p, m100, V_PLAN):.0f}J")

    print("② MinSnap 轨迹(风不改变参考轨迹,只扰动跟踪)...")
    trajs = {}
    for nm, P in plans.items():
        traj, wps_enu, step = make_traj(P, v_avg=6.0)
        trajs[nm] = (traj, wps_enu)

    print("③ 风格子飞行(2 路线 × 6 风况)...")
    cells = []
    ref_x = {}
    for wname, wvec in WINDS:
        row = {"wind": wname, "wind_vec": list(wvec)}
        for nm in ("climb_dist", "detour_m100"):
            traj, wps_enu = trajs[nm]
            t0 = time.time()
            t, x, v = fly_wind(traj, wps_enu[0], wvec)
            E, T, Pm, _ = flown_energy(t, v, power)
            margin = min_dist_to_wall(np.asarray(x))
            err = float(np.linalg.norm(np.asarray(x)[-1] - wps_enu[-1]))
            row[nm] = {"E_J": round(E, 0), "T_s": round(T, 1), "meanP_W": round(Pm, 0),
                       "wall_margin_m": round(margin, 2), "goal_err_m": round(err, 2)}
            if wname == "nowind":
                ref_x[nm] = np.asarray(x)
            else:
                X = np.asarray(x); R = ref_x[nm]
                n = min(len(X), len(R))
                row[nm]["max_dev_from_nowind_m"] = round(
                    float(np.linalg.norm(X[:n] - R[:n], axis=1).max()), 2)
            print(f"  [{wname}] {nm}: E={E:.0f}J margin={margin:.2f}m err={err:.2f}m "
                  f"[{time.time()-t0:.0f}s]")
        ec, ed = row["climb_dist"]["E_J"], row["detour_m100"]["E_J"]
        row["savings_pct"] = round(100 * (ec - ed) / ec, 2)
        row["sign_holds"] = bool(ed < ec)
        row["both_safe"] = bool(row["climb_dist"]["wall_margin_m"] > 0
                                and row["detour_m100"]["wall_margin_m"] > 0)
        row["both_tracked"] = bool(row["climb_dist"]["goal_err_m"] < 5
                                   and row["detour_m100"]["goal_err_m"] < 5)
        print(f"  [{wname}] 省能 {row['savings_pct']:+.2f}% sign={row['sign_holds']} "
              f"safe={row['both_safe']} tracked={row['both_tracked']}")
        cells.append(row)

    judged = [c for c in cells if c["wind"] in JUDGED]
    supported = all(c["sign_holds"] and c["both_safe"] and c["both_tracked"] for c in judged)
    result = {
        "preregistered": {
            "criteria": "4 judged wind cells (head/cross x 3/6 m/s): savings sign holds "
                        "AND wall_margin>0 AND goal_err<5m for both routes",
            "judged_cells": sorted(JUDGED), "extra_cells": ["nowind", "tail6"],
            "ruler": "frozen M100Em on ground-frame flown kinematics (no wind energy term; "
                     "iter5 falsified that channel)"},
        "scenario": {"wall": "half_y=16, h=12, s=(10,0,-3), g=(72,0,-3), margin=2.0, v_plan=8"},
        "cells": cells,
        "verdict_hint": "SUPPORTED" if supported else "check cells",
    }
    with open(os.path.join(HERE, "pl_iter2_wind_robust.json"), "w") as f:
        json.dump(result, f, ensure_ascii=False, indent=1,
                  default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print(f"verdict_hint={result['verdict_hint']}  JSON 落盘 experiments/pl_iter2_wind_robust.json")


if __name__ == "__main__":
    main()
