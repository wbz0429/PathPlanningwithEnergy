"""pl_iter2b_penetration.py — iter2 裁决辅助:量化风况下绕行路线对墙盒的侵入深度。
(pl_iter2 主实验发现 detour 在 head6/cross3/cross6 下 wall_margin=0;本脚本回答
 "擦墙还是穿墙":逐时刻检测严格入盒 + 最大侵入深度。判据见主实验预注册。)
输出:experiments/pl_iter2b_penetration.json
"""
import os, sys, json

HERE = os.path.dirname(os.path.abspath(__file__))
EM100 = os.path.dirname(HERE)
sys.path.insert(0, os.path.dirname(EM100))
sys.path.insert(0, EM100)
sys.path.insert(0, HERE)
import numpy as np
import physics_eval as pe
from planning_experiment import M100Em
from wall_experiment import clean_wall_map
from sim_flight import make_traj
from pl_iter2_wind_robust import fly_wind

WINDS = [("head3", (-3., 0., 0.)), ("head6", (-6., 0., 0.)),
         ("cross3", (0., 3., 0.)), ("cross6", (0., 6., 0.))]


def main():
    src = open(os.path.join(EM100, "state", "best_featurize.py")).read()
    m100 = M100Em(src, payload=250.)
    vg, esdf, _ = clean_wall_map(16., 12.)
    s = np.array([10., 0., -3.]); g = np.array([72., 0., -3.])
    p, _ = pe.energy_astar(vg, esdf, m100, s, g, velocity=8.0,
                           safety_margin=2.0, max_expand=1500000)
    P = np.array([np.asarray(w, float).reshape(-1)[:3] for w in p])
    traj, wps, _ = make_traj(P, v_avg=6.0)
    lo = np.array([39., -16., 0.]); hi = np.array([42., 16., 12.])
    out = {"detour_side_max_abs_y": round(float(np.abs(P[:, 1]).max()), 1),
           "detour_side_sign": float(np.sign(P[np.abs(P[:, 1]).argmax(), 1])),
           "cells": {}}
    for wname, wvec in WINDS:
        t, x, v = fly_wind(traj, wps[0], wvec)
        X = np.asarray(x)
        inside = np.all((X > lo) & (X < hi), axis=1)
        cell = {"steps_inside": int(inside.sum()),
                "frac_inside_pct": round(float(inside.mean() * 100), 2)}
        if inside.any():
            depth = np.minimum(X[inside] - lo, hi - X[inside]).min(axis=1)
            cell["max_penetration_m"] = round(float(depth.max()), 2)
        out["cells"][wname] = cell
        print(wname, cell)
    with open(os.path.join(HERE, "pl_iter2b_penetration.json"), "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("JSON 落盘 experiments/pl_iter2b_penetration.json")


if __name__ == "__main__":
    main()
