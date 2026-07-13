"""robustness_suite.py — 把所有鲁棒性/审计检查固化落盘(此前部分只在对话里跑过,不合格)。
① 速度鲁棒性(4-12 m/s):分叉+省能是否全程成立
② 起终点鲁棒性(4 种接近方式)
③ 因果消融(爬升项挖掉 → 决策退化)结果落盘
④ 公平性对照:BEMT 质量改成 M100 的 2.65kg(2.4+0.25 载荷)——排除"baseline 质量弄错才翻墙"
全部结果写 experiments/robustness_suite.json,可复现。
"""
import os, sys, copy, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import physics_eval as pe
from planning_experiment import M100Em, DistanceEm, score_path
from wall_experiment import clean_wall_map

HERE = os.path.dirname(os.path.abspath(__file__))
FULL = open(os.path.join(HERE, "state", "best_featurize.py")).read()
ABLATED = (FULL.replace("climb = np.maximum(vz, 0.0)", "climb = np.zeros_like(vz)")
               .replace("desc = np.minimum(vz, 0.0)", "desc = np.zeros_like(vz)"))


def plan_and_judge(vg, esdf, em, s, g, V, yard, yardV):
    p, _ = pe.energy_astar(vg, esdf, em, s, g, velocity=float(V), safety_margin=0.6, max_expand=1500000)
    if not p or len(p) < 2:
        return None
    P = np.array([np.asarray(w, float).reshape(-1)[:3] for w in p])
    alt, ydev = float(-P[:, 2].min()), float(np.abs(P[:, 1]).max())
    return {"choice": "翻越" if alt > ydev + 2 else "绕行", "alt": round(alt, 1),
            "ydev": round(ydev, 1), "E_yard": round(score_path(p, yard, yardV), 0)}


def main():
    m100 = M100Em(FULL, payload=250.)
    m100_abl = M100Em(ABLATED, payload=250.)
    dist = DistanceEm()
    out = {}

    # ① 速度鲁棒性(窄墙 12)
    vg, esdf, bemt = clean_wall_map(12., 12.)
    rows = []
    for V in [4, 6, 8, 10, 12]:
        d = plan_and_judge(vg, esdf, dist, np.array([10., 0., -3.]), np.array([72., 0., -3.]), V, m100, V)
        m = plan_and_judge(vg, esdf, m100, np.array([10., 0., -3.]), np.array([72., 0., -3.]), V, m100, V)
        sv = round(100 * (d["E_yard"] - m["E_yard"]) / d["E_yard"], 1)
        rows.append({"v": V, "dist": d["choice"], "m100": m["choice"], "save%": sv})
        print(f"① v={V}: 距离={d['choice']} M100={m['choice']} 省{sv}%")
    out["velocity_sweep"] = rows

    # ② 起终点鲁棒性(中墙 16, v=8)
    vg, esdf, bemt = clean_wall_map(16., 12.)
    V = 8.0
    cases = [("正对", [10, 0, -3], [72, 0, -3]), ("斜进", [10, -12, -3], [72, 10, -4]),
             ("偏置", [8, 8, -2], [70, -6, -5]), ("高起点", [10, 0, -6], [72, 0, -3])]
    rows = []
    for nm, s, g in cases:
        s, g = np.array(s, float), np.array(g, float)
        d = plan_and_judge(vg, esdf, dist, s, g, V, m100, V)
        m = plan_and_judge(vg, esdf, m100, s, g, V, m100, V)
        sv = round(100 * (d["E_yard"] - m["E_yard"]) / d["E_yard"], 1)
        rows.append({"case": nm, "dist": d["choice"], "m100": m["choice"], "save%": sv})
        print(f"② {nm}: 距离={d['choice']} M100={m['choice']} 省{sv}%")
    out["startgoal_sweep"] = rows

    # ③ 因果消融(中墙 16, v=8)+ held-out ARE
    import m100_eval as me
    ns = {"np": np}; exec(FULL, ns); are_full = me.evaluate(ns["featurize"], seed=0, test_frac=0.33)["energy_ARE"]
    ns = {"np": np}; exec(ABLATED, ns); are_abl = me.evaluate(ns["featurize"], seed=0, test_frac=0.33)["energy_ARE"]
    s, g = np.array([10., 0., -3.]), np.array([72., 0., -3.])
    abl = plan_and_judge(vg, esdf, m100_abl, s, g, V, m100, V)
    ful = plan_and_judge(vg, esdf, m100, s, g, V, m100, V)
    out["ablation"] = {"ARE_full%": round(are_full * 100, 2), "ARE_ablated%": round(are_abl * 100, 2),
                       "full_choice": ful["choice"], "ablated_choice": abl["choice"]}
    print(f"③ 消融: ARE {are_full*100:.2f}%→{are_abl*100:.2f}%  决策 {ful['choice']}→{abl['choice']}")

    # ④ 公平性:BEMT 质量 1.5 → 2.65kg(M100 2.4 + 载荷 0.25)
    bemt_m100 = copy.deepcopy(bemt)
    bemt_m100.params.mass = 2.65
    b15 = plan_and_judge(vg, esdf, bemt, s, g, V, m100, V)
    b265 = plan_and_judge(vg, esdf, bemt_m100, s, g, V, m100, V)
    out["bemt_mass_fairness"] = {"BEMT@1.5kg": b15["choice"], "BEMT@2.65kg": b265["choice"]}
    print(f"④ BEMT质量公平性: 1.5kg={b15['choice']}  2.65kg(M100质量)={b265['choice']}")
    print("   → 若两者都翻越:质量参数不是翻墙的因,爬升盲才是(与消融一致)")

    json.dump(out, open(os.path.join(HERE, "experiments", "robustness_suite.json"), "w"),
              ensure_ascii=False, indent=2)
    print("\n全部落盘 experiments/robustness_suite.json")


if __name__ == "__main__":
    main()
