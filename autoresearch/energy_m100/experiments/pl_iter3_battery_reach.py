"""pl_iter3_battery_reach.py — 规划层 iter3:H3 电量约束可达集(预注册实验)

═══ 预注册(先写判据,后跑实验)═══════════════════════════════════════════
假设 H3:给定能量预算 B,能量感知规划(energy_astar+M100代价)的可达目标集
  比距离规划(同规划器+长度代价,M100 尺评能耗)大。注意:逐目标 E_energy ≤ E_dist
  由最优性近乎必然 → 可证伪内容是**幅度**:扩大量是否达到实用量级、集中在哪、
  随预算怎么变。
设置(冻结于此):墙场景(half_y=16,h=12);start=(10,0,-3);目标网格 x∈{45,50,...,75}
  × y∈{-25,-20,...,25} × z=-3(墙后 7×11=77 个);规划 v=8(真实巡航)、裕度 2.0m
  (H2 教训:0.6m 不可飞;两规划器同裕度=公平);尺=冻结 M100Em(payload 250g)。
判据:
  (1) SUPPORTED 当且仅当:存在预算 B 使 ΔN(B)=|能量可达∖距离可达| ≥ 5 个目标
      且 ΔN/N_dist(B) ≥ 10%,并且保守速度复评(v=4/12,路径不重规划、只重新计价,
      对能量规划器不利的下界)下峰值相对扩大 ≥ 5% 仍成立;
  (2) REFUTED 当 v=8 峰值相对扩大 < 5%;
  (3) 介于其间 → INCONCLUSIVE,报实测数。
  全预算扫描曲线 + 差集目标的空间分布必须整体报告(不许只报峰值点)。
诚实边界:目标网格只放墙后(墙前两规划器路径相同,稀释比例;绝对数同报);
  速度敏感性用重计价 = 能量规划器优势的下界(它在 v≠8 下本可重规划得更好)。
═══════════════════════════════════════════════════════════════════════

用法:.venv/bin/python experiments/pl_iter3_battery_reach.py
输出:experiments/pl_iter3_battery_reach.json
"""
import os, sys, json, time, itertools

HERE = os.path.dirname(os.path.abspath(__file__))
EM100 = os.path.dirname(HERE)
sys.path.insert(0, os.path.dirname(EM100))
sys.path.insert(0, EM100)
import numpy as np
import physics_eval as pe
from planning_experiment import M100Em, DistanceEm, score_path
from wall_experiment import clean_wall_map

V_PLAN = 8.0
V_RESCORE = [4.0, 12.0]
MARGIN = 2.0
GOALS_X = [45, 50, 55, 60, 65, 70, 75]
GOALS_Y = [-25, -20, -15, -10, -5, 0, 5, 10, 15, 20, 25]
GOAL_Z = -3.0
START = np.array([10., 0., -3.])


def classify(P):
    max_alt = float(-np.asarray(P)[:, 2].min())
    max_y = float(np.abs(np.asarray(P)[:, 1]).max())
    return ("climb" if max_alt > 10 else "low"), round(max_alt, 1), round(max_y, 1)


def main():
    src = open(os.path.join(EM100, "state", "best_featurize.py")).read()
    m100 = M100Em(src, payload=250.); dist = DistanceEm()
    vg, esdf, _ = clean_wall_map(16., 12.)

    goals, rows = [], []
    for gx, gy in itertools.product(GOALS_X, GOALS_Y):
        goals.append(np.array([float(gx), float(gy), GOAL_Z]))
    print(f"规划 {len(goals)} 目标 × 2 规划器(v={V_PLAN}, margin={MARGIN})...")
    t0 = time.time()
    for i, g in enumerate(goals):
        row = {"x": g[0], "y": g[1]}
        pd, _ = pe.energy_astar(vg, esdf, dist, START, g, velocity=V_PLAN,
                                safety_margin=MARGIN, max_expand=1500000)
        pe_, ce = pe.energy_astar(vg, esdf, m100, START, g, velocity=V_PLAN,
                                  safety_margin=MARGIN, max_expand=1500000)
        if not pd or not pe_:
            row["fail"] = True; rows.append(row); continue
        row["E_dist"] = round(score_path(pd, m100, V_PLAN), 1)
        row["E_energy"] = round(float(ce), 1)
        row["mode_dist"], row["alt_dist"], row["ymax_dist"] = classify(pd)
        row["mode_energy"], row["alt_energy"], row["ymax_energy"] = classify(pe_)
        for v in V_RESCORE:
            row[f"E_dist@v{int(v)}"] = round(score_path(pd, m100, v), 1)
            row[f"E_energy@v{int(v)}"] = round(score_path(pe_, m100, v), 1)
        rows.append(row)
        if (i + 1) % 11 == 0:
            print(f"  {i+1}/{len(goals)} [{time.time()-t0:.0f}s]")

    ok = [r for r in rows if not r.get("fail")]
    print(f"成功 {len(ok)}/{len(rows)}")

    def sweep(ekey_d, ekey_e):
        Ed = np.array([r[ekey_d] for r in ok]); Ee = np.array([r[ekey_e] for r in ok])
        Bs = np.linspace(Ee.min(), Ed.max() * 1.05, 120)
        curve = []
        for B in Bs:
            nd = int((Ed <= B).sum()); ne = int((Ee <= B).sum())
            dn = int(((Ee <= B) & (Ed > B)).sum())
            curve.append({"B": round(float(B), 0), "N_dist": nd, "N_energy": ne, "dN": dn,
                          "rel_pct": round(100 * dn / nd, 1) if nd else None})
        valid = [c for c in curve if c["N_dist"] >= 5 and c["rel_pct"] is not None]
        peak = max(valid, key=lambda c: c["rel_pct"]) if valid else None
        peak_abs = max(curve, key=lambda c: c["dN"]) if curve else None
        return curve, peak, peak_abs

    curve8, peak8, peak8abs = sweep("E_dist", "E_energy")
    sens = {}
    for v in V_RESCORE:
        _, pk, pka = sweep(f"E_dist@v{int(v)}", f"E_energy@v{int(v)}")
        sens[f"v{int(v)}"] = {"peak_rel": pk, "peak_abs": pka}

    # 差集目标的空间结构(在 v=8 绝对峰值预算处)
    Bstar = peak8abs["B"]
    flipped = [{"x": r["x"], "y": r["y"], "E_dist": r["E_dist"], "E_energy": r["E_energy"],
                "mode_dist": r["mode_dist"], "mode_energy": r["mode_energy"]}
               for r in ok if r["E_energy"] <= Bstar < r["E_dist"]]

    supported = (peak8abs["dN"] >= 5 and peak8 and peak8["rel_pct"] >= 10.0
                 and all(s["peak_rel"] and s["peak_rel"]["rel_pct"] >= 5.0 for s in sens.values()))
    refuted = bool(peak8 is None or peak8["rel_pct"] < 5.0)
    result = {
        "preregistered": {
            "criteria": "SUPPORTED iff exists B: dN>=5 AND rel>=10% (N_dist>=5) AND "
                        "conservative rescore v=4/12 keeps peak rel>=5%; REFUTED iff peak rel<5%@v8",
            "goal_grid": {"x": GOALS_X, "y": GOALS_Y, "z": GOAL_Z}, "v_plan": V_PLAN,
            "margin": MARGIN, "rescore_v": V_RESCORE},
        "per_goal": rows,
        "budget_curve_v8": curve8,
        "peak_rel_v8": peak8, "peak_abs_v8": peak8abs,
        "speed_sensitivity_rescore": sens,
        "flipped_goals_at_Bstar": {"B": Bstar, "goals": flipped},
        "verdict_hint": "SUPPORTED" if supported else ("REFUTED" if refuted else "INCONCLUSIVE-zone"),
    }
    with open(os.path.join(HERE, "pl_iter3_battery_reach.json"), "w") as f:
        json.dump(result, f, ensure_ascii=False, indent=1,
                  default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print(f"peak_rel@v8={peak8}  peak_abs@v8={peak8abs}")
    print(f"sens={ {k: v['peak_rel'] for k, v in sens.items()} }")
    print(f"verdict_hint={result['verdict_hint']}  JSON 落盘 experiments/pl_iter3_battery_reach.json")


if __name__ == "__main__":
    main()
