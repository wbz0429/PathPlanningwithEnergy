"""pl_iter5_savings_dist.py — 规划层 iter5:H5 省能分布(预注册实验)

═══ 预注册(先写判据,后跑实验)═══════════════════════════════════════════
假设 H5:能量感知规划相对距离规划的省能,在随机城市块场景系综上不是单点轶事
  ("4-22%")而是有结构的分布——由场景几何(是否存在迫使距离规划爬升的高/宽阻挡)
  决定。交付 = 分布统计(中位数/四分位/份额),不是最好单点。
场景生成器(确定性,seed=0..19,预注册于此):楼群区 x∈[25,60]×y∈[-20,20];
  每景 N=rng.integers(3,7) 栋楼;楼心均匀采样,footprint w_x,w_y~U[4,14]m,
  高 h~U[4,12]m(栅格顶棚约束);允许重叠(复杂形状);start=(5,0,-3),goal=(75,0,-3);
  规划 v=8、margin=2.0(可飞裕度,两规划器同);尺=冻结 M100Em(250g)。
判据:
  (1) SUPPORTED 当且仅当:成功场景中 ≥25% 达到省能 ≥3%(能量感知在随机系综的
      非平凡份额上有实质价值),且中位省能 ≥0(不倒贴),且该份额在保守重计价
      (v=4/12,不重规划,能量侧下界)下 v=12 侧仍 ≥25%(v=4 已知萎缩,报告不设门);
  (2) REFUTED 当 <10% 场景达 ≥3%(价值只存在于精心构造的几何,随机系综撑不起);
  (3) 10-25% → INCONCLUSIVE-zone,报实测;
  (4) 全 20 景逐景数据 + 分布统计 + 场景特征相关(最高楼/阻挡宽度 vs 省能)整体报告;
      规划失败景记 fail 并计数(不悄悄丢弃)。
═══════════════════════════════════════════════════════════════════════

用法:.venv/bin/python experiments/pl_iter5_savings_dist.py
输出:experiments/pl_iter5_savings_dist.json
"""
import os, sys, copy, json, time

HERE = os.path.dirname(os.path.abspath(__file__))
EM100 = os.path.dirname(HERE)
sys.path.insert(0, os.path.dirname(EM100))
sys.path.insert(0, EM100)
import numpy as np
import physics_eval as pe
from planning_experiment import M100Em, DistanceEm, score_path

N_SCENES = 20
V_PLAN = 8.0
V_RESCORE = [4.0, 12.0]
MARGIN = 2.0
START = np.array([5., 0., -3.]); GOAL = np.array([75., 0., -3.])
_BASE = None


def gen_scene(seed):
    rng = np.random.default_rng(seed)
    n = int(rng.integers(3, 7))
    boxes = []
    for _ in range(n):
        cx = float(rng.uniform(25, 60)); cy = float(rng.uniform(-20, 20))
        wx = float(rng.uniform(4, 14)); wy = float(rng.uniform(4, 14))
        h = float(rng.uniform(4, 12))
        boxes.append({"x0": round(cx - wx / 2, 1), "x1": round(cx + wx / 2, 1),
                      "y0": round(cy - wy / 2, 1), "y1": round(cy + wy / 2, 1),
                      "h": round(h, 1)})
    return boxes


def build_map(boxes):
    global _BASE
    if _BASE is None:
        _BASE = pe.get_grounded_map()
    vg0, esdf0, _ = _BASE
    vg = copy.deepcopy(vg0); vg.grid[:] = 0
    for iz in range(vg.grid.shape[2]):
        if vg.grid_to_world((0, 0, iz))[2] > -0.6:
            vg.grid[:, :, iz] = 1
    for b in boxes:
        for xw in np.arange(b["x0"], b["x1"] + .25, .5):
            for yw in np.arange(b["y0"], b["y1"] + .25, .5):
                for zw in np.arange(-b["h"], 0.1, .5):
                    idx = vg.world_to_grid(np.array([xw, yw, zw]))
                    if vg.is_valid_index(idx):
                        vg.grid[idx] = 1
    esdf = type(esdf0)(vg); esdf.compute()
    return vg, esdf


def classify(P):
    P = np.asarray(P)
    return {"max_alt": round(float(-P[:, 2].min()), 1),
            "max_abs_y": round(float(np.abs(P[:, 1]).max()), 1),
            "length": round(float(np.sum(np.linalg.norm(np.diff(P, axis=0), axis=1))), 1)}


def main():
    src = open(os.path.join(EM100, "state", "best_featurize.py")).read()
    m100 = M100Em(src, payload=250.); dist = DistanceEm()
    rows = []
    t0 = time.time()
    for seed in range(N_SCENES):
        boxes = gen_scene(seed)
        vg, esdf = build_map(boxes)
        row = {"seed": seed, "n_bldg": len(boxes), "boxes": boxes,
               "max_h": max(b["h"] for b in boxes)}
        pd, _ = pe.energy_astar(vg, esdf, dist, START, GOAL, velocity=V_PLAN,
                                safety_margin=MARGIN, max_expand=1500000)
        pen, ce = pe.energy_astar(vg, esdf, m100, START, GOAL, velocity=V_PLAN,
                                  safety_margin=MARGIN, max_expand=1500000)
        if not pd or not pen:
            row["fail"] = True; rows.append(row)
            print(f"seed {seed}: FAIL"); continue
        Pd = np.array([np.asarray(w, float).reshape(-1)[:3] for w in pd])
        Pn = np.array([np.asarray(w, float).reshape(-1)[:3] for w in pen])
        ed = score_path(pd, m100, V_PLAN); en = float(ce)
        row["E_dist"], row["E_energy"] = round(ed, 1), round(en, 1)
        row["sav_pct"] = round(100 * (ed - en) / ed, 2)
        row["path_dist"], row["path_energy"] = classify(Pd), classify(Pn)
        for v in V_RESCORE:
            e_d = score_path(pd, m100, v); e_n = score_path(pen, m100, v)
            row[f"sav_pct@v{int(v)}"] = round(100 * (e_d - e_n) / e_d, 2)
        rows.append(row)
        print(f"seed {seed}: n={len(boxes)} maxh={row['max_h']:.0f} sav={row['sav_pct']}% "
              f"(dist alt{row['path_dist']['max_alt']} / energy alt{row['path_energy']['max_alt']}) "
              f"[{time.time()-t0:.0f}s]")

    ok = [r for r in rows if not r.get("fail")]
    sav = np.array([r["sav_pct"] for r in ok])
    stats = {"n_ok": len(ok), "n_fail": len(rows) - len(ok),
             "median": round(float(np.median(sav)), 2),
             "q25": round(float(np.percentile(sav, 25)), 2),
             "q75": round(float(np.percentile(sav, 75)), 2),
             "min": round(float(sav.min()), 2), "max": round(float(sav.max()), 2),
             "frac_ge3pct": round(float((sav >= 3).mean()), 3),
             "frac_zero": round(float((np.abs(sav) < 0.5).mean()), 3)}
    for v in V_RESCORE:
        sv = np.array([r[f"sav_pct@v{int(v)}"] for r in ok])
        stats[f"median@v{int(v)}"] = round(float(np.median(sv)), 2)
        stats[f"frac_ge3pct@v{int(v)}"] = round(float((sv >= 3).mean()), 3)

    supported = (stats["frac_ge3pct"] >= 0.25 and stats["median"] >= 0
                 and stats["frac_ge3pct@v12"] >= 0.25)
    refuted = stats["frac_ge3pct"] < 0.10
    result = {"preregistered": {
                  "criteria": "SUPPORTED iff frac(sav>=3%)>=25% AND median>=0 AND frac@v12>=25%; "
                              "REFUTED iff frac<10%; else INCONCLUSIVE-zone",
                  "generator": "seeds 0..19, buildings x[25,60] y[-20,20] wx,wy U[4,14] h U[4,12] "
                               "N U{3..6}, start(5,0,-3) goal(75,0,-3), v8, margin 2.0"},
              "per_scene": rows, "stats": stats,
              "verdict_hint": "SUPPORTED" if supported else ("REFUTED" if refuted else "INCONCLUSIVE-zone")}
    with open(os.path.join(HERE, "pl_iter5_savings_dist.json"), "w") as f:
        json.dump(result, f, ensure_ascii=False, indent=1,
                  default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print(f"stats={stats}")
    print(f"verdict_hint={result['verdict_hint']}  JSON 落盘 experiments/pl_iter5_savings_dist.json")


if __name__ == "__main__":
    main()
