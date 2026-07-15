# -*- coding: utf-8 -*-
"""large_scale_savings.py — 无人值守大规模省能分布实验(把 H5 从 n=20 扩到 n=250,收紧 CI)。
每个随机城市场景:距离规划 vs 真机M100规划,记省能% + 决策。增量落盘 JSONL(可断点续、中途可看)。
跑完自动出:统计(中位/四分位/≥3%占比/Wilson CI)+ 直方图。
用法:python large_scale_savings.py [N]        # 跑到 seed N-1(默认 250)
      python large_scale_savings.py plot       # 只根据现有 JSONL 出图+统计
"""
import os, sys, copy, json, time, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import physics_eval as pe
from planning_experiment import M100Em, DistanceEm, score_path

HERE = os.path.dirname(os.path.abspath(__file__))
JL = os.path.join(HERE, "experiments", "large_scale_savings.jsonl")
V_PLAN = 8.0; V_RESCORE = [4.0, 12.0]; MARGIN = 2.0
START = np.array([5., 0., -3.]); GOAL = np.array([75., 0., -3.])
_BASE = None


def gen_scene(seed):
    rng = np.random.default_rng(seed)
    n = int(rng.integers(3, 7)); boxes = []
    for _ in range(n):
        cx = float(rng.uniform(25, 60)); cy = float(rng.uniform(-20, 20))
        wx = float(rng.uniform(4, 14)); wy = float(rng.uniform(4, 14)); h = float(rng.uniform(4, 12))
        boxes.append({"x0": round(cx-wx/2, 1), "x1": round(cx+wx/2, 1),
                      "y0": round(cy-wy/2, 1), "y1": round(cy+wy/2, 1), "h": round(h, 1)})
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
        for xw in np.arange(b["x0"], b["x1"]+.25, .5):
            for yw in np.arange(b["y0"], b["y1"]+.25, .5):
                for zw in np.arange(-b["h"], 0.1, .5):
                    idx = vg.world_to_grid(np.array([xw, yw, zw]))
                    if vg.is_valid_index(idx):
                        vg.grid[idx] = 1
    esdf = type(esdf0)(vg); esdf.compute()
    return vg, esdf


def done_seeds():
    if not os.path.exists(JL):
        return set()
    s = set()
    for ln in open(JL):
        try:
            s.add(json.loads(ln)["seed"])
        except Exception:
            pass
    return s


def run(N):
    src = open(os.path.join(HERE, "state", "best_featurize.py")).read()
    m100 = M100Em(src, payload=250.); dist = DistanceEm()
    have = done_seeds()
    print(f"已完成 {len(have)} 个场景,续跑至 seed {N-1}", flush=True)
    t0 = time.time()
    for seed in range(N):
        if seed in have:
            continue
        try:
            boxes = gen_scene(seed); vg, esdf = build_map(boxes)
            row = {"seed": seed, "n_bldg": len(boxes), "max_h": max(b["h"] for b in boxes)}
            pd, _ = pe.energy_astar(vg, esdf, dist, START, GOAL, velocity=V_PLAN, safety_margin=MARGIN, max_expand=1500000)
            pen, ce = pe.energy_astar(vg, esdf, m100, START, GOAL, velocity=V_PLAN, safety_margin=MARGIN, max_expand=1500000)
            if not pd or not pen:
                row["fail"] = True
            else:
                ed = score_path(pd, m100, V_PLAN); en = float(ce)
                row["sav_pct"] = round(100*(ed-en)/ed, 2) if ed > 0 else 0.0
                for v in V_RESCORE:
                    e_d = score_path(pd, m100, v); e_n = score_path(pen, m100, v)
                    row[f"sav@v{int(v)}"] = round(100*(e_d-e_n)/e_d, 2) if e_d > 0 else 0.0
            with open(JL, "a") as f:
                f.write(json.dumps(row) + "\n"); f.flush()
            el = time.time()-t0
            print(f"seed {seed}: 省能={row.get('sav_pct','FAIL')}%  ({len(have)+seed+1}/{N}, {el:.0f}s)", flush=True)
        except Exception as e:
            with open(JL, "a") as f:
                f.write(json.dumps({"seed": seed, "error": str(e)[:80]}) + "\n")
            print(f"seed {seed}: 异常 {str(e)[:60]}", flush=True)
    stats_and_plot()


def wilson(k, n, z=1.96):
    if n == 0:
        return (0, 0)
    p = k/n; d = 1+z*z/n
    c = (p+z*z/(2*n))/d; hw = z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return (round(100*(c-hw), 1), round(100*(c+hw), 1))


def stats_and_plot():
    rows = [json.loads(l) for l in open(JL)] if os.path.exists(JL) else []
    sav = [r["sav_pct"] for r in rows if "sav_pct" in r]
    if not sav:
        print("无有效数据"); return
    sav = np.array(sav); n = len(sav)
    ge3 = int((sav >= 3).sum())
    lo, hi = wilson(ge3, n)
    print(f"\n=== 省能分布(n={n})===")
    print(f"  中位 {np.median(sav):.1f}%  Q75 {np.percentile(sav,75):.1f}%  Q90 {np.percentile(sav,90):.1f}%  max {sav.max():.1f}%")
    print(f"  ≥3% 占比 {100*ge3/n:.0f}%  (Wilson 95% CI [{lo}%,{hi}%])  ← n=20 时是 [9%,49%]")
    print(f"  =0% 占比 {100*int((sav<0.5).sum())/n:.0f}%(有低空走廊,能量最优≈最短路)")
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "Heiti TC"]; plt.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(figsize=(9, 4.6))
    ax.hist(sav, bins=np.arange(-0.5, max(12, sav.max()+1), 1), color="tab:green", alpha=.8, edgecolor="white")
    ax.axvline(np.median(sav), color="k", ls="--", label=f"中位 {np.median(sav):.1f}%")
    ax.set_xlabel("能量感知规划省能 %(vs 最短距离)"); ax.set_ylabel("场景数")
    ax.set_title(f"省能分布(n={n} 随机城市场景,v=8):≥3% 占比 {100*ge3/n:.0f}% CI[{lo},{hi}]%")
    ax.legend()
    plt.tight_layout()
    p = os.path.join(HERE, "experiments", "large_scale_savings.png")
    plt.savefig(p, dpi=120); print(f"直方图存 {p}")
    json.dump({"n": n, "median": round(float(np.median(sav)), 2), "q75": round(float(np.percentile(sav, 75)), 2),
               "ge3_pct": round(100*ge3/n, 1), "wilson_ci": [lo, hi], "zero_pct": round(100*int((sav < 0.5).sum())/n, 1)},
              open(os.path.join(HERE, "experiments", "large_scale_savings_stats.json"), "w"), ensure_ascii=False, indent=2)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "plot":
        stats_and_plot()
    else:
        run(int(sys.argv[1]) if len(sys.argv) > 1 else 250)
