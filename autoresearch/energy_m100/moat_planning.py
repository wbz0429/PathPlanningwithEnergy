# -*- coding: utf-8 -*-
"""moat_planning.py — 在路径规划这个核心任务上直接验数据优势(反杀"随机拟合都和你一样")。
论点:随机拟合的能耗模型能达到相同 ARE,但当规划代价时会不会做出相同的规划决策?
做法:造 N 个"同样准(留出ARE≈1.9%)"的随机拟合能耗模型,各自当 energy_astar 的代价,在墙场景规划;
比较它们与"真机验证模型"的:(a)决策(翻/绕) (b)路径能耗(用真机模型这把公认尺子评)。
若随机模型规划出更差/翻墙的路 → 数据优势在路径规划上直接体现;若都一样 → 诚实承认。
输出 experiments/pub/护城河_规划决策.png
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import physics_eval as pe
from planning_experiment import M100Em, DistanceEm, score_path
from wall_experiment import clean_wall_map
from autoresearch_energy_real import LIB, evaluate
from pubstyle import PAL, save
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__)); EXP = os.path.join(HERE, "experiments")

# LIB 项名 → 能在 featurize 源码里重建的 numpy 表达式(状态字典 s)
EXPR = {
    "v": "s['v_h']", "v2": "s['v_h']**2", "v3": "s['v_h']**3", "vz": "s['v_z']",
    "climb+": "np.maximum(0.,s['v_z'])", "|az|": "np.abs(s['a_z'])", "ah": "s['a_h']",
    "omega": "s['omega']", "payload": "s['payload']", "wind": "s['wind']",
    "v*pay": "s['v_h']*s['payload']", "v2*pay": "s['v_h']**2*s['payload']",
}


def featurize_src(terms):
    cols = ",".join(["np.ones_like(s['v_h'])"] + [EXPR[t] for t in terms])
    return f"def featurize(s):\n import numpy as np\n return np.column_stack([{cols}])"


def held_out_ARE(terms):
    fn = lambda s: np.column_stack([np.ones_like(s["v_h"])] + [LIB[t](s) for t in terms])
    return float(np.mean([evaluate(fn, seed=k)["energy_ARE"] for k in (5, 6, 7)]))


def main():
    rng = np.random.default_rng(0); keys = list(LIB)
    vg, esdf, bemt = clean_wall_map(16., 12.)
    s0 = np.array([10., 0., -3.]); g0 = np.array([72., 0., -3.]); V = 8.0
    # 公认尺子:真机验证模型(评所有路的能耗都用它)
    ours_terms = ["v", "v2", "v3", "climb+", "payload", "v2*pay"]
    yard = M100Em(featurize_src(ours_terms), payload=250., name="真机")

    def plan_and_eval(em, tag):
        p, _ = pe.energy_astar(vg, esdf, em, s0, g0, velocity=V, safety_margin=0.6, max_expand=1500000)
        if not p or len(p) < 2:
            return None
        P = np.array([np.asarray(w, float).reshape(-1)[:3] for w in p])
        alt, ydev = -P[:, 2].min(), np.abs(P[:, 1]).max()
        E = score_path(p, yard, V)   # 用真机尺子评这条路的能耗
        return {"tag": tag, "choice": "翻越" if alt > ydev + 2 else "绕行",
                "alt": round(float(alt), 1), "E_yard": round(E, 0)}

    rows = [plan_and_eval(yard, "★真机验证模型")]
    print(f"★真机验证模型: {rows[0]['choice']} 最高{rows[0]['alt']}m 能耗{rows[0]['E_yard']}J\n")

    # N 个同样准的随机拟合模型当代价
    got = 0; tries = 0
    while got < 6 and tries < 120:
        tries += 1
        k = int(rng.integers(3, 9)); terms = list(rng.choice(keys, size=k, replace=False))
        are = held_out_ARE(terms)
        if are >= 0.021:   # 只要和我们一样准的
            continue
        em = M100Em(featurize_src(terms), payload=250., name=f"rand{got+1}")
        r = plan_and_eval(em, f"随机#{got+1}")
        if r:
            r["ARE%"] = round(are*100, 2); rows.append(r); got += 1
            print(f"  {r['tag']}(ARE {r['ARE%']}%): {r['choice']} 最高{r['alt']}m 能耗{r['E_yard']}J")

    ourE = rows[0]["E_yard"]
    worse = [r for r in rows[1:] if r["E_yard"] > ourE + 50]
    diff_choice = [r for r in rows[1:] if r["choice"] != rows[0]["choice"]]
    print(f"\n→ {len(rows)-1} 个同样准的随机模型当规划代价:")
    print(f"   决策与真机模型不同的: {len(diff_choice)}  |  规划出更费电路的(>+50J): {len(worse)}")
    if worse or diff_choice:
        print("   → 数据优势在路径规划上体现:同样准但规划决策/能耗不同,真机模型更省")
    else:
        print("   → 诚实:此场景随机模型规划决策也一样,数据优势不体现在规划决策上")

    json.dump({"ours_E": ourE, "rows": rows, "n_diff_choice": len(diff_choice), "n_worse": len(worse)},
              open(os.path.join(EXP, "moat_planning.json"), "w"), ensure_ascii=False, indent=2)
    _fig(rows, ourE)


def _fig(rows, ourE):
    names = [r["tag"] for r in rows]; E = [r["E_yard"] for r in rows]; ch = [r["choice"] for r in rows]
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    colors = [PAL["green"] if n.startswith("★") else (PAL["red"] if ch[i] != rows[0]["choice"] or E[i] > ourE+50 else PAL["blue"])
              for i, n in enumerate(names)]
    y = np.arange(len(names))
    ax.barh(y, E, color=colors)
    ax.axvline(ourE, color=PAL["green"], ls=(0, (4, 3)), lw=1.5)
    for i, (e, c) in enumerate(zip(E, ch)):
        ax.text(e + 15, y[i], f"{e:.0f}J·{c}", va="center", fontsize=8)
    ax.set_yticks(y); ax.set_yticklabels(names, fontsize=9); ax.invert_yaxis()
    ax.set_xlabel("规划出的路径能耗(真机尺子评,J)——越低越好")
    ax.set_title("同样准(ARE≈1.9%)的能耗模型当规划代价,规划决策/能耗对比\n(绿=真机验证模型;红=决策不同或更费电)")
    plt.tight_layout(); save(fig, "护城河_规划决策")
    print("图存 pub/护城河_规划决策")


if __name__ == "__main__":
    main()
