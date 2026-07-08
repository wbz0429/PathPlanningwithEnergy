"""
compare_milestone.py — 里程碑横向对比(答辩生死线)

① 算法横向对比:loop-best vs 业界规划器(A* / 能量加权A* / 默认RRT* / RRT-Connect),同一诚实评测器。
② 优化方法对比:LLM-loop vs 随机搜索(同预算,仅Layer-1参数、默认平滑器)——证明"是 loop 的代码级能力在干活,不是评测器形式化"。

用法: python compare_milestone.py [里程碑号N] [random预算B]
输出: experiments/fig_ms<N>_compare.png + 打印对比表
"""
import os, sys, json, random, time
os.environ.setdefault("MPLBACKEND", "Agg")
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["PingFang SC", "Heiti SC", "Arial Unicode MS", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False

import physics_eval as pe, evaluator as ev, benchmark_planning as bp
from planning.config import PlanningConfig

EXP = os.path.join(_HERE, "experiments")
N = sys.argv[1] if len(sys.argv) > 1 else "1"
BUDGET = int(sys.argv[2]) if len(sys.argv) > 2 else 15
TUNED = json.load(open(os.path.join(_HERE, "state", "best.json")))["config"]
SM = open(os.path.join(_HERE, "state", "best_smoother.py")).read()
DEFAULT = dict(step_size=1.5, max_iterations=5000, goal_sample_rate=0.4, search_radius=4.0,
               weight_energy=0.6, weight_distance=0.3, weight_time=0.1)


def _astar_score(energy_weighted):
    """A* / 能量加权A* 的剖面能耗 score(确定性)。"""
    vg, esdf, em = pe.get_grounded_map()
    vstar = pe._GCACHE.get("vstar") or pe.optimal_cruise_speed(em)[0]
    total = 0.0
    for sc in ev.SCENARIOS:
        cfg = PlanningConfig(**dict(ev.BASE, dubins_turning_radius=1.5, planning_timeout=15.0))
        if energy_weighted:
            p, _ = pe.energy_astar(vg, esdf, em, sc["start"], sc["goal"])
        else:
            p = bp.AStarPlanner(vg, esdf, cfg).plan(sc["start"], sc["goal"])
        p = bp.smooth_path(p, esdf, 1.0) if p else None
        total += pe.energy_with_profile(p, em, vstar) if p else 3000.0
    return total


def _score(ov, smoother=None):
    r = pe.evaluate(ov, runs=3, seed0=0, smoother_src=smoother)
    return r["score"]


print("=" * 64); print(f"  Milestone {N} 横向对比"); print("=" * 64)

# ① 算法横向对比
print("\n① 算法横向对比(同一诚实评测器,越低越好):")
algos = []
for name, fn in [
    ("A* 最短", lambda: _astar_score(False)),
    ("能量加权A*", lambda: _astar_score(True)),
    ("默认 RRT*", lambda: _score(DEFAULT)),
    ("RRT-Connect", lambda: _score(dict(DEFAULT, use_rrt_connect=True))),
    ("LLM-loop best", lambda: _score(TUNED, SM)),
]:
    t = time.time(); s = fn(); algos.append((name, s))
    print(f"   {name:16s} score={s:8.1f}  ({time.time()-t:.0f}s)")

# ② 优化方法对比(生死线):随机搜索同预算
print(f"\n② 优化方法对比(随机搜索预算 B={BUDGET},仅参数+默认平滑器):")
rng = random.Random(2026)
SPACE = {"step_size": (1.0, 5.0), "max_iterations": (3000, 8000),
         "goal_sample_rate": (0.1, 0.6), "search_radius": (3.0, 7.0)}
rand_best = 9e18; rand_curve = []
for i in range(BUDGET):
    c = {k: (rng.randint(int(lo), int(hi)) if k == "max_iterations" else round(rng.uniform(lo, hi), 3))
         for k, (lo, hi) in SPACE.items()}
    c["use_rrt_connect"] = rng.random() < 0.5
    s = _score(c)   # 默认平滑器:随机搜索不会写代码
    rand_best = min(rand_best, s); rand_curve.append(rand_best)
    print(f"   [{i+1:02d}/{BUDGET}] rrt_connect={c['use_rrt_connect']} step={c['step_size']} -> {s:.0f} | best={rand_best:.0f}")

manual = _score(DEFAULT)
llm = dict(algos)["LLM-loop best"]
opt = [("人工默认", manual), ("随机搜索(同预算)", rand_best), ("LLM-loop", llm)]
print(f"\n   人工={manual:.0f} | 随机搜索best={rand_best:.0f} | LLM-loop={llm:.0f}")

# ---- 出图 ----
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.2))
names = [a[0] for a in algos]; scores = [a[1] for a in algos]
cols = ["#7f7f7f", "#2ca02c", "#ff7f0e", "#9467bd", "#1f77b4"]
ax1.bar(names, scores, color=cols)
for i, s in enumerate(scores): ax1.text(i, s, f"{s:.0f}", ha="center", va="bottom", fontsize=9)
ax1.set_ylabel("score(剖面能耗和,越低越好)"); ax1.set_title("① 算法横向对比(诚实评测器)")
ax1.tick_params(axis="x", labelsize=8, rotation=12); ax1.grid(alpha=0.3, axis="y")

on = [o[0] for o in opt]; os_ = [o[1] for o in opt]
ax2.bar(on, os_, color=["#ff7f0e", "#8c564b", "#1f77b4"])
for i, s in enumerate(os_): ax2.text(i, s, f"{s:.0f}", ha="center", va="bottom", fontsize=9)
ax2.set_ylabel("score"); ax2.set_title(f"② 优化方法对比:LLM-loop vs 随机搜索(同预算 B={BUDGET})")
ax2.tick_params(axis="x", labelsize=8); ax2.grid(alpha=0.3, axis="y")
fig.suptitle(f"Milestone {N} · 横向对比:业界算法 + 优化方法生死线", fontsize=13, weight="bold")
fig.tight_layout()
out = os.path.join(EXP, f"fig_ms{N}_compare.png")
fig.savefig(out, dpi=140); plt.close(fig)
print(f"\n[saved] {out}")
json.dump({"algos": algos, "optimizers": opt, "rand_curve": rand_curve, "budget": BUDGET},
          open(os.path.join(EXP, f"compare_ms{N}.json"), "w"), indent=2, ensure_ascii=False)
