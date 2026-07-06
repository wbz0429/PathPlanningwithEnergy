"""
report.py — 从实验日志生成中期答辩图

Generates the two midterm figures:
  1. fig_convergence.png  — 优化收敛曲线(每轮 score + 至今最优 vs_astar),
                            读 experiments/log.jsonl(经典搜索);若有 agent_log.jsonl 则叠加。
  2. fig_before_after.png — 优化前后对比(默认 vs best,三场景能耗柱状 + A* 基线线),
                            用 evaluator 在留出种子(seed0=100)上重评,honest held-out。

用法:  python report.py [runs]     (runs 默认 5,用于前后对比重评)
"""
import os
import sys
import json

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import matplotlib.pyplot as plt
import evaluator as ev

# 中文字体(macOS 常见 CJK 字体回退,避免方块)
plt.rcParams["font.sans-serif"] = [
    "PingFang SC", "Heiti SC", "STHeiti", "Arial Unicode MS",
    "Hiragino Sans GB", "Songti SC", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False

EXP = os.path.join(_HERE, "experiments")
HELD_OUT_SEED = 100
DEFAULT_CONFIG = dict(step_size=1.5, max_iterations=5000, goal_sample_rate=0.4, search_radius=4.0,
                      dubins_turning_radius=1.5, weight_energy=0.6, weight_distance=0.3,
                      weight_time=0.1, planning_timeout=10.0, energy_aware=True)


def _load_jsonl(path):
    if not os.path.exists(path):
        return []
    return [json.loads(l) for l in open(path) if l.strip()]


def fig_convergence():
    cls = _load_jsonl(os.path.join(EXP, "log.jsonl"))
    agent = _load_jsonl(os.path.join(EXP, "agent_log.jsonl"))
    if not cls and not agent:
        print("[skip] 无 log.jsonl / agent_log.jsonl"); return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    def series(recs):
        its = [r["iter"] for r in recs]
        sc = [r["score"] if r.get("score") else np.nan for r in recs]
        best = np.inf; bs = []
        for s in sc:
            if not np.isnan(s) and s < best:
                best = s
            bs.append(best)
        # 至今最优对应的 vs_astar
        bestvs = np.inf; bvs = []; curvs = np.inf
        bsc = np.inf
        for r in recs:
            s = r.get("score"); v = r.get("vs_astar")
            if s and v and s < bsc:
                bsc = s; curvs = v
            bvs.append(curvs if curvs != np.inf else np.nan)
        return its, sc, bs, bvs

    if cls:
        its, sc, bs, bvs = series(cls)
        ax1.scatter(its, sc, s=22, c="#bbb", label="每轮 score(经典搜索)", zorder=2)
        ax1.plot(its, bs, "-o", ms=3, c="#1f77b4", label="至今最优 score", zorder=3)
        ax2.plot(its, bvs, "-o", ms=3, c="#1f77b4", label="经典参数搜索")
    if agent:
        its, sc, bs, bvs = series(agent)
        ax1.plot(its, bs, "-s", ms=3, c="#d62728", label="至今最优 score(智能体闭环·Mock离线)")
        ax2.plot(its, bvs, "-s", ms=3, c="#d62728", label="智能体闭环(Mock离线)")

    ax1.set_yscale("log")
    ax1.set_xlabel("迭代 iteration"); ax1.set_ylabel("score = 三场景总能耗(J,失败重罚)")
    ax1.set_title("(a) 搜索过程:score 收敛"); ax1.legend(fontsize=8); ax1.grid(alpha=0.3)

    ax2.axhline(1.0, ls="--", c="green", lw=1.2, label="A* 最优 (=1.0)")
    ax2.set_xlabel("迭代 iteration"); ax2.set_ylabel("至今最优 vs A* 能耗比")
    ax2.set_title("(b) 至今最优路径质量(越接近 1 越好)"); ax2.legend(fontsize=8); ax2.grid(alpha=0.3)

    fig.suptitle("无人机路径规划自优化 — 收敛曲线", fontsize=13, weight="bold")
    fig.tight_layout()
    out = os.path.join(EXP, "fig_convergence.png")
    fig.savefig(out, dpi=140); plt.close(fig)
    print(f"[saved] {out}")


def _median_plan_time(ov, k=3):
    """场景A(窄通道)规划耗时中位数。均值会被少数失败run的超时甩尾污染,故用中位数(稳健)。"""
    import time, random
    from planning.config import PlanningConfig
    from planning.rrt_star import RRTStar
    vg, esdf, em = ev._get_map()
    ck = dict(ev.BASE); ck.update(ov)
    ck.setdefault("energy_aware", True); ck.setdefault("flight_velocity", 2.0)
    start = np.array([0., 0., -3.]); goal = np.array([70., 0., -3.])
    ts = []
    for r in range(k):
        random.seed(HELD_OUT_SEED + r); np.random.seed(HELD_OUT_SEED + r)
        p = RRTStar(vg, esdf, PlanningConfig(**ck), energy_model=em)
        t0 = time.time(); p.plan(start, goal); ts.append(time.time() - t0)
    return float(np.median(ts))


def fig_before_after(runs=5):
    """三方法对比:默认 RRT* / 参数搜索 best / RRT-Connect(Stage-4) + A* 参考。
    讲清故事:纯参数搜索撞天花板(能耗换可靠),算法改动 RRT-Connect 才破权衡(能耗↓+规划快~60×)。"""
    bpath = os.path.join(EXP, "best.json")
    best_cfg = json.load(open(bpath))["config"]
    best_ov = dict(best_cfg, planning_timeout=10.0, energy_aware=True)
    connect_ov = dict(DEFAULT_CONFIG, use_rrt_connect=True)

    astar = ev.astar_baseline()
    scs = ["A_straight", "B_diag_up", "C_diag_down"]

    methods = [
        ("默认 RRT*",       "#ff7f0e", DEFAULT_CONFIG),
        ("参数搜索 best",   "#9467bd", best_ov),
        ("RRT-Connect",     "#1f77b4", connect_ov),
    ]
    results = []
    for name, _, ov in methods:
        print(f"[eval] 留出种子 seed0={HELD_OUT_SEED}, runs={runs} — {name} ...")
        results.append(ev.evaluate(ov, runs=runs, seed0=HELD_OUT_SEED))

    def e(res, name):
        return res["detail"].get(name, {}).get("energy_mean") or 0.0

    e_astar = [astar[s]["energy"] for s in scs]
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    ax1, ax2, ax3 = axes

    # (a) 三场景能耗
    x = np.arange(len(scs)); w = 0.25
    for k, (name, color, _) in enumerate(methods):
        vals = [e(results[k], s) for s in scs]
        ax1.bar(x + (k - 1) * w, vals, w, label=name, color=color)
    for i, ea in enumerate(e_astar):
        ax1.hlines(ea, x[i]-0.42, x[i]+0.42, colors="green", ls="--", lw=1.5,
                   label="A* 最优" if i == 0 else None)
    ax1.set_xticks(x); ax1.set_xticklabels(["A 直穿", "B 对角上", "C 对角下"])
    ax1.set_ylabel("能耗 (J)"); ax1.set_title("(a) 三场景能耗 vs A*")
    ax1.legend(fontsize=8); ax1.grid(alpha=0.3, axis="y")

    # (b) vsA* 能耗比 + 成功率
    labels = [m[0] for m in methods] + ["A* 最优"]
    vs = [r.get("vs_astar") or 0 for r in results] + [1.0]
    sr = [r["min_success"] for r in results] + [1.0]
    colors = [m[1] for m in methods] + ["green"]
    ax2.bar(labels, vs, color=colors, alpha=0.88)
    for i, (v, s) in enumerate(zip(vs, sr)):
        ax2.text(i, v + 0.004, f"{v:.3f}\n{s:.0%}成功", ha="center", fontsize=8)
    ax2.axhline(1.0, ls="--", c="green", lw=1)
    ax2.set_ylabel("vs A* 能耗比"); ax2.set_ylim(0.95, max(vs) + 0.06)
    ax2.set_title("(b) 能耗比 + 成功率")
    ax2.tick_params(axis="x", labelsize=8, rotation=12); ax2.grid(alpha=0.3, axis="y")

    # (c) 场景A规划耗时中位数(log,稳健于失败run甩尾)
    ct = [_median_plan_time(ov) for _, _, ov in methods]
    ax3.bar([m[0] for m in methods], ct, color=[m[1] for m in methods], alpha=0.88)
    for i, c in enumerate(ct):
        ax3.text(i, c, f"{c:.2f}s", ha="center", va="bottom", fontsize=9)
    if ct[-1] > 0:
        ax3.set_title(f"(c) 场景A规划耗时中位数  (RRT-Connect 快 ~{ct[0]/ct[-1]:.0f}×)")
    ax3.set_yscale("log"); ax3.set_ylabel("规划耗时 (s, log)")
    ax3.tick_params(axis="x", labelsize=8, rotation=12); ax3.grid(alpha=0.3, axis="y")

    fig.suptitle("无人机路径规划自优化 — 方法对比(held-out seed=100)",
                 fontsize=13, weight="bold")
    fig.tight_layout()
    out = os.path.join(EXP, "fig_before_after.png")
    fig.savefig(out, dpi=140); plt.close(fig)
    print(f"[saved] {out}")
    for (name, _, _), r, c in zip(methods, results, ct):
        print(f"  {name:14s} vsA*={r.get('vs_astar'):.4f} succ={r['min_success']:.0%} "
              f"planA_median={c:.2f}s")


if __name__ == "__main__":
    runs = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    fig_convergence()
    fig_before_after(runs)
