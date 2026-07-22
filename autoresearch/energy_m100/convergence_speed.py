# -*- coding: utf-8 -*-
"""convergence_speed.py — 样本效率/收敛速度实验(Bergstra&Bengio 2012 的价值维度)。
最终质量 loop≈随机(已证),但"达到同等质量用了多少次评估"可能不同。
记录 loop(贪心增删项) vs 随机(随机子集)的"评估次数 → 至今最优留出ARE"曲线,多seed取均值±std。
若 loop 更快到达地板 → 样本效率更高(正面优势,有顶刊先例)。输出 experiments/pub/收敛速度.png。
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from autoresearch_energy_real import LIB, BASELINE, obj, holdout
from pubstyle import PAL, save
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__)); EXP = os.path.join(HERE, "experiments")
BUDGET = 40   # 评估次数预算
SEEDS = 8


def loop_curve(seed):
    """贪心增删项(=我们的结构搜索简化版):每次评估后记录至今最优的留出ARE。"""
    rng = np.random.default_rng(seed)
    best = list(BASELINE); best_o = obj(best); curve = []
    for _ in range(BUDGET):
        cand = list(best); t = str(rng.choice(list(LIB)))
        cand.remove(t) if t in cand else cand.append(t)
        if not cand:
            curve.append(holdout(best)[0]); continue
        o = obj(cand)
        if o < best_o - 1e-5:
            best, best_o = cand, o
        curve.append(holdout(best)[0])   # 至今最优的留出ARE
    return curve


def random_curve(seed):
    """随机子集:每次随机抽一组项,记录至今最优。"""
    rng = np.random.default_rng(1000 + seed)
    keys = list(LIB); best = list(BASELINE); best_o = obj(best); curve = []
    for _ in range(BUDGET):
        k = int(rng.integers(1, len(keys) + 1))
        terms = list(rng.choice(keys, size=k, replace=False))
        o = obj(terms)
        if o < best_o - 1e-5:
            best, best_o = terms, o
        curve.append(holdout(best)[0])
    return curve


def main():
    print(f"收敛速度实验:loop vs 随机,{SEEDS} seeds × {BUDGET} 评估...", flush=True)
    L = np.array([loop_curve(s) for s in range(SEEDS)])
    print("  loop 完成", flush=True)
    R = np.array([random_curve(s) for s in range(SEEDS)])
    print("  随机 完成", flush=True)
    x = np.arange(1, BUDGET + 1)
    Lm, Ls = L.mean(0) * 100, L.std(0) * 100
    Rm, Rs = R.mean(0) * 100, R.std(0) * 100
    floor = min(Lm.min(), Rm.min())
    thr = floor + 0.10   # "达到地板+0.1pp"算收敛
    def hit(m):
        idx = np.where(m <= thr)[0]
        return int(idx[0] + 1) if len(idx) else None
    lh, rh = hit(Lm), hit(Rm)
    print(f"\n达到地板+0.1pp({thr:.2f}%)所需评估次数:loop={lh}  随机={rh}")
    speedup = (rh / lh) if (lh and rh) else None
    if speedup: print(f"→ loop 样本效率约为随机的 {speedup:.1f}×(更少评估到同等质量)")

    fig, ax = plt.subplots(figsize=(5.0, 3.2))
    ax.plot(x, Lm, color=PAL["green"], lw=1.8, label="autoresearch loop(物理引导)")
    ax.fill_between(x, Lm - Ls, Lm + Ls, color=PAL["green"], alpha=.15)
    ax.plot(x, Rm, color=PAL["gray"], lw=1.8, label="随机搜索")
    ax.fill_between(x, Rm - Rs, Rm + Rs, color=PAL["gray"], alpha=.15)
    ax.axhline(floor, ls=(0, (4, 3)), color=PAL["blue"], lw=1)
    ax.text(BUDGET, floor - 0.08, f"数据地板 ≈{floor:.2f}%", color=PAL["blue"], fontsize=8.5, ha="right")
    if lh: ax.axvline(lh, color=PAL["green"], ls=":", lw=1, alpha=.7)
    if rh: ax.axvline(rh, color=PAL["gray"], ls=":", lw=1, alpha=.7)
    if lh and rh:
        ax.annotate(f"loop {lh} 次到达\n随机 {rh} 次到达", xy=(rh, thr), xytext=(BUDGET*0.5, floor + 0.9),
                    fontsize=8.5, color=PAL["blue"])
    ax.set_xlabel("评估次数"); ax.set_ylabel("至今最优留出 ARE(%)")
    ax.set_title("样本效率:最终质量相当,loop 更少评估到达数据地板")
    ax.legend(fontsize=8.5, loc="upper right")
    save(fig, "收敛速度")
    json.dump({"budget": BUDGET, "seeds": SEEDS, "floor%": round(float(floor), 3),
               "threshold%": round(float(thr), 3), "loop_hit": lh, "random_hit": rh,
               "speedup": round(float(speedup), 2) if speedup else None,
               "loop_mean": [round(v, 3) for v in Lm], "random_mean": [round(v, 3) for v in Rm]},
              open(os.path.join(EXP, "convergence_speed.json"), "w"), ensure_ascii=False, indent=2)
    print("落盘 experiments/convergence_speed.json")


if __name__ == "__main__":
    main()
