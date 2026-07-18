# -*- coding: utf-8 -*-
"""tradeoff_fig.py — 把“翻越 vs 绕行”的能量权衡画死:
翻越 = 少走水平距离 + 多付爬升能耗(随墙宽几乎不变);
绕行 = 无爬升 + 多走水平距离(随墙宽线性上升)。
两条线交叉点 = 相图里 M100 从“绕行”翻转到“翻越”的临界墙宽。
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from planning_experiment import M100Em, DistanceEm, score_path
plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "Heiti TC"]; plt.rcParams["axes.unicode_minus"] = False
HERE = os.path.dirname(os.path.abspath(__file__)); EXP = os.path.join(HERE, "experiments")

m100 = M100Em(open(os.path.join(HERE, "state", "best_featurize.py")).read(), payload=250.)
V = 8.0; H = 13.0        # 墙高 13m,起(10,0,-3)终(72,0,-3)


def seg_energy(path):
    return score_path([np.asarray(p, float) for p in path], m100, V)


def over_path():
    # 翻越:爬到墙顶+2,水平穿过,再降回
    return [np.array([10, 0, -3]), np.array([30, 0, -(H+2)]), np.array([52, 0, -(H+2)]), np.array([72, 0, -3])]


def over_flat_equiv():
    # 同水平投影、但不爬升(用来分解出“爬升罚”)
    return [np.array([10, 0, -3]), np.array([30, 0, -3]), np.array([52, 0, -3]), np.array([72, 0, -3])]


def around_path(hy):
    return [np.array([10, 0, -3]), np.array([38, hy+3, -3]), np.array([44, hy+3, -3]), np.array([72, 0, -3])]


widths = np.arange(8, 30.1, 2.0)
E_over = seg_energy(over_path())                 # 翻越,与墙宽无关
E_over_flat = seg_energy(over_flat_equiv())      # 翻越的水平部分
E_climb = E_over - E_over_flat                    # 爬升罚
E_around = np.array([seg_energy(around_path(hy)) for hy in widths])

# 交叉点(绕行=翻越)
diff = E_around - E_over
cross = None
for i in range(len(widths)-1):
    if diff[i] < 0 <= diff[i+1] or diff[i] <= 0 < diff[i+1]:
        cross = widths[i] + (widths[i+1]-widths[i]) * (0-diff[i])/(diff[i+1]-diff[i]); break

fig, ax = plt.subplots(figsize=(10, 5.4))
ax.axhline(E_over, color="tab:red", lw=2.2, label=f"翻越(总能耗,几乎不随墙宽变)")
ax.axhline(E_over_flat, color="tab:red", ls=":", lw=1.2, alpha=.7)
ax.fill_between(widths, E_over_flat, E_over, color="tab:red", alpha=.12)
ax.text(8.3, (E_over+E_over_flat)/2, f"翻越的爬升罚\n≈{E_climb:.0f} J", fontsize=9, color="tab:red", va="center")
ax.text(8.3, E_over_flat-180, "翻越的水平部分", fontsize=8.5, color="tab:red", alpha=.8, va="center")
ax.plot(widths, E_around, "o-", color="tab:green", lw=2.2, label="绕行(总能耗,随墙宽线性↑)")
if cross:
    ax.axvline(cross, color="k", ls="--", alpha=.6)
    ax.text(cross+0.2, E_around.min()+120, f"临界墙宽\n≈{cross:.1f}m", fontsize=10, fontweight="bold")
    ax.annotate("← 窄墙:绕行省\n(M100 绕行)", (widths[0]+1, E_around.min()+60), fontsize=10, color="tab:green")
    ax.annotate("宽墙:翻越省 →\n(M100 也翻墙)", (widths[-1]-6, E_over+60), fontsize=10, color="tab:red")
ax.set_xlabel("墙半宽 (m)——越宽,绕行要多走越多"); ax.set_ylabel("M100 真机能耗 (J)")
ax.set_title("翻越 vs 绕行的能量权衡:\n翻越=少走距离但多付爬升罚(平);绕行=无爬升但多走距离(升)→ 交叉点决定选谁")
ax.legend(loc="lower right", fontsize=9); ax.grid(alpha=.25)
plt.tight_layout()
p = os.path.join(EXP, "tradeoff.png"); plt.savefig(p, dpi=125)
print(f"图存 {p}")
json.dump({"E_over": round(float(E_over)), "E_over_flat": round(float(E_over_flat)),
           "E_climb_penalty": round(float(E_climb)), "crossover_halfwidth_m": round(float(cross), 1) if cross else None,
           "widths": widths.tolist(), "E_around": [round(float(e)) for e in E_around]},
          open(os.path.join(EXP, "tradeoff.json"), "w"), ensure_ascii=False, indent=2)
print(f"翻越总 {E_over:.0f}J(水平 {E_over_flat:.0f} + 爬升罚 {E_climb:.0f});临界墙半宽 ≈{cross:.1f}m" if cross else "无交叉")
