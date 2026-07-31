# -*- coding: utf-8 -*-
"""complexity_regularity_fig.py — 「引导优势 × 有效组合复杂度」规律图(论文版)。
核心主张(全部由我们的实验支撑):
  引导搜索相对随机的优势,取决于「有效组合复杂度」,而非「名义空间大小」。
  证据:
    ① 能耗域(低有效维度):引导打平随机,z=−0.35(真实端点)。
    ② 负控制(我们自己的受控实验):名义空间 K=10→110 膨胀,
       引导优势仍 ≈ 0(complexity_scaling_fair,6 seeds)——名义膨胀不是杠杆。
    ③ 规划器域(代码结构,高有效复杂度):引导显著胜随机 8.1%,z=−2.38,
       0% 随机跑超过(planner_significance,新鲜复现)。
    ④ 文献:低有效维→随机追平(Bergstra'12);~15-20 维临界(REMBO);
        程序空间→引导胜(FunSearch/AlphaEvolve)。
输出 experiments/pub/复杂度规律.png
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib.pyplot as plt
from pubstyle import PAL, save

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.join(HERE, "experiments")

# 受控负控制:名义空间膨胀不产生优势(我们的实验)
fair = json.load(open(os.path.join(EXP, "complexity_scaling_fair.json")))["rows"]
ks = [r["K"] for r in fair]
advs = [r["advantage_pp"] for r in fair]

fig, ax = plt.subplots(figsize=(7.0, 3.9))
# 文献规律示意曲线(有效复杂度 x;逻辑斯蒂形,0→30%)
x = np.linspace(0, 10, 200)
y = 30 / (1 + np.exp(-(x - 5) * 0.9))
ax.plot(x, y, color=PAL["gray"], lw=1.6, ls=(0, (5, 2)), label="规律(文献综合,示意)")
ax.fill_between(x, 0, y, color=PAL["gray"], alpha=.06)

# 文献标注
lit = [(1.3, 1, "Bergstra&Bengio'12\n低有效维→随机追平"),
       (4.6, 8, "REMBO/贝叶斯\n~15-20维临界"),
       (8.6, 26, "FunSearch/AlphaEvolve\n程序空间")]
for lx, ly, t in lit:
    ax.scatter([lx], [ly], s=28, color=PAL["gray"], zorder=3)
    ax.annotate(t, (lx, ly), (lx, ly+3.0), fontsize=7, color="#666", ha="center")

# ★ 我们:能耗域(低有效维,真实端点)——打平
ax.scatter([1.2], [0.2], s=150, color=PAL["red"], marker="o", zorder=5, edgecolors="white", linewidths=1.5)
ax.annotate("★我们·能耗域(实测)\n打平 z=−0.35", (1.2, 0.2), (2.4, -8.2),
            fontsize=8.5, color=PAL["red"], ha="center", fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=PAL["red"], lw=1))

# ★ 我们:规划器域(高有效复杂度,真实端点)——显著胜(新鲜复现 8.1%)
ax.scatter([8.2], [8.1], s=150, color=PAL["green"], marker="o", zorder=5, edgecolors="white", linewidths=1.5)
ax.annotate("★我们·规划器域(实测)\n代码结构→胜随机 8.1%,z=−2.38,\n0% 随机跑超过", (8.2, 8.1), (7.3, 13.5),
            fontsize=8.5, color=PAL["green"], ha="center", fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=PAL["green"], lw=1))

# 负控制小图:名义空间 K=10→110 膨胀,优势 ≈ 0
axins = ax.inset_axes([0.06, 0.55, 0.34, 0.34])
axins.axhline(0, color=PAL["gray"], lw=.8, ls=(0, (4, 3)))
axins.plot(ks, advs, "o-", color=PAL["blue"], lw=1.4, ms=4)
axins.set_ylim(-0.3, 0.5)
axins.set_xticks([10, 110]); axins.tick_params(labelsize=6)
axins.set_ylabel("优势(pp)", fontsize=6.5)
axins.set_xlabel("名义 K 膨胀", fontsize=6.5)
axins.set_title("负控制:名义膨胀无优势", fontsize=7, color="#444")

ax.set_xlabel("有效组合复杂度(小 → 大)")
ax.set_ylabel("引导搜索相对随机的优势(%)")
ax.set_xticks([]); ax.set_ylim(-9, 34)
ax.set_title("引导搜索的优势取决于「有效组合复杂度」,而非名义空间大小\n"
             "(两端点=我们实测;负控制=我们受控实验;曲线=文献综合)", fontsize=10)
ax.legend(loc="upper left", fontsize=8)
plt.tight_layout(); save(fig, "复杂度规律")
print("图存 pub/复杂度规律")
