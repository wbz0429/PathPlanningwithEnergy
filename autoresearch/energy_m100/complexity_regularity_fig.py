# -*- coding: utf-8 -*-
"""complexity_regularity_fig.py — "引导搜索优势 vs 搜索空间复杂度"规律图。
规律曲线来自文献(Bergstra2012/REMBO/NAS去混淆/FunSearch),诚实标注为文献综合;
我们两个真实端点(能耗域打平、规划器域胜27%)标在曲线上——这两个点是我们实测的。
输出 experiments/pub/复杂度规律.png
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib.pyplot as plt
from pubstyle import PAL, save

fig, ax = plt.subplots(figsize=(6.6, 3.7))
# 文献规律示意曲线(小空间~0优势,随复杂度上升;逻辑斯蒂形)
x = np.linspace(0, 10, 200)
y = 30 / (1 + np.exp(-(x - 5) * 0.9))    # 0→~30% 的上升曲线
ax.plot(x, y, color=PAL["gray"], lw=1.6, ls=(0, (5, 2)), label="规律(文献综合,示意)")
ax.fill_between(x, 0, y, color=PAL["gray"], alpha=.06)

# 文献边界标注点(示意位置)
lit = [(1.2, 1, "Bergstra&Bengio'12\n低有效维→随机追平"),
       (4.6, 8, "REMBO/贝叶斯\n~15-20维临界"),
       (8.6, 27, "FunSearch/AlphaEvolve\n程序空间>宇宙原子数")]
for lx, ly, t in lit:
    ax.scatter([lx], [ly], s=28, color=PAL["gray"], zorder=3)
    ax.annotate(t, (lx, ly), (lx-0.3, ly+3.5), fontsize=7, color="#666", ha="center")

# ★ 我们的两个真实端点(标注收进画布内,避免出框被裁)
ax.scatter([1.0], [0.3], s=150, color=PAL["red"], marker="o", zorder=5, edgecolors="white", linewidths=1.5)
ax.annotate("★我们·能耗域(实测)\n十几个线性项(小)→打平 z=−0.35", (1.0, 0.3), (3.2, -7.5),
            fontsize=8.5, color=PAL["red"], ha="center", fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=PAL["red"], lw=1))
ax.scatter([8.2], [27], s=150, color=PAL["green"], marker="o", zorder=5, edgecolors="white", linewidths=1.5)
ax.annotate("★我们·规划器域(实测)\n代码空间(大)→胜随机27% z=−2.38", (8.2, 27), (7.6, 17),
            fontsize=8.5, color=PAL["green"], ha="center", fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=PAL["green"], lw=1))

ax.set_xlabel("搜索空间复杂度(小 → 大)")
ax.set_ylabel("引导搜索相对随机的优势(%)")
ax.set_xticks([]); ax.set_ylim(-9, 34)
ax.set_title("引导搜索的优势随搜索空间复杂度增长\n(规律=文献综合;两端点=我们实测)", fontsize=10.5)
ax.legend(loc="upper left", fontsize=8.5)
plt.tight_layout(); save(fig, "复杂度规律")
print("图存 pub/复杂度规律")
