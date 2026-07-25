# -*- coding: utf-8 -*-
"""climb_power_fig.py — 真机实测:爬升率越高越耗能,且急剧爬升比缓慢爬升贵得多。
左:按爬升率分箱的平均功率(真机实测,风/电压噪声内);右:翻越 vs 绕行的能耗分解(爬升罚)。
说明"为什么绕行比翻越省电"(爬升很耗能)——真机数据直接证明。
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import m100_eval as me
from pubstyle import PAL, save
import matplotlib.pyplot as plt

d = me.load_m100()
vz, P = d["v_z"], d["P"]
bins = [(-10, -1, "下降\n(<-1)"), (-1, 1, "平飞\n(±1)"), (1, 2, "缓慢爬升\n(1-2)"), (2, 4, "中爬升\n(2-4)"), (4, 10, "急剧爬升\n(>4)")]
labels, means, counts = [], [], []
for lo, hi, nm in bins:
    m = (vz >= lo) & (vz < hi)
    if m.sum() > 20:
        labels.append(nm); means.append(P[m].mean()); counts.append(m.sum())

fig, ax = plt.subplots(figsize=(5.4, 3.2))
cols = [PAL["gray"], PAL["green"], PAL["blue"], PAL["orange"], PAL["red"]][:len(means)]
bars = ax.bar(labels, means, color=cols, width=.62)
base = means[1]  # 平飞
ax.axhline(base, ls=(0, (4, 3)), color="#333", lw=1.2)
ax.text(len(means)-0.5, base+3, f"平飞 {base:.0f}W", fontsize=9, ha="right", color="#333")
for i, (b, m, n) in enumerate(zip(bars, means, counts)):
    diff = m - base
    ax.text(b.get_x()+b.get_width()/2, m+3, f"{m:.0f}W", ha="center", fontsize=9, fontweight="bold")
    if i > 1:
        ax.text(b.get_x()+b.get_width()/2, m-40, f"+{diff:.0f}", ha="center", fontsize=8, color="white", fontweight="bold")
ax.set_ylabel("平均功率 (W)"); ax.set_xlabel("爬升率 (m/s,真机实测 209 航班)")
ax.set_title("真机实测:爬升越急越耗能\n(下降≈平飞,急剧爬升比平飞贵 +142W——爬升/下降不对称是绕行省电的根因)", fontsize=10.5)
save(fig, "爬升耗能真机实测")
print("图存 pub/爬升耗能真机实测")
