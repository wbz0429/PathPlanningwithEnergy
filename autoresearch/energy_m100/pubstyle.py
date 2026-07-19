# -*- coding: utf-8 -*-
"""pubstyle.py — 顶会级中文图风格(CVPR/ICML 排版规范)。import 即生效。
特征:CJK 无衬线 + despine(去上/右边框)+ 色盲友好配色 + 浅网格 + 矢量 PDF + 300dpi。
用法:from pubstyle import PAL, save;  plt.subplots... ;  save(fig, "name")
"""
import os, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

# 选一个可用的干净中文无衬线
for _f in ("PingFang HK", "PingFang SC", "Heiti TC", "Hiragino Sans GB", "Arial Unicode MS"):
    if any(_f in x.name for x in font_manager.fontManager.ttflist):
        _CJK = _f; break
else:
    _CJK = "Arial Unicode MS"

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": [_CJK, "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 10.5, "axes.labelsize": 12, "axes.titlesize": 12, "legend.fontsize": 10,
    "xtick.labelsize": 10, "ytick.labelsize": 10, "axes.unicode_minus": False,
    "axes.linewidth": 0.9, "xtick.major.width": 0.9, "ytick.major.width": 0.9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.22, "grid.linewidth": 0.6,
    "legend.frameon": False, "figure.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.03,
})

# 色盲友好调色板(Seaborn deep 系)
PAL = {"blue": "#4C72B0", "red": "#C44E52", "green": "#55A868", "purple": "#8172B3",
       "orange": "#DD8452", "gray": "#8C8C8C", "teal": "#028090", "navy": "#1E2961"}

_OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "experiments", "pub")
os.makedirs(_OUT, exist_ok=True)


def save(fig, name):
    fig.savefig(os.path.join(_OUT, name + ".pdf"))
    fig.savefig(os.path.join(_OUT, name + ".png"), dpi=300)
    plt.close(fig)
    print(f"  → pub/{name}.pdf + .png")
