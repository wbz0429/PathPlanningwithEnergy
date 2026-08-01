# -*- coding: utf-8 -*-
"""protocol_overview_fig.py — 论文 Fig 1(teaser):可信自动科研协议四阶段 + 每阶段实测数字。
一图看懂贡献:知识库 → 零拟合迁移(全失败)→ refit 排行(选最优)→ 防作弊 loop → 能力边界 + 诊断。
输出 experiments/pub/协议总览_Fig1.png
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib.pyplot as plt
from pubstyle import PAL, save
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

HERE = os.path.dirname(os.path.abspath(__file__)); EXP = os.path.join(HERE, "experiments")
J = lambda n: json.load(open(os.path.join(EXP, n)))

# 取关键数字
zoo = {r["code"]: r["val_ARE_percent"] for r in J("model_zoo_leaderboard.json")["results"] if "error" not in r}
ap = J("as_published_benchmark.json")["results"]
ap_range = f"{min(r['energy_ARE_pct'] for r in ap):.0f}–{max(r['energy_ARE_pct'] for r in ap):.0f}"
eg = J("significance_test.json")["energy_domain"]; pl = J("planner_significance.json")

fig, ax = plt.subplots(figsize=(11, 4.2))
ax.set_xlim(0, 11); ax.set_ylim(0, 4.2); ax.axis("off")


def box(x, y, w, h, title, body, color, tsize=9, bsize=7.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08",
                                fc="white", ec=color, lw=1.6))
    ax.text(x + w/2, y + h - 0.32, title, ha="center", fontsize=tsize, fontweight="bold", color=color)
    ax.text(x + w/2, y + h - 0.72, body, ha="center", va="top", fontsize=bsize, color="#222", linespacing=1.35)


def arrow(x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", color="#888", lw=1.6,
                                 mutation_scale=16))


# Stage 1: 知识库
box(0.3, 2.4, 2.3, 1.6, "① 文献知识库", "14 个已发表\n无人机能耗模型\n(物理/数据/混合)", PAL["gray"])
# Stage 2: 零拟合迁移
box(3.1, 2.4, 2.3, 1.6, "② 零拟合迁移", f"直接套用全失败\n能量 ARE {ap_range}%\n→ 必须 refit", PAL["red"])
# Stage 3: refit 排行 → 起点
box(5.9, 2.4, 2.3, 1.6, "③ refit 排行", f"数据驱动最优\nTseng 1.90%\n物理模型 ~7%", PAL["blue"])
# Stage 4: 防作弊 loop
box(8.7, 2.4, 2.1, 1.6, "④ 防作弊 loop", "冻结评测器\nkeep/revert\n诚实报负结果", PAL["green"])
arrow(2.6, 3.2, 3.05, 3.2); arrow(5.4, 3.2, 5.85, 3.2); arrow(8.2, 3.2, 8.65, 3.2)

# 底部:能力边界 + 诊断
box(0.3, 0.3, 5.0, 1.7, "⑤ 能力边界(实测)",
    f"能耗域打平 z={eg['loop_z_score']}(9 档 K 全预测 tie)\n"
    f"规划器域胜 {pl['win_pct']}% z={pl['loop_z']}(跨空间代码逃逸)",
    PAL["purple"])
box(5.9, 0.3, 4.9, 1.7, "头部空间诊断(Algorithm 1)",
    "跑 loop 前预测引导是否胜随机\n饱和空间→跳过(省预算);可逃逸空间→投入(+8.1%)",
    PAL["orange"])
arrow(2.6, 2.4, 2.6, 2.02)   # 起点 → 边界
arrow(7.0, 2.4, 6.3, 2.02)   # refit → 诊断

# 底部链接线
ax.add_patch(FancyArrowPatch((5.35, 0.3+0.85), (5.85, 0.3+0.85), arrowstyle="-|>", color="#888", lw=1.6, mutation_scale=16))

ax.set_title("可信自动科研协议:知识库 → 零拟合证明 refit 必需 → refit 选起点 → 防作弊 loop → 能力边界 + 头部空间诊断",
             fontsize=11, fontweight="bold", pad=8)
save(fig, "协议总览_Fig1")
print("图存 pub/协议总览_Fig1")
