# -*- coding: utf-8 -*-
"""arch_fig.py — 两张图:
  (1) 架构图 arch:AI4S 框架的节点+数据流(LLM提议/外搜/harness实验/冻结评测器/keep-revert/知识库),Transformer论文风的方块+箭头。
  (2) 突破叙事图 breakthrough:autoresearch 怎么突破阈值(6.88%→卡4.2%→外搜→1.9%),体现框架重要性。
输出 experiments/pub/ 矢量 PDF+PNG。
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from pubstyle import PAL, save

HERE = os.path.dirname(os.path.abspath(__file__)); EXP = os.path.join(HERE, "experiments")


def box(ax, x, y, w, h, txt, fc, ec, tc="white", fs=10, bold=True, sub=None):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06",
                                fc=fc, ec=ec, lw=1.6))
    ax.text(x+w/2, y+h/2+(0.055 if sub else 0), txt, ha="center", va="center",
            fontsize=fs, color=tc, fontweight="bold" if bold else "normal", zorder=5)
    if sub:
        ax.text(x+w/2, y+h/2-0.075, sub, ha="center", va="center", fontsize=fs-2.5, color=tc, alpha=.9, zorder=5)


def arrow(ax, p1, p2, color="#333", rad=0.0, lw=1.8, ls="-"):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=16, color=color,
                                 lw=lw, ls=ls, connectionstyle=f"arc3,rad={rad}", zorder=3))


def architecture():
    fig, ax = plt.subplots(figsize=(9.2, 5.2)); ax.set_xlim(0, 10); ax.set_ylim(0, 6.2); ax.axis("off")
    # 配色:LLM=紫, 外搜=橙, harness=蓝, 评测器=青(冻结), 决策=绿, 知识库=灰
    LLM, EXT, HAR, EVAL, DEC, KB = PAL["purple"], PAL["orange"], PAL["blue"], PAL["teal"], PAL["green"], PAL["gray"]
    # 主循环行(y=3.3)
    box(ax, 0.3, 3.1, 1.7, 1.0, "① LLM 提议", LLM, LLM, sub="改写 featurize 代码")
    box(ax, 2.5, 3.1, 1.7, 1.0, "② Harness\n跑实验", HAR, HAR, sub="沙箱执行候选")
    box(ax, 4.7, 3.1, 1.9, 1.0, "③ 冻结评测器", EVAL, EVAL, sub="真机功率·留出·不可改")
    box(ax, 7.1, 3.1, 1.9, 1.0, "④ keep / revert", DEC, DEC, sub="留出退则回滚")
    # 主流箭头
    arrow(ax, (2.0, 3.6), (2.5, 3.6)); arrow(ax, (4.2, 3.6), (4.7, 3.6)); arrow(ax, (6.6, 3.6), (7.1, 3.6))
    # 回环:decision → LLM(走上方大弧,不穿框)
    arrow(ax, (8.05, 4.1), (1.15, 4.1), color=DEC, rad=0.42, lw=1.7)
    ax.text(4.6, 5.55, "主循环:提议 → 实验 → 评测 → 接受/回滚", fontsize=9.5, color=DEC, ha="center", style="italic", fontweight="bold")
    # 知识库(底部,被读写虚线)
    box(ax, 3.15, 1.5, 3.0, 0.75, "⑤ 知识库 KNOWLEDGE", KB, KB, sub="洞见/失败判据/查新裁决 持久化")
    arrow(ax, (7.9, 3.1), (5.7, 2.25), color=KB, rad=0.15, lw=1.2, ls=(0,(3,2)))   # decision 写入
    arrow(ax, (3.4, 2.25), (1.1, 3.1), color=KB, rad=0.15, lw=1.2, ls=(0,(3,2)))   # 读回 LLM
    # 外搜(左上分支,撞瓶颈触发)
    box(ax, 0.3, 5.0, 3.0, 0.95, "★ 外搜 / 文献检索", EXT, EXT, sub="撞瓶颈→查新→改方向")
    arrow(ax, (2.9, 4.1), (1.9, 5.0), color=EXT, rad=-0.15, lw=1.6)   # 从 LLM 上去
    arrow(ax, (1.5, 5.0), (1.0, 4.1), color=EXT, rad=0.25, lw=1.6)    # 洞见回灌 LLM
    ax.text(4.9, 5.15, "触发:留出误差停滞 / “发现”待判新", fontsize=8, color=EXT, ha="left")
    # 输入/输出
    box(ax, 0.3, 0.3, 2.5, 0.75, "真机 M100 数据", "#eef4f7", EVAL, tc="#1a2333", fs=9, sub="209 航班 · P=V·|I|")
    arrow(ax, (2.8, 0.68), (5.4, 3.1), color=EVAL, rad=-0.12, lw=1.3)
    box(ax, 7.2, 0.3, 2.5, 0.75, "输出:模型 + 边界", "#f0faf4", DEC, tc="#1a2333", fs=9, sub="+ 诚实负结果")
    arrow(ax, (8.05, 3.1), (8.45, 1.05), color=DEC, rad=0.0, lw=1.3)
    ax.set_title("可信 AI4S 框架架构:五节点闭环 + 外搜逃逸支路", fontsize=12.5, fontweight="bold", pad=12)
    # 图例
    for i, (c, t) in enumerate([(LLM, "LLM 提议"), (HAR, "Harness 实验"), (EVAL, "冻结评测器"), (DEC, "keep/revert"), (EXT, "外搜"), (KB, "知识库")]):
        ax.add_patch(plt.Rectangle((0.3+i*1.6, 0.0), 0.28, 0.16, fc=c)); ax.text(0.64+i*1.6, 0.08, t, fontsize=7.5, va="center")
    save(fig, "框架架构图")


def breakthrough():
    fig, ax = plt.subplots(figsize=(9.2, 4.4)); ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, 5)
    # 阶梯式突破:每一关一个方块 + ARE 数字
    stages = [
        (0.3, "起点\n稳态BEMT", "6.88%", PAL["gray"], "1,v,v² 几乎解释不了功率"),
        (2.3, "① 加物理核\n(LLM提议)", "4.25%", PAL["purple"], "动量理论 T^1.5+v³+爬降"),
        (4.3, "⚠ 卡阈值\n7变体全卡", "~4.2%", PAL["red"], "只在物理形里搜→局部最优"),
        (6.3, "★ 外搜突破\n(文献检索)", "1.93%", PAL["orange"], "查到:缺线性payload非花哨物理"),
        (8.3, "收敛\n(诚实边界)", "1.86%", PAL["green"], "= 随机搜索(结构搜索无益)"),
    ]
    y0 = 2.2
    for i, (x, name, are, c, note) in enumerate(stages):
        h = 0.95
        box(ax, x, y0, 1.5, h, name, c, c, fs=9.5, sub=None)
        ax.text(x+0.75, y0-0.35, are, ha="center", fontsize=13, fontweight="bold", color=c)
        ax.text(x+0.75, y0+h+0.28, note, ha="center", fontsize=7.3, color="#555", wrap=True)
        if i < len(stages)-1:
            nx = stages[i+1][0]
            col = PAL["orange"] if i == 2 else "#666"
            arrow(ax, (x+1.5, y0+h/2), (nx, y0+h/2), color=col, lw=2.2 if i == 2 else 1.6)
    # 高亮"外搜突破"这一跳
    ax.annotate("", xy=(6.3, 1.5), xytext=(4.55, 1.5),
                arrowprops=dict(arrowstyle="-|>", color=PAL["orange"], lw=2.5))
    ax.text(5.4, 1.15, "外搜逃出局部最优\n(降 2.3 个百分点)", ha="center", fontsize=9, color=PAL["orange"], fontweight="bold")
    ax.text(5.0, 4.5, "autoresearch 如何突破阈值:局部搜索卡壳 → 外搜识别正确方向 → 一步突破",
            ha="center", fontsize=12.5, fontweight="bold")
    ax.text(5.0, 0.35, "关键:突破不来自“更花哨的模型”,而来自框架的外搜环节——这正是框架相对朴素调参/随机搜索的价值所在",
            ha="center", fontsize=8.5, color="#444", style="italic")
    save(fig, "突破阈值叙事")


if __name__ == "__main__":
    architecture()
    breakthrough()
