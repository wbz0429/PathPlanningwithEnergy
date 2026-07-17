# -*- coding: utf-8 -*-
"""loop_process.py — 展示 autoresearch loop 的“全过程”:迭代轨迹 + KEEP/REVERT + 瓶颈如何处理 + 文献检索触发。
这是我们真正的贡献(过程/方法学)的正脸。数据全部来自 agent_log.jsonl + KNOWLEDGE.md(真实记录)。
输出:experiments/loop_process.png + loop_process.json(瓶颈处理表)。
"""
import os, json
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "Heiti TC"]; plt.rcParams["axes.unicode_minus"] = False
HERE = os.path.dirname(os.path.abspath(__file__)); EXP = os.path.join(HERE, "experiments")

# 真实迭代轨迹(留出 ARE %,决定)——来自 agent_log.jsonl
TRAJ = [
    (0, 6.88, "base", "稳态BEMT基线(1,v,v²)"),
    (1, 4.25, "KEEP", "动量理论核 T^1.5+T²/V+v³+爬降"),
    (2, 4.91, "REVERT", "Glauert前飞桥→去掉主导项,更差"),
    (3, 4.24, "KEEP", "推力标定型面 T·v²(勉强)"),
    (4, 4.23, "REVERT", "轴向诱导sqrt→增益低于阈"),
    (5, 4.38, "REVERT", "风→空速→留出反而变差"),
    (6, 1.93, "KEEP", "物理核+裸线性payload(决定性)"),
    (7, 1.93, "REVERT", "payload变体→与iter6相同"),
    (8, 1.96, "KEEP*", "纯线性库≈物理+payload(诊断)"),
    (9, 1.96, "REVERT", "物理叠线性→共线,留出退"),
    (10, 1.86, "END", "capstone:loop≈随机(能力边界)"),
]

# 瓶颈 → 如何处理 → 结果(来自 KNOWLEDGE.md Episode 边界)
BOTTLENECKS = [
    {"瓶颈": "物理非线性形卡在 ~4.24%,加花哨物理形不降",
     "怎么处理": "Episode边界触发文献检索:查这个数据集的既定最优模型",
     "调研发现": "Tseng 多项式回归(数据集论文自带)是前沿;缺口可能是 wind + 多项式垂直项,非更花哨的物理",
     "结果": "改变方向:不再堆物理非线性,转去干净测 wind"},
    {"瓶颈": "wind 到底有没有用?(数据集专装风速计)",
     "怎么处理": "EXPLORE:空速 v_air=√(v_h²+wind²) 替换地速,单独测",
     "调研发现": "per-sample R² 微升但留出能量 ARE 4.24→4.38 反而变差(quadrature 风向抵消)",
     "结果": "iter5 REVERT:对能量指标 wind quadrature 耦合无益,记入知识库不再试"},
    {"瓶颈": "物理形式到底赢不赢纯线性回归?",
     "怎么处理": "决定性诊断:纯线性库 LIB vs 物理核+payload,同评测器对比",
     "调研发现": "纯线性 1.909% ≈ 物理+payload 1.929%;物理叠线性反而退(共线过拟合)",
     "结果": "核心 finding:物理先验不转化为更低 ARE(能力边界·创新点3)"},
    {"瓶颈": "loop 相对简单基线到底有没有必要?",
     "怎么处理": "生死线:random_subset / greedy 同预算对比 + R=40 显著性",
     "调研发现": "loop 1.86% 与随机 1.87±0.03 打平(z=−0.35);只略胜贪心",
     "结果": "诚实报告:能耗域结构搜索≈随机(域相关能力边界)"},
]


def main():
    fig = plt.figure(figsize=(15, 9))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.15, 1], hspace=0.32)

    # ---- 上:迭代轨迹 ----
    ax = fig.add_subplot(gs[0])
    xs = [t[0] for t in TRAJ]; ys = [t[1] for t in TRAJ]
    ax.plot(xs, ys, "-", color="#888", lw=1.5, zorder=1)
    for it, are, dec, note in TRAJ:
        if dec == "KEEP":
            ax.scatter(it, are, s=130, color="#27ae60", zorder=3, marker="o", edgecolors="white", linewidths=1.5)
        elif dec.startswith("KEEP"):
            ax.scatter(it, are, s=130, color="#27ae60", zorder=3, marker="o", alpha=.5, edgecolors="white", linewidths=1.5)
        elif dec == "REVERT":
            ax.scatter(it, are, s=150, color="#c0392b", zorder=3, marker="X", edgecolors="white", linewidths=1)
        else:
            ax.scatter(it, are, s=150, color="#2c3e50", zorder=3, marker="s", edgecolors="white", linewidths=1)
    # 关键事件标注
    ax.annotate("① 动量理论核\n6.88→4.25", (1, 4.25), (1.1, 5.6), fontsize=8.5, color="#27ae60",
                arrowprops=dict(arrowstyle="->", color="#27ae60"))
    ax.annotate("卡在~4.24%\n→文献检索:Tseng多项式=前沿", (3, 4.24), (2.4, 3.0), fontsize=8.5, color="#c47f1a",
                arrowprops=dict(arrowstyle="->", color="#c47f1a"))
    ax.annotate("wind测试→REVERT\n(quadrature无益)", (5, 4.38), (5.1, 5.5), fontsize=8.5, color="#c0392b",
                arrowprops=dict(arrowstyle="->", color="#c0392b"))
    ax.annotate("② 物理核+裸payload\n决定性 4.24→1.93", (6, 1.93), (6.2, 3.2), fontsize=9, color="#27ae60",
                arrowprops=dict(arrowstyle="->", color="#27ae60"), fontweight="bold")
    ax.annotate("③ 纯线性≈物理\n物理不赢回归(能力边界)", (8, 1.96), (8.0, 3.0), fontsize=8.5, color="#2c3e50",
                arrowprops=dict(arrowstyle="->", color="#2c3e50"))
    ax.annotate("loop≈随机\nz=−0.35", (10, 1.86), (9.3, 0.9), fontsize=8.5, color="#2c3e50",
                arrowprops=dict(arrowstyle="->", color="#2c3e50"))
    ax.set_xlabel("迭代 iteration"); ax.set_ylabel("留出能量 ARE %(越低越好)")
    ax.set_title("autoresearch loop 全过程:每轮 KEEP/REVERT + 瓶颈如何被处理(真实 agent_log)", fontsize=13, fontweight="bold")
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([0],[0],marker="o",color="w",markerfacecolor="#27ae60",markersize=11,label="KEEP(接受)"),
                       Line2D([0],[0],marker="X",color="w",markerfacecolor="#c0392b",markersize=11,label="REVERT(回滚,留判据)"),
                       Line2D([0],[0],marker="s",color="w",markerfacecolor="#2c3e50",markersize=10,label="里程碑/终点")],
              fontsize=9, loc="upper right")
    ax.set_ylim(0.5, 7.4); ax.grid(alpha=.25)

    # ---- 下:loop 机制流程图 ----
    ax2 = fig.add_subplot(gs[1]); ax2.axis("off"); ax2.set_xlim(0, 10); ax2.set_ylim(0, 3)
    boxes = [(0.5, "读状态\n(冻结评测器/知识库)"), (2.4, "提假设\n改一次featurize"),
             (4.3, "沙箱评测\nsearch+留出ARE"), (6.2, "keep / revert\n留出退则回滚"),
             (8.1, "记账+提交\nagent_log/git")]
    for x, txt in boxes:
        b = FancyBboxPatch((x, 1.6), 1.5, 0.9, boxstyle="round,pad=0.05", fc="#eaf4f7", ec="#028090", lw=1.5)
        ax2.add_patch(b); ax2.text(x+0.75, 2.05, txt, ha="center", va="center", fontsize=8.5)
    for i in range(len(boxes)-1):
        ax2.add_patch(FancyArrowPatch((boxes[i][0]+1.5, 2.05), (boxes[i+1][0], 2.05),
                     arrowstyle="->", mutation_scale=15, color="#028090"))
    # 回环 + episode 边界分支
    ax2.add_patch(FancyArrowPatch((8.85, 1.6), (3.15, 1.6), arrowstyle="->", mutation_scale=15,
                 color="#028090", connectionstyle="arc3,rad=0.25"))
    ax2.text(5.5, 0.75, "循环", fontsize=9, color="#028090", ha="center")
    # episode 边界 → 文献检索
    eb = FancyBboxPatch((3.3, 0.05), 3.4, 0.7, boxstyle="round,pad=0.05", fc="#fef3e6", ec="#c47f1a", lw=1.5)
    ax2.add_patch(eb); ax2.text(5.0, 0.4, "★ Episode边界 / 撞瓶颈 → 触发文献检索(deep-research)\n把带来源的洞见写回知识库,改变下一轮方向",
                                ha="center", va="center", fontsize=8.5, color="#8a5a12")
    ax2.add_patch(FancyArrowPatch((3.15, 1.55), (4.3, 0.78), arrowstyle="->", mutation_scale=13, color="#c47f1a"))
    ax2.set_title("loop 机制:读→提→评→keep/revert→记账,循环;撞瓶颈则触发文献检索改向", fontsize=12, fontweight="bold", y=0.98)

    plt.savefig(os.path.join(EXP, "loop_process.png"), dpi=125, bbox_inches="tight")
    print("图存 experiments/loop_process.png")
    json.dump({"trajectory": [{"iter": t[0], "val_ARE": t[1], "decision": t[2], "note": t[3]} for t in TRAJ],
               "bottlenecks": BOTTLENECKS}, open(os.path.join(EXP, "loop_process.json"), "w"), ensure_ascii=False, indent=2)
    print("瓶颈处理表存 experiments/loop_process.json")
    print("\n=== 瓶颈 → 处理 → 结果 ===")
    for b in BOTTLENECKS:
        print(f"· {b['瓶颈']}\n   处理:{b['怎么处理']}\n   发现:{b['调研发现']}\n   结果:{b['结果']}\n")


if __name__ == "__main__":
    main()
