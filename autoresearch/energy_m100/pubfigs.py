# -*- coding: utf-8 -*-
"""pubfigs.py — 用 pubstyle 把核心图重绘成顶会级中文图(从落盘数据,不重跑实验)。输出 experiments/pub/。"""
import os, json
import numpy as np
import matplotlib.pyplot as plt
from pubstyle import PAL, save

HERE = os.path.dirname(os.path.abspath(__file__)); EXP = os.path.join(HERE, "experiments")
J = lambda n: json.load(open(os.path.join(EXP, n)))


def noise_floor():
    d = J("domain_value.json")["noise_floor"]["rows"]
    nf = [r["n_feat"] for r in d]; sr = [r["search%"] for r in d]; vl = [r["val%"] for r in d]
    floor = min(vl)
    fig, ax = plt.subplots(figsize=(3.5, 2.7))
    ax.plot(nf, sr, "o-", color=PAL["blue"], ms=4, lw=1.4, label="拟合(训练)")
    ax.plot(nf, vl, "s-", color=PAL["red"], ms=4, lw=1.4, label="留出")
    ax.axhline(floor, ls=(0, (4, 3)), color=PAL["green"], lw=1.2)
    ax.text(nf[-1], floor - 0.3, f"数据地板 ≈{floor:.1f}%", color=PAL["green"], fontsize=8.5, ha="right")
    ax.annotate("过拟合\n(52 项输给 6 项)", xy=(nf[-1], vl[-1]), xytext=(30, 3.5), fontsize=8, color=PAL["red"],
                ha="center", arrowprops=dict(arrowstyle="->", color=PAL["red"], lw=0.8))
    ax.set_xlabel("模型容量(特征项数)"); ax.set_ylabel("能量 ARE(%)")
    ax.set_ylim(0.8, 7.4); ax.legend(loc="upper right")
    save(fig, "噪声地板")


def framework_ablation():
    d = J("framework_ablation.json")["local_optimum"]
    phys = d["phys_basin"]; esc = d["escape_ARE"]
    names = list(phys.keys())[::-1]; vals = [phys[n] for n in names]
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    y = np.arange(len(names))
    ax.barh(y, vals, color=PAL["red"], alpha=.85, height=.62)
    ax.barh([-1.1], [esc], color=PAL["green"], height=.62)
    for i, v in enumerate(vals):
        ax.text(v + 0.06, y[i], f"{v:.2f}", va="center", fontsize=8)
    ax.text(esc + 0.06, -1.1, f"{esc:.2f}", va="center", fontsize=9, fontweight="bold", color=PAL["green"])
    ax.axvline(esc, color=PAL["green"], ls=(0, (4, 3)), lw=1)
    ax.set_yticks(list(y) + [-1.1]); ax.set_yticklabels(names + ["★ 物理核+线性payload"], fontsize=8.5)
    ax.set_xlabel("留出能量 ARE(%)")
    ax.text(np.mean(vals), len(names) - 1.5, "局部最优盆地\n(7 变体全卡)", fontsize=9, color=PAL["red"], ha="center")
    ax.set_xlim(0, 5)
    save(fig, "框架消融_局部最优")


def savings_dist():
    rows = [json.loads(l) for l in open(os.path.join(EXP, "large_scale_savings.jsonl"))]
    sav = np.array([r["sav_pct"] for r in rows if "sav_pct" in r])
    fig, ax = plt.subplots(figsize=(4.4, 2.7))
    ax.hist(sav, bins=np.arange(-0.5, sav.max() + 1.5, 1), color=PAL["green"], alpha=.85, edgecolor="white", lw=.5)
    med = np.median(sav)
    ax.axvline(med, color=PAL["navy"], ls=(0, (4, 3)), lw=1.2)
    ge3 = 100 * (sav >= 3).mean()
    ax.text(0.55, 0.9, f"n={len(sav)}　中位 {med:.0f}%\n≥3% 占比 {ge3:.0f}%(CI[15.5,25.4]%)",
            transform=ax.transAxes, fontsize=9, va="top")
    ax.set_xlabel("能量感知规划省能(%,vs 最短距离)"); ax.set_ylabel("场景数")
    save(fig, "省能分布_n250")


def tradeoff():
    d = J("tradeoff.json")
    w = np.array(d["widths"]); Ea = np.array(d["E_around"]); Eo = d["E_over"]; Ef = d["E_over_flat"]; cx = d["crossover_halfwidth_m"]
    fig, ax = plt.subplots(figsize=(4.6, 2.9))
    ax.axhline(Eo, color=PAL["red"], lw=1.8, label="翻越(固定爬升罚)")
    ax.fill_between(w, Ef, Eo, color=PAL["red"], alpha=.10)
    ax.plot(w, Ea, "o-", color=PAL["green"], ms=3.5, lw=1.6, label="绕行(随墙宽↑)")
    ax.axvline(cx, color=PAL["gray"], ls=(0, (4, 3)), lw=1)
    ax.text(cx + .3, Ea.min() + 80, f"临界≈{cx:.0f}m", fontsize=9, color=PAL["gray"])
    ax.text(w[0] + .3, (Eo + Ef) / 2, f"爬升罚≈{Eo-Ef:.0f}J", fontsize=8.5, color=PAL["red"], va="center")
    ax.set_xlabel("墙半宽(m)"); ax.set_ylabel("M100 能耗(J)"); ax.legend(loc="lower right", fontsize=9)
    save(fig, "翻越绕行权衡")


def main():
    print("重绘顶会级中文图 → experiments/pub/")
    noise_floor(); framework_ablation(); savings_dist(); tradeoff()
    print("完成。")


if __name__ == "__main__":
    main()
