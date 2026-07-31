# -*- coding: utf-8 -*-
"""headroom_prediction_fig.py — 头部空间诊断「预测 vs 实测」图(论文 Fig 7)。
───────────────────────────────────────────────────────────────────────────
x = 设置(能耗域各 K 档 + 规划器域),y = 优势(pp)。
  · 散点 = 实测优势(能耗: complexity_scaling_fair 的 advantage_pp;
           规划: planner_significance 的 (random_mean−loop)/random_mean×100)。
  · 颜色/符号 = 诊断预测(tie=灰, guided-wins=绿)。
若所有"预测 tie"的点都靠近 0、所有"预测 guided-wins"的点都>0 → 诊断正确。
───────────────────────────────────────────────────────────────────────────
输出: experiments/pub/头部空间诊断_预测vs实测.png
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from pubstyle import PAL, save
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__)); EXP = os.path.join(HERE, "experiments")


def main():
    fair = json.load(open(os.path.join(EXP, "complexity_scaling_fair.json")))["rows"]
    diag = json.load(open(os.path.join(EXP, "headroom_diagnostic.json")))["rows"]
    pl = json.load(open(os.path.join(EXP, "planner_significance.json")))

    # 能耗域:每 K 档的实测优势 + 诊断预测
    points = []
    diag_by_K = {r["K"]: r["predicted"] for r in diag}
    for r in fair:
        points.append({"label": f"能耗 K={r['K']}", "adv": r["advantage_pp"],
                       "pred": diag_by_K.get(r["K"], "?"), "err": max(r["guided_std"], r["random_std"])})

    # 规划器域:实测优势 = (random_mean − loop)/random_mean×100;误差也换算成 pp
    adv_pl = 100 * (pl["random_mean"] - pl["loop"]) / pl["random_mean"]
    err_pl = 100 * pl["random_std"] / pl["random_mean"]      # 分数单位 → pp
    try:
        pl_diag = json.load(open(os.path.join(EXP, "planner_headroom.json")))
        pred_pl = pl_diag.get("predicted_in_space", "?")
    except Exception:
        pred_pl = "?"
    points.append({"label": "规划器域(跨空间)", "adv": adv_pl, "pred": pred_pl, "err": err_pl})

    fig, ax = plt.subplots(figsize=(7.0, 3.4))
    xs = np.arange(len(points))
    for i, p in enumerate(points):
        c = PAL["green"] if p["pred"] == "guided-wins" else PAL["gray"]
        ax.errorbar(i, p["adv"], yerr=p["err"], fmt="o", color=c, ms=8, capsize=3, lw=1.5)
        ax.text(i, p["adv"] + (p["err"] or 0.05) + 0.15, f"{p['adv']:+.2f}pp",
                ha="center", fontsize=8, fontweight="bold",
                color=PAL["green"] if p["adv"] > 0 else "#444")
    ax.axhline(0, color="#333", lw=1.2)
    ax.set_xticks(xs); ax.set_xticklabels([p["label"] for p in points], fontsize=8.5)
    ax.set_ylabel("引导相对随机优势 (pp)")
    ax.set_ylim(min(-2.5, min(p["adv"] for p in points) - 1), max(10, max(p["adv"] for p in points) + 2))
    ax.set_title("头部空间诊断:预测(颜色)vs 实测(位置)\n"
                 "灰=预测打平(应≈0),绿=预测引导胜(应>0);规划器域为跨空间逃逸", fontsize=10.5)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=PAL["gray"], label="诊断预测:打平"),
                       Patch(color=PAL["green"], label="诊断预测:引导胜")],
              fontsize=8.5, loc="upper right")
    fig.tight_layout()
    save(fig, "头部空间诊断_预测vs实测")
    print("图存 pub/头部空间诊断_预测vs实测")


if __name__ == "__main__":
    main()
