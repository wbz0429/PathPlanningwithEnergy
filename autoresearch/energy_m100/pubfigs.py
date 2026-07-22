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
    ax.text(esc + 0.08, -1.1, f"{esc:.2f}(外搜逃逸)", va="center", fontsize=9, fontweight="bold", color=PAL["green"])
    ax.axvline(esc, color=PAL["green"], ls=(0, (4, 3)), lw=1)
    ax.set_yticks(list(y) + [-1.1]); ax.set_yticklabels(names + ["★ 物理核+线性payload"], fontsize=8.5)
    ax.set_xlabel("留出能量 ARE(%)")
    # 盆地标注放在柱区左下的空白处(2.0~3.2 之间无柱),避开数值标签
    ax.text(2.55, 1.5, "局部最优盆地\n(7 变体全卡 ~4.3%)", fontsize=8.5, color=PAL["red"], ha="center",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=PAL["red"], lw=0.6, alpha=0.9))
    ax.set_xlim(0, 5.2)
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


def phase_diagram():
    d = J("phase_diagram.json"); W, Vs, sv = d["widths"], d["vels"], np.array(d["save"]); ar = np.array(d["m100_around"])
    fig, ax = plt.subplots(figsize=(4.6, 2.7))
    im = ax.imshow(sv, aspect="auto", origin="lower", cmap="YlGn", vmin=0,
                   extent=[W[0]-2, W[-1]+2, Vs[0]-2, Vs[-1]+2])
    for i, v in enumerate(Vs):
        for j, hy in enumerate(W):
            ax.text(hy, v, f"{sv[i,j]:.0f}", ha="center", va="center", fontsize=8,
                    color="black" if sv[i,j] < 12 else "white")
    ax.set_xlabel("墙半宽(m)"); ax.set_ylabel("巡航速度(m/s)")
    cb = fig.colorbar(im, ax=ax, pad=.02); cb.set_label("省能(%)", fontsize=10); cb.outline.set_visible(False)
    save(fig, "操作包络相图")


def px4_ab():
    M = np.load(os.path.join(HERE, "px4_integration", "px4_traj.npz"))["pos"]
    D = np.load(os.path.join(HERE, "px4_integration", "px4_traj_dist.npz"))["pos"]
    ab = J(os.path.join("..", "px4_integration", "px4_ab_comparison.json")) if os.path.exists(os.path.join(EXP,"..","px4_integration","px4_ab_comparison.json")) else json.load(open(os.path.join(HERE,"px4_integration","px4_ab_comparison.json")))
    from matplotlib.patches import Rectangle
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.6, 2.7), gridspec_kw={"width_ratios": [2, 1]})
    a1.add_patch(Rectangle((22, 0), 3, 5, color=PAL["gray"], alpha=.4))
    a1.add_patch(Rectangle((45, 0), 3, 12, color=PAL["gray"], alpha=.4))
    a1.plot(D[:, 0], -D[:, 2], color=PAL["red"], lw=1.6, label="距离(翻越)")
    a1.plot(M[:, 0], -M[:, 2], color=PAL["green"], lw=1.6, label="M100(绕行)")
    a1.set_xlabel("x 北(m)"); a1.set_ylabel("高度(m)"); a1.legend(fontsize=8.5, loc="upper right")
    E = [ab["distance"]["E"], ab["m100"]["E"]]
    a2.bar(["距离\n翻越", "M100\n绕行"], E, color=[PAL["red"], PAL["green"]], width=.6)
    for i, e in enumerate(E): a2.text(i, e, f"{e}", ha="center", va="bottom", fontsize=9)
    a2.set_ylabel("M100 能耗(J)"); a2.set_ylim(0, max(E)*1.16)
    a2.text(.5, .5, f"省 {ab['saving_pct']}%", transform=a2.transAxes, ha="center", fontsize=11, fontweight="bold", color=PAL["navy"])
    save(fig, "PX4真飞控_AB对比")


def corridor():
    T = np.load(os.path.join(EXP, "corridor_trajs.npz"))
    from matplotlib.patches import Rectangle
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.8, 2.7))
    for ax in (a1, a2):
        pass
    a1.add_patch(Rectangle((22, 0), 3, 5, color=PAL["gray"], alpha=.4)); a1.text(23.5, 5.3, "楼A 5m", fontsize=7.5, ha="center")
    a1.add_patch(Rectangle((45, 0), 3, 12, color=PAL["gray"], alpha=.4)); a1.text(46.5, 12.3, "楼B 12m", fontsize=7.5, ha="center")
    a2.add_patch(Rectangle((22, -26), 3, 52, color=PAL["gray"], alpha=.3)); a2.add_patch(Rectangle((45, -10), 3, 20, color=PAL["gray"], alpha=.4))
    for base, c, lb in [("距离最短", PAL["red"], "距离"), ("真机M100", PAL["green"], "M100")]:
        X = T[base + "_x"]
        a1.plot(X[:, 0], X[:, 2], color=c, lw=1.6, label=lb); a2.plot(X[:, 0], X[:, 1], color=c, lw=1.6)
    a1.set_xlabel("x 北(m)"); a1.set_ylabel("高度(m)"); a1.set_title("侧视", fontsize=10); a1.legend(fontsize=8.5)
    a2.set_xlabel("x 北(m)"); a2.set_ylabel("y 东(m)"); a2.set_title("俯视", fontsize=10)
    save(fig, "城市走廊_混合决策")


def crossdomain_sig():
    e = J("significance_test.json")["energy_domain"]; p = J("planner_significance.json")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.6, 2.6))
    # 能耗域
    m, sd, lp = e["random_mean%"], e["random_std%"], e["loop_ARE%"]
    a1.axvspan(m-1.96*sd, m+1.96*sd, color=PAL["blue"], alpha=.12)
    a1.axvline(m, color=PAL["gray"], ls=(0, (3, 3)), lw=1)
    a1.axvline(lp, color=PAL["green"], lw=2)
    # 文字挪到 loop 线左侧、避免被竖线穿过,加白底
    a1.text(lp-0.006, .78, f"loop {lp}%\nz={e['loop_z_score']}", transform=a1.get_xaxis_transform(), fontsize=8.5,
            ha="right", color=PAL["green"], bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85))
    a1.set_xlim(m-3.2*sd, m+3.2*sd); a1.set_yticks([]); a1.set_xlabel("留出 ARE(%)")
    a1.set_title("能耗域(封顶)→ 打平", fontsize=10)
    # 规划器域
    m2, sd2, lp2 = p["random_mean"], p["random_std"], p["loop"]
    a2.axvspan(m2-1.96*sd2, m2+1.96*sd2, color=PAL["blue"], alpha=.12)
    a2.axvline(m2, color=PAL["gray"], ls=(0, (3, 3)), lw=1)
    a2.axvline(lp2, color=PAL["green"], lw=2)
    a2.text(lp2+12, .78, f"loop {lp2:.0f}\nz={p['loop_z']}", transform=a2.get_xaxis_transform(), fontsize=8.5,
            ha="left", color=PAL["green"], bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85))
    a2.set_xlim(min(lp2, m2-3.2*sd2)-40, m2+3.2*sd2); a2.set_yticks([]); a2.set_xlabel("能耗 score")
    a2.set_title("规划器域(有改进空间)→ 显著胜", fontsize=10)
    for a in (a1, a2): a.spines["left"].set_visible(False)
    save(fig, "两域显著性")


def hero():
    import sys; sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
    import physics_eval as pe
    from planning_experiment import M100Em, DistanceEm
    from wall_experiment import clean_wall_map
    from matplotlib.patches import Rectangle
    src = open(os.path.join(HERE, "state", "best_featurize.py")).read()
    m100 = M100Em(src, payload=250.); dist = DistanceEm()
    vg, esdf, bemt = clean_wall_map(16., 12.); s = np.array([10., 0., -3.]); g = np.array([72., 0., -3.])
    P = {}
    for nm, em in [("距离", dist), ("BEMT", bemt), ("M100", m100)]:
        p, _ = pe.energy_astar(vg, esdf, em, s, g, velocity=8., safety_margin=0.6, max_expand=1500000)
        P[nm] = np.array([np.asarray(w, float).reshape(-1)[:3] for w in p])
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.8, 2.7))
    cols = {"距离": PAL["red"], "BEMT": PAL["orange"], "M100": PAL["green"]}
    a1.add_patch(Rectangle((39, 0), 3, 12, color=PAL["gray"], alpha=.4))
    a2.add_patch(Rectangle((39, -16), 3, 32, color=PAL["gray"], alpha=.3))
    for nm, X in P.items():
        a1.plot(X[:, 0], -X[:, 2], color=cols[nm], lw=1.5, label=nm); a2.plot(X[:, 0], X[:, 1], color=cols[nm], lw=1.5)
    a1.set_xlabel("x(m)"); a1.set_ylabel("高度(m)"); a1.set_title("侧视:距离/BEMT 翻墙,M100 绕行", fontsize=9.5); a1.legend(fontsize=8.5)
    a2.set_xlabel("x(m)"); a2.set_ylabel("y(m)"); a2.set_title("俯视:M100 横向绕行", fontsize=9.5)
    save(fig, "真机代价改变决策")


def main():
    print("重绘顶会级中文图 → experiments/pub/")
    noise_floor(); framework_ablation(); savings_dist(); tradeoff()
    phase_diagram(); px4_ab(); corridor(); crossdomain_sig(); hero()
    print("完成。")


if __name__ == "__main__":
    main()
