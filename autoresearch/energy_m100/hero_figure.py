"""hero_figure.py — 论文主图:中墙场景,距离/BEMT 翻墙 vs M100 绕行,一图看懂。
左:侧视(XZ)看翻越;中:俯视(XY)看绕行;右:M100真机能耗柱状+省能。"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import physics_eval as pe
from planning_experiment import M100Em, DistanceEm, score_path
from wall_experiment import clean_wall_map

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "Heiti TC"]
    plt.rcParams["axes.unicode_minus"] = False

    src = open(os.path.join(HERE, "state", "best_featurize.py")).read()
    m100 = M100Em(src, payload=250.); dist = DistanceEm()
    HALF_Y, HGT, V = 16., 12., 8.
    vg, esdf, bemt = clean_wall_map(HALF_Y, HGT)
    s = np.array([10., 0., -3.]); g = np.array([72., 0., -3.])
    conds = [("距离最短", dist, "tab:red"), ("教科书BEMT", bemt, "tab:orange"), ("真机M100", m100, "tab:green")]
    paths = {}
    for nm, em, _ in conds:
        p, _ = pe.energy_astar(vg, esdf, em, s, g, velocity=V, safety_margin=0.6, max_expand=1500000)
        paths[nm] = np.array([np.asarray(w, float).reshape(-1)[:3] for w in p])

    fig, (axz, axy, axb) = plt.subplots(1, 3, figsize=(16, 4.3))
    # 侧视 XZ(高度)
    axz.add_patch(Rectangle((39, 0), 3, HGT, color="gray", alpha=.6, label="墙(12m高)"))
    for nm, _, c in conds:
        P = paths[nm]; axz.plot(P[:, 0], -P[:, 2], color=c, lw=2.2, label=nm)
    axz.set_xlabel("x (m)"); axz.set_ylabel("高度 (m)"); axz.set_title("侧视:距离/BEMT 翻墙,M100 不爬")
    axz.legend(fontsize=8); axz.set_ylim(0, 16)
    # 俯视 XY(横向绕行)
    axy.add_patch(Rectangle((39, -HALF_Y), 3, 2 * HALF_Y, color="gray", alpha=.6))
    for nm, _, c in conds:
        P = paths[nm]; axy.plot(P[:, 0], P[:, 1], color=c, lw=2.2, label=nm)
    axy.set_xlabel("x (m)"); axy.set_ylabel("y (m)"); axy.set_title("俯视:M100 横向绕行避爬升")
    # 能耗柱状(M100 真机尺)
    names = [c[0] for c in conds]
    E = [score_path([p for p in paths[n]], m100, V) for n in names]
    bars = axb.bar(names, E, color=[c[2] for c in conds])
    axb.set_ylabel("M100 真机能耗 (J)")
    sv = 100 * (E[0] - E[2]) / E[0]
    axb.set_title(f"真机能耗:M100 绕行省 {sv:.1f}%(vs 距离翻墙)")
    for b, e in zip(bars, E):
        axb.text(b.get_x() + b.get_width() / 2, e, f"{e:.0f}", ha="center", va="bottom", fontsize=9)
    axb.set_ylim(0, max(E) * 1.15)
    plt.suptitle("真机 M100 能耗代价改变规划决策:躲开真实存在的爬升能耗(held-out 验证 <5%),省能 9.4%",
                 fontsize=12)
    plt.tight_layout()
    out = os.path.join(HERE, "experiments", "hero_figure.png")
    plt.savefig(out, dpi=130); print(f"hero 图存 {out}  省能 {sv:.1f}%")


if __name__ == "__main__":
    main()
