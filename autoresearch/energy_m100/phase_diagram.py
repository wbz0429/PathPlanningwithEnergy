"""
phase_diagram.py — 表征"能量感知规划何时相对最短路有价值"。
扫 (墙宽 × 巡航速度),画省能% 相图 + 标出 M100 从'绕行'翻转到'翻越'的边界。
把单点结果升级为操作包络表征:窄墙/高速 → 省得多;宽墙 → 连M100也翻墙 → 省能归零。
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import physics_eval as pe
from planning_experiment import M100Em, DistanceEm, score_path
from wall_experiment import clean_wall_map

HERE = os.path.dirname(os.path.abspath(__file__))
WIDTHS = [12, 16, 20, 24, 28, 32]
VELS = [4, 8, 12]


def run():
    src = open(os.path.join(HERE, "state", "best_featurize.py")).read()
    m100 = M100Em(src, payload=250.); dist = DistanceEm()
    s = np.array([10., 0., -3.]); g = np.array([72., 0., -3.])
    maps = {hy: clean_wall_map(float(hy), 12.) for hy in WIDTHS}
    save = np.zeros((len(VELS), len(WIDTHS)))
    m100_around = np.zeros_like(save, dtype=bool)
    for i, V in enumerate(VELS):
        for j, hy in enumerate(WIDTHS):
            vg, esdf, _ = maps[hy]
            res = {}
            for nm, em in [("d", dist), ("m", m100)]:
                p, _ = pe.energy_astar(vg, esdf, em, s, g, velocity=float(V),
                                       safety_margin=0.6, max_expand=1500000)
                P = np.array([np.asarray(w, float).reshape(-1)[:3] for w in p])
                res[nm] = (p, -P[:, 2].min(), np.abs(P[:, 1]).max())
            sv = 100 * (score_path(res["d"][0], m100, float(V)) -
                        score_path(res["m"][0], m100, float(V))) / score_path(res["d"][0], m100, float(V))
            save[i, j] = sv
            m100_around[i, j] = res["m"][2] > res["m"][1] + 2   # M100 绕行?
            print(f"v={V} 半宽{hy}: 省{sv:+.1f}%  M100={'绕行' if m100_around[i,j] else '翻越'}")
    json.dump({"widths": WIDTHS, "vels": VELS, "save": save.tolist(),
               "m100_around": m100_around.tolist()},
              open(os.path.join(HERE, "experiments", "phase_diagram.json"), "w"), indent=2)
    _plot(save, m100_around)


def _plot(save, m100_around):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "Heiti TC"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(figsize=(8, 4))
    im = ax.imshow(save, aspect="auto", origin="lower", cmap="YlGn",
                   extent=[WIDTHS[0]-2, WIDTHS[-1]+2, VELS[0]-2, VELS[-1]+2], vmin=0)
    for i, V in enumerate(VELS):
        for j, hy in enumerate(WIDTHS):
            txt = f"{save[i,j]:.0f}%\n{'绕' if m100_around[i,j] else '翻'}"
            ax.text(hy, V, txt, ha="center", va="center", fontsize=8)
    ax.set_xlabel("墙半宽 (m) — 越宽绕行越远"); ax.set_ylabel("巡航速度 (m/s)")
    ax.set_title("能量感知规划省能% 相图(vs 最短路):窄墙/高速→绕行省能;宽墙→连M100也翻墙")
    fig.colorbar(im, label="省能% (M100尺)")
    plt.tight_layout()
    out = os.path.join(HERE, "experiments", "phase_diagram.png")
    plt.savefig(out, dpi=120); print(f"相图存 {out}")


if __name__ == "__main__":
    run()
