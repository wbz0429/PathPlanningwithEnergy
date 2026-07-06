"""
visualize_trajectory.py — 把规划器真实输出的三维轨迹画进场景(替代 AirSim 的可视化角色)

Renders the ACTUAL planned trajectories through the actual Blocks obstacle scene,
comparing A* / default RRT* / RRT-Connect on scenario A. Three views:
  3D  |  俯视(X-Y)  |  侧视(X-高度), 高度 = -z(NED),地面=0。
不需要 AirSim。场景与评测器 evaluator.py 完全一致(同一张地图、同一 A* 基线)。

用法: python visualize_trajectory.py
输出: experiments/fig_trajectory_sceneA.png
"""
import os
import sys
import random

os.environ.setdefault("MPLBACKEND", "Agg")
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

plt.rcParams["font.sans-serif"] = ["PingFang SC", "Heiti SC", "STHeiti",
                                   "Arial Unicode MS", "Hiragino Sans GB", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False

import evaluator as ev
import benchmark_planning as bp
from planning.config import PlanningConfig
from planning.rrt_star import RRTStar

START = np.array([0., 0., -3.])
GOAL = np.array([70., 0., -3.])


def _cfg(**kw):
    base = dict(ev.BASE)
    base.update(dict(step_size=1.5, max_iterations=5000, goal_sample_rate=0.4,
                     search_radius=4.0, dubins_turning_radius=1.5, weight_energy=0.6,
                     weight_distance=0.3, weight_time=0.1, planning_timeout=10.0,
                     energy_aware=True, flight_velocity=2.0))
    base.update(kw)
    return PlanningConfig(**base)


def _plan_all():
    vg, esdf, em = ev._get_map()
    out = {}
    # A*
    ap = bp.AStarPlanner(vg, esdf, _cfg())
    pa = ap.plan(START, GOAL)
    if pa:
        pa = bp.smooth_path(pa, esdf, 1.0)
    out["A* 最优"] = (pa, "green")
    # default RRT*
    random.seed(100); np.random.seed(100)
    pr = RRTStar(vg, esdf, _cfg(), energy_model=em).plan(START, GOAL)
    out["默认 RRT*"] = (pr, "orange")
    # RRT-Connect
    random.seed(100); np.random.seed(100)
    pc = RRTStar(vg, esdf, _cfg(use_rrt_connect=True), energy_model=em).plan(START, GOAL)
    out["RRT-Connect"] = (pc, "royalblue")
    return out


def _len_energy(path, em):
    if not path:
        return None, None
    L = bp.compute_path_length(path)
    E, _ = em.compute_energy_for_path(path, velocity=2.0)
    return L, E


def _box_faces(x0, x1, y0, y1, a0, a1):
    """返回长方体 6 个面的顶点(用于 3D Poly3DCollection),坐标 (x, y, 高度)。"""
    v = [(x0, y0, a0), (x1, y0, a0), (x1, y1, a0), (x0, y1, a0),
         (x0, y0, a1), (x1, y0, a1), (x1, y1, a1), (x0, y1, a1)]
    idx = [(0, 1, 2, 3), (4, 5, 6, 7), (0, 1, 5, 4),
           (2, 3, 7, 6), (1, 2, 6, 5), (0, 3, 7, 4)]
    return [[v[i] for i in f] for f in idx]


def main():
    vg, esdf, em = ev._get_map()
    paths = _plan_all()
    obs = bp.BLOCKS_OBSTACLES

    fig = plt.figure(figsize=(19, 6))
    ax3d = fig.add_subplot(1, 3, 1, projection="3d")
    axxy = fig.add_subplot(1, 3, 2)
    axxz = fig.add_subplot(1, 3, 3)

    # ---- 障碍物(墙),高度 = -z ----
    for o in obs:
        x0, x1 = o["x_range"]; y0, y1 = o["y_range"]
        a0, a1 = -o["z_range"][1], -o["z_range"][0]   # z(-10,0)->高度(0,10)
        for f in _box_faces(x0, x1, y0, y1, a0, a1):
            ax3d.add_collection3d(Poly3DCollection([f], alpha=0.12,
                                  facecolor="gray", edgecolor="none"))
        axxy.add_patch(Rectangle((x0, y0), x1-x0, y1-y0, alpha=0.35, color="gray"))
        axxz.add_patch(Rectangle((x0, a0), x1-x0, a1-a0, alpha=0.35, color="gray"))

    # ---- 三条真实轨迹 ----
    legend_txt = []
    for name, (path, color) in paths.items():
        L, E = _len_energy(path, em)
        tag = f"{name}: {L:.0f}m / {E:.0f}J" if path else f"{name}: 规划失败"
        legend_txt.append(tag)
        if not path:
            continue
        P = np.array(path)
        alt = -P[:, 2]
        ax3d.plot(P[:, 0], P[:, 1], alt, "-", color=color, lw=2.2, label=tag)
        axxy.plot(P[:, 0], P[:, 1], "-", color=color, lw=2.2, label=tag)
        axxz.plot(P[:, 0], alt, "-", color=color, lw=2.2, label=tag)

    # 起点/终点
    for ax, xy in [(axxy, (START[1], GOAL[1])), (axxz, (-START[2], -GOAL[2]))]:
        ax.scatter([START[0]], [xy[0]], c="black", marker="o", s=60, zorder=5)
        ax.scatter([GOAL[0]], [xy[1]], c="red", marker="*", s=160, zorder=5)
    ax3d.scatter([START[0]], [START[1]], [-START[2]], c="black", marker="o", s=50)
    ax3d.scatter([GOAL[0]], [GOAL[1]], [-GOAL[2]], c="red", marker="*", s=140)

    ax3d.set_title("(a) 3D 轨迹(灰=墙)")
    ax3d.set_xlabel("X (m)"); ax3d.set_ylabel("Y (m)"); ax3d.set_zlabel("高度 -z (m)")
    ax3d.view_init(elev=22, azim=-60)

    axxy.axhline(0, color="k", lw=0.5, ls=":")
    axxy.set_title("(b) 俯视 X-Y:谁横向绕行?")
    axxy.set_xlabel("X (m)"); axxy.set_ylabel("Y (m)")
    axxy.legend(fontsize=8, loc="upper left"); axxy.grid(alpha=0.3)

    axxz.axhline(0, color="saddlebrown", lw=1.5, label="地面 (高度0)")
    axxz.set_title("(c) 侧视 X-高度:谁在垂直机动?")
    axxz.set_xlabel("X (m)"); axxz.set_ylabel("高度 -z (m)")
    axxz.legend(fontsize=8, loc="upper right"); axxz.grid(alpha=0.3)

    fig.suptitle("场景A 真实规划轨迹穿越障碍场景(无需 AirSim) — 长度/能耗见图例",
                 fontsize=13, weight="bold")
    fig.tight_layout()
    out = os.path.join(_HERE, "experiments", "fig_trajectory_sceneA.png")
    fig.savefig(out, dpi=140); plt.close(fig)
    print("[saved]", out)
    for t in legend_txt:
        print("  ", t)


if __name__ == "__main__":
    main()
