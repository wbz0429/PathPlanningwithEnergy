"""
animate_flight.py — 优化后轨迹的 3D 飞越动画(无需 AirSim,pillow 出 GIF)

把 loop 优化出来的最优配置(RRT-Connect + 平滑器 + 速度剖面)在场景上规划的轨迹,
做成无人机沿路径飞的 3D 动画:墙 + 无人机点 + 拖尾(按高度上色),视角缓慢旋转。
用法: python animate_flight.py [scenario A|B|C]   输出 experiments/flight_<scn>.gif
"""
import os, sys, json, random
os.environ.setdefault("MPLBACKEND", "Agg")
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
plt.rcParams["font.sans-serif"] = ["PingFang SC", "Heiti SC", "Arial Unicode MS", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False

import physics_eval as pe, evaluator as ev, benchmark_planning as bp, candidate as cand
from planning.config import PlanningConfig
from planning.rrt_star import RRTStar

EXP = os.path.join(_HERE, "experiments")
TUNED = json.load(open(os.path.join(_HERE, "state", "best.json")))["config"]
SCNS = {"A": (np.array([0., 0., -3.]), np.array([70., 0., -3.]), "A 直穿(翻墙)"),
        "B": (np.array([0., 0., -3.]), np.array([70., 20., -3.]), "B 对角上"),
        "C": (np.array([0., 0., -3.]), np.array([70., -25., -3.]), "C 对角下")}


def _box_faces(x0, x1, y0, y1, a0, a1):
    v = [(x0, y0, a0), (x1, y0, a0), (x1, y1, a0), (x0, y1, a0),
         (x0, y0, a1), (x1, y0, a1), (x1, y1, a1), (x0, y1, a1)]
    idx = [(0, 1, 2, 3), (4, 5, 6, 7), (0, 1, 5, 4), (2, 3, 7, 6), (1, 2, 6, 5), (0, 3, 7, 4)]
    return [[v[i] for i in f] for f in idx]


def _plan(s, g):
    vg, esdf, em = pe.get_grounded_map()
    orig = RRTStar._smooth_path
    try:
        sm = open(os.path.join(_HERE, "state", "best_smoother.py")).read()
        RRTStar._smooth_path = cand.make_patch_method(cand.load_smoother(sm))
        b = dict(ev.BASE); b.update(dict(dubins_turning_radius=1.5, planning_timeout=15.0,
                 energy_aware=True, flight_velocity=2.0)); b.update(TUNED)
        random.seed(0); np.random.seed(0)
        return RRTStar(vg, esdf, PlanningConfig(**b), energy_model=em).plan(s, g)
    finally:
        RRTStar._smooth_path = orig


def densify(path, step=0.6):
    P = [np.asarray(p, float) for p in path]
    out = [P[0]]
    for i in range(len(P) - 1):
        d = np.linalg.norm(P[i+1] - P[i]); n = max(1, int(d / step))
        for k in range(1, n + 1):
            out.append(P[i] + (P[i+1] - P[i]) * k / n)
    return np.array(out)


def main(scn="A"):
    s, g, title = SCNS[scn]
    path = _plan(s, g)
    if not path:
        print("规划失败"); return
    P = densify(path)
    X, Y, ALT = P[:, 0], P[:, 1], -P[:, 2]

    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(111, projection="3d")
    for o in bp.BLOCKS_OBSTACLES:
        x0, x1 = o["x_range"]; y0, y1 = o["y_range"]; a0, a1 = -o["z_range"][1], -o["z_range"][0]
        for f in _box_faces(x0, x1, y0, y1, a0, a1):
            ax.add_collection3d(Poly3DCollection([f], alpha=0.12, facecolor="gray", edgecolor="none"))
    ax.plot(X, Y, ALT, "-", color="#ddd", lw=1)                      # 完整路径淡影
    ax.scatter([s[0]], [s[1]], [-s[2]], c="black", s=40)
    ax.scatter([g[0]], [g[1]], [-g[2]], c="red", marker="*", s=140)
    trail, = ax.plot([], [], [], "-", color="#1f77b4", lw=2.5)
    drone = ax.scatter([], [], [], c="#d62728", s=70)
    ax.set_xlim(0, 72); ax.set_ylim(-28, 28); ax.set_zlim(0, 16)
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("高度(m)")

    N = len(P)
    def upd(f):
        i = min(N - 1, int(f))
        trail.set_data(X[:i+1], Y[:i+1]); trail.set_3d_properties(ALT[:i+1])
        drone._offsets3d = ([X[i]], [Y[i]], [ALT[i]])
        ax.view_init(elev=22, azim=-70 + f * 0.6)
        ax.set_title(f"{title} · 优化后轨迹飞越(无需 AirSim) · 高度 {ALT[i]:.1f}m")
        return trail, drone

    ani = animation.FuncAnimation(fig, upd, frames=N, interval=80, blit=False)
    out = os.path.join(EXP, f"flight_{scn}.gif")
    ani.save(out, writer="pillow", fps=12)
    plt.close(fig)
    print(f"[saved] {out}  ({N} 帧, 路径长 {bp.compute_path_length(path):.0f}m, 高度峰 {ALT.max():.1f}m)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "A")
