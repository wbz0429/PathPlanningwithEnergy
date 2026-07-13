"""render_videos.py — 把动力学飞行(RotorPy)渲染成 MP4:红=距离代价,绿=真机M100代价,双机同屏 + 实时功率。
输入:experiments/sim_flight_trajs.npz(单墙)、corridor_trajs.npz(城市走廊)。20fps 实时速度。
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import imageio_ffmpeg
matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from mpl_toolkits.mplot3d import Axes3D  # noqa
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False
HERE = os.path.dirname(os.path.abspath(__file__))


def draw_box(ax, x0, x1, y0, y1, h, color="gray", alpha=.45):
    v = np.array([[x0,y0,0],[x1,y0,0],[x1,y1,0],[x0,y1,0],
                  [x0,y0,h],[x1,y0,h],[x1,y1,h],[x0,y1,h]])
    faces = [[0,1,2,3],[4,5,6,7],[0,1,5,4],[2,3,7,6],[1,2,6,5],[0,3,7,4]]
    ax.add_collection3d(Poly3DCollection([v[f] for f in faces], color=color, alpha=alpha, linewidths=.3))


def render(npz_path, boxes, pairs, title, out_name, step=5):
    D = np.load(npz_path)
    # 对齐两条轨迹长度(取最长,短的用最后位置补)
    series = {}
    for base, _c in pairs:
        series[base] = (D[base + "_t"], D[base + "_x"], D[base + "_P"])
    T = max(s[0][-1] for s in series.values())
    n = max(len(s[0]) for s in series.values())
    fig = plt.figure(figsize=(13, 5.4))
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    axp = fig.add_subplot(1, 2, 2)
    for (x0, x1, y0, y1, h) in boxes:
        draw_box(ax, x0, x1, y0, y1, h)
    lines, dots, plines = {}, {}, {}
    for base, c in pairs:
        t, x, P = series[base]
        lines[base], = ax.plot([], [], [], color=c, lw=2, label=base)
        dots[base], = ax.plot([], [], [], "o", color=c, ms=7)
        plines[base], = axp.plot([], [], color=c, lw=1.4,
                                 label=f"{base}(均{P.mean():.0f}W)")
    allx = np.vstack([series[b][1] for b, _ in pairs])
    ax.set_xlim(allx[:, 0].min() - 2, allx[:, 0].max() + 2)
    ax.set_ylim(allx[:, 1].min() - 4, allx[:, 1].max() + 4)
    ax.set_zlim(0, max(15, allx[:, 2].max() + 1))
    ax.set_xlabel("x(m)"); ax.set_ylabel("y(m)"); ax.set_zlabel("alt(m)")
    ax.legend(fontsize=8, loc="upper left"); ax.view_init(elev=22, azim=-60)
    axp.set_xlim(0, T); axp.set_ylim(0, 900)
    axp.set_xlabel("t (s)"); axp.set_ylabel("M100 功率 (W)")
    axp.legend(fontsize=8); axp.set_title("飞行瞬时功率(真机模型)")
    fig.suptitle(title); fig.tight_layout()
    frames = range(0, n, step)

    def update(fi):
        arts = []
        for base, _c in pairs:
            t, x, P = series[base]
            i = min(fi, len(t) - 1)
            lines[base].set_data(x[:i, 0], x[:i, 1]); lines[base].set_3d_properties(x[:i, 2])
            dots[base].set_data([x[i, 0]], [x[i, 1]]); dots[base].set_3d_properties([x[i, 2]])
            plines[base].set_data(t[:i], P[:min(i, len(P))])
            arts += [lines[base], dots[base], plines[base]]
        return arts

    ani = FuncAnimation(fig, update, frames=frames, blit=False)
    out = os.path.join(HERE, "experiments", out_name)
    ani.save(out, writer=FFMpegWriter(fps=20, bitrate=2200))
    plt.close(fig)
    print(f"视频存 {out}  (时长~{len(list(frames))/20:.0f}s)")


if __name__ == "__main__":
    render(os.path.join(HERE, "experiments", "sim_flight_trajs.npz"),
           boxes=[(39, 42, -16, 16, 12)],
           pairs=[("翻越(距离代价)", "tab:red"), ("绕行(M100代价)", "tab:green")],
           title="单墙场景·动力学实飞:距离翻墙(功率飙870W) vs M100绕行(省6.1%)",
           out_name="video_wall.mp4")
    render(os.path.join(HERE, "experiments", "corridor_trajs.npz"),
           boxes=[(22, 25, -26, 26, 5), (45, 48, -10, 10, 12)],
           pairs=[("距离最短", "tab:red"), ("真机M100", "tab:green")],
           title="城市走廊·逐障碍混合决策:M100 翻矮楼A、绕高楼B(实飞省4.4%)",
           out_name="video_corridor.mp4")
