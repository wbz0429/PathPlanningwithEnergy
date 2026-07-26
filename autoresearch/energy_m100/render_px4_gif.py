# -*- coding: utf-8 -*-
"""render_px4_gif.py — 把 PX4 真飞控栈飞行(px4_traj*.npz)重渲染成投影可读的高清 GIF。
原 gif 只有 640px、字号 8pt,投到幕布上看不清;这里按 dpi=150、字号×1.8 重画,
左:3D 轨迹(距离翻越 vs M100 绕行同屏),右:M100 模型瞬时功率。
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
PX4 = os.path.join(HERE, "px4_integration")
FS = 1.8            # 字号放大系数(投影用)
BOXES = [(22, 25, -26, 26, 5), (45, 48, -10, 10, 12)]   # 楼A 5m / 楼B 12m


def draw_box(ax, x0, x1, y0, y1, h):
    v = np.array([[x0, y0, 0], [x1, y0, 0], [x1, y1, 0], [x0, y1, 0],
                  [x0, y0, h], [x1, y0, h], [x1, y1, h], [x0, y1, h]])
    faces = [[0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4], [2, 3, 7, 6], [1, 2, 6, 5], [0, 3, 7, 4]]
    ax.add_collection3d(Poly3DCollection([v[f] for f in faces], color="gray", alpha=.45, linewidths=.3))


def main(step=7, dpi=150):
    m100 = np.load(os.path.join(PX4, "px4_traj.npz"))
    dist = np.load(os.path.join(PX4, "px4_traj_dist.npz"))
    pw = np.load(os.path.join(PX4, "px4_traj_power.npz"))
    tracks = [("距离最短(翻越)", dist["t"], dist["pos"], "tab:red"),
              ("真机M100(绕行)", m100["t"], m100["pos"], "tab:green")]

    fig = plt.figure(figsize=(13, 5.4))
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    axp = fig.add_subplot(1, 2, 2)
    for b in BOXES:
        draw_box(ax, *b)

    lines, dots = {}, {}
    for nm, t, x, c in tracks:
        lines[nm], = ax.plot([], [], [], color=c, lw=2 * FS, label=nm)
        dots[nm], = ax.plot([], [], [], "o", color=c, ms=7 * FS)
    allx = np.vstack([x for _, _, x, _ in tracks])
    ax.set_xlim(allx[:, 0].min() - 2, allx[:, 0].max() + 2)
    ax.set_ylim(allx[:, 1].min() - 4, allx[:, 1].max() + 4)
    ax.set_zlim(0, max(15, (-allx[:, 2]).max() + 1))
    ax.set_xlabel("x 北(m)", fontsize=10 * FS); ax.set_ylabel("y 东(m)", fontsize=10 * FS)
    ax.set_zlabel("高度(m)", fontsize=10 * FS); ax.tick_params(labelsize=8.5 * FS)
    ax.legend(fontsize=8 * FS, loc="upper left"); ax.view_init(elev=22, azim=-60)

    tp, P = pw["t"], pw["P"]
    pline, = axp.plot([], [], color="tab:green", lw=1.4 * FS, label=f"真机M100 模型功率(均{P.mean():.0f}W)")
    axp.set_xlim(0, tp[-1]); axp.set_ylim(0, 900)
    axp.set_xlabel("t (s)", fontsize=10 * FS); axp.set_ylabel("功率 (W)", fontsize=10 * FS)
    axp.tick_params(labelsize=8.5 * FS); axp.legend(fontsize=8 * FS)
    axp.set_title("PX4 飞出轨迹的瞬时功率(真机能耗模型)", fontsize=11 * FS)
    fig.suptitle("PX4 SITL + Gazebo Harmonic 真飞控栈:同固件同脚本 A/B,M100 绕行省 14.3%", fontsize=12 * FS)
    fig.tight_layout()

    n = max(len(t) for _, t, _, _ in tracks)
    frames = range(0, n, step)

    def update(fi):
        arts = []
        for nm, t, x, _c in tracks:
            i = min(fi, len(t) - 1)
            lines[nm].set_data(x[:i, 0], x[:i, 1]); lines[nm].set_3d_properties(-x[:i, 2])
            dots[nm].set_data([x[i, 0]], [x[i, 1]]); dots[nm].set_3d_properties([-x[i, 2]])
            arts += [lines[nm], dots[nm]]
        j = min(fi, len(tp) - 1)
        pline.set_data(tp[:j], P[:j]); arts.append(pline)
        return arts

    out_mp4 = os.path.join(HERE, "experiments", "px4_flight_hq.mp4")
    FuncAnimation(fig, update, frames=frames, blit=False).save(
        out_mp4, writer=FFMpegWriter(fps=20, bitrate=2400), dpi=dpi)
    plt.close(fig)
    print(f"高清视频存 {out_mp4}({len(list(frames))} 帧)")
    print("接着用 ffmpeg 调色板转 GIF → experiments/gifs/px4_flight.gif")


if __name__ == "__main__":
    main()
