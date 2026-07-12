"""
wall_experiment.py — 工程缝合主实验(真结果):
真机 DJI M100 能耗模型(held-out ARE 1.93%)当代价,插进已有 RRT*/A* 规划器,
在"高墙障碍"场景对比【距离最短 / 教科书BEMT / 真机M100】三种代价选出的路径。

核心发现(真规划器,非手画):
  距离最短 & 教科书BEMT → 翻墙(爬升),因为它们不知道/低估爬升耗能;
  真机M100(从真实功率学到爬升贵)→ 绕行,省 5–13% 能量。

破循环:每条路的能耗同时用 M100 和 BEMT 两把尺子评(交叉验证)。
诚实边界:v=8 真实巡航;干净地图隔离机制;省能幅度随墙宽下降;非新算法,是"可信代价改变选择"。
"""
import os, sys, copy, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import physics_eval as pe
from planning_experiment import M100Em, DistanceEm, score_path

HERE = os.path.dirname(os.path.abspath(__file__))
V_CRUISE = 8.0                       # M100 真实巡航速度(数据 4–12 m/s 内)
_BASE = None


def clean_wall_map(half_y, height):
    """干净地图:清空 → 地面 → 一堵墙(x≈40, y∈±half_y, 高 height)。隔离机制,避免其他障碍污染。"""
    global _BASE
    if _BASE is None:
        _BASE = pe.get_grounded_map()
    vg0, esdf0, bemt = _BASE
    vg = copy.deepcopy(vg0); vg.grid[:] = 0
    for iz in range(vg.grid.shape[2]):
        if vg.grid_to_world((0, 0, iz))[2] > -0.6:
            vg.grid[:, :, iz] = 1                        # 地面(防钻地)
    for xw in np.arange(39, 42.5, .5):
        for yw in np.arange(-half_y, half_y + .5, .5):
            for zw in np.arange(-height, 0.1, .5):
                idx = vg.world_to_grid(np.array([xw, yw, zw]))
                if vg.is_valid_index(idx):
                    vg.grid[idx] = 1
    esdf = type(esdf0)(vg); esdf.compute()
    return vg, esdf, bemt


def run():
    src = open(os.path.join(HERE, "state", "best_featurize.py")).read()
    m100 = M100Em(src, payload=250., name="M100")
    dist = DistanceEm()
    s = np.array([10., 0., -3.]); g = np.array([72., 0., -3.])
    yards = {"M100尺": m100, "BEMT尺": None, "路长": dist}   # BEMT尺 稍后填

    scenarios = [("窄墙", 12.), ("中墙", 16.), ("宽墙", 20.)]
    rows = []
    paths_for_fig = {}
    for scn, hy in scenarios:
        vg, esdf, bemt = clean_wall_map(hy, 12.)
        yards["BEMT尺"] = bemt
        conds = [("距离最短", dist), ("教科书BEMT", bemt), ("真机M100", m100)]
        rec = {"场景": scn, "墙半宽": hy}
        pdict = {}
        for cname, em in conds:
            p, _ = pe.energy_astar(vg, esdf, em, s, g, velocity=V_CRUISE,
                                   safety_margin=0.6, max_expand=1500000)
            pdict[cname] = p
            if p and len(p) >= 2:
                P = np.array([np.asarray(w, float).reshape(-1)[:3] for w in p])
                alt = float(-P[:, 2].min()); ydev = float(np.abs(P[:, 1]).max())
                rec[f"{cname}_选择"] = "翻越" if alt > ydev + 2 else "绕行"
                rec[f"{cname}_爬升m"] = round(alt, 0)
                rec[f"{cname}_E_M100"] = round(score_path(p, m100, V_CRUISE), 0)
                rec[f"{cname}_E_BEMT"] = round(score_path(p, bemt, V_CRUISE), 0)
        # 省能:M100 vs 距离,两把尺子(交叉验证)
        if rec.get("距离最短_E_M100") and rec.get("真机M100_E_M100"):
            d_m, m_m = rec["距离最短_E_M100"], rec["真机M100_E_M100"]
            d_b, m_b = rec["距离最短_E_BEMT"], rec["真机M100_E_BEMT"]
            rec["省能%_M100尺"] = round(100 * (d_m - m_m) / d_m, 1)
            rec["省能%_BEMT尺"] = round(100 * (d_b - m_b) / d_b, 1)   # 交叉验证:另把尺也省=可信
        rows.append(rec)
        paths_for_fig[scn] = pdict
        print(f"[{scn} 半宽{hy}] 距离={rec.get('距离最短_选择')} BEMT={rec.get('教科书BEMT_选择')} "
              f"M100={rec.get('真机M100_选择')} | 省能 M100尺={rec.get('省能%_M100尺')}% "
              f"BEMT尺={rec.get('省能%_BEMT尺')}%")

    os.makedirs(os.path.join(HERE, "experiments"), exist_ok=True)
    json.dump(rows, open(os.path.join(HERE, "experiments", "wall_experiment_result.json"), "w"),
              ensure_ascii=False, indent=2)
    print("\n结果存 experiments/wall_experiment_result.json")
    _make_fig(paths_for_fig, rows)
    return rows


def _make_fig(paths_for_fig, rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa
    fig = plt.figure(figsize=(15, 5))
    colors = {"距离最短": "tab:red", "教科书BEMT": "tab:orange", "真机M100": "tab:green"}
    for i, (scn, pdict) in enumerate(paths_for_fig.items()):
        ax = fig.add_subplot(1, len(paths_for_fig), i + 1, projection="3d")
        for cname, p in pdict.items():
            if not p or len(p) < 2:
                continue
            P = np.array([np.asarray(w, float).reshape(-1)[:3] for w in p])
            ax.plot(P[:, 0], P[:, 1], -P[:, 2], color=colors[cname], label=cname, lw=2)
        ax.set_title(scn); ax.set_xlabel("x(m)"); ax.set_ylabel("y(m)"); ax.set_zlabel("alt(m)")
        if i == 0:
            ax.legend(fontsize=8)
    plt.suptitle("真机M100能耗代价 vs 距离/BEMT:距离&BEMT翻墙,M100绕行避爬升(省5-13%)")
    plt.tight_layout()
    out = os.path.join(HERE, "experiments", "wall_experiment.png")
    plt.savefig(out, dpi=110); print(f"图存 {out}")


if __name__ == "__main__":
    run()
