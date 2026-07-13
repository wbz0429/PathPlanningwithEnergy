"""corridor_experiment.py — 城市走廊场景(旗舰演示):一条配送路线上两栋楼,逼出混合决策。
  楼A(x≈22-25):矮而宽(5m高, y±26)→ 绕行极远,翻越便宜 → 该翻;
  楼B(x≈45-48):高而窄(12m高, y±10)→ 翻越费电,绕行近   → 该绕。
预期:距离代价 = 全翻(直线最短);真机M100代价 = 翻A + 绕B(逐障碍聪明决策)。
再用 RotorPy 动力学实飞两条路线,验证混合决策的省能过动力学仍成立。
"""
import os, sys, copy, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import physics_eval as pe
from planning_experiment import M100Em, DistanceEm, score_path
from planning_connect import PowerModel
from sim_flight import make_traj, fly, flown_energy

HERE = os.path.dirname(os.path.abspath(__file__))
_BASE = None
# 楼盒(世界坐标):(x0,x1,y0,y1,高度)
BLDG_A = (22., 25., -26., 26., 5.)     # 矮宽
BLDG_B = (45., 48., -10., 10., 12.)    # 高窄


def clean_corridor_map():
    global _BASE
    if _BASE is None:
        _BASE = pe.get_grounded_map()
    vg0, esdf0, bemt = _BASE
    vg = copy.deepcopy(vg0); vg.grid[:] = 0
    for iz in range(vg.grid.shape[2]):
        if vg.grid_to_world((0, 0, iz))[2] > -0.6:
            vg.grid[:, :, iz] = 1                          # 地面
    for (x0, x1, y0, y1, h) in (BLDG_A, BLDG_B):
        for xw in np.arange(x0, x1 + .25, .5):
            for yw in np.arange(y0, y1 + .25, .5):
                for zw in np.arange(-h, 0.1, .5):
                    idx = vg.world_to_grid(np.array([xw, yw, zw]))
                    if vg.is_valid_index(idx):
                        vg.grid[idx] = 1
    esdf = type(esdf0)(vg); esdf.compute()
    return vg, esdf, bemt


def classify(P):
    """每栋楼的决策:楼x范围内的最大高度/最大|y| → 翻或绕。"""
    out = {}
    for nm, (x0, x1, y0, y1, h) in [("A矮宽", BLDG_A), ("B高窄", BLDG_B)]:
        m = (P[:, 0] > x0 - 4) & (P[:, 0] < x1 + 4)
        if not m.any():
            out[nm] = "?"; continue
        alt = float(-P[m, 2].max() if P[m, 2].min() > 0 else -P[m, 2].min())
        ydev = float(np.abs(P[m, 1]).max())
        out[nm] = f"翻({alt:.0f}m)" if alt > h - 1 else f"绕(y{ydev:.0f}m)"
    return out


def min_dist_to_boxes(x_enu):
    d_all = np.full(len(x_enu), 1e9)
    for (x0, x1, y0, y1, h) in (BLDG_A, BLDG_B):
        lo = np.array([x0, y0, 0.]); hi = np.array([x1, y1, h])
        d = np.maximum(lo - x_enu, 0) + np.maximum(x_enu - hi, 0)
        d_all = np.minimum(d_all, np.linalg.norm(d, axis=1))
    return float(d_all.min())


def main():
    src = open(os.path.join(HERE, "state", "best_featurize.py")).read()
    m100 = M100Em(src, payload=250.); dist = DistanceEm(); power = PowerModel(src, "M100")
    vg, esdf, bemt = clean_corridor_map()
    s = np.array([5., 0., -3.]); g = np.array([75., 0., -3.]); V = 8.0
    out = {}

    print("=== ① 规划层对比(裕度0.6, v=8)===")
    for nm, em in [("距离最短", dist), ("教科书BEMT", bemt), ("真机M100", m100)]:
        p, _ = pe.energy_astar(vg, esdf, em, s, g, velocity=V, safety_margin=0.6, max_expand=2000000)
        P = np.array([np.asarray(w, float).reshape(-1)[:3] for w in p])
        cls = classify(P); E = score_path(p, m100, V); L = score_path(p, dist)
        out[nm] = {"决策": cls, "E_M100": round(E, 0), "路长": round(L, 0)}
        print(f"  {nm}: A={cls['A矮宽']} B={cls['B高窄']}  路长{L:.0f}m  M100能耗{E:.0f}J")
    sv = 100 * (out["距离最短"]["E_M100"] - out["真机M100"]["E_M100"]) / out["距离最短"]["E_M100"]
    out["规划省能%"] = round(sv, 1)
    print(f"  规划层省能(M100 vs 距离): {sv:+.1f}%")

    print("\n=== ② 动力学实飞(裕度2.0, RotorPy)===")
    trajs = {}
    for nm, em in [("距离最短", dist), ("真机M100", m100)]:
        p, _ = pe.energy_astar(vg, esdf, em, s, g, velocity=V, safety_margin=2.0, max_expand=2000000)
        P = np.array([np.asarray(w, float).reshape(-1)[:3] for w in p])
        traj, wps, step = make_traj(P, v_avg=6.0)
        t, x, v = fly(traj, wps[0])
        E, T, Pm, Pser = flown_energy(t, v, power)
        margin = min_dist_to_boxes(x)
        cls = classify(np.column_stack([x[:, 0], x[:, 1], -x[:, 2]]))
        trajs[nm + "_t"] = np.asarray(t); trajs[nm + "_x"] = np.asarray(x); trajs[nm + "_P"] = Pser
        out[nm + "_飞行"] = {"决策": cls, "E": round(E, 0), "时长": round(T, 1),
                             "均功率": round(Pm, 0), "最近距离": round(margin, 2)}
        print(f"  {nm}: A={cls['A矮宽']} B={cls['B高窄']} 飞行{T:.1f}s 均{Pm:.0f}W E={E:.0f}J 距楼最近{margin:.2f}m")
    svf = 100 * (out["距离最短_飞行"]["E"] - out["真机M100_飞行"]["E"]) / out["距离最短_飞行"]["E"]
    out["飞行省能%"] = round(svf, 1)
    print(f"\n=== ③ 结论: 规划省能{sv:+.1f}% → 动力学实飞省能{svf:+.1f}% ===")

    json.dump(out, open(os.path.join(HERE, "experiments", "corridor_result.json"), "w"),
              ensure_ascii=False, indent=2)
    np.savez(os.path.join(HERE, "experiments", "corridor_trajs.npz"), **trajs)
    _fig(trajs, out)


def _fig(trajs, out):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "Heiti TC"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(17, 4.2))
    for (x0, x1, y0, y1, h), lb in [(BLDG_A, "楼A 5m宽"), (BLDG_B, "楼B 12m窄")]:
        a1.add_patch(Rectangle((x0, 0), x1 - x0, h, color="gray", alpha=.6))
        a1.text((x0 + x1) / 2, h + .3, lb, ha="center", fontsize=8)
        a2.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, color="gray", alpha=.6))
    colors = {"距离最短": "tab:red", "真机M100": "tab:green"}
    for base, c in colors.items():
        X = trajs[base + "_x"]; t = trajs[base + "_t"]; P = trajs[base + "_P"]
        a1.plot(X[:, 0], X[:, 2], color=c, lw=2, label=base)
        a2.plot(X[:, 0], X[:, 1], color=c, lw=2)
        a3.plot(t, P, color=c, lw=1.5, label=f"{base} 均{P.mean():.0f}W")
    a1.set_xlabel("x (m)"); a1.set_ylabel("高度 (m)"); a1.set_title("侧视:M100 翻矮楼A、不翻高楼B"); a1.legend(fontsize=8)
    a2.set_xlabel("x (m)"); a2.set_ylabel("y (m)"); a2.set_title("俯视:M100 只在高楼B处绕行")
    a3.set_xlabel("t (s)"); a3.set_ylabel("功率 (W)"); a3.set_title("飞行瞬时功率(真机模型)"); a3.legend(fontsize=8)
    plt.suptitle(f"城市走廊·逐障碍混合决策:距离=全翻,M100=翻矮楼+绕高楼 | "
                 f"规划省能{out['规划省能%']}% → 动力学实飞省能{out['飞行省能%']}%")
    plt.tight_layout()
    p = os.path.join(HERE, "experiments", "corridor.png")
    plt.savefig(p, dpi=125); print(f"图存 {p}")


if __name__ == "__main__":
    main()
