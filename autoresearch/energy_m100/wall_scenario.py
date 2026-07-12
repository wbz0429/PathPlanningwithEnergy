"""
wall_scenario.py — 自定义"高墙"场景:一堵 15m 高墙挡在起终点之间,逼出"翻越 vs 绕行"的能量权衡。
用真规划器 energy_astar,三代价(距离/BEMT/真机M100)各规划,看是否产生不同路径 + 省能。
这是最稳的工程缝合演示:真高差 → 能量最优 ≠ 最短路。
"""
import os, sys, copy, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import physics_eval as pe
from planning_experiment import M100Em, DistanceEm, score_path

HERE = os.path.dirname(os.path.abspath(__file__))


def build_wall_map(half_y=10.0, height=15.0):
    """复制接地图,加一堵墙(x≈40, y∈[-half_y,half_y], 地面到 height 米高),重算 ESDF。
    半宽越大→越难抄角/绕行→逼出真正的"翻越 vs 长绕行"选择。"""
    vg0, esdf0, bemt = pe.get_grounded_map()
    vg = copy.deepcopy(vg0)
    added = 0
    for xw in np.arange(39, 42.5, 0.5):
        for yw in np.arange(-half_y, half_y + 0.5, 0.5):
            for zw in np.arange(-height, 0.1, 0.5):        # 地面到 height 高
                idx = vg.world_to_grid(np.array([xw, yw, zw]))
                if vg.is_valid_index(idx) and vg.grid[idx] != 1:
                    vg.grid[idx] = 1; added += 1
    ESDFClass = type(esdf0)
    esdf = ESDFClass(vg); esdf.compute()
    return vg, esdf, bemt, added


def main():
    print("建高墙地图...")
    vg, esdf, bemt, added = build_wall_map()
    print(f"墙加了 {added} 体素")
    m100 = M100Em(open(os.path.join(HERE, "state", "best_featurize.py")).read(), payload=250., name="M100")
    dist = DistanceEm()
    # 起终点:墙两侧,低空 3m(z=-3),y=0(正对墙)
    s = np.array([10., 0., -3.]); g = np.array([72., 0., -3.])
    print(f"起点={s} 终点={g}(墙在 x=40, 15m高)\n")
    yard_m100 = m100; yard_len = dist
    res = {}
    for name, em in [("距离", dist), ("BEMT", bemt), ("M100", m100)]:
        t0 = time.time()
        path, _ = pe.energy_astar(vg, esdf, em, s, g, max_expand=600000)
        dt = time.time() - t0
        if path and len(path) >= 2:
            P = np.array([np.asarray(p, float).reshape(-1)[:3] for p in path])
            max_alt = float(-P[:, 2].min())         # 最高飞到多少m
            max_y = float(np.abs(P[:, 1]).max())     # 最大横向偏移
            L = score_path(path, yard_len)
            Em = score_path(path, yard_m100)
            res[name] = dict(n=len(P), L=L, Em=Em, alt=max_alt, ydev=max_y)
            topo = "翻越" if max_alt > 8 else ("绕行" if max_y > 8 else "直穿?")
            print(f"  {name:5s}: {len(P)}点 {dt:.0f}s 路长={L:.0f}m 最高={max_alt:.0f}m 横偏={max_y:.0f}m "
                  f"→[{topo}] M100能耗={Em:.0f}J")
        else:
            print(f"  {name:5s}: 规划失败"); res[name] = None
    # 省能
    if res.get("距离") and res.get("M100"):
        sv = 100 * (res["距离"]["Em"] - res["M100"]["Em"]) / res["距离"]["Em"]
        print(f"\n省能(M100代价 vs 距离,M100尺子): {sv:+.1f}%")
        print(f"路径不同? 距离最高{res['距离']['alt']:.0f}m vs M100最高{res['M100']['alt']:.0f}m")
    import json
    json.dump({k: v for k, v in res.items()}, open(os.path.join(HERE, "experiments", "wall_scenario_result.json"), "w"), indent=2)
    return res


if __name__ == "__main__":
    main()
