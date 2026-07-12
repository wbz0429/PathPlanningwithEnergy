"""
batch_compare.py — 批量跑三代价对比,找出"能量最优 ≠ 最短路"的场景,并出省能表。
三代价:距离最短 / 教科书BEMT / 真机M100。同一 energy_astar,只换 em(干净消融)。
双模型交叉验证:每条路的能耗同时用 M100 和 BEMT 两把尺子评(破循环)。
"""
import os, sys, time, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import physics_eval as pe
from planning_experiment import M100Em, DistanceEm, score_path

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    vg, esdf, bemt = pe.get_grounded_map()
    m100 = M100Em(open(os.path.join(HERE, "state", "best_featurize.py")).read(), payload=250., name="M100")
    m100_phys = M100Em(open(os.path.join(HERE, "state", "best_featurize.py")).read(),
                       payload=250., physics_climb=True, name="M100+物理爬升")
    conds = [("距离", DistanceEm()), ("BEMT", bemt), ("M100", m100)]
    yard = {"M100": m100, "BEMT": bemt, "len": DistanceEm()}

    rows = []
    seeds = list(range(12))
    scs = pe.gen_scenarios(seeds)
    print(f"跑 {len(scs)} 场景 × 3 代价 ...")
    for i, sc in enumerate(scs):
        s, g = sc["start"], sc["goal"]
        rec = {"scene": sc["name"], "start": np.round(s, 1).tolist(), "goal": np.round(g, 1).tolist()}
        paths = {}
        for name, em in conds:
            try:
                p, _ = pe.energy_astar(vg, esdf, em, s, g)
            except Exception as e:
                p = None
            paths[name] = p
            if p and len(p) >= 2:
                rec[f"{name}_len"] = round(score_path(p, yard["len"]), 1)
                rec[f"{name}_Em100"] = round(score_path(p, yard["M100"]), 0)
                rec[f"{name}_Ebemt"] = round(score_path(p, yard["BEMT"]), 0)
        # 省能:M100代价 vs 距离,在 M100 尺子下
        if "距离_Em100" in rec and "M100_Em100" in rec and rec["距离_Em100"] > 0:
            rec["省能%_M100尺"] = round(100 * (rec["距离_Em100"] - rec["M100_Em100"]) / rec["距离_Em100"], 1)
            # 路径是否不同(长度差异>2m 视为不同拓扑)
            rec["路径不同"] = abs(rec.get("M100_len", 0) - rec.get("距离_len", 0)) > 2
        rows.append(rec)
        diff = rec.get("省能%_M100尺", "?")
        print(f"  {sc['name']}: 省能={diff}%  路径不同={rec.get('路径不同','?')}")

    json.dump(rows, open(os.path.join(HERE, "experiments", "batch_compare_result.json"), "w"),
              ensure_ascii=False, indent=2)
    # 汇总
    saves = [r["省能%_M100尺"] for r in rows if "省能%_M100尺" in r]
    diffs = [r for r in rows if r.get("路径不同")]
    print(f"\n=== 汇总 ===")
    print(f"有效场景: {len(saves)}  平均省能: {np.mean(saves):.1f}%  最大: {max(saves):.1f}%")
    print(f"能量最优≠最短路的场景: {len(diffs)}/{len(saves)}")
    print(f"结果存 experiments/batch_compare_result.json")


if __name__ == "__main__":
    main()
