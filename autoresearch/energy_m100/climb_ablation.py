"""climb_ablation.py — 因果消融:证明"爬升校准"就是分叉的因,堵住"BEMT是稻草人"质疑。
把 M100 模型的爬升项(climb/desc)挖掉 → 得到"爬升盲"的 M100 变体。
预期:①held-out ARE 变差(爬升项对真实精度有贡献)②规划器变得像 BEMT 一样翻墙(而非绕行)。
这排除了"M100 因别的原因绕行"的可能——因就是它从真数据学到的爬升代价。
不改冻结的 best_featurize.py,用字符串生成变体(受控实验,非改尺子)。
"""
import os, sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import physics_eval as pe
import m100_eval as me
from planning_experiment import M100Em, DistanceEm, score_path
from wall_experiment import clean_wall_map

HERE = os.path.dirname(os.path.abspath(__file__))
FULL = open(os.path.join(HERE, "state", "best_featurize.py")).read()
# 消融:爬升/下降项置零 → 爬升盲
ABLATED = (FULL.replace("climb = np.maximum(vz, 0.0)", "climb = np.zeros_like(vz)  # 消融")
               .replace("desc = np.minimum(vz, 0.0)", "desc = np.zeros_like(vz)  # 消融"))


def held_out_are(src):
    ns = {"np": np}; exec(src, ns)
    return me.evaluate(ns["featurize"], seed=0, test_frac=0.33)["energy_ARE"]


def main():
    print("=== ① held-out 真实精度:爬升项有没有用 ===")
    are_full = held_out_are(FULL); are_abl = held_out_are(ABLATED)
    print(f"  完整 M100       : ARE {are_full*100:.2f}%")
    print(f"  爬升盲 M100(消融): ARE {are_abl*100:.2f}%   (变差 {100*(are_abl-are_full):.2f}pp → 爬升项确实拟合真实能耗)")

    print("\n=== ② 规划决策:爬升项是不是分叉的因 ===")
    m_full = M100Em(FULL, payload=250.); m_abl = M100Em(ABLATED, payload=250.); dist = DistanceEm()
    vg, esdf, bemt = clean_wall_map(16., 12.); V = 8.0
    s = np.array([10., 0., -3.]); g = np.array([72., 0., -3.])
    for nm, em in [("距离最短", dist), ("完整M100", m_full), ("爬升盲M100", m_abl), ("教科书BEMT", bemt)]:
        p, _ = pe.energy_astar(vg, esdf, em, s, g, velocity=V, safety_margin=0.6, max_expand=1500000)
        P = np.array([np.asarray(w, float).reshape(-1)[:3] for w in p])
        alt, ydev = -P[:, 2].min(), np.abs(P[:, 1]).max()
        choice = "翻越" if alt > ydev + 2 else "绕行"
        print(f"  {nm:10s}: {choice}  (爬升{alt:.0f}m 横偏{ydev:.0f}m)")
    print("\n→ 若'爬升盲M100'翻墙(像BEMT),而'完整M100'绕行 = 爬升校准就是分叉的因,BEMT非稻草人(它就是爬升盲)")


if __name__ == "__main__":
    main()
