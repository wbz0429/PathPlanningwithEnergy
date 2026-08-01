# -*- coding: utf-8 -*-
"""headroom_diagnostic.py — 「头部空间诊断」:跑 loop 前预测引导搜索是否值得。
───────────────────────────────────────────────────────────────────────────
动机:论文主张"引导优势取决于有效组合复杂度"。若能在跑 loop 之前用一个便宜
指标预测引导是否胜随机,就把"我们报告了边界"升级为"我们提供了预测边界的工具"。
这是一个真正的算法/方法学贡献,值得做。

指标(全部在 search 划分上算,不碰 val):
  对候选池里每个单项 term,算 单效用 search ARE = [1, term] 拟合的 ARE。
  记 U = {u_i} 为单项效用集合(越小越好)。
  1.  spread  = std(U)                     —— 单项效用离散度(信息集中度)
  2.  best_single = min(U)                 —— 最好单项的 ARE
  3.  组合搜索能到 best_subset(取预算内随机抽子集的最优,同 fair 实验)
  4.  synergy = best_single − best_subset  —— 组合相对单体的增益(负=组合反而差)

  预测规则(假设):
    · 若 spread 大 + synergy 小 → 低有效维度(少数项主导)→ 随机很快找到 → 引导≈随机(打平)
    · 若 spread 小 + synergy 大 → 高有效复杂度(组合才有效)→ 引导(带方向)胜随机
  输出一个 "predicted_headroom" 分数,供两域对照。
───────────────────────────────────────────────────────────────────────────
"""
import os, sys, json, itertools
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from autoresearch_energy_real import evaluate
from complexity_scaling_fair import build_pool, _feat, _search_are

HERE = os.path.dirname(os.path.abspath(__file__)); EXP = os.path.join(HERE, "experiments")


def single_term_utilities(pool):
    """每个单项的 search ARE(越小越好)。"""
    return {t: _search_are(pool, [t]) for t in pool}


def best_random_subset(pool, budget=30, seeds=(0,1,2,3,4)):
    """预算内均匀抽子集的最优 search ARE。多 seed 取中位数(稳定,去单次噪声)。"""
    bests = []
    for seed in seeds:
        rng = np.random.default_rng(1000 + seed)
        keys = np.array(list(pool))
        best = 1e9
        for _ in range(budget):
            k = int(rng.integers(1, min(len(keys), 8) + 1))
            terms = list(rng.choice(keys, size=k, replace=False))
            best = min(best, _search_are(pool, terms))
        bests.append(best)
    return float(np.median(bests))


def diagnostic(pool, budget=30):
    util = single_term_utilities(pool)
    U = np.array(list(util.values())) * 100.0      # fraction → percent
    spread = float(U.std())
    best_single = float(U.min())
    best_subset = best_random_subset(pool, budget) * 100.0
    synergy = best_single - best_subset          # 负=组合也不如单体
    # 预测:spread 大且 synergy 小 → 打平;否则有 headroom
    # (归一化:spread 除以最佳单项 ARE,做成相对量)
    rel_spread = spread / max(best_single, 1e-9)
    headroom = max(0.0, -synergy) / max(best_single, 1e-9)   # 组合相对单体的相对增益
    predicted = "tie" if rel_spread > 0.3 and headroom < 0.05 else "guided-wins"
    return {"n_pool": len(pool), "spread": round(spread, 4),
            "rel_spread": round(rel_spread, 3), "best_single": round(best_single, 4),
            "best_subset": round(best_subset, 4), "synergy": round(synergy, 4),
            "headroom": round(headroom, 3), "predicted": predicted}


def main():
    print("头部空间诊断:跑 loop 前预测引导是否胜随机(能耗域示范)")
    # 不同名义 K 下诊断:看是否都预测 "tie"(匹配 fair 实验的 ≈0 优势)
    rows = []
    for K in (10, 20, 40, 70, 110):
        pool = build_pool(K)
        d = diagnostic(pool)
        rows.append({"K": K, **d})
        print(f"  K={K:3d}: spread={d['rel_spread']:.2f}  best_single={d['best_single']:.3f}%  "
              f"best_subset={d['best_subset']:.3f}%  headroom={d['headroom']:.2f}  → 预测 {d['predicted']}")
    json.dump({"rows": rows}, open(os.path.join(EXP, "headroom_diagnostic.json"), "w"),
              ensure_ascii=False, indent=2)
    print(f"\n落盘 {EXP}/headroom_diagnostic.json")
    print("\n若能耗域各 K 都预测 tie → 诊断成立:名义膨胀不改变有效维度 → 预测打平,与 fair 实验一致。")
    print("下一步:把同一诊断搬到规划器域(代码结构),验证它预测 guided-wins(z=−2.38)。")


if __name__ == "__main__":
    main()
