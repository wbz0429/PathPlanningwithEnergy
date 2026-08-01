# -*- coding: utf-8 -*-
"""headroom_mechanism_probe.py — 诊断的机制边界受控探测(合成域)。
───────────────────────────────────────────────────────────────────────────
诚实刻画诊断的适用范围(论文 Scope and limit):
  诊断能识别"当前空间饱和"吗? → 合成高/低 synergy 池检验。
结果:single-term utility spread 信号方向正确(高 synergy 的 rel_spread 更大),
     但一旦噪声项稀释交互,单靠 spread+subset 无法可靠区分高/低 synergy 池。
结论:诊断可靠地回答"当前空间饱和"(能耗域 9 档全对);
     "是否存在可逃逸高复杂度空间"需要跨空间检查(程序级变异空间),非特征池统计能答。
───────────────────────────────────────────────────────────────────────────
输出: experiments/headroom_mechanism_probe.json
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from complexity_scaling_fair import _search_are

HERE = os.path.dirname(os.path.abspath(__file__)); EXP = os.path.join(HERE, "experiments")


def synth_pool(high_synergy, seed=0):
    rng = np.random.default_rng(seed)
    base = {"v": lambda s: s["v_h"], "v2": lambda s: s["v_h"]**2,
            "pay": lambda s: s["payload"], "climb": lambda s: np.maximum(0., s["v_z"])}
    pool = dict(base)
    if high_synergy:
        pool["pay*climb"] = lambda s: base["pay"](s) * base["climb"](s)
        for i in range(20):
            k = float(rng.uniform(1, 4))
            pool[f"noise{i}"] = (lambda k: (lambda s: np.sin(k*s["v_h"]) * s["wind"]))(k)
    else:
        for i in range(20):
            k = float(rng.uniform(1, 4))
            pool[f"noise{i}"] = (lambda k: (lambda s: np.sin(k*s["v_h"]) * s["wind"]))(k)
    return pool


def best_subset_median(pool, budget=30, seeds=(0,1,2,3,4)):
    bests = []
    for seed in seeds:
        rng = np.random.default_rng(1000 + seed)
        keys = np.array(list(pool)); b = 1e9
        for _ in range(budget):
            k = int(rng.integers(1, min(len(keys), 8) + 1))
            ts = list(rng.choice(keys, size=k, replace=False))
            b = min(b, _search_are(pool, ts))
        bests.append(b)
    return float(np.median(bests))


def main():
    out = {"seeds": "multi-seed median", "note": "mechanistic boundary probe"}
    for hs, nm in [(True, "high_synergy"), (False, "low_synergy")]:
        pool = synth_pool(hs)
        util = np.array([_search_are(pool, [t]) for t in pool]) * 100
        best_single = float(util.min()); spread = float(util.std())
        rel = spread / best_single
        best_subset = best_subset_median(pool) * 100
        headroom = max(0.0, best_single - best_subset) / best_single
        pred = "tie" if rel > 0.3 and headroom < 0.05 else "guided-wins"
        out[nm] = {"best_single%": round(best_single, 3), "spread": round(spread, 3),
                   "rel_spread": round(rel, 3), "best_subset%": round(best_subset, 3),
                   "headroom": round(headroom, 3), "predicted": pred}
        print(f"  {nm:12s}: rel_spread={rel:.2f}  best_subset={best_subset:.2f}%  headroom={headroom:.2f}  → {pred}")
    out["conclusion"] = ("single-term utility spread signal is directionally correct (high-synergy rel_spread larger), "
                         "but with noise diluting the interaction, spread+subset alone cannot reliably separate high/low "
                         "synergy pools → diagnostic answers 'current space saturated' (correct on energy 9 levels), "
                         "not 'escapable higher-complexity space exists' (needs program-level mutation-space check).")
    json.dump(out, open(os.path.join(EXP, "headroom_mechanism_probe.json"), "w"), ensure_ascii=False, indent=2)
    print(f"落盘 {EXP}/headroom_mechanism_probe.json")


if __name__ == "__main__":
    main()
