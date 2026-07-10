"""
baselines_energy.py — M100 能耗模型 autoresearch 的对标 + 防线(与 radar 平行)。
全在冻结评测器(真实功率 held-out 能量ARE)上比。
  生死线:loop vs 同预算随机子集;必要性:loop vs 贪心前向选择;
  另含 全模型(所有项)与 稳态BEMT(已知物理)两个参照。
"""
import numpy as np
from m100_eval import evaluate
from autoresearch_energy_real import LIB, BASELINE, feat, obj, holdout, search


def greedy_forward():
    cur, rem = [], set(LIB)
    best_o = obj(BASELINE)
    while rem:
        o, t = min((obj(cur + [t]) if (cur + [t]) else best_o, t) for t in rem)
        if o < best_o - 1e-5:
            cur.append(t); rem.discard(t); best_o = o
        else:
            break
    return cur or list(BASELINE)


def random_subset(iters=60, seed=1):
    rng = np.random.default_rng(seed)
    best = list(BASELINE); best_o = obj(best)
    keys = list(LIB)
    for _ in range(iters):
        k = int(rng.integers(1, len(keys) + 1))
        terms = list(rng.choice(keys, size=k, replace=False))
        o = obj(terms)
        if o < best_o - 1e-5:
            best, best_o = terms, o
    return best


if __name__ == "__main__":
    print("真 M100 · 能耗模型对标(留出 seeds 5,6,7 复核)\n")
    rows = []
    rows.append(("稳态BEMT(1,v,v²·已知物理)", list(BASELINE)))
    rows.append(("全模型(所有项)", list(LIB)))
    rows.append(("随机子集(60预算·生死线)", random_subset(60)))
    rows.append(("贪心前向(必要性)", greedy_forward()))
    best, keeps = search(iters=60)
    rows.append((f"autoresearch loop({keeps}KEEP)", best))

    print(f"{'方法':<28}{'留出ARE':>9}{'R²':>8}   模型")
    print("-" * 78)
    res = {}
    for name, terms in rows:
        are, r2 = holdout(terms)
        res[name] = are
        print(f"{name:<28}{are*100:>8.2f}%{r2:>+8.3f}   [{'+'.join(terms)[:34]}]")

    loop = [v for k, v in res.items() if k.startswith("autoresearch")][0]
    rnd = res["随机子集(60预算·生死线)"]; grd = res["贪心前向(必要性)"]
    print("\n=== 防线(留出 ARE,越低越好)===")
    print(f"① 生死线 vs 随机: loop={loop*100:.2f}% 随机={rnd*100:.2f}% → "
          + ("✅ loop 胜" if loop < rnd - 1e-4 else "⚠️ 未明显胜随机"))
    gap = (grd - loop) / grd if grd else 0
    print(f"② 必要性 vs 贪心: loop={loop*100:.2f}% 贪心={grd*100:.2f}%(差{gap*100:+.1f}%) → "
          + ("✅ loop 胜" if gap > 0.05 else
             "⚠️ 与贪心相当——结构搜索必要性弱(同radar);LLM价值在代码级featurize进化(candidate_energy),非选项"))
