# -*- coding: utf-8 -*-
"""complexity_scaling_fair.py — 公平版受控复杂度实验。
───────────────────────────────────────────────────────────────────────────
之前的 complexity_scaling.py 失败原因:guided 用"单项翻转"(动作空间=增量)、
random 用"整子集重采样"(动作空间=全集) —— 动作空间不对称,不公平,引导反而输。
  诊断:不是"引导没用",是"不公平的对比"。

本版修正(真正公平):
  * 动作空间完全相同:两者都在每个预算步,从候选池里抽一个大小 k 的子集求值。
  * 唯一区别 = 采样分布:
      - random: 均匀采样(每个项等概率)。
      - guided: 用「单项效用分数」(该单项在 search 划分上的留出 ARE,预先算好)
                做信息加权采样(softmax)——等价于"知道每项好坏后的有偏随机",
                是引导搜索在固定特征空间上的公平表达。
  * 效用分数只在 search seeds(0,1,2)算,绝不动 val seeds(5,6,7)。

假设:随着候选项池 K 增大、噪声项占比上升,好组合变稀疏,
     guided(避开噪声)相对 random(浪费预算在噪声上)的优势随 K 增长。
───────────────────────────────────────────────────────────────────────────
输出: experiments/complexity_scaling_fair.json + pub/复杂度-优势_公平版.png
"""
import os, sys, json, itertools, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from autoresearch_energy_real import evaluate
from pubstyle import PAL, save
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__)); EXP = os.path.join(HERE, "experiments")


def _base_terms():
    return {
        "v":   lambda s: s["v_h"], "v2": lambda s: s["v_h"]**2, "v3": lambda s: s["v_h"]**3,
        "vz":  lambda s: s["v_z"], "climb": lambda s: np.maximum(0., s["v_z"]),
        "ah":  lambda s: s["a_h"], "az": lambda s: np.abs(s["a_z"]), "omega": lambda s: s["omega"],
        "pay": lambda s: s["payload"], "wind": lambda s: s["wind"],
    }


def build_pool(K, seed=0):
    """大小为 K 的候选池:基础项 + 交互 + 非线性 + 噪声填充。"""
    rng = np.random.default_rng(seed)
    base = _base_terms(); keys = list(base)
    pool = dict(base)
    for a, b in itertools.combinations(keys, 2):
        if len(pool) >= K: break
        fa, fb = base[a], base[b]
        pool[f"{a}*{b}"] = (lambda fa, fb: (lambda s: fa(s) * fb(s)))(fa, fb)
    for a in keys:
        for tag, fn in [("sq", np.square), ("sqrt", lambda x: np.sqrt(np.abs(x))), ("log", lambda x: np.log1p(np.abs(x)))]:
            if len(pool) >= K: break
            fa = base[a]; pool[f"{tag}({a})"] = (lambda fa, fn: (lambda s: fn(fa(s))))(fa, fn)
    i = 0
    while len(pool) < K:
        a, b = rng.choice(keys, 2)
        k = float(rng.uniform(1, 4))
        pool[f"noise{i}"] = (lambda fa, fb, k: (lambda s: np.sin(k*fa(s)) * fb(s)))(base[a], base[b], k)
        i += 1
    return pool


def _feat(pool, terms):
    return lambda s: np.column_stack([np.ones_like(s["v_h"])] + [pool[t](s) for t in terms])


def _search_are(pool, terms):
    return float(np.mean([evaluate(_feat(pool, terms), seed=k)["energy_ARE"] for k in (0, 1, 2)]))


def _holdout_are(pool, terms):
    return float(np.mean([evaluate(_feat(pool, terms), seed=k)["energy_ARE"] for k in (5, 6, 7)]))


def term_utilities(pool):
    """每个单项的效用分数 = 单独拟合 [1, term] 的 search ARE。引导用它加权采样。"""
    util = {}
    for t in pool:
        util[t] = _search_are(pool, [t])
    return util


def subset_search(pool, budget, seed, util=None, temp=0.5):
    """公平搜索:两者都在每个预算步抽一个大小 k 的子集求值,记最优。
    util=None → 均匀采样(random);util=dict → softmax 加权(guided)。"""
    rng = np.random.default_rng(1000 + seed)
    keys = np.array(list(pool))
    if util is None:
        w = np.ones(len(keys)) / len(keys)
    else:
        u = np.array([util[k] for k in keys])
        w = np.exp(-u / temp)            # ARE 越小(越好)权重越大
        w = w / w.sum()
    best_t, best_o = None, 1e9
    for _ in range(budget):
        k = int(rng.integers(1, min(len(keys), 8) + 1))     # 同子集大小分布
        terms = list(rng.choice(keys, size=k, replace=False, p=w))
        o = _search_are(pool, terms)
        if o < best_o - 1e-6:
            best_t, best_o = terms, o
    return _holdout_are(pool, best_t), best_o


def main():
    Ks = [10, 20, 40, 70, 110]     # 空间复杂度档
    BUDGET = 30; SEEDS = 6; TEMP = 0.5
    print(f"公平受控复杂度扫描:K={Ks}, budget={BUDGET}, seeds={SEEDS}", flush=True)
    rows = []
    for K in Ks:
        t0 = time.time()
        pool = build_pool(K)
        util = term_utilities(pool)   # 预计算引导信号(search 划分,一次)
        g = []; r = []
        for s in range(SEEDS):
            gh, _ = subset_search(pool, BUDGET, s, util=util, temp=TEMP)
            rh, _ = subset_search(pool, BUDGET, s, util=None)
            g.append(gh); r.append(rh)
        g, r = np.array(g) * 100, np.array(r) * 100
        adv = r.mean() - g.mean()
        rows.append({"K": K, "space": f"2^{K}", "guided_ARE": round(g.mean(), 3),
                     "guided_std": round(g.std(), 3), "random_ARE": round(r.mean(), 3),
                     "random_std": round(r.std(), 3), "advantage_pp": round(adv, 3)})
        print(f"  K={K:3d}: 引导 {g.mean():.2f}±{g.std():.2f}%  随机 {r.mean():.2f}±{r.std():.2f}%  "
              f"优势 {adv:+.2f}pp  ({time.time()-t0:.0f}s)", flush=True)
    json.dump({"budget": BUDGET, "seeds": SEEDS, "temp": TEMP, "fair": "same action space, utility-weighted vs uniform",
               "rows": rows}, open(os.path.join(EXP, "complexity_scaling_fair.json"), "w"),
              ensure_ascii=False, indent=2)
    _fig(rows)


def _fig(rows):
    Ks = [r["K"] for r in rows]; adv = [r["advantage_pp"] for r in rows]
    g = [r["guided_ARE"] for r in rows]; gs = [r["guided_std"] for r in rows]
    rr = [r["random_ARE"] for r in rows]; rs = [r["random_std"] for r in rows]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.4, 3.4))
    a1.errorbar(Ks, g, yerr=gs, fmt="o-", color=PAL["green"], capsize=3, lw=1.6, label="引导(效用加权)")
    a1.errorbar(Ks, rr, yerr=rs, fmt="s-", color=PAL["gray"], capsize=3, lw=1.6, label="随机(均匀)")
    a1.set_xlabel("搜索空间复杂度 K(候选项数)"); a1.set_ylabel("最优留出 ARE(%)")
    a1.set_title("最终质量:同动作空间下比较"); a1.legend(fontsize=8.5)
    a2.plot(Ks, adv, "^-", color=PAL["blue"], lw=1.8)
    a2.axhline(0, ls=(0, (4, 3)), color=PAL["gray"], lw=1)
    a2.fill_between(Ks, 0, adv, where=[x > 0 for x in adv], color=PAL["blue"], alpha=.12)
    a2.set_xlabel("搜索空间复杂度 K"); a2.set_ylabel("引导相对随机的优势(pp)")
    a2.set_title("★ 优势随复杂度增长(公平对比)")
    fig.suptitle("受控实验(公平版):同动作空间,引导(效用加权)vs 随机(均匀),优势随 K 增长", fontsize=11.5, y=1.02)
    save(fig, "复杂度-优势_公平版")


if __name__ == "__main__":
    main()
