"""
baselines.py — 对标 baseline + 两条答辩防线。全在冻结评测器(OSPA/MOTA)+ 留出场景上比。

要回答的两个问题:
  ① 生死线:联合 autoresearch loop 能不能赢【同预算随机搜索】?(证明是"搜索"在干活,非评测器形式化)
  ② 必要性:联合 loop 能不能赢【坐标下降贪心】?(答辩会问"参数联合耦合到需要 LLM/进化吗?
     还是逐个贪心调就够?"——若贪心追平,必要性存疑,须诚实退到"评测方法学 + 组件进化"当贡献)
另含算法组件 baseline:自适应 DBSCAN(KNN 自适应 eps,对标 RadarScenes SOTA 聚类思路)。
"""
import numpy as np
from dataclasses import replace
from scipy.spatial import cKDTree
from pipeline import PipelineParams, dbscan, cluster_centroids
from smoke_test import gen_scene, evaluate_sequence
from autoresearch_radar import HARD, TRAIN, HELDOUT, BOUNDS, search as joint_search


def eval_cfg(params, seeds, cluster_fn=None):
    ospas, motas = [], []
    for s in seeds:
        m = evaluate_sequence(gen_scene(seed=s, **HARD), params, cluster_fn=cluster_fn)
        ospas.append(m["OSPA_mean"]); motas.append(m["MOTA"])
    return float(np.mean(ospas)), float(np.min(motas))


# --- 算法组件 baseline:自适应 DBSCAN(每帧 KNN 自适应 eps)---
def adaptive_cluster(points_xy, params, k=4):
    X = np.asarray(points_xy, float).reshape(-1, 2)
    if len(X) < k + 2:
        return cluster_centroids(X, params.eps, params.min_samples)
    d, _ = cKDTree(X).query(X, k=k + 1)
    eps = float(np.median(d[:, -1]))              # 每帧动态 eps = k-NN 距离中位数
    lab = dbscan(X, max(0.4, eps), params.min_samples)
    if lab.max() < 0:
        return np.empty((0, 2))
    return np.array([X[lab == c].mean(0) for c in range(lab.max() + 1)])


# --- baseline:随机搜索(同预算,生死线)---
def random_search(iters=60, seed=1):
    rng = np.random.default_rng(seed)
    best = PipelineParams(); best_o, best_m = eval_cfg(best, TRAIN)
    for _ in range(iters):
        d = {k: (int(rng.integers(lo, hi + 1)) if isint else float(rng.uniform(lo, hi)))
             for k, (lo, hi, isint) in BOUNDS.items()}
        o, m = eval_cfg(PipelineParams(**d), TRAIN)
        if o < best_o - 1e-4 and m >= best_m - 0.02:
            best, best_o, best_m = PipelineParams(**d), o, m
    return best


# --- baseline:坐标下降贪心(必要性 baseline)---
def coordinate_descent(passes=3, grid=6):
    best = PipelineParams(); best_o, best_m = eval_cfg(best, TRAIN)
    for _ in range(passes):
        for k, (lo, hi, isint) in BOUNDS.items():
            vals = range(int(lo), int(hi) + 1) if isint else np.linspace(lo, hi, grid)
            for v in vals:
                cand = replace(best, **{k: (int(v) if isint else float(v))})
                o, m = eval_cfg(cand, TRAIN)
                if o < best_o - 1e-4 and m >= best_m - 0.02:
                    best, best_o, best_m = cand, o, m
    return best


def _row(name, params, cluster_fn=None):
    to, tm = eval_cfg(params, TRAIN, cluster_fn)
    ho, hm = eval_cfg(params, HELDOUT, cluster_fn)
    return name, to, tm, ho, hm


if __name__ == "__main__":
    print("跑对标(硬场景,训练5 seed / 留出3 seed)... 需约半分钟\n")
    rows = []
    rows.append(_row("默认(固定参数)", PipelineParams()))
    rows.append(_row("自适应DBSCAN(KNN-eps)", PipelineParams(), adaptive_cluster))
    rows.append(_row("随机搜索(60预算·生死线)", random_search(60)))
    rows.append(_row("坐标下降贪心(必要性)", coordinate_descent()))
    (_, _, _), (pJ, _, _), keeps = joint_search(iters=60)
    rows.append(_row(f"联合autoresearch(60轮/{keeps}KEEP)", pJ))

    print(f"{'方法':<26}{'训练OSPA':>9}{'训练MOTA':>9}{'留出OSPA':>9}{'留出MOTA':>9}")
    print("-" * 62)
    for name, to, tm, ho, hm in rows:
        print(f"{name:<26}{to:>9.3f}{tm:>9.3f}{ho:>9.3f}{hm:>9.3f}")

    # 两条防线判决(以留出 OSPA 为准)
    d = {r[0]: r for r in rows}
    jo = [r for r in rows if r[0].startswith("联合")][0][3]
    ro = d.get("随机搜索(60预算·生死线)", (None,)*5)[3]
    go = d.get("坐标下降贪心(必要性)", (None,)*5)[3]
    print("\n=== 防线判决(留出 OSPA,越低越好)===")
    print(f"① 生死线  vs 随机搜索: 联合={jo:.3f} 随机={ro:.3f} → "
          + ("✅ 联合胜(搜索在干活)" if jo < ro - 1e-3 else "⚠️ 未明显胜随机,生死线告警"))
    gap = (go - jo) / go if go else 0.0        # 联合相对贪心的改进
    if gap > 0.05:
        verdict = "✅ 联合明显胜贪心(参数耦合,联合搜索有必要)"
    elif gap > 0.005:
        verdict = (f"⚠️ 仅微弱胜贪心({gap*100:.1f}%,耦合弱)——'必要性'不成立于纯调参;"
                   "真数据/代码级组件进化上须重验,否则诚实退到'作弊不了评测方法学 + 组件进化'当贡献")
    else:
        verdict = "⚠️ 贪心追平/更好 → 联合搜索'必要性'不成立,退到'评测方法学 + 组件进化'当贡献"
    print(f"② 必要性  vs 坐标下降: 联合={jo:.3f} 贪心={go:.3f}(差 {gap*100:+.1f}%) → {verdict}")
    print("\n(真数据到后:把 gen_scene 换 radar_adapter.stream_sequence,同一张表在 RadarScenes 上重跑。)")
