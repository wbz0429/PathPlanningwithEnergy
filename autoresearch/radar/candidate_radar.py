"""
candidate_radar.py — 代码级可进化组件:聚类 cluster(points, ctx) + 关联 associate(track_pos, det_pos, ctx)。
这是 radar autoresearch 里【贪心调参到不了、只有 LLM proposer 能做】的部分:进化出新的
聚类/关联启发式(不只调 eps/Q/R)。经沙箱(AST 白名单 + 超时)注入 pipeline,绝不改 pipeline 源码。

契约:
  cluster(points(N,2), ctx) -> np.ndarray(K,2)     簇质心=检测。ctx.eps/min_samples 可用;命名空间给 dbscan、np。
  associate(track_pos(N,2), det_pos(M,2), ctx) -> [(track_idx, det_idx), ...]  门内配对。ctx.gate 可用;给 linear_sum_assignment、np。
安全:load_* 先过 sandbox.check_code(禁 os/sys/open/eval/dunder,只许 numpy/math/scipy);
      contract_test_* 注册前契约测试;评测时外层可用 sandbox.time_limit 兜死循环。
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # 找 autoresearch/sandbox.py
import numpy as np
from scipy.optimize import linear_sum_assignment
from pipeline import dbscan, PipelineParams
from sandbox import check_code

DEFAULT_CLUSTER_SRC = '''
def cluster(points, ctx):
    """默认:固定 eps DBSCAN 质心。"""
    if len(points) == 0:
        return np.empty((0, 2))
    lab = dbscan(points, ctx.eps, ctx.min_samples)
    if lab.max() < 0:
        return np.empty((0, 2))
    return np.array([points[lab == c].mean(0) for c in range(lab.max() + 1)])
'''

DEFAULT_ASSOCIATE_SRC = '''
def associate(track_pos, det_pos, ctx):
    """默认:门控匈牙利。返回 [(track_idx, det_idx), ...]。"""
    D = np.linalg.norm(track_pos[:, None, :] - det_pos[None, :, :], axis=2)
    BIG = 1e6
    Dc = np.where(D <= ctx.gate, D, BIG)
    ri, ci = linear_sum_assignment(Dc)
    return [(int(r), int(c)) for r, c in zip(ri, ci) if Dc[r, c] < BIG]
'''


def _load(src, name, extra_ns):
    ok, reason = check_code(src)
    if not ok:
        raise ValueError(f"sandbox reject: {reason}")
    ns = {"np": np, "numpy": np, **extra_ns}
    exec(compile(src, f"<{name}>", "exec"), ns)
    fn = ns.get(name)
    if not callable(fn):
        raise ValueError(f"source must define callable `{name}`")
    return fn


def load_cluster(src):
    return _load(src, "cluster", {"dbscan": dbscan})


def load_associate(src):
    return _load(src, "associate", {"linear_sum_assignment": linear_sum_assignment})


def contract_test_cluster(fn):
    ctx = PipelineParams()
    pts = np.array([[0, 0], [0.1, 0], [0, 0.1], [10, 10], [10.1, 10.]])
    out = np.asarray(fn(pts, ctx), float)
    assert out.ndim == 2 and out.shape[1] == 2, "cluster 须返回 (K,2)"
    assert len(out) == 0 or np.all(np.isfinite(out)), "质心须有限"
    e = np.asarray(fn(np.empty((0, 2)), ctx), float).reshape(-1, 2)   # 空输入不崩
    assert e.shape[1] == 2


def contract_test_associate(fn):
    ctx = PipelineParams()
    tp = np.array([[0, 0], [10, 0.]]); dp = np.array([[0.2, 0], [10.1, 0.]])
    pairs = fn(tp, dp, ctx)
    assert isinstance(pairs, (list, tuple)), "associate 须返回 list"
    seen_t, seen_d = set(), set()
    for pr in pairs:
        ti, di = int(pr[0]), int(pr[1])
        assert 0 <= ti < len(tp) and 0 <= di < len(dp), "配对索引越界"
        assert ti not in seen_t and di not in seen_d, "配对须一对一"
        seen_t.add(ti); seen_d.add(di)


# 一个"进化候选"示例(LLM 可能提的):DBSCAN 后合并过近质心(抗过分割),再按点数加权
EXAMPLE_CLUSTER_SRC = '''
def cluster(points, ctx):
    if len(points) == 0:
        return np.empty((0, 2))
    lab = dbscan(points, ctx.eps, ctx.min_samples)
    if lab.max() < 0:
        return np.empty((0, 2))
    cents = np.array([points[lab == c].mean(0) for c in range(lab.max() + 1)])
    # 合并互相距离 < eps 的质心(减少同一目标被切成多簇)
    keep = np.ones(len(cents), bool)
    for i in range(len(cents)):
        if not keep[i]:
            continue
        for j in range(i + 1, len(cents)):
            if keep[j] and np.linalg.norm(cents[i] - cents[j]) < ctx.eps:
                cents[i] = (cents[i] + cents[j]) / 2.0
                keep[j] = False
    return cents[keep]
'''


if __name__ == "__main__":
    from smoke_test import gen_scene, evaluate_sequence

    # 1) 默认组件:加载 + 契约 + 注入后与内建默认一致
    cl = load_cluster(DEFAULT_CLUSTER_SRC); asso = load_associate(DEFAULT_ASSOCIATE_SRC)
    contract_test_cluster(cl); contract_test_associate(asso)
    base = evaluate_sequence(gen_scene(seed=0), PipelineParams())
    inj = evaluate_sequence(gen_scene(seed=0), PipelineParams(), cluster_fn=cl, associate_fn=asso)
    assert abs(base["OSPA_mean"] - inj["OSPA_mean"]) < 1e-9, "注入默认应与内建一致"
    print(f"默认组件注入 == 内建 ✓  (OSPA={inj['OSPA_mean']:.3f})")

    # 2) 进化候选:合并过近质心 —— 契约 + 能评测(在硬场景上看效果)
    ec = load_cluster(EXAMPLE_CLUSTER_SRC); contract_test_cluster(ec)
    HARD = dict(clutter=(12, 26), noise=0.55, pts_per_target=(4, 9))
    d0 = np.mean([evaluate_sequence(gen_scene(seed=s, **HARD), PipelineParams())["OSPA_mean"] for s in (0, 1, 2)])
    de = np.mean([evaluate_sequence(gen_scene(seed=s, **HARD), PipelineParams(), cluster_fn=ec)["OSPA_mean"] for s in (0, 1, 2)])
    print(f"进化聚类候选(合并过近质心): 默认OSPA={d0:.3f} → 候选OSPA={de:.3f} ({100*(d0-de)/d0:+.1f}%)")

    # 3) 沙箱拦截恶意代码
    for bad, why in [("import os\ndef cluster(points, ctx):\n return os.getcwd()", "import os"),
                     ("def cluster(points, ctx):\n return points.__class__", "dunder")]:
        try:
            load_cluster(bad); print(f"❌ 未拦截: {why}")
        except ValueError as e:
            print(f"沙箱拦截 [{why}] ✓  ({str(e)[:40]})")

    print("\n✅ 代码级组件进化接口就绪 —— cluster/associate 可经沙箱注入进化。")
    print("   这是贪心调参到不了的地方:LLM proposer 进化组件即插进 evaluate_sequence,同一冻结尺子评。")
