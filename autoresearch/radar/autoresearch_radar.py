"""
autoresearch_radar.py — 雷达跟踪管线的 autoresearch 联合调优闭环(keep/revert)。
用 smoke_test.evaluate_sequence 当【冻结评测器】(OSPA/MOTA 是尺子,不可碰),
联合调 PipelineParams(eps,min_samples,q,r,gate,n_confirm,max_miss),
训练 seed 上 propose→evaluate→keep/revert,留出 seed 复核过拟合。

本文件用随机扰动搜索演示【机制】(证明 loop 能压过默认参数);真正的 autoresearch
是把"提议"换成 LLM 研究员(同能耗那套 skill),评测器/纪律不变。

诚实纪律(同能耗):KEEP 当且仅当 OSPA 更低 且 MOTA 不显著退;留出验证防过拟合;
score 只对训练 seed。评测器冻结——loop 只能动 PipelineParams。
"""
import numpy as np
from dataclasses import replace
from pipeline import PipelineParams
from smoke_test import gen_scene, evaluate_sequence

# 硬场景(多杂波/大噪声/目标点少)→ 默认参数留有余地,才看得出 loop 有没有用
HARD = dict(clutter=(12, 26), noise=0.55, pts_per_target=(4, 9))
TRAIN = (0, 1, 2, 3, 4)
HELDOUT = (100, 200, 300)

BOUNDS = {  # (lo, hi, is_int)
    "eps": (0.5, 4.0, False), "min_samples": (1, 6, True),
    "q": (0.05, 10.0, False), "r": (0.05, 3.0, False), "gate": (1.0, 6.0, False),
    "n_confirm": (1, 5, True), "max_miss": (1, 8, True),
}


def eval_cfg(params, seeds, scene_kw=HARD):
    """在若干 seed 上评估 → (平均OSPA, 最差MOTA)。流式、内存安全。"""
    ospas, motas = [], []
    for s in seeds:
        m = evaluate_sequence(gen_scene(seed=s, **scene_kw), params)
        ospas.append(m["OSPA_mean"]); motas.append(m["MOTA"])
    return float(np.mean(ospas)), float(np.min(motas))


def perturb(params, rng):
    """复制 best,随机扰动 1-2 个旋钮(bounds 内)。"""
    d = params.__dict__.copy()
    keys = rng.choice(list(BOUNDS), size=rng.integers(1, 3), replace=False)
    for k in keys:
        lo, hi, is_int = BOUNDS[k]
        if is_int:
            d[k] = int(rng.integers(lo, hi + 1))
        else:
            d[k] = float(np.clip(d[k] * rng.uniform(0.6, 1.6) + rng.normal(0, 0.15), lo, hi))
    return PipelineParams(**d)


def search(iters=60, seed=0):
    rng = np.random.default_rng(seed)
    best = PipelineParams()
    b_ospa, b_mota = eval_cfg(best, TRAIN)
    base = (best, b_ospa, b_mota)
    keeps = 0
    for it in range(iters):
        cand = perturb(best, rng)
        o, mo = eval_cfg(cand, TRAIN)
        # KEEP: OSPA 更低 且 MOTA 不显著退(硬约束,防"降OSPA却丢目标")
        if o < b_ospa - 1e-4 and mo >= b_mota - 0.02:
            best, b_ospa, b_mota = cand, o, mo
            keeps += 1
    return base, (best, b_ospa, b_mota), keeps


if __name__ == "__main__":
    (p0, o0, m0), (pB, oB, mB), keeps = search(iters=60)
    print(f"=== 联合调优闭环(硬场景,{len(TRAIN)} 训练 seed,60 轮,{keeps} 次 KEEP)===")
    print(f"默认参数  训练: OSPA={o0:.3f}  MOTA={m0:.3f}")
    print(f"调优后    训练: OSPA={oB:.3f}  MOTA={mB:.3f}   (OSPA 降 {100*(o0-oB)/o0:+.1f}%)")
    # 留出复核(防过拟合)
    o0h, m0h = eval_cfg(p0, HELDOUT)
    oBh, mBh = eval_cfg(pB, HELDOUT)
    print(f"\n留出 seed(100,200,300) 复核:")
    print(f"默认      留出: OSPA={o0h:.3f}  MOTA={m0h:.3f}")
    print(f"调优后    留出: OSPA={oBh:.3f}  MOTA={mBh:.3f}   (OSPA 降 {100*(o0h-oBh)/o0h:+.1f}%)")
    print(f"\n调优出的参数: eps={pB.eps:.2f} min_samples={pB.min_samples} q={pB.q:.2f} "
          f"r={pB.r:.2f} gate={pB.gate:.2f} n_confirm={pB.n_confirm} max_miss={pB.max_miss}")
    improved = oBh < o0h - 1e-3 and mBh >= m0h - 0.03
    print("\n" + ("✅ loop 在留出场景上也压过默认参数 —— 链路'autoresearch→尺子→正结果'跑通(合成)。"
                  if improved else "⚠️ 留出上未稳定改进(默认可能已近最优或场景太易)。"))
