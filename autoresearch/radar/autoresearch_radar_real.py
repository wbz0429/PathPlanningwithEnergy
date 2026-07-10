"""
autoresearch_radar_real.py — 在【真 RadarScenes】上跑 autoresearch 联合调优 loop(keep/revert)。
冻结评测器 = OSPA/MOTA(作弊不了);按【序列】划分 train/held-out(防过拟合);
联合调 PipelineParams,propose→evaluate→keep/revert。随机扰动演示机制,LLM proposer 后接。

score = mean_OSPA − 0.3·mean_MOTA(越低越好:低 OSPA + 高 MOTA);
KEEP 当 score 更低 且 min_MOTA 不显著退。真数据上 FP 抑制 → OSPA↓ 且 MOTA↑,一致。
"""
import os, glob, itertools
import numpy as np
from radar_adapter import stream_sequence
from smoke_test import evaluate_sequence
from pipeline import PipelineParams
from autoresearch_radar import BOUNDS, perturb

ROOT = os.path.expanduser("~/datasets/radar_scenes/extracted/RadarScenes/data")
SEQS = sorted(glob.glob(ROOT + "/sequence_*"))
TRAIN = SEQS[0:4]
HELDOUT = SEQS[4:7]
TRAIN_FRAMES = 500          # 搜索时每序列帧数(快)
VAL_FRAMES = 900            # 留出复核帧数(更全)


def eval_cfg(params, seqs, max_frames):
    ospas, motas = [], []
    for s in seqs:
        m = evaluate_sequence(itertools.islice(stream_sequence(s, vr_thresh=0.5), max_frames),
                              params, ospa_c=2.0, mota_gate=2.0)
        ospas.append(m["OSPA_mean"]); motas.append(m["MOTA"])
    return float(np.mean(ospas)), float(np.mean(motas)), float(np.min(motas))


def score_of(ospa, mota):
    return ospa - 0.3 * mota


def search(iters=50, seed=0):
    rng = np.random.default_rng(seed)
    best = PipelineParams()
    bo, bm, bmin = eval_cfg(best, TRAIN, TRAIN_FRAMES)
    bs = score_of(bo, bm)
    base = (best, bo, bm)
    keeps = 0
    for it in range(iters):
        cand = perturb(best, rng)
        o, m, mn = eval_cfg(cand, TRAIN, TRAIN_FRAMES)
        s = score_of(o, m)
        if s < bs - 1e-4 and mn >= bmin - 0.05:      # KEEP:score降 且 最差序列MOTA不显著退
            best, bo, bm, bmin, bs = cand, o, m, mn, s
            keeps += 1
    return base, (best, bo, bm), keeps


if __name__ == "__main__":
    print(f"真 RadarScenes autoresearch loop | 训练{len(TRAIN)}序列 留出{len(HELDOUT)}序列 | 50轮\n")
    (p0, o0, m0), (pB, oB, mB), keeps = search(iters=50)
    print(f"[{keeps} 次 KEEP]")
    print(f"默认参数   训练: OSPA={o0:.3f} MOTA={m0:+.3f}")
    print(f"loop调优   训练: OSPA={oB:.3f} MOTA={mB:+.3f}   (OSPA {100*(o0-oB)/o0:+.1f}%, MOTA {mB-m0:+.2f})")
    # 留出复核
    v0o, v0m, _ = eval_cfg(p0, HELDOUT, VAL_FRAMES)
    vBo, vBm, _ = eval_cfg(pB, HELDOUT, VAL_FRAMES)
    print(f"\n留出序列复核({len(HELDOUT)}条,{VAL_FRAMES}帧):")
    print(f"默认       留出: OSPA={v0o:.3f} MOTA={v0m:+.3f}")
    print(f"loop调优   留出: OSPA={vBo:.3f} MOTA={vBm:+.3f}   (OSPA {100*(v0o-vBo)/v0o:+.1f}%, MOTA {vBm-v0m:+.2f})")
    print(f"\n调优参数: eps={pB.eps:.2f} min_samples={pB.min_samples} q={pB.q:.2f} r={pB.r:.2f} "
          f"gate={pB.gate:.2f} n_confirm={pB.n_confirm} max_miss={pB.max_miss}")
    ok = vBo < v0o - 1e-3 and vBm > v0m - 0.05
    print("\n" + ("✅ loop 在【真 RadarScenes 留出序列】上压过默认参数 —— 真基准正结果 + 泛化。"
                  if ok else "⚠️ 留出未稳定改进,需调 score/预算。"))
