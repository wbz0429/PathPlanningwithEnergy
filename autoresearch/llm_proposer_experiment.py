# -*- coding: utf-8 -*-
"""llm_proposer_experiment.py — 真 LLM(Claude)当 in-loop proposer 的 best-shot 引导 vs 随机对照。
规划器域(平滑器代码空间,有 headroom)。评测器冻结,proposer 只提候选代码,keep/revert。

用法:
  score '<smoother_src>'          # 给一个候选打分(我=proposer 每轮调它)
  save <tag> <are> '<src>'        # 记录一个候选到 ledger
  best                            # 打印当前最优 k 个(best-shot 上下文)
  ledger                          # 打印全部记录
数据落 experiments/llm_proposer_ledger.jsonl(每轮:tag/score/success/proposer/note/src_hash)
"""
import os, sys, json, hashlib, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import physics_eval as pe

LEDGER = os.path.join(HERE, "experiments", "llm_proposer_ledger.jsonl")
TUNED = json.load(open(os.path.join(HERE, "state", "best.json")))["config"]


def score_src(src):
    """冻结评测器给平滑器候选打分。返回 (score, success)。越低越好。"""
    r = pe.evaluate(TUNED, runs=3, seed0=0, smoother_src=src)
    return r.get("score", 9e9), r.get("min_success", 0.0)


def holdout_src(src):
    """留出验证(不同种子),防过拟合训练种子。"""
    r = pe.evaluate(TUNED, runs=3, seed0=100, smoother_src=src)
    return r.get("score", 9e9), r.get("min_success", 0.0)


def log(tag, src, proposer, note):
    sc, su = score_src(src); hsc, hsu = holdout_src(src)
    rec = {"t": round(time.time(), 1), "tag": tag, "proposer": proposer,
           "score": round(sc, 1), "success": round(su, 3),
           "holdout_score": round(hsc, 1), "holdout_success": round(hsu, 3),
           "note": note, "src_hash": hashlib.md5(src.encode()).hexdigest()[:8],
           "src_len": len(src)}
    os.makedirs(os.path.dirname(LEDGER), exist_ok=True)
    with open(LEDGER, "a") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    # 存候选代码
    cdir = os.path.join(HERE, "experiments", "llm_candidates"); os.makedirs(cdir, exist_ok=True)
    open(os.path.join(cdir, f"{tag}.py"), "w").write(src)
    print(f"[{tag}] {proposer}: score={sc:.0f} success={su:.2f} | held-out={hsc:.0f}/{hsu:.2f} | {note}")
    return rec


def load_ledger():
    if not os.path.exists(LEDGER):
        return []
    return [json.loads(l) for l in open(LEDGER)]


def best_k(k=2, proposer=None):
    rows = [r for r in load_ledger() if r["success"] >= 0.999 and (proposer is None or r["proposer"] == proposer)]
    rows.sort(key=lambda r: r["score"])
    return rows[:k]


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "ledger"
    if cmd == "best":
        for r in best_k(3):
            print(f"  {r['tag']}: score={r['score']} ({r['proposer']}) {r['note']}")
    elif cmd == "ledger":
        rows = load_ledger()
        print(f"共 {len(rows)} 条候选")
        for r in rows:
            print(f"  {r['tag']:14s} {r['proposer']:8s} score={r['score']:8.0f} succ={r['success']:.2f} ho={r['holdout_score']:8.0f}")
