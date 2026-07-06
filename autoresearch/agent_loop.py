"""
agent_loop.py — LLM 研究智能体自优化闭环(本课题核心)

The closed research loop: propose -> apply -> evaluate -> keep/revert -> log.
把用户在 Stage 2-4 手动做的"读结果→提假设→改参数或改代码→留出验证→取舍"自动化。
- 提议由 Proposer 完成(MockProposer 离线 / LLMProposer 真调 API)。
- 参数改动直接进 evaluator;代码改动经 sandbox 校验 + 契约测试 + 类级 monkeypatch
  注入 RRTStar._smooth_path(不改 drone_sim 源码)+ 墙钟超时保护。
- keep/revert 用与 search.py 一致的单指标 score(越低越好)。

用法:
  python agent_loop.py --proposer mock --iters 12 --runs 3 --code
  ANTHROPIC_API_KEY=... python agent_loop.py --proposer llm --iters 20 --runs 3 --code
"""
import os
import sys
import json
import time
import argparse

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import evaluator as ev                       # 固定评测器(内部已把 drone_sim 加入 path)
from planning.rrt_star import RRTStar        # 代码进化的 monkeypatch 目标
import sandbox
import candidate
from actions import Action, clip_params, normalize_weights
from proposer import MockProposer, LLMProposer

DEFAULT_CONFIG = dict(
    step_size=1.5, max_iterations=5000, goal_sample_rate=0.4, search_radius=4.0,
    dubins_turning_radius=1.5, weight_energy=0.6, weight_distance=0.3, weight_time=0.1,
)
FIXED = dict(planning_timeout=10.0, energy_aware=True)
EVAL_TIMEOUT_S = 220.0   # 单次 evaluate 墙钟上限(防候选平滑器死循环)

_ORIG_SMOOTH = RRTStar._smooth_path


def _run_eval(overrides, smoother_src, runs, seed0):
    """
    评测一组 overrides;smoother_src=None 用原生 _smooth_path,否则注入候选。
    返回 evaluator 结果 dict;非法/崩溃返回 {"_bad": reason}。
    """
    if smoother_src is None:
        return ev.evaluate(overrides, runs=runs, seed0=seed0)

    ok, reason = sandbox.check_code(smoother_src)
    if not ok:
        return {"_bad": f"INVALID:{reason}"}
    try:
        fn = candidate.load_smoother(smoother_src)
        candidate.contract_test(fn)
    except Exception as e:
        return {"_bad": f"CONTRACT:{type(e).__name__}:{e}"}

    RRTStar._smooth_path = candidate.make_patch_method(fn)
    try:
        with sandbox.time_limit(EVAL_TIMEOUT_S):
            return ev.evaluate(overrides, runs=runs, seed0=seed0)
    except Exception as e:
        return {"_bad": f"CRASH:{type(e).__name__}:{e}"}
    finally:
        RRTStar._smooth_path = _ORIG_SMOOTH


def _cfg_from_action(action: Action, best_cfg: dict):
    """set_params -> 合并裁剪后的参数;rewrite_smoother -> 沿用最优参数。返回 overrides(含 FIXED)。"""
    cfg = dict(best_cfg)
    if action.kind == "set_params":
        clean, _ = clip_params(action.params or {})
        cfg.update(clean)
        cfg = normalize_weights(cfg)
    return {**cfg, **FIXED}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--proposer", choices=["mock", "llm"], default="mock")
    ap.add_argument("--iters", type=int, default=12)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--model", default="claude-opus-4-8")
    ap.add_argument("--code", action="store_true", help="允许代码级改动(rewrite_smoother)")
    ap.add_argument("--seed0", type=int, default=0)
    args = ap.parse_args()

    logdir = os.path.join(_HERE, "experiments")
    gendir = os.path.join(_HERE, "generated")
    os.makedirs(logdir, exist_ok=True)
    os.makedirs(gendir, exist_ok=True)
    logf = open(os.path.join(logdir, "agent_log.jsonl"), "w")

    proposer = MockProposer() if args.proposer == "mock" else LLMProposer(model=args.model)
    strategy = ""
    pmd = os.path.join(_HERE, "program.md")
    if os.path.exists(pmd):
        strategy = open(pmd).read()

    astar = ev.astar_baseline()
    astar_total = sum(v["energy"] for v in astar.values())
    print(f"[agent-loop] proposer={args.proposer} iters={args.iters} runs={args.runs} "
          f"code={'on' if args.code else 'off'} | A* 基线={astar_total:.0f}J", flush=True)

    # ---- 基线评测(第 0 轮) ----
    res0 = _run_eval({**DEFAULT_CONFIG, **FIXED}, None, args.runs, args.seed0)
    best = {"config": dict(DEFAULT_CONFIG), "score": res0["score"], "res": res0,
            "smoother_src": None, "smoother_name": "default"}
    history = []

    def _record(it, phase, action, res, decision, reason):
        good = "_bad" not in res
        rec = {
            "iter": it, "phase": phase,
            "action": action.to_log() if action else {"kind": "baseline", "name": "default"},
            "score": round(res["score"], 2) if good else None,
            "energy_total": res.get("energy_total") if good else None,
            "vs_astar": round(res["vs_astar"], 4) if good and res.get("vs_astar") else None,
            "min_success": res.get("min_success") if good else None,
            "decision": decision, "reason": reason,
            "smoother": best["smoother_name"],
        }
        logf.write(json.dumps(rec, ensure_ascii=False) + "\n"); logf.flush()
        history.append({"iter": it, "kind": rec["action"]["kind"], "name": rec["action"]["name"],
                        "score": res["score"] if good else 9e9, "vs_astar": rec["vs_astar"],
                        "min_success": rec["min_success"], "decision": decision, "reason": reason})
        vs = f"{rec['vs_astar']}" if rec["vs_astar"] else "--"
        et = f"{res.get('energy_total'):.0f}J" if good and res.get("energy_total") else "FAIL"
        print(f"[{it:02d}] {phase:14s} score={ (res['score'] if good else float('nan')):9.0f} "
              f"E={et:>8s} vsA*={vs:>6} succ={rec['min_success']} -> {decision}"
              f"{'  ('+reason+')' if reason else ''}", flush=True)

    _record(0, "baseline", None, res0, "KEEP", "")

    t_all = time.time()
    for it in range(1, args.iters):
        ctx = {
            "iteration": it, "n_iters": args.iters, "astar_total": astar_total,
            "best": {"config": best["config"], "score": best["score"],
                     "vs_astar": best["res"].get("vs_astar"),
                     "min_success": best["res"].get("min_success"),
                     "energy_total": best["res"].get("energy_total"),
                     "smoother_name": best["smoother_name"]},
            "history": history, "allow_code": args.code,
            "current_smoother_src": best["smoother_src"] or candidate.DEFAULT_SMOOTHER_SRC,
            "strategy": strategy,
        }
        try:
            action = proposer.propose(ctx)
        except Exception as e:
            print(f"[{it:02d}] proposer error: {type(e).__name__}: {e}", flush=True)
            _record(it, "propose_err", Action("set_params", "propose_error", str(e)),
                    {"_bad": "PROPOSER"}, "SKIP", f"{type(e).__name__}")
            continue

        overrides = _cfg_from_action(action, best["config"])
        smoother_src = action.code if action.kind == "rewrite_smoother" else best["smoother_src"]
        res = _run_eval(overrides, smoother_src, args.runs, args.seed0)

        if "_bad" in res:
            _record(it, action.kind, action, {"score": 9e9, **res}, "REVERT", res["_bad"])
            continue

        improved = res["score"] < best["score"] - 1e-9
        if improved:
            new_cfg = {k: v for k, v in overrides.items() if k not in FIXED}
            best = {"config": new_cfg, "score": res["score"], "res": res,
                    "smoother_src": action.code if action.kind == "rewrite_smoother" else best["smoother_src"],
                    "smoother_name": action.name if action.kind == "rewrite_smoother" else best["smoother_name"]}
            # 持久化最优
            json.dump({"config": new_cfg, "smoother_name": best["smoother_name"],
                       "result": {k: res[k] for k in ("score", "energy_total", "vs_astar", "min_success")}},
                      open(os.path.join(logdir, "agent_best.json"), "w"), indent=2, default=float)
            if best["smoother_src"]:
                open(os.path.join(gendir, "best_smoother.py"), "w").write(best["smoother_src"])
            _record(it, action.kind, action, res, "KEEP", "score improved <== NEW BEST")
        else:
            _record(it, action.kind, action, res, "REVERT", "no improvement")

    logf.close()
    print(f"\n[done] {(time.time()-t_all)/60:.1f}min  best score={best['score']:.0f} "
          f"vsA*={best['res'].get('vs_astar')} smoother={best['smoother_name']}", flush=True)
    print(f"[best config] {json.dumps(best['config'], ensure_ascii=False)}", flush=True)


if __name__ == "__main__":
    main()
