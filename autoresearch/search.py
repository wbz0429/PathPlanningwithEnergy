"""
autoresearch 循环本体 (相当于 karpathy autoresearch 的驱动器)

提议 -> 评测(evaluator) -> 单指标 keep/revert -> 记 JSONL 日志。
策略：先随机探索找 100% 成功区域，再在最优点附近局部扰动压能耗。

用法：  python search.py <n_iters> <runs>
"""
import os, sys, json, time, random
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import numpy as np
import evaluator as ev

# ---- 搜索空间（可改参数 + 范围）----
SPACE = {
    "step_size":             (1.0, 3.0,  "float"),
    "max_iterations":        (3000, 8000, "int"),
    "goal_sample_rate":      (0.1, 0.6,  "float"),
    "search_radius":         (3.0, 7.0,  "float"),
    "dubins_turning_radius": (1.0, 2.5,  "float"),
}
FIXED = {"planning_timeout": 10.0, "energy_aware": True}

DEFAULT = dict(step_size=1.5, max_iterations=5000, goal_sample_rate=0.4,
               search_radius=4.0, dubins_turning_radius=1.5,
               weight_energy=0.6, weight_distance=0.3, weight_time=0.1)


def _sample_weights(rng):
    w = [rng.random() + 1e-3 for _ in range(3)]
    s = sum(w)
    return {"weight_energy": w[0]/s, "weight_distance": w[1]/s, "weight_time": w[2]/s}


def random_config(rng):
    c = {}
    for k, (lo, hi, typ) in SPACE.items():
        v = rng.uniform(lo, hi)
        c[k] = int(round(v)) if typ == "int" else round(v, 3)
    c.update(_sample_weights(rng))
    return c


def perturb(base, rng, scale=0.25):
    c = dict(base)
    for k, (lo, hi, typ) in SPACE.items():
        span = (hi - lo) * scale
        v = base[k] + rng.gauss(0, span)
        v = max(lo, min(hi, v))
        c[k] = int(round(v)) if typ == "int" else round(v, 3)
    if rng.random() < 0.5:  # 偶尔也扰动权重
        c.update(_sample_weights(rng))
    return c


def to_overrides(c):
    o = dict(c); o.update(FIXED); return o


def main():
    n_iters = int(sys.argv[1]) if len(sys.argv) > 1 else 36
    runs = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    n_explore = max(4, n_iters // 3)
    rng = random.Random(12345)

    logdir = os.path.join(_HERE, "experiments")
    os.makedirs(logdir, exist_ok=True)
    logpath = os.path.join(logdir, "log.jsonl")
    bestpath = os.path.join(logdir, "best.json")
    logf = open(logpath, "w")

    a = ev.astar_baseline()
    astar_total = sum(v["energy"] for v in a.values())
    print(f"[autoresearch] A* baseline 总能耗={astar_total:.1f}J | "
          f"n_iters={n_iters} runs={runs} explore={n_explore}", flush=True)

    best = None
    t_all = time.time()
    for i in range(n_iters):
        if i == 0:
            cand = DEFAULT; phase = "default"
        elif i < n_explore:
            cand = random_config(rng); phase = "explore"
        else:
            cand = perturb(best["config"], rng); phase = "exploit"

        t0 = time.time()
        res = ev.evaluate(to_overrides(cand), runs=runs)
        dt = time.time() - t0

        improved = best is None or res["score"] < best["score"]
        rec = {"iter": i, "phase": phase, "config": cand,
               "score": round(res["score"], 2), "energy_total": res["energy_total"],
               "vs_astar": round(res["vs_astar"], 4) if res["vs_astar"] else None,
               "min_success": res["min_success"], "eval_s": round(dt, 1),
               "improved": improved}
        logf.write(json.dumps(rec) + "\n"); logf.flush()

        if improved:
            best = {"config": cand, "score": res["score"], "res": res}
            json.dump({"config": cand, "result": res}, open(bestpath, "w"),
                      indent=2, default=float)

        et = res["energy_total"]
        et_s = f"{et:.0f}J" if et else "FAIL"
        flag = "  <== NEW BEST" if improved else ""
        vs = res["vs_astar"]
        print(f"[{i:02d}/{n_iters}] {phase:8s} score={res['score']:9.0f} "
              f"E={et_s:>7s} vsA*={vs:.3f} succ={res['min_success']:.0%} "
              f"({dt:.0f}s){flag}", flush=True)

    print(f"\n[done] {(time.time()-t_all)/60:.1f}min  best score={best['score']:.0f} "
          f"energy={best['res']['energy_total']} vsA*={best['res']['vs_astar']:.3f}", flush=True)
    print(f"[best config] {json.dumps(best['config'])}", flush=True)
    logf.close()


if __name__ == "__main__":
    main()
