"""
robustness_check.py — "优化是否有意义"的鲁棒性门(construct validity)

问题:在我们评测器上赢 ≠ 真有意义(可能在钻 BEMT 的空子,如"翻墙"依赖降落近免费的 v* 假设)。
方法:把 loop-best 与 RRT-Connect 基线的**同一批路径**,在**扰动的巡航速度 v***(不让 agent 改物理,
     只是我们做验证)下重算剖面能耗,看 loop 的优势是否跨扰动稳健。
     - 优势跨所有 v* 都在 → 真改进(稳健)。
     - 某些 v* 下优势消失/反转 → 该 gain 对模型假设敏感(可能是 artifact),答辩需说明。

用法: python robustness_check.py
"""
import os, sys, json, random
os.environ.setdefault("MPLBACKEND", "Agg")
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import numpy as np
import physics_eval as pe, evaluator as ev, benchmark_planning as bp, candidate as cand
from planning.config import PlanningConfig
from planning.rrt_star import RRTStar

TUNED = json.load(open(os.path.join(_HERE, "state", "best.json")))["config"]
SM = open(os.path.join(_HERE, "state", "best_smoother.py")).read()
DEF = dict(step_size=1.5, max_iterations=5000, goal_sample_rate=0.4, search_radius=4.0,
           weight_energy=0.6, weight_distance=0.3, weight_time=0.1, use_rrt_connect=True)


def _plan(ov, smoother, s, g, seed=0):
    vg, esdf, em = pe.get_grounded_map()
    orig = RRTStar._smooth_path
    try:
        if smoother:
            RRTStar._smooth_path = cand.make_patch_method(cand.load_smoother(smoother))
        b = dict(ev.BASE); b.update(dict(dubins_turning_radius=1.5, planning_timeout=15.0,
                 energy_aware=True, flight_velocity=2.0)); b.update(ov)
        random.seed(seed); np.random.seed(seed)
        return RRTStar(vg, esdf, PlanningConfig(**b), energy_model=em).plan(s, g)
    finally:
        RRTStar._smooth_path = orig


if __name__ == "__main__":
    vg, esdf, em = pe.get_grounded_map()
    vstar = pe._GCACHE.get("vstar") or pe.optimal_cruise_speed(em)[0]
    loop_paths, conn_paths = [], []
    for sc in ev.SCENARIOS:
        loop_paths.append(_plan(TUNED, SM, sc["start"], sc["goal"]))
        conn_paths.append(_plan(DEF, None, sc["start"], sc["goal"]))

    print("=" * 60)
    print(f"  鲁棒性门:loop-best vs RRT-Connect,扰动巡航速度 v*(标定 {vstar:.1f} m/s)")
    print("=" * 60)
    robust = True
    for fac in (0.6, 0.8, 1.0, 1.25, 1.5):
        vt = vstar * fac
        lE = sum(pe.energy_with_profile(p, em, vt) if p else 3000 for p in loop_paths)
        cE = sum(pe.energy_with_profile(p, em, vt) if p else 3000 for p in conn_paths)
        adv = (cE - lE) / cE * 100
        ok = lE < cE
        robust = robust and ok
        print(f"  v*={vt:5.1f} ({fac:.2f}x): loop={lE:7.0f}  RRT-Connect={cE:7.0f}  "
              f"loop优势={adv:+5.1f}%  {'✓' if ok else '✗ 优势消失!'}")
    print("\n结论:", "✅ loop 优势跨所有扰动稳健 → 真改进,非模型 artifact"
          if robust else "⚠️ 某些扰动下优势消失 → 该 gain 对模型假设敏感,需在报告中标注")
