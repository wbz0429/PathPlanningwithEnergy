"""stage-2 评测：默认参数 + 留出种子，量化代码改写的效果。"""
import os, sys, json
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import evaluator as ev

DEFAULT = dict(step_size=1.5, max_iterations=5000, goal_sample_rate=0.4, search_radius=4.0,
               dubins_turning_radius=1.5, weight_energy=0.6, weight_distance=0.3,
               weight_time=0.1, planning_timeout=10.0, energy_aware=True)
runs = int(sys.argv[1]) if len(sys.argv) > 1 else 10

a = ev.astar_baseline(); at = sum(v["energy"] for v in a.values())
print(f"[A* total] {at:.1f}J  (held-out seed0=100, runs={runs})")
r = ev.evaluate(DEFAULT, runs=runs, seed0=100)
print(f"[default + 当前代码] score={r['score']:.0f} energy_total={r['energy_total']} "
      f"vsA*={r['vs_astar']} min_succ={r['min_success']:.0%}")
for n, d in r["detail"].items():
    print(f"    {n:12s} succ={d['success']:.0%} energy={d.get('energy_mean')} "
          f"compute={d.get('compute_s'):.1f}s")
