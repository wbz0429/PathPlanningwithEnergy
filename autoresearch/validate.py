"""
留出验证：用「未参与搜索的随机种子 + 更多 runs」复核 best 配置，
对比默认配置，防止过拟合。读取 experiments/best.json。

用法： python validate.py [runs]
"""
import os, sys, json
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import evaluator as ev

DEFAULT = dict(step_size=1.5, max_iterations=5000, goal_sample_rate=0.4,
               search_radius=4.0, dubins_turning_radius=1.5,
               weight_energy=0.6, weight_distance=0.3, weight_time=0.1,
               planning_timeout=10.0, energy_aware=True)
HELD_OUT_SEED = 100  # 搜索用 seed0=0；这里用 100，保证留出


def show(tag, res):
    print(f"\n[{tag}] score={res['score']:.0f}  energy_total="
          f"{res['energy_total']}  vsA*={res['vs_astar']}  min_succ={res['min_success']:.0%}")
    for name, d in res["detail"].items():
        print(f"    {name:12s} succ={d['success']:.0%} "
              f"energy={d.get('energy_mean')} compute={d.get('compute_s'):.1f}s")


def main():
    runs = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    a = ev.astar_baseline()
    astar_total = sum(v["energy"] for v in a.values())
    print(f"[A* baseline 总能耗] {astar_total:.1f}J  (held-out seed0={HELD_OUT_SEED}, runs={runs})")

    best = json.load(open(os.path.join(_HERE, "experiments", "best.json")))["config"]
    best_ov = dict(best); best_ov.setdefault("planning_timeout", 10.0)

    print("\n=== 默认配置（留出复核）===")
    show("default", ev.evaluate(DEFAULT, runs=runs, seed0=HELD_OUT_SEED))

    print("\n=== autoresearch best（留出复核）===")
    show("best", ev.evaluate(best_ov, runs=runs, seed0=HELD_OUT_SEED))
    print(f"\n[best config] {json.dumps(best, ensure_ascii=False)}")


if __name__ == "__main__":
    main()
