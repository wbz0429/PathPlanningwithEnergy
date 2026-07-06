"""
autoresearch 固定评测器 (相当于 karpathy autoresearch 里不可改的 prepare.py)

职责：
- 一次性构建已知地图 + ESDF + A* 最优 baseline（缓存复用）
- 给定一组参数 overrides，跑 N 次 RRT* 规划，聚合成"单一可比指标"
- 评测可复现：每次规划前固定随机种子

不依赖 AirSim。指标 = 在"全部场景 100% 成功"约束下的平均能耗(J)。
"""
import os, sys, time, random
os.environ.setdefault("MPLBACKEND", "Agg")
_HERE = os.path.dirname(os.path.abspath(__file__))
_DRONE = os.path.join(os.path.dirname(_HERE), "drone_sim")
sys.path.insert(0, _DRONE)

import numpy as np
from planning.config import PlanningConfig
from mapping.esdf import ESDF
from planning.rrt_star import RRTStar
from energy.physics_model import PhysicsEnergyModel
import benchmark_planning as bp

FLIGHT_VELOCITY = 2.0

# 评测场景（沿用仓库 benchmark_report.md 的三场景，保证可比 & 可信）
SCENARIOS = [
    {"name": "A_straight",  "start": np.array([0., 0., -3.]),  "goal": np.array([70., 0., -3.])},
    {"name": "B_diag_up",   "start": np.array([0., 0., -3.]),  "goal": np.array([70., 20., -3.])},
    {"name": "C_diag_down", "start": np.array([0., 0., -3.]),  "goal": np.array([70., -25., -3.])},
]

# 评测器固定的"基线配置"——地图相关参数固定，不参与搜索
BASE = dict(
    voxel_size=0.5, grid_size=(180, 120, 40), origin=(-10.0, -30.0, -15.0),
    max_depth=25.0, safety_margin=1.0,
)

_CACHE = {}


def _get_map():
    """构建地图+ESDF（一次性缓存）"""
    if "vg" not in _CACHE:
        cfg = PlanningConfig(**BASE)
        vg = bp.build_known_map(cfg)
        esdf = ESDF(vg); esdf.compute()
        em = PhysicsEnergyModel()
        _CACHE["vg"], _CACHE["esdf"], _CACHE["em"] = vg, esdf, em
    return _CACHE["vg"], _CACHE["esdf"], _CACHE["em"]


def astar_baseline():
    """A* 最优 baseline（确定性，缓存）"""
    if "astar" in _CACHE:
        return _CACHE["astar"]
    vg, esdf, em = _get_map()
    cfg = PlanningConfig(**BASE)
    out = {}
    for sc in SCENARIOS:
        planner = bp.AStarPlanner(vg, esdf, cfg)
        path = planner.plan(sc["start"], sc["goal"])
        if path:
            path = bp.smooth_path(path, esdf, cfg.safety_margin)
            L = bp.compute_path_length(path)
            E, T = em.compute_energy_for_path(path, velocity=FLIGHT_VELOCITY)
            out[sc["name"]] = {"length": L, "energy": E, "time": T}
    _CACHE["astar"] = out
    return out


def evaluate(overrides: dict, runs: int = 3, scenarios=None, seed0: int = 0, verbose=False):
    """
    评测一组参数。返回单一可比指标 + 明细。

    overrides: 覆盖 PlanningConfig 的参数（搜索空间）
    返回 dict: score(越低越好), energy_mean, vs_astar, min_success, compute_s, detail
    """
    vg, esdf, em = _get_map()
    astar = astar_baseline()
    scs = scenarios or SCENARIOS

    cfg_kwargs = dict(BASE)
    cfg_kwargs.update(overrides)
    cfg_kwargs.setdefault("energy_aware", True)
    cfg_kwargs.setdefault("flight_velocity", FLIGHT_VELOCITY)

    detail = {}
    energies_per_sc = []
    vs_astar_list = []
    min_success = 1.0
    t_start = time.time()

    for sc in scs:
        cfg = PlanningConfig(**cfg_kwargs)
        succ = 0; Ls = []; Es = []; cts = []
        for r in range(runs):
            random.seed(seed0 + r); np.random.seed(seed0 + r)
            planner = RRTStar(vg, esdf, cfg, energy_model=em)
            t0 = time.time()
            path = planner.plan(sc["start"], sc["goal"])
            cts.append(time.time() - t0)
            if path and len(path) >= 2:
                succ += 1
                Ls.append(bp.compute_path_length(path))
                E, _ = em.compute_energy_for_path(path, velocity=FLIGHT_VELOCITY)
                Es.append(E)
        sr = succ / runs
        min_success = min(min_success, sr)
        if Es:
            e_mean = float(np.mean(Es))
            energies_per_sc.append(e_mean)
            if sc["name"] in astar:
                vs_astar_list.append(e_mean / astar[sc["name"]]["energy"])
            detail[sc["name"]] = {
                "success": sr, "energy_mean": e_mean, "energy_std": float(np.std(Es)),
                "length_mean": float(np.mean(Ls)), "compute_s": float(np.mean(cts)),
            }
        else:
            detail[sc["name"]] = {"success": sr, "energy_mean": None, "compute_s": float(np.mean(cts))}
        if verbose:
            print(f"    [{sc['name']}] success={sr:.0%} "
                  f"energy={detail[sc['name']].get('energy_mean')} "
                  f"compute={np.mean(cts):.2f}s")

    # 单一指标（越低越好，软约束）：
    #   每个场景：100% 成功 -> 贡献其平均能耗(J)；否则 -> PENALTY*(2-成功率)，
    #   PENALTY 远大于真实能耗，使搜索先把成功率推到 100%，再压能耗。
    PENALTY = 10000.0
    contribs = []
    all_success = True
    for sc in scs:
        d = detail[sc["name"]]
        if d["success"] >= 1.0 and d.get("energy_mean") is not None:
            contribs.append(d["energy_mean"])
        else:
            all_success = False
            contribs.append(PENALTY * (2.0 - d["success"]))
    score = float(np.sum(contribs))
    energy_total = float(np.sum(energies_per_sc)) if all_success and len(energies_per_sc) == len(scs) else None

    return {
        "score": score,
        "energy_total": energy_total,
        "energy_mean_per_sc": float(np.mean(energies_per_sc)) if energies_per_sc else None,
        "vs_astar": float(np.mean(vs_astar_list)) if vs_astar_list else None,
        "min_success": min_success,
        "compute_s": time.time() - t_start,
        "detail": detail,
    }


if __name__ == "__main__":
    print("=" * 64)
    print("  评测器标定：A* baseline + 默认 RRT* 配置")
    print("=" * 64)
    a = astar_baseline()
    print("[A* baseline]")
    for k, v in a.items():
        print(f"  {k}: len={v['length']:.1f}m energy={v['energy']:.1f}J")

    default_overrides = dict(
        step_size=1.5, max_iterations=5000, goal_sample_rate=0.4,
        search_radius=4.0, planning_timeout=15.0,
        weight_energy=0.6, weight_distance=0.3, weight_time=0.1,
        dubins_turning_radius=1.5,
    )
    print(f"\n[默认配置评测] runs=3, timeout=15s ...")
    res = evaluate(default_overrides, runs=3, verbose=True)
    print(f"\n[结果] score(总能耗)={res['energy_total']} "
          f"vs_astar={res['vs_astar']} min_success={res['min_success']:.0%} "
          f"用时={res['compute_s']:.1f}s")
