"""
physics_eval.py — Phase A: 评测器物理保真升级(不改旧 evaluator.py)

修的两个坑(见 LAYERS.md Layer-0):
  1. 地面硬约束:旧地图 z>0(NED 地下)是"自由空间",A* 靠钻地 -0.8m 取得"最优"。
     这里把 z >= ground_z 标为占据,禁止地下飞。
  2. (下一步)能量加权最优锚点替代最短-A*;独立碰撞复核;冻结 kinodynamic。

本文件先落地 (1) 地面约束 + 验证:看 A*/RRT*/RRT-Connect 在有地面后
是否不再钻地、路径与能耗如何变化(预期:实心墙 Row1/Row4 逼出"翻墙 vs 绕行"能量权衡)。
"""
import os
import sys
import copy

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import evaluator as ev
import benchmark_planning as bp
from mapping.esdf import ESDF

GROUND_Z = -0.5   # NED: 世界 z >= -0.5(海拔 <= 0.5m)标为地面/不可飞

_GCACHE = {}


def add_ground_plane(vg, ground_z: float = GROUND_Z) -> int:
    """把 world-z >= ground_z 的体素标为占据(地面)。返回新增占据数。"""
    origin_z = vg.config.origin[2]
    vs = vg.config.voxel_size
    nz = vg.config.grid_size[2]
    # world_z(iz) = origin_z + iz*vs ;求 world_z >= ground_z 的最小 iz
    iz0 = int(np.ceil((ground_z - origin_z) / vs))
    iz0 = max(0, min(nz, iz0))
    before = int(np.sum(vg.grid == 1))
    vg.grid[:, :, iz0:] = 1
    return int(np.sum(vg.grid == 1)) - before, iz0


def get_grounded_map():
    """构建带地面约束的地图 + ESDF(缓存)。复用 evaluator 的 BASE 配置。"""
    if "vg" not in _GCACHE:
        from planning.config import PlanningConfig
        cfg = PlanningConfig(**dict(ev.BASE))
        vg = bp.build_known_map(cfg)
        added, iz0 = add_ground_plane(vg, GROUND_Z)
        esdf = ESDF(vg); esdf.compute()
        em = ev._get_map()[2]  # 复用能量模型
        _GCACHE.update(vg=vg, esdf=esdf, em=em, added=added, iz0=iz0)
        print(f"[grounded map] 地面 z>={GROUND_Z} 起(iz>={iz0}),新增占据体素 {added}")
    return _GCACHE["vg"], _GCACHE["esdf"], _GCACHE["em"]


def energy_astar(vg, esdf, em, start, goal, velocity=2.0, safety_margin=1.0,
                 max_expand=400000):
    """
    能量加权 A*:边代价 = 该移动方向的 BEMT 能耗(恒速下只依赖方向,预计算 26 个)。
    返回 (path, total_energy)。这是"离散能量最优参考",替代最短-A* 作为锚点。
    """
    import heapq
    vs = vg.config.voxel_size
    si = tuple(vg.world_to_grid(start)); gi = tuple(vg.world_to_grid(goal))

    # 预计算 26 个方向的 (grid偏移, 位移距离, 能耗) —— 恒速下与位置无关
    dirs = []
    e_per_m = []
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                if dx == dy == dz == 0:
                    continue
                disp = np.array([dx, dy, dz], float) * vs
                d = np.linalg.norm(disp)
                e, _ = em.compute_energy_for_segment(np.zeros(3), disp, velocity)
                dirs.append((dx, dy, dz, e))
                e_per_m.append(e / d)
    e_per_m_min = min(e_per_m)   # 可采纳启发式的每米能耗下界

    def h(idx):
        return e_per_m_min * np.linalg.norm(np.array(gi) - np.array(idx)) * vs

    g = {si: 0.0}; came = {}; closed = set()
    openset = [(h(si), si)]
    expanded = 0
    while openset:
        _, cur = heapq.heappop(openset)
        if cur == gi:
            break
        if cur in closed:
            continue
        closed.add(cur); expanded += 1
        if expanded > max_expand:
            break
        for dx, dy, dz, e in dirs:
            nb = (cur[0]+dx, cur[1]+dy, cur[2]+dz)
            if nb in closed or not vg.is_valid_index(nb):
                continue
            if vg.grid[nb[0], nb[1], nb[2]] == 1:      # 占据(含地面/墙)
                continue
            if esdf.get_distance(vg.grid_to_world(nb)) < safety_margin:
                continue
            ng = g[cur] + e
            if nb not in g or ng < g[nb]:
                g[nb] = ng; came[nb] = cur
                heapq.heappush(openset, (ng + h(nb), nb))
    if gi not in came and gi != si:
        return None, None
    # 回溯
    path = [np.array(vg.grid_to_world(gi))]
    c = gi
    while c in came:
        c = came[c]; path.append(np.array(vg.grid_to_world(c)))
    path.reverse()
    return path, g.get(gi)


def optimal_cruise_speed(em, vmin=0.5, vmax=25.0, n=250):
    """能量最优巡航速度 v*(每米能耗 U 形曲线最低点)。"""
    vsw = np.linspace(vmin, vmax, n)
    epm = [em.compute_electrical_power(np.array([v, 0., 0.])) / v for v in vsw]
    return float(vsw[int(np.argmin(epm))]), float(min(epm))


def energy_with_profile(path, em, v_star, a_lat=3.0, speed_fn=None, accel=True):
    """
    速度剖面能量。转弯按侧向加速度上限限速(kinodynamic 可行性,冻结约束)。
    [S3a 解冻速度] speed_fn 给定时,用 agent 进化的速度剖面,但**裁到 [0.5, vcap]**(可行性冻结,防作弊:
    不许超转弯上限);accel=True 时对**提速计入动能代价**(减速不回收)→ 频繁变速费能,平滑速度剖面才省。
    仍全用冻结 BEMT 计价。
    """
    if not path or len(path) < 2:
        return None
    P = [np.asarray(p, float) for p in path]
    seg = [P[i+1] - P[i] for i in range(len(P)-1)]
    L = [float(np.linalg.norm(s)) for s in seg]
    d = [seg[i]/L[i] if L[i] > 1e-9 else seg[i] for i in range(len(seg))]
    vcap = [v_star] * len(seg)
    for i in range(1, len(seg)):
        theta = np.arccos(np.clip(np.dot(d[i-1], d[i]), -1, 1))
        R = max(0.3, min(L[i-1], L[i]) / max(theta, 1e-3))
        vturn = min(v_star, np.sqrt(a_lat * R))
        vcap[i-1] = min(vcap[i-1], vturn); vcap[i] = min(vcap[i], vturn)
    # 速度剖面:默认走可行上限;agent 进化的 speed_fn 被裁到 [0.5, vcap](不许超上限=不许作弊)
    if speed_fn is not None:
        try:
            sp = speed_fn([p.copy() for p in P], float(v_star), list(vcap))
            speeds = [float(np.clip(sp[i], 0.5, vcap[i])) for i in range(len(seg))]
        except Exception:
            speeds = [max(0.5, v) for v in vcap]
    else:
        speeds = [max(0.5, v) for v in vcap]
    E = 0.0
    for i in range(len(seg)):
        e, _ = em.compute_energy_for_segment(P[i], P[i+1], speeds[i])
        E += e
    if accel:   # 提速的动能代价(从静止起飞;减速不回收)→ 鼓励平滑速度剖面
        m = em.params.mass; eff = max(0.1, em.params.motor_efficiency * em.params.esc_efficiency)
        v_prev = 0.0
        for v in speeds:
            if v > v_prev:
                E += 0.5 * m * (v**2 - v_prev**2) / eff
            v_prev = v
    return E


def gen_scenarios(seeds):
    """
    [S3b] 程序化场景生成器(固定、agent 不可改 = 出题权在生成器,不作弊)。
    每个 seed 生成一对 起点(左)/终点(右) —— 随机 y 与高度,强制跨越 Blocks 墙群、无碰、A* 可解。
    返回 [{name, start, goal}]。seeds 固定 → 可复现;train/test 用不同 seed 池 → 留出泛化测试。
    """
    from planning.config import PlanningConfig
    vg, esdf, em = get_grounded_map()
    cfg = PlanningConfig(**dict(ev.BASE, planning_timeout=15.0))
    out = []
    for sd in seeds:
        rng = np.random.default_rng(int(sd))
        for _ in range(60):
            s = np.array([rng.uniform(-5, 12), rng.uniform(-20, 20), rng.uniform(-4, -2)])
            g = np.array([rng.uniform(58, 72), rng.uniform(-20, 20), rng.uniform(-4, -2)])
            if esdf.get_distance(s) < 1.2 or esdf.get_distance(g) < 1.2:
                continue
            p = bp.AStarPlanner(vg, esdf, cfg).plan(s, g)
            if p and len(p) >= 2:
                out.append({"name": f"gen{sd}", "start": s, "goal": g})
                break
    return out


_SCEN = {}
def get_train():
    """训练场景 = 原 A/B/C + 3 个生成场景(loop 在这上面优化)。"""
    if "train" not in _SCEN:
        _SCEN["train"] = list(ev.SCENARIOS) + gen_scenarios([300, 301, 302])
    return _SCEN["train"]
def get_test():
    """留出测试场景 = 3 个不同 seed 的生成场景(只验证泛化,不参与优化)。"""
    if "test" not in _SCEN:
        _SCEN["test"] = gen_scenarios([400, 401, 402])
    return _SCEN["test"]


def _path_collision_free(path, esdf, safety_margin, step=0.25):
    """独立碰撞复核:沿每段密采样,任一点 ESDF < margin 即判碰(不信任 planner 自报成功)。"""
    for i in range(len(path) - 1):
        a = np.asarray(path[i], float); b = np.asarray(path[i+1], float)
        d = np.linalg.norm(b - a)
        if d < 1e-9:
            continue
        for t in np.arange(0.0, 1.0 + 1e-9, step / max(d, 1e-6)):
            if esdf.get_distance(a + t * (b - a)) < safety_margin:
                return False
    return True


def evaluate(overrides: dict, runs: int = 3, seed0: int = 0, smoother_src=None,
             sampler_src=None, speed_src=None, scenarios=None, verbose=False):
    """
    Phase A 诚实评测器:地面约束地图 + 速度剖面能量(转弯限速,v*) + 独立碰撞复核。
    仍冻结 BEMT/kinodynamic/尺子;overrides 只应含 Layer-1 可动键。
    返回 dict:score(越低越好), energy_total, vs_anchor, min_success, detail。
    """
    import random, contextlib
    from planning.config import PlanningConfig
    from planning.rrt_star import RRTStar
    import candidate as cand
    import sandbox

    vg, esdf, em = get_grounded_map()
    if "vstar" not in _GCACHE:
        _GCACHE["vstar"] = optimal_cruise_speed(em)[0]
    vstar = _GCACHE["vstar"]
    scs = scenarios or ev.SCENARIOS
    sm = ev.BASE["safety_margin"]

    ck = dict(ev.BASE); ck.update(overrides)
    ck.setdefault("energy_aware", True); ck.setdefault("flight_velocity", 2.0)

    def _bad(reason):
        return {"score": 9e9, "energy_total": None, "min_success": 0.0,
                "detail": {}, "vstar": vstar, "_bad": reason}

    # 代码候选:AST 沙箱 + 契约测试 + 崩溃/超时守卫(机械强制,不靠自觉)
    orig_smooth = RRTStar._smooth_path
    orig_rs = RRTStar._random_sample
    orig_ss = RRTStar._smart_sample
    orig_cb = RRTStar._compute_sampling_bounds
    has_code = (smoother_src is not None) or (sampler_src is not None) or (speed_src is not None)
    speed_fn = None
    if speed_src is not None:
        ok, reason = sandbox.check_code(speed_src)
        if not ok:
            return _bad(f"INVALID(speed):{reason}")
        try:
            speed_fn = cand.load_speed(speed_src); cand.contract_test_speed(speed_fn)
        except Exception as e:
            return _bad(f"CONTRACT(speed):{type(e).__name__}:{e}")

    if smoother_src is not None:
        ok, reason = sandbox.check_code(smoother_src)
        if not ok:
            return _bad(f"INVALID(smoother):{reason}")
        try:
            fn = cand.load_smoother(smoother_src); cand.contract_test(fn)
        except Exception as e:
            return _bad(f"CONTRACT(smoother):{type(e).__name__}:{e}")
        RRTStar._smooth_path = cand.make_patch_method(fn)

    if sampler_src is not None:
        ok, reason = sandbox.check_code(sampler_src)
        if not ok:
            return _bad(f"INVALID(sampler):{reason}")
        try:
            sfn = cand.load_sampler(sampler_src); cand.contract_test_sampler(sfn)
        except Exception as e:
            return _bad(f"CONTRACT(sampler):{type(e).__name__}:{e}")
        rs_patch = cand.sampler_random_patch(sfn)
        def _cb_wrap(self, s, g, _orig=orig_cb):
            self._samp_start = s; self._samp_goal = g
            return _orig(self, s, g)
        def _ss_patch(self, start, goal, nodes, iteration, _rs=rs_patch):
            self._samp_start = start; self._samp_goal = goal; self._samp_iter = iteration
            return _rs(self)
        RRTStar._random_sample = rs_patch
        RRTStar._smart_sample = _ss_patch
        RRTStar._compute_sampling_bounds = _cb_wrap

    detail = {}
    guard = sandbox.time_limit(300) if has_code else contextlib.nullcontext()
    try:
        with guard:
            for sc in scs:
                succ = 0; Es = []
                for r in range(runs):
                    random.seed(seed0 + r); np.random.seed(seed0 + r)
                    path = RRTStar(vg, esdf, PlanningConfig(**ck), energy_model=em).plan(sc["start"], sc["goal"])
                    if path and len(path) >= 2 and _path_collision_free(path, esdf, sm):
                        succ += 1
                        Es.append(energy_with_profile(path, em, vstar, speed_fn=speed_fn))
                sr = succ / runs
                detail[sc["name"]] = {"success": sr,
                                      "energy_mean": float(np.mean(Es)) if Es else None}
                if verbose:
                    print(f"    [{sc['name']}] succ={sr:.0%} E={detail[sc['name']]['energy_mean']}")
    except Exception as e:
        return _bad(f"CRASH:{type(e).__name__}:{e}")
    finally:
        RRTStar._smooth_path = orig_smooth
        RRTStar._random_sample = orig_rs
        RRTStar._smart_sample = orig_ss
        RRTStar._compute_sampling_bounds = orig_cb

    PENALTY = 3000.0
    contribs, all_ok, min_s = [], True, 1.0
    for sc in scs:
        dd = detail[sc["name"]]; min_s = min(min_s, dd["success"])
        if dd["success"] >= 1.0 and dd["energy_mean"] is not None:
            contribs.append(dd["energy_mean"])
        else:
            all_ok = False; contribs.append(PENALTY * (2.0 - dd["success"]))
    score = float(np.sum(contribs))
    return {"score": score,
            "energy_total": float(np.sum(contribs)) if all_ok else None,
            "min_success": min_s, "detail": detail, "vstar": vstar}


if __name__ == "__main__":
    import random
    from planning.config import PlanningConfig
    from planning.rrt_star import RRTStar

    S = np.array([0., 0., -3.]); G = np.array([70., 0., -3.])

    def cfg(**k):
        b = dict(ev.BASE)
        b.update(dict(step_size=1.5, max_iterations=5000, goal_sample_rate=0.4,
                      search_radius=4.0, dubins_turning_radius=1.5, weight_energy=0.6,
                      weight_distance=0.3, weight_time=0.1, planning_timeout=15.0,
                      energy_aware=True, flight_velocity=2.0))
        b.update(k); return PlanningConfig(**b)

    def stats(tag, path, em):
        if not path or len(path) < 2:
            print(f"  {tag:16s} 规划失败"); return
        P = np.array(path); alt = -P[:, 2]
        L = bp.compute_path_length(path)
        E, _ = em.compute_energy_for_path(path, velocity=2.0)
        under = "  ⚠️钻地!" if alt.min() < -0.05 else ""
        print(f"  {tag:16s} 长度={L:5.1f}m 能耗={E:6.0f}J 高度=[{alt.min():5.1f},{alt.max():5.1f}]m{under}")

    print("=" * 70)
    print("  Phase A 验证:加地面约束后,场景A 三方法轨迹(留出 seed=100)")
    print("=" * 70)
    vg, esdf, em = get_grounded_map()

    print("\n[A* 最优(能量?其实是最短)]")
    ap = bp.AStarPlanner(vg, esdf, cfg())
    pa = ap.plan(S, G)
    if pa:
        pa = bp.smooth_path(pa, esdf, 1.0)
    stats("A*", pa, em)

    random.seed(100); np.random.seed(100)
    stats("默认 RRT*", RRTStar(vg, esdf, cfg(), energy_model=em).plan(S, G), em)
    random.seed(100); np.random.seed(100)
    stats("RRT-Connect", RRTStar(vg, esdf, cfg(use_rrt_connect=True), energy_model=em).plan(S, G), em)

    print("\n=== 锚点对比:最短-A* vs 能量最优-A*(有地面)===")
    for name, s, gg in [("A 直穿", S, G),
                        ("B 对角上", np.array([0., 0., -3.]), np.array([70., 20., -3.])),
                        ("C 对角下", np.array([0., 0., -3.]), np.array([70., -25., -3.]))]:
        ap = bp.AStarPlanner(vg, esdf, cfg())
        ps = ap.plan(s, gg)
        if ps:
            ps = bp.smooth_path(ps, esdf, 1.0)
        es, _ = em.compute_energy_for_path(ps, velocity=2.0) if ps else (None, None)
        pe, _raw = energy_astar(vg, esdf, em, s, gg)
        pe_s = bp.smooth_path(pe, esdf, 1.0) if pe else None
        ee, _ = em.compute_energy_for_path(pe_s, velocity=2.0) if pe_s else (None, None)
        if es and ee:
            head = (es - ee) / es * 100
            altA = -np.array(ps)[:, 2]; altE = -np.array(pe_s)[:, 2]
            print(f"  {name}: 最短-A*={es:6.0f}J(高度≤{altA.max():.1f}m) | "
                  f"能量最优={ee:6.0f}J(高度≤{altE.max():.1f}m) | headroom={head:+.1f}%")
        else:
            print(f"  {name}: 规划失败(A*={es}, energyA*={ee})")
    print("\nheadroom>0 = 存在比最短更省能的路径 → agent 有真实优化空间(第②重天花板松动)")
