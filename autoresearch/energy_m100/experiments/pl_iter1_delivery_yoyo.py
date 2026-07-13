"""pl_iter1_delivery_yoyo.py — 规划层 iter1:H1 配送顺序防悠悠球(预注册实验)

═══ 预注册(先写判据,后跑实验)═══════════════════════════════════════════
假设 H1:多点屋顶配送、不等高链(高12m-谷2~4m-中6~10m)几何下,能量最优访问顺序
  会避开"降进低谷再重爬"的 yo-yo(距离序 = 最近邻 = 高→谷→中 会 yo-yo),
  因为爬升有真премium(~58-75 J/爬升米,held-out 验证的+114W 爬升盲机制)而
  多飞的水平米只花 32-124 J/m(随速度降)。
判据(SUPPORTED 需全部满足,且必须是规划器级证据):
  (1) 存在合理城市几何(楼高≤15m——本图受栅格 z∈[-15,5] 限制取≤12m,跨度≤80m,
      楼间距 d≥6m)使 能量最优顺序 ≠ 距离最优顺序(最近邻);
  (2) 用能量序相对距离序省能 ≥3%(M100Em 尺,payload=250g);
  (3) 规划器级(energy_astar 逐腿规划)复核后 (1)(2) 仍成立;
  (4) 翻转在扫描速度 4-12 m/s 内至少 2 个相邻速度点上持续(防单点侥幸)。
  找不到满足 (1)-(4) 的几何 → REFUTED(同样有价值,报翻转区边界)。
扫描全范围(不许只报翻转点):d∈{6,8,10,12,15,20,25}m,h_A=12,h_C∈{6,8,10},
  h_B∈{2,4},v∈{4,6,8,10,12} m/s;任务=开放路线(送完停在最后一点)为主判据,
  闭合巡回(回到基站)为次要报告项。
选 headline 几何规则(防 cherry-pick,选最保守而非最大省能):
  在 v=8 开放路线翻转且省能≥3% 的几何里选 d 最大(=城市最稀疏、最不刻意)者;
  若 v=8 无翻转,取有翻转的速度中最低者再按同规则选,并如实报"翻转仅在高速区"。
诚实边界:直线腿爬升角大时 v_z 超 M100 数据范围(|v_z|>3 m/s)= 模型外推,逐条标记;
  开放路线两顺序总下降相等(设计使然)→ 主判据不依赖"下降也费电"这一较弱模型行为;
  闭合巡回的省能里有下降项贡献,分解报告。
═══════════════════════════════════════════════════════════════════════

用法:.venv/bin/python experiments/pl_iter1_delivery_yoyo.py
输出:experiments/pl_iter1_delivery_yoyo.json(结论必须能从此 JSON 复现)
"""
import os, sys, copy, json, time, itertools

HERE = os.path.dirname(os.path.abspath(__file__))
EM100 = os.path.dirname(HERE)                       # energy_m100/
AR = os.path.dirname(EM100)                         # autoresearch/
sys.path.insert(0, AR)
sys.path.insert(0, EM100)
import numpy as np
import physics_eval as pe
from planning_experiment import M100Em, DistanceEm, score_path

V_SWEEP = [4, 6, 8, 10, 12]
D_SWEEP = [6, 8, 10, 12, 15, 20, 25]
HC_SWEEP = [6, 8, 10]
HB_SWEEP = [2, 4]
H_A = 12.0
X1 = 10.0            # 第一栋楼中心 x
CLEAR = 1.5          # 配送悬停点 = 屋顶上方 1.5m
START_ALT = 2.0      # 基站悬停高度
VZ_DATA_MAX = 3.0    # M100 数据内 |v_z| 上界,超出=外推
SAFETY = 0.6         # 与 wall/corridor 实验一致
PLANNER_V = [4, 8, 12]   # 规划器级复核速度(端点+真实巡航)

ORDERS = list(itertools.permutations("ABC"))


def waypoints(d, h_b, h_c):
    """S + 三栋楼配送点(NED,z=-alt)。A 高楼在前,B 低谷居中,C 中楼在后。"""
    return {
        "S": np.array([0.0, 0.0, -START_ALT]),
        "A": np.array([X1, 0.0, -(H_A + CLEAR)]),
        "B": np.array([X1 + d, 0.0, -(h_b + CLEAR)]),
        "C": np.array([X1 + 2 * d, 0.0, -(h_c + CLEAR)]),
    }


def leg_energy_straight(em, p, q, v):
    """候选层:直线腿能量 + v_z 外推标记。"""
    disp = q - p
    e, t = em.compute_energy_for_segment(np.zeros(3), disp, v)
    vz = abs(float(-disp[2]) / max(float(np.linalg.norm(disp)) / v, 1e-9))
    return e, bool(vz > VZ_DATA_MAX)


def tour_energy_straight(em, wps, order, v, closed):
    seq = ["S"] + list(order) + (["S"] if closed else [])
    E, extrap = 0.0, False
    for a, b in zip(seq[:-1], seq[1:]):
        e, ex = leg_energy_straight(em, wps[a], wps[b], v)
        E += e; extrap = extrap or ex
    return E, extrap


def tour_length(wps, order, closed):
    seq = ["S"] + list(order) + (["S"] if closed else [])
    return sum(float(np.linalg.norm(wps[b] - wps[a])) for a, b in zip(seq[:-1], seq[1:]))


def nn_order(wps, keys=("A", "B", "C")):
    """距离 baseline:最近邻(物流标准启发式),从 S 出发,3D 直线距离。"""
    left, cur, out = list(keys), "S", []
    while left:
        nxt = min(left, key=lambda k: float(np.linalg.norm(wps[k] - wps[cur])))
        out.append(nxt); left.remove(nxt); cur = nxt
    return tuple(out)


def stage1_scan(em):
    rows = []
    for d, h_b, h_c, v in itertools.product(D_SWEEP, HB_SWEEP, HC_SWEEP, V_SWEEP):
        wps = waypoints(d, h_b, h_c)
        rec = {"d": d, "h_B": h_b, "h_C": h_c, "v": v}
        nn = nn_order(wps)
        rec["dist_order_NN"] = "".join(nn)
        for closed in (False, True):
            tag = "closed" if closed else "open"
            E = {"".join(o): tour_energy_straight(em, wps, o, v, closed) for o in ORDERS}
            L = {"".join(o): tour_length(wps, o, closed) for o in ORDERS}
            e_order = min(E, key=lambda k: E[k][0])
            l_order = min(L, key=lambda k: L[k])
            e_nn, e_best = E["".join(nn)][0], E[e_order][0]
            rec[f"{tag}_E_orders"] = {k: round(val[0], 1) for k, val in E.items()}
            rec[f"{tag}_len_min_order"] = l_order
            rec[f"{tag}_energy_order"] = e_order
            rec[f"{tag}_flip"] = e_order != "".join(nn)
            rec[f"{tag}_savings_pct"] = round(100 * (e_nn - e_best) / e_nn, 2)
            rec[f"{tag}_extrap"] = E[e_order][1] or E["".join(nn)][1]
        rows.append(rec)
    return rows


# ───────────────────────── Stage 2: 规划器级复核 ─────────────────────────

def city_map(d, h_b, h_c):
    """干净地图:地面 + 三根 3×3m 柱楼(A=12m, B=h_b, C=h_c)。"""
    vg0, esdf0, _ = pe.get_grounded_map()
    vg = copy.deepcopy(vg0); vg.grid[:] = 0
    for iz in range(vg.grid.shape[2]):
        if vg.grid_to_world((0, 0, iz))[2] > -0.6:
            vg.grid[:, :, iz] = 1
    for xc, h in ((X1, H_A), (X1 + d, h_b), (X1 + 2 * d, h_c)):
        for xw in np.arange(xc - 1.5, xc + 1.6, .5):
            for yw in np.arange(-1.5, 1.6, .5):
                for zw in np.arange(-h, 0.1, .5):
                    idx = vg.world_to_grid(np.array([xw, yw, zw]))
                    if vg.is_valid_index(idx):
                        vg.grid[idx] = 1
    esdf = type(esdf0)(vg); esdf.compute()
    return vg, esdf


def plan_leg(vg, esdf, em, p, q, v):
    path, cost = pe.energy_astar(vg, esdf, em, p, q, velocity=v,
                                 safety_margin=SAFETY, max_expand=1500000)
    return path, cost


def path_vz_extrap(path, v):
    for a, b in zip(path[:-1], path[1:]):
        seg = np.asarray(b, float) - np.asarray(a, float)
        L = float(np.linalg.norm(seg))
        if L < 1e-9:
            continue
        if abs(-seg[2]) / (L / v) > VZ_DATA_MAX:
            return True
    return False


def stage2_verify(em, dist_em, d, h_b, h_c):
    wps = waypoints(d, h_b, h_c)
    vg, esdf = city_map(d, h_b, h_c)
    for k, w in wps.items():
        assert esdf.get_distance(w) >= SAFETY, f"waypoint {k} 不可达(ESDF<{SAFETY})"
    keys = list(wps)
    out = {"geometry": {"d": d, "h_A": H_A, "h_B": h_b, "h_C": h_c,
                        "span_m": X1 + 2 * d + 1.5, "safety_margin": SAFETY}}

    # 距离规划器:无向腿一次(代价=长度,与速度无关)
    dist_paths, t0 = {}, time.time()
    for a, b in itertools.combinations(keys, 2):
        p, c = plan_leg(vg, esdf, dist_em, wps[a], wps[b], 8.0)
        assert p, f"距离腿 {a}->{b} 规划失败"
        dist_paths[(a, b)] = p
    print(f"  距离腿 6 条规划完 {time.time()-t0:.0f}s")

    def dpath(a, b):
        return dist_paths[(a, b)] if (a, b) in dist_paths else dist_paths[(b, a)][::-1]

    per_v = []
    for v in PLANNER_V:
        t0 = time.time()
        epaths = {}
        for a in keys:
            for b in keys:
                if a != b and not (b == "S" and a == "S"):
                    epaths[(a, b)] = plan_leg(vg, esdf, em, wps[a], wps[b], v)
        rec = {"v": v}
        for closed in (False, True):
            tag = "closed" if closed else "open"
            seqs = {"".join(o): ["S"] + list(o) + (["S"] if closed else []) for o in ORDERS}
            # 距离系统:最近邻序(规划腿长度) + 最短路径腿;能量用 M100 尺评
            nn = []
            left, cur = [k for k in keys if k != "S"], "S"
            while left:
                nxt = min(left, key=lambda k: sum(
                    float(np.linalg.norm(np.asarray(q, float) - np.asarray(p, float)))
                    for p, q in zip(dpath(cur, k)[:-1], dpath(cur, k)[1:])))
                nn.append(nxt); left.remove(nxt); cur = nxt
            nn = tuple(nn)
            seq_nn = seqs["".join(nn)]
            E_dist_sys = sum(score_path(dpath(a, b), em, v)
                             for a, b in zip(seq_nn[:-1], seq_nn[1:]))
            # 能量系统:能量规划腿,枚举全部顺序取最优
            E_orders = {}
            extrap = {}
            for name, seq in seqs.items():
                tot, ex = 0.0, False
                for a, b in zip(seq[:-1], seq[1:]):
                    path, cost = epaths[(a, b)]
                    assert path, f"能量腿 {a}->{b} v={v} 规划失败"
                    tot += cost
                    ex = ex or path_vz_extrap(path, v)
                E_orders[name] = tot; extrap[name] = ex
            e_order = min(E_orders, key=E_orders.get)
            rec[f"{tag}_dist_order_NN"] = "".join(nn)
            rec[f"{tag}_energy_order"] = e_order
            rec[f"{tag}_flip"] = e_order != "".join(nn)
            rec[f"{tag}_E_dist_system"] = round(E_dist_sys, 1)
            rec[f"{tag}_E_energy_system"] = round(E_orders[e_order], 1)
            rec[f"{tag}_savings_pct"] = round(
                100 * (E_dist_sys - E_orders[e_order]) / E_dist_sys, 2)
            # 纯顺序效应分解:同一批能量规划腿,只换顺序
            rec[f"{tag}_order_only_pct"] = round(
                100 * (E_orders["".join(nn)] - E_orders[e_order]) / E_orders["".join(nn)], 2)
            rec[f"{tag}_E_orders"] = {k: round(x, 1) for k, x in E_orders.items()}
            rec[f"{tag}_extrap"] = extrap[e_order] or extrap["".join(nn)]
        per_v.append(rec)
        print(f"  v={v}: open flip={rec['open_flip']} 省{rec['open_savings_pct']}% "
              f"(纯顺序 {rec['open_order_only_pct']}%) | closed flip={rec['closed_flip']} "
              f"省{rec['closed_savings_pct']}% [{time.time()-t0:.0f}s]")
    out["per_v"] = per_v
    return out


def main():
    src = open(os.path.join(EM100, "state", "best_featurize.py")).read()
    em = M100Em(src, payload=250.0, name="M100")
    dist_em = DistanceEm()

    # Stage 0: 机制刻画(归因用)
    char = {"level_J_per_m": {}, "climb_premium_J_per_m@v8": {}, "desc_premium_J_per_m@v8": {}}
    for v in V_SWEEP:
        e, t = em.compute_energy_for_segment(np.zeros(3), np.array([100., 0, 0]), v)
        char["level_J_per_m"][v] = round(e / 100, 1)
    e0, _ = em.compute_energy_for_segment(np.zeros(3), np.array([10., 0, 0]), 8)
    for dz in (2, 4, 6, 8):
        eu, _ = em.compute_energy_for_segment(np.zeros(3), np.array([10., 0, -dz]), 8)
        ed, _ = em.compute_energy_for_segment(np.zeros(3), np.array([10., 0, dz]), 8)
        char["climb_premium_J_per_m@v8"][dz] = round((eu - e0) / dz, 1)
        char["desc_premium_J_per_m@v8"][dz] = round((ed - e0) / dz, 1)

    print("Stage 1: 候选层全扫描(直线腿)...")
    rows = stage1_scan(em)
    flips = [r for r in rows if r["open_flip"]]
    print(f"  {len(rows)} 配置,open 翻转 {len(flips)} 个")

    # headline 选择(预注册规则:v=8 翻转且≥3% 里 d 最大;否则最低翻转速度)
    pick, pick_rule = None, ""
    cand8 = [r for r in flips if r["v"] == 8 and r["open_savings_pct"] >= 3.0]
    if cand8:
        pick = max(cand8, key=lambda r: r["d"])
        pick_rule = "v=8 开放路线翻转且省能≥3% 中 d 最大"
    else:
        for v in V_SWEEP:
            cv = [r for r in flips if r["v"] == v and r["open_savings_pct"] >= 3.0]
            if cv:
                pick = max(cv, key=lambda r: r["d"])
                pick_rule = f"v=8 无 →取最低翻转速度 v={v} 中 d 最大(翻转仅在该速度区)"
                break
    result = {"preregistered": {
                  "criteria": "flip(open,NN基线) & savings>=3% & planner-level & >=2 adjacent speeds",
                  "scan": {"d": D_SWEEP, "h_A": H_A, "h_B": HB_SWEEP, "h_C": HC_SWEEP,
                           "v": V_SWEEP, "planner_v": PLANNER_V},
                  "headline_rule": "v=8翻转且>=3%中d最大, 否则最低翻转速度"},
              "model_characterization": char,
              "stage1_all_rows": rows}

    if pick is None:
        result["verdict_hint"] = "候选层无 flip+3% → REFUTED(无需规划器级)"
        print("候选层无满足判据的几何 → REFUTED")
    else:
        print(f"Stage 2: 规划器级复核 headline 几何 d={pick['d']} h_B={pick['h_B']} "
              f"h_C={pick['h_C']}({pick_rule})...")
        result["headline_pick"] = {k: pick[k] for k in ("d", "h_B", "h_C", "v",
                                   "open_savings_pct", "dist_order_NN", "open_energy_order")}
        result["headline_rule_applied"] = pick_rule
        result["stage2_planner"] = stage2_verify(em, dist_em, pick["d"], pick["h_B"], pick["h_C"])

    with open(os.path.join(HERE, "pl_iter1_delivery_yoyo.json"), "w") as f:
        json.dump(result, f, ensure_ascii=False, indent=1,
                  default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print("JSON 落盘 experiments/pl_iter1_delivery_yoyo.json")


if __name__ == "__main__":
    main()
