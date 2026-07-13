"""pl_iter4_gravity_glide.py — 规划层 iter4:H4 借重力(预注册实验)

═══ 预注册(先写判据,后跑实验)═══════════════════════════════════════════
假设 H4:出发点高于目标(12m→2m,降 10m)时,能量最优垂直剖面显著异于直线插值
  (存在"陡滑翔+平飞"两相结构或最优下沉率),且省能 ≥2%(种子判据)。
先验张力(诚实声明):iter2 探针显示 v=8 时下降相对平飞有 +20~38 J/m 溢价
  (冻结模型凸于 v_z)→ 若全速度域皆凸且极小在 v_z≈0,则直线缓坡(最浅下降)最优,
  H4 应 REFUTED,机制="真机数据在巡航速度下没有滑翔甜点"。本实验全域刻画后裁决。
设置:自由空间(无障碍,不用 A*——26 连通网格把坡度量化到 45°,是离散化伪影不是证据;
  验证层用 RotorPy 动力学,符合证据标准"规划器或动力学")。H=10m 固定;
  D∈{20,40,60,80}m;v∈{4,8,12} m/s;剖面族 = 直线 ramp(基线)vs 两相滑翔
  (θ∈{5,10,15,20,30,45,60}°,glide-first / glide-last 两向);尺=冻结 M100Em(250g)。
判据:
  (1) SUPPORTED 当且仅当:候选层存在 (D,v) 使最优两相剖面比 ramp 省 ≥2%,
      且该赢家经 RotorPy 动力学实飞(同 v_avg,MinSnap+SE3,飞出运动学喂冻结尺)
      优势符号保持且 ≥2%,且赢家 |v_z| ≤ 3 m/s(数据内;超出→降级 INCONCLUSIVE 并标外推);
  (2) REFUTED 当候选层全网格无剖面胜 ramp ≥2%(尺不承认滑翔结构,无需动力学),
      或候选层赢但动力学杀掉;
  (3) 附带报告(不进判据):P(v_h,v_z) 曲面 + 每个 v_h 的最优下沉率 v_z*;
      early vs late 在分段可加尺下的简并性(数值验证);动力学下 early vs late 差
      (加速度/平滑效应,描述性)。全网格结果整体报告。
═══════════════════════════════════════════════════════════════════════

用法:.venv/bin/python experiments/pl_iter4_gravity_glide.py
输出:experiments/pl_iter4_gravity_glide.json
"""
import os, sys, json, math

HERE = os.path.dirname(os.path.abspath(__file__))
EM100 = os.path.dirname(HERE)
sys.path.insert(0, os.path.dirname(EM100))
sys.path.insert(0, EM100)
import numpy as np
from planning_experiment import M100Em, score_path
from planning_connect import PowerModel

H_DROP = 10.0
Z_HI, Z_LO = -12.0, -2.0
D_SWEEP = [20., 40., 60., 80.]
V_SWEEP = [4., 8., 12.]
THETAS = [5., 10., 15., 20., 30., 45., 60.]
VZ_DATA_MAX = 3.0


def ramp(D):
    return [np.array([0., 0., Z_HI]), np.array([D, 0., Z_LO])]


def two_phase(D, theta_deg, glide_first):
    xb = H_DROP / math.tan(math.radians(theta_deg))
    if xb >= D - 1e-6:
        return None
    if glide_first:
        return [np.array([0., 0., Z_HI]), np.array([xb, 0., Z_LO]), np.array([D, 0., Z_LO])]
    return [np.array([0., 0., Z_HI]), np.array([D - xb, 0., Z_HI]), np.array([D, 0., Z_LO])]


def max_vz(path, v):
    m = 0.0
    for a, b in zip(path[:-1], path[1:]):
        seg = b - a; L = float(np.linalg.norm(seg))
        if L > 1e-9:
            m = max(m, abs(float(-seg[2])) / (L / v))
    return m


def main():
    src = open(os.path.join(EM100, "state", "best_featurize.py")).read()
    em = M100Em(src, payload=250.)
    power = PowerModel(src, "M100")

    # (3a) P(v_h, v_z) 曲面 + v_z*
    surface = []
    vz_grid = np.arange(-3.0, 1.01, 0.25)
    for vh in range(2, 13):
        st = dict(v_h=np.full_like(vz_grid, float(vh)), v_z=vz_grid,
                  a_h=np.zeros_like(vz_grid), a_z=np.zeros_like(vz_grid),
                  omega=np.zeros_like(vz_grid), payload=np.full_like(vz_grid, 250.),
                  wind=np.zeros_like(vz_grid), speed=np.hypot(vh, vz_grid))
        P = np.maximum(0.0, power.predict(st))
        surface.append({"v_h": vh, "P_at_vz": {round(float(z), 2): round(float(p), 1)
                                               for z, p in zip(vz_grid, P)},
                        "vz_star": round(float(vz_grid[int(np.argmin(P))]), 2),
                        "P_level": round(float(P[np.argmin(np.abs(vz_grid))]), 1),
                        "P_min": round(float(P.min()), 1)})
    print("v_z* per v_h:", {s['v_h']: s['vz_star'] for s in surface})

    # (1) 候选层网格
    rows, best = [], None
    for D in D_SWEEP:
        for v in V_SWEEP:
            e_ramp = score_path(ramp(D), em, v)
            rec = {"D": D, "v": v, "E_ramp": round(e_ramp, 1),
                   "ramp_vz": round(max_vz(ramp(D), v), 2), "profiles": []}
            for th in THETAS:
                for first in (True, False):
                    p = two_phase(D, th, first)
                    if p is None:
                        continue
                    e = score_path(p, em, v)
                    rec["profiles"].append({
                        "theta": th, "order": "glide_first" if first else "glide_last",
                        "E": round(e, 1), "sav_pct": round(100 * (e_ramp - e) / e_ramp, 2),
                        "max_vz": round(max_vz(p, v), 2)})
            wins = [q for q in rec["profiles"] if q["sav_pct"] >= 2.0]
            rec["best"] = max(rec["profiles"], key=lambda q: q["sav_pct"])
            rec["early_late_max_diff_pct"] = round(max(
                abs(a["sav_pct"] - b["sav_pct"]) for a in rec["profiles"] for b in rec["profiles"]
                if a["theta"] == b["theta"] and a["order"] != b["order"]) if wins or True else 0, 3)
            rows.append(rec)
            if wins and (best is None or rec["best"]["sav_pct"] > best[2]["sav_pct"]):
                best = (D, v, rec["best"])
            print(f"D={D:.0f} v={v:.0f}: ramp={e_ramp:.0f}J best={rec['best']['order']}"
                  f"@{rec['best']['theta']}° sav={rec['best']['sav_pct']}% vz={rec['best']['max_vz']}")

    result = {"preregistered": {
                  "criteria": "SUPPORTED iff candidate win>=2% AND RotorPy-flown win>=2% same sign "
                              "AND winner |v_z|<=3; REFUTED iff no candidate win>=2% anywhere",
                  "grid": {"D": D_SWEEP, "v": V_SWEEP, "theta": THETAS, "H": H_DROP}},
              "P_surface": surface, "candidate_grid": rows}

    # (2) 动力学复核(仅当候选层有 ≥2% 赢家)
    if best is None:
        result["verdict_hint"] = "REFUTED (no candidate profile beats ramp by >=2%)"
        print("候选层无 ≥2% 赢家 → REFUTED,无需动力学")
    else:
        D, v, bp = best
        print(f"动力学复核 headline: D={D} v={v} {bp['order']}@{bp['theta']}° (cand +{bp['sav_pct']}%)")
        from sim_flight import make_traj, fly, flown_energy
        flights = {}
        for name, path in [("ramp", ramp(D)),
                           ("winner", two_phase(D, bp["theta"], bp["order"] == "glide_first")),
                           ("winner_rev", two_phase(D, bp["theta"], bp["order"] != "glide_first"))]:
            traj, wps, _ = make_traj(np.array(path), v_avg=min(v, 8.0))
            t, x, vv = fly(traj, wps[0])
            E, T, Pm, _ = flown_energy(t, vv, power)
            flights[name] = {"E_J": round(E, 0), "T_s": round(T, 1), "meanP_W": round(Pm, 0)}
            print(f"  {name}: E={E:.0f}J T={T:.1f}s")
        er, ew = flights["ramp"]["E_J"], flights["winner"]["E_J"]
        result["dynamics"] = {"headline": {"D": D, "v": v, **bp}, "flights": flights,
                              "flown_sav_pct": round(100 * (er - ew) / er, 2)}
        ok = result["dynamics"]["flown_sav_pct"] >= 2.0
        extrap = bp["max_vz"] > VZ_DATA_MAX
        result["verdict_hint"] = ("INCONCLUSIVE (winner relies on |v_z|>3 extrapolation)" if ok and extrap
                                  else "SUPPORTED" if ok else "REFUTED (dynamics kills candidate win)")
    with open(os.path.join(HERE, "pl_iter4_gravity_glide.json"), "w") as f:
        json.dump(result, f, ensure_ascii=False, indent=1,
                  default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print(f"verdict_hint={result['verdict_hint']}  JSON 落盘 experiments/pl_iter4_gravity_glide.json")


if __name__ == "__main__":
    main()
