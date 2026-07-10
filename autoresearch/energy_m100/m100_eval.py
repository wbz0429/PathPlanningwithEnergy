"""
m100_eval.py — 无人机能耗建模的【冻结·作弊不了评测器】。真值 = DJI M100 真实电测功率 P=V·|I|;
指标 = held-out 飞行上的预测误差(per-sample R² + per-flight 能量 ARE)。
按【飞行】划分 train/test(防跨飞行泄漏)。这是尺子——autoresearch 只能改能耗模型(featurize),碰这里=作弊。

与 radar 完全平行:autoresearch 优化算法(能耗模型)→ 对真数据的冻结指标 → held-out。
数据:~/datasets/m100/flights.csv(Rodrigues 2021,209 真机飞行,CC-BY)。
"""
import os, csv
import numpy as np

DATA = os.path.expanduser("~/datasets/m100/flights.csv")
CACHE = os.path.expanduser("~/datasets/m100/m100_cache.npz")
G = 9.81
STATE_KEYS = ("v_h", "v_z", "a_h", "a_z", "omega", "payload", "wind", "speed")


def load_m100():
    if os.path.exists(CACHE):
        z = np.load(CACHE)
        return {k: z[k] for k in z.files}
    acc = {k: [] for k in ("flight", "V", "I", "vx", "vy", "vz", "ax", "ay", "az",
                           "wx", "wy", "wz", "payload", "wind", "t", "speed")}
    with open(DATA) as f:
        r = csv.reader(f); hdr = next(r); idx = {n: i for i, n in enumerate(hdr)}; NC = len(hdr)
        def g(row, n):
            try: return float(row[idx[n]])
            except: return np.nan
        for row in r:
            if len(row) != NC or not row[idx["route"]].startswith("R"):
                continue
            acc["flight"].append(int(float(row[idx["flight"]])))
            for k, col in [("V", "battery_voltage"), ("I", "battery_current"),
                           ("vx", "velocity_x"), ("vy", "velocity_y"), ("vz", "velocity_z"),
                           ("ax", "linear_acceleration_x"), ("ay", "linear_acceleration_y"),
                           ("az", "linear_acceleration_z"), ("wx", "angular_x"), ("wy", "angular_y"),
                           ("wz", "angular_z"), ("payload", "payload"), ("wind", "wind_speed"),
                           ("t", "time"), ("speed", "speed")]:
                acc[k].append(g(row, col))
    a = {k: np.array(v, float) for k, v in acc.items()}
    P = a["V"] * np.abs(a["I"])
    d = dict(flight=a["flight"].astype(int), P=P,
             v_h=np.hypot(a["vx"], a["vy"]), v_z=a["vz"],
             a_h=np.hypot(a["ax"], a["ay"]), a_z=a["az"] + G,
             omega=np.sqrt(a["wx"]**2 + a["wy"]**2 + a["wz"]**2),
             payload=a["payload"], wind=a["wind"], speed=a["speed"], t=a["t"])
    finite_keys = ("P", "v_h", "v_z", "a_h", "a_z", "omega", "payload", "wind", "speed", "t")
    m = np.all([np.isfinite(d[k]) for k in finite_keys], axis=0) & (P > 50)
    d = {k: v[m] for k, v in d.items()}
    np.savez(CACHE, **d)
    return d


def _state(d, mask):
    return {k: d[k][mask] for k in STATE_KEYS}


def evaluate(featurize_fn, seed=0, test_frac=0.3):
    """拟合 train 线性系数,评 held-out 飞行。返回 {r2, energy_ARE, n_feat, n_test_flights}。"""
    d = load_m100()
    flights = np.unique(d["flight"])
    rng = np.random.default_rng(seed); rng.shuffle(flights)
    ntr = int((1 - test_frac) * len(flights))
    trf = set(flights[:ntr].tolist())
    tr = np.array([f in trf for f in d["flight"]]); te = ~tr
    with np.errstate(all="ignore"):
        Xtr = np.nan_to_num(np.asarray(featurize_fn(_state(d, tr)), float), posinf=0, neginf=0)
        Xte = np.nan_to_num(np.asarray(featurize_fn(_state(d, te)), float), posinf=0, neginf=0)
        # 列标准化(存 train 统计),消除病态/共线的溢出;【常数列/截距保持不变】
        mu = Xtr.mean(0); sd = Xtr.std(0)
        const = sd < 1e-9
        mu[const] = 0.0; sd[const] = 1.0          # 截距列 (1-0)/1=1 保持
        Xtr = (Xtr - mu) / sd; Xte = (Xte - mu) / sd
        ytr, yte = d["P"][tr], d["P"][te]
        XtX = Xtr.T @ Xtr
        lam = 1e-4 * np.trace(XtX) / max(1, Xtr.shape[1])
        beta = np.linalg.solve(XtX + lam * np.eye(Xtr.shape[1]), Xtr.T @ ytr)
        pred = Xte @ beta
        r2 = 1 - np.sum((yte - pred) ** 2) / np.sum((yte - yte.mean()) ** 2)
        # per-flight 能量 ARE(积分功率,论文标准)
        ares = []
        for f in flights[ntr:]:
            fm = d["flight"] == f
            Xf = np.nan_to_num(np.asarray(featurize_fn(_state(d, fm)), float), posinf=0, neginf=0)
            pf = ((Xf - mu) / sd) @ beta
            tt = d["t"][fm]; dt = np.clip(np.diff(tt, prepend=tt[0]), 0, 1)
            Em, Ep = np.sum(d["P"][fm] * dt), np.sum(pf * dt)
            if Em > 0:
                ares.append(abs(Ep - Em) / Em)
    return {"r2": float(r2), "energy_ARE": float(np.mean(ares)),
            "n_feat": int(Xtr.shape[1]), "n_test_flights": int(len(flights) - ntr)}


# 默认能耗模型 = 稳态 BEMT 式 U 形:P ≈ b0 + b1·v + b2·v²(仅水平速度)
def default_featurize(s):
    return np.column_stack([np.ones_like(s["v_h"]), s["v_h"], s["v_h"] ** 2])


if __name__ == "__main__":
    d = load_m100()
    print(f"M100 载入: {len(d['P'])} 样本, {len(np.unique(d['flight']))} 飞行, "
          f"功率均值 {d['P'].mean():.0f}W")
    m = evaluate(default_featurize, seed=0)
    print(f"基线(稳态 BEMT: 1,v,v²): held-out R²={m['r2']:.3f}  能量ARE={m['energy_ARE']*100:.1f}%  "
          f"({m['n_feat']}特征, {m['n_test_flights']}测试飞行)")
