"""
planning_connect.py — B:把 M100 真机锚定的能耗模型接进能量感知路径规划(闭合到 UAV 场景)。
思路:loop 发现的 best featurize(真实 DJI 功率验证过)→ 固化成 PowerModel.predict(state) →
用它当路径能耗代价 → 比"真机锚定 vs 稳态BEMT vs 距离"三种代价下,能量最优路线是否不同。

诚实边界(写进记录):规划增益是"相对 + 真实模型锚定",非绝对最优;两模型都拟合在 M100,
差别在特征集(稳态BEMT只有v,忽略爬升/载荷;M100锚定含爬升/载荷)→ 决定它们对"爬升路线"定价不同。
"""
import os
import numpy as np
from candidate_energy import DEFAULT_FEATURIZE_SRC, load_featurize
from m100_eval import load_m100, STATE_KEYS, evaluate

HERE = os.path.dirname(os.path.abspath(__file__))
G = 9.81


class PowerModel:
    """把一个 featurize 源码 + M100 数据 → 预测功率 P(state) 的模型(部署用全量拟合)。"""
    def __init__(self, featurize_src, name="model"):
        self.name = name
        self.fn = load_featurize(featurize_src)
        d = load_m100()
        with np.errstate(all="ignore"):
            X = np.nan_to_num(np.asarray(self.fn({k: d[k] for k in STATE_KEYS}), float), posinf=0, neginf=0)
            self.mu = X.mean(0); self.sd = X.std(0)
            const = self.sd < 1e-9; self.mu[const] = 0.0; self.sd[const] = 1.0
            Xs = (X - self.mu) / self.sd
            XtX = Xs.T @ Xs
            lam = 1e-4 * np.trace(XtX) / max(1, Xs.shape[1])
            self.beta = np.linalg.solve(XtX + lam * np.eye(Xs.shape[1]), Xs.T @ d["P"])
        # 记录:held-out ARE(诚实)
        self.holdout_ARE = float(np.mean([evaluate(self.fn, seed=k)["energy_ARE"] for k in (5, 6, 7)]))

    def predict(self, state):
        X = np.nan_to_num(np.asarray(self.fn(state), float), posinf=0, neginf=0)
        return ((X - self.mu) / self.sd) @ self.beta


def path_state(waypoints, v_cruise=8.0, payload=250.0, a_lat=3.0):
    """3D 折线 → 每段的运动学状态 dict(供 PowerModel.predict)。
    转弯处按侧向加速度上限 a_lat 限速(急转→减速→动能变化),爬升给 v_z。"""
    W = np.asarray(waypoints, float)
    segs = np.diff(W, axis=0)
    L = np.linalg.norm(segs, axis=1)
    Lh = np.linalg.norm(segs[:, :2], axis=1)
    dz = segs[:, 2]
    # 转弯角 → 段速(急转限速)
    v = np.full(len(segs), v_cruise)
    for i in range(1, len(segs)):
        d0 = segs[i - 1] / (L[i - 1] + 1e-9); d1 = segs[i] / (L[i] + 1e-9)
        turn = np.arccos(np.clip(d0 @ d1, -1, 1))
        if turn > 0.1:
            r = max(0.5, L[i] / (2 * np.sin(turn / 2 + 1e-6)))
            v[i] = min(v_cruise, np.sqrt(a_lat * r))
    t = L / np.maximum(v, 0.5)
    v_h = Lh / np.maximum(t, 1e-3)
    v_z = dz / np.maximum(t, 1e-3)
    a_h = np.abs(np.diff(v, prepend=v[0])) / np.maximum(t, 1e-3)
    N = len(segs)
    z = np.zeros(N)
    return dict(v_h=v_h, v_z=v_z, a_h=a_h, a_z=z, omega=z,
                payload=np.full(N, payload), wind=z, speed=v), t


def path_energy(waypoints, model, **kw):
    """真机锚定路径能耗(J)= Σ 预测功率 × 段时长。"""
    st, t = path_state(waypoints, **kw)
    P = np.maximum(0.0, model.predict(st))     # 功率非负
    return float(np.sum(P * t))


def path_length(waypoints):
    return float(np.sum(np.linalg.norm(np.diff(np.asarray(waypoints, float), axis=0), axis=1)))


# 场景:start→goal 间有一堵墙,三种拓扑路线
def scenario_routes():
    start = [0, 0, -3.]; goal = [40, 0, -3.]
    return {
        "翻墙(爬到15m)": [start, [15, 0, -15.], [25, 0, -15.], goal],
        "左绕(横向)":   [start, [15, 18, -3.], [25, 18, -3.], goal],
        "宽绕(更远)":   [start, [12, 28, -3.], [28, 28, -3.], goal],
    }


def make_figure(routes, rows, m_bemt, m_m100, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    for f in ("PingFang SC", "Heiti SC", "STHeiti", "Arial Unicode MS"):
        if any(f in x.name for x in font_manager.fontManager.ttflist):
            plt.rcParams["font.sans-serif"] = [f]; break
    plt.rcParams["axes.unicode_minus"] = False
    names = list(routes)
    colors = ["#c0392b", "#2c6fbb", "#27ae60"]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.3))
    for nm, c in zip(names, colors):
        W = np.asarray(routes[nm], float)
        ax[0].plot(W[:, 0], W[:, 1], "-o", color=c, label=nm, ms=4)
        ax[1].plot(W[:, 0], -W[:, 2], "-o", color=c, ms=4)      # -z = 高度
    ax[0].set_title("俯视 XY"); ax[0].set_xlabel("x (m)"); ax[0].set_ylabel("y (m)"); ax[0].legend(fontsize=8); ax[0].grid(alpha=.3)
    ax[1].set_title("侧视 XZ(高度)"); ax[1].set_xlabel("x (m)"); ax[1].set_ylabel("高度 (m)"); ax[1].grid(alpha=.3)
    # 能耗对比条
    x = np.arange(len(names)); w = 0.38
    eb = [rows[n][1] for n in names]; em = [rows[n][2] for n in names]
    ax[2].bar(x - w/2, eb, w, label=f"稳态BEMT(ARE {m_bemt.holdout_ARE*100:.1f}%)", color="#95a5a6")
    ax[2].bar(x + w/2, em, w, label=f"M100锚定(ARE {m_m100.holdout_ARE*100:.1f}%)", color="#e67e22")
    ax[2].axvline(np.argmin(eb) - w/2, ls=":", color="#95a5a6"); ax[2].axvline(np.argmin(em) + w/2, ls=":", color="#e67e22")
    ax[2].set_xticks(x); ax[2].set_xticklabels([n[:4] for n in names], fontsize=8)
    ax[2].set_title("能耗(J):BEMT选翻墙 vs M100锚定选左绕"); ax[2].set_ylabel("能耗 (J)"); ax[2].legend(fontsize=8); ax[2].grid(axis="y", alpha=.3)
    fig.suptitle("B:真机锚定能耗模型接入路径规划 —— 改变能量最优路线选择", fontsize=12)
    fig.tight_layout(); fig.savefig(out, dpi=110); print("[图]", out)


if __name__ == "__main__":
    m_bemt = PowerModel(DEFAULT_FEATURIZE_SRC, "稳态BEMT(仅v)")
    best_src = open(os.path.join(HERE, "state", "best_featurize.py")).read()
    m_m100 = PowerModel(best_src, "M100锚定(含爬升/载荷)")
    print(f"稳态BEMT   模型: held-out ARE={m_bemt.holdout_ARE*100:.2f}%")
    print(f"M100锚定   模型: held-out ARE={m_m100.holdout_ARE*100:.2f}%  ← 真机验证更准的能耗代价\n")

    routes = scenario_routes()
    print(f"{'路线':<16}{'距离m':>8}{'稳态BEMT能耗':>14}{'M100锚定能耗':>14}")
    print("-" * 54)
    rows = {}
    for name, wp in routes.items():
        d = path_length(wp); eb = path_energy(wp, m_bemt); em = path_energy(wp, m_m100)
        rows[name] = (d, eb, em)
        print(f"{name:<16}{d:>8.1f}{eb:>14.0f}{em:>14.0f}")
    best_dist = min(rows, key=lambda k: rows[k][0])
    best_bemt = min(rows, key=lambda k: rows[k][1])
    best_m100 = min(rows, key=lambda k: rows[k][2])
    print(f"\n最短距离选: {best_dist}")
    print(f"稳态BEMT选: {best_bemt}")
    print(f"M100锚定选: {best_m100}")
    print("\n" + ("✅ 真机锚定能耗代价改变了能量最优路线选择 —— 规划建立在真实测量上,不是手搭BEMT。"
                  if best_m100 != best_bemt or best_m100 != best_dist
                  else "⚠️ 三代价选同一路线(此场景区分度不够,需换场景)。"))
    os.makedirs(os.path.join(HERE, "experiments"), exist_ok=True)
    make_figure(routes, rows, m_bemt, m_m100, os.path.join(HERE, "experiments", "fig_planning_connect.png"))
