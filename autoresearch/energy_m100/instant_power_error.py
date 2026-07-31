# -*- coding: utf-8 -*-
"""instant_power_error.py — 瞬时功率误差分析:补足"单指标 ARE"的盲区。
───────────────────────────────────────────────────────────────────────────
ARE 是积分量,正负瞬时误差会抵消 → 单看 ARE 可能掩盖"某状态系统性偏差"。
本实验在留出航班上逐点计算预测 vs 真实瞬时功率:
  (1) 整体 RMSE / MAE / bias(均值误差)
  (2) 分爬升率状态箱的预测均值 vs 真实均值(直接看爬升标定误差)
  (3) 三模型并排:我们的 best vs Tseng2022 vs 教科书 BEMT
───────────────────────────────────────────────────────────────────────────
输出:
  experiments/instant_power_error.json
  experiments/pub/瞬时功率误差.png
"""
import os, sys, json, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from m100_eval import load_m100, _state
from candidate_energy import load_featurize
from energy_model_zoo import get_model_by_code


def fit_beta(fn, d, tr_mask):
    """训练集拟合并返回 (beta, mu, sd),与 m100_eval 相同的标准化逻辑。"""
    Xtr = np.nan_to_num(np.asarray(fn(_state(d, tr_mask)), float), posinf=0, neginf=0)
    mu = Xtr.mean(0); sd = Xtr.std(0)
    const = sd < 1e-9
    mu[const] = 0.0; sd[const] = 1.0
    Xtr = (Xtr - mu) / sd
    ytr = d["P"][tr_mask]
    XtX = Xtr.T @ Xtr
    lam = 1e-4 * np.trace(XtX) / max(1, Xtr.shape[1])
    beta = np.linalg.solve(XtX + lam * np.eye(Xtr.shape[1]), Xtr.T @ ytr)
    return beta, mu, sd


def predict(fn, beta, mu, sd, d, mask):
    X = np.nan_to_num(np.asarray(fn(_state(d, mask)), float), posinf=0, neginf=0)
    return ((X - mu) / sd) @ beta


def main():
    d = load_m100()
    flights = np.unique(d["flight"])
    rng = np.random.default_rng(5); rng.shuffle(flights)
    ntr = int(0.7 * len(flights))
    tr = np.array([f in flights[:ntr].tolist() for f in d["flight"]])
    te = ~tr

    # our_loop = loop 真产物(iter6 物理核+payload);tseng2022 = 人类专家;bemt = 教科书
    our_loop_fn = get_model_by_code("physics_plus_payload")[2]
    models = [
        ("our_loop",  "我们的 loop(物理+payload)", our_loop_fn),
        ("tseng2022", "Tseng 2022(人类专家)",      get_model_by_code("tseng_2022")[2]),
        ("bemt",      "教科书 BEMT",              get_model_by_code("bemt_steady")[2]),
    ]

    # 爬升率状态箱(与真机实测一致)
    vz = d["v_z"]
    bins = [(-10, -1, "下降"), (-1, 1, "平飞"), (1, 2, "缓慢爬升"), (2, 4, "中爬升"), (4, 10, "急剧爬升")]
    out = {"n_test_flights": int(flights[ntr:].size), "n_samples": int(te.sum()),
           "models": {}}

    for code, name, fn in models:
        beta, mu, sd = fit_beta(fn, d, tr)
        p_hat = predict(fn, beta, mu, sd, d, te)
        p_true = d["P"][te]
        res = p_hat - p_true
        rec = {
            "rmse_W": float(np.sqrt(np.mean(res ** 2))),
            "mae_W": float(np.mean(np.abs(res))),
            "bias_W": float(np.mean(res)),
            "r2": float(1 - np.sum(res ** 2) / np.sum((p_true - p_true.mean()) ** 2)),
            "mean_true_W": float(p_true.mean()),
            "mean_pred_W": float(p_hat.mean()),
            "by_state": [],
        }
        for lo, hi, nm in bins:
            m = (vz[te] >= lo) & (vz[te] < hi)
            if m.sum() < 10:
                continue
            rec["by_state"].append({
                "state": nm, "n": int(m.sum()),
                "true_W": float(p_true[m].mean()), "pred_W": float(p_hat[m].mean()),
                "bias_W": float(np.mean(p_hat[m] - p_true[m])),
                "bias_pct": float(100 * np.mean(p_hat[m] - p_true[m]) / p_true[m].mean()),
            })
        out["models"][code] = {"name": name, **rec}
        print(f"  {name:18s} RMSE={rec['rmse_W']:.1f}W  MAE={rec['mae_W']:.1f}W  bias={rec['bias_W']:+.1f}W  R²={rec['r2']:.3f}")

    # 存 JSON
    os.makedirs(os.path.join(HERE, "experiments"), exist_ok=True)
    jp = os.path.join(HERE, "experiments", "instant_power_error.json")
    json.dump({"generated": time.strftime("%Y-%m-%d %H:%M"), **out},
              open(jp, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"存 {jp}")

    # 图:分状态箱 true vs pred(三模型)
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from pubstyle import PAL
    states = out["models"]["our_loop"]["by_state"]
    xs = [b["state"] for b in states]
    fig, ax = plt.subplots(figsize=(8.5, 3.6))
    width = 0.25
    codes = ["bemt", "tseng2022", "our_loop"]
    colmap = {"bemt": PAL["gray"], "tseng2022": PAL["blue"], "our_loop": PAL["green"]}
    for j, code in enumerate(codes):
        bs = out["models"][code]["by_state"]
        vals = [b["pred_W"] for b in bs]
        x = np.arange(len(xs)) + (j - 1) * width
        ax.bar(x, vals, width, color=colmap[code], alpha=.85,
               label=out["models"][code]["name"] + f" (R²={out['models'][code]['r2']:.2f})")
    true_vals = [b["true_W"] for b in states]
    ax.plot(np.arange(len(xs)), true_vals, "k--o", lw=1.6, ms=5, label="真机实测")
    for i, v in enumerate(true_vals):
        ax.text(i - 0.02, v + 12, f"{v:.0f}", fontsize=8, ha="center", color="#222")
    ax.set_xticks(np.arange(len(xs))); ax.set_xticklabels(xs)
    ax.set_ylabel("平均功率 (W)"); ax.set_xlabel("飞行状态(爬升率分箱)")
    ax.set_title("瞬时功率预测 vs 真机实测:分状态剖面(留出航班)", fontsize=11)
    ax.legend(fontsize=8.5)
    fig.tight_layout()
    fp = os.path.join(HERE, "experiments", "pub", "瞬时功率误差.png")
    os.makedirs(os.path.dirname(fp), exist_ok=True)
    fig.savefig(fp, dpi=200); plt.close(fig)
    print(f"图存 {fp}")


if __name__ == "__main__":
    main()
