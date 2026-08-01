# -*- coding: utf-8 -*-
"""as_published_benchmark.py — 零调整(原样)文献模型在 M100 上的基准评测。
───────────────────────────────────────────────────────────────────────────
对每个 as-published 模型,在留出航班上直接算功率预测(不拟合任何系数),
指标:RMSE(W)、MAPE、per-flight 能量 ARE(与 refit zoo 同口径可对比)。
同时列出 refit 版本的 ARE 做对比列 → 直观展示"不拟合 vs 拟合"的迁移差距。
───────────────────────────────────────────────────────────────────────────
输出:
  experiments/as_published_benchmark.json
  experiments/pub/as_published_vs_refit.png
"""
import os, sys, json, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from m100_eval import load_m100
from as_published_models import AS_PUBLISHED, predict_published

# refit 版本的留出 ARE(来自 model_zoo_leaderboard,唯一可比口径)
REFIT_ARE = {
    "momentum_hover":    None,   # 无 refit 版
    "momentum_forward":  None,
    "tseng_simplified":  None,   # 对应 Tseng 2022 refit = 1.90%
}


def main():
    d = load_m100()
    flights = np.unique(d["flight"])
    rng = np.random.default_rng(5); rng.shuffle(flights)
    ntr = int(0.7 * len(flights))
    te = np.array([f not in flights[:ntr].tolist() for f in d["flight"]])
    P, t = d["P"], d["t"]

    out = []
    for code, name, pred, src in AS_PUBLISHED:
        p_hat = predict_published(code, d, te)
        p_true = P[te]
        res = p_hat - p_true
        rmse = float(np.sqrt(np.mean(res**2)))
        mape = float(np.mean(np.abs(res) / np.maximum(p_true, 1))) * 100
        # per-flight 能量 ARE
        ares = []
        for f in flights[ntr:]:
            fm = d["flight"] == f
            pf = predict_published(code, d, fm)
            tt = t[fm]; dt = np.clip(np.diff(tt, prepend=tt[0]), 0, 1)
            Em, Ep = np.sum(P[fm] * dt), np.sum(pf * dt)
            if Em > 0:
                ares.append(abs(Ep - Em) / Em)
        rec = {"code": code, "name": name, "source": src,
               "rmse_W": rmse, "mape_pct": mape,
               "energy_ARE_pct": float(np.mean(ares)) * 100,
               "mean_true_W": float(p_true.mean()),
               "mean_pred_W": float(p_hat.mean()),
               "refit_ARE_pct": REFIT_ARE.get(code)}
        out.append(rec)
        print(f"  {name:34s} RMSE={rmse:6.1f}W  MAPE={mape:5.1f}%  能量ARE={rec['energy_ARE_pct']:5.1f}%"
              f"  预测均{rec['mean_pred_W']:4.0f}W vs 真机{rec['mean_true_W']:4.0f}W")

    os.makedirs(os.path.join(HERE, "experiments"), exist_ok=True)
    jp = os.path.join(HERE, "experiments", "as_published_benchmark.json")
    json.dump({"generated": time.strftime("%Y-%m-%d %H:%M"), "n_test_flights": int(flights[ntr:].size),
               "results": out}, open(jp, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"存 {jp}")

    # 图:as-published 预测均值 vs 真机均值
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from pubstyle import PAL
    fig, ax = plt.subplots(figsize=(8, 3.6))
    names = [r["name"] for r in out]
    ypos = np.arange(len(names))
    true = out[0]["mean_true_W"]
    ax.barh(ypos, [r["mean_pred_W"] for r in out], height=.5, color=PAL["blue"], label="as-published 预测功率均值")
    ax.axvline(true, color=PAL["red"], lw=2, label=f"真机实测均值 {true:.0f}W")
    for i, r in enumerate(out):
        ax.text(r["mean_pred_W"] + 5, i, f"{r['mean_pred_W']:.0f}W (ARE {r['energy_ARE_pct']:.0f}%)",
                va="center", fontsize=8.5)
    ax.set_yticks(ypos); ax.set_yticklabels(names, fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlabel("平均功率 (W)")
    ax.set_title("零调整(as-published)文献模型直接套用 M100:预测 vs 真机", fontsize=11)
    ax.legend(fontsize=9, loc="lower right")
    fig.tight_layout()
    fp = os.path.join(HERE, "experiments", "pub", "as_published_vs_refit.png")
    os.makedirs(os.path.dirname(fp), exist_ok=True)
    fig.savefig(fp, dpi=200); plt.close(fig)
    print(f"图存 {fp}")


if __name__ == "__main__":
    main()
