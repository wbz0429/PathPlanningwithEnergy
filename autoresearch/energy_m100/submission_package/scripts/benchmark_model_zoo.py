# -*- coding: utf-8 -*-
"""benchmark_model_zoo.py — 把 energy_model_zoo 的全部模型在冻结 M100 评测器上跑一遍,生成排行榜。
───────────────────────────────────────────────────────────────────────────
这是 pre-loop 前置阶段:文献模型 → 系统评测 → 选最优基线 → 送入 loop。
每改一次模型知识库就跑一次,保持榜单最新。
───────────────────────────────────────────────────────────────────────────
输出:
  experiments/model_zoo_leaderboard.json  — 完整结果(含每个 seed 的 ARE/R²)
  experiments/pub/模型知识库排行榜.png     — 条形图(中文,论文级)
  stdout 打印排行榜
───────────────────────────────────────────────────────────────────────────
"""
import os, sys, json, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from m100_eval import evaluate
from energy_model_zoo import get_all_models, get_rankable_models, export_model_src


def benchmark_all(search_seeds=(0, 1, 2), val_seeds=(5, 6, 7)):
    """对所有注册模型评估。返回按 val_ARE 升序排序的结果列表。"""
    models = get_all_models()
    results = []
    for code, name, fn, paper, category, rankable in models:
        t0 = time.time()
        print(f"  {name:30s} [{code}] ", end="", flush=True)
        try:
            s = [evaluate(fn, seed=k) for k in search_seeds]
            v = [evaluate(fn, seed=k) for k in val_seeds]
            elap = time.time() - t0
            r = {
                "code": code, "name": name, "paper": paper, "category": category,
                "rankable": rankable, "_fn": fn,
                "search_ARE_percent": float(np.mean([m["energy_ARE"] for m in s]) * 100),
                "val_ARE_percent":    float(np.mean([m["energy_ARE"] for m in v]) * 100),
                "val_r2":             float(np.mean([m["r2"] for m in v])),
                "n_feat":             int(s[0]["n_feat"]),
                "n_test_flights":     int(s[0]["n_test_flights"]),
                "search_seeds":       [float(m["energy_ARE"]) * 100 for m in s],
                "val_seeds":          [float(m["energy_ARE"]) * 100 for m in v],
                "time_sec":           round(elap, 1),
            }
            results.append(r)
            print(f"→ val_ARE={r['val_ARE_percent']:.2f}%  R²={r['val_r2']:.3f}  ({r['time_sec']:.1f}s)")
        except Exception as e:
            print(f"✗ FAIL: {e}")
            results.append({"code": code, "name": name, "paper": paper, "category": category,
                            "error": str(e)})
    results.sort(key=lambda r: r.get("val_ARE_percent", 9e9))
    return results


def print_leaderboard(results):
    """打印格式化排行榜(占位符标记⚠️)。"""
    print("\n" + "=" * 110)
    print(f"{'排名':<4} {'模型':<34} {'类别':<12} {'留出ARE':>9} {'R²':>7} {'特征数':>6} {'来源':<34}")
    print("-" * 110)
    for i, r in enumerate(results, 1):
        tag = " [!]占位" if not r.get("rankable", True) else ""
        if "error" in r:
            print(f"{i:<4} {r['name']+tag:<34} {'FAIL':<12} {'—':>9} {'—':>7} {'—':>6} {r['error'][:34]}")
        else:
            print(f"{i:<4} {r['name']+tag:<34} {r['category']:<12} {r['val_ARE_percent']:>8.2f}% "
                  f"{r['val_r2']:>7.3f} {r['n_feat']:>6} {r['paper'][:34]}")
    print("=" * 110)
    # 最优基线只从可排行模型中选
    rankable = [r for r in results if r.get("rankable", True) and "error" not in r]
    best = rankable[0] if rankable else None
    if best:
        print(f"\n★ 文献最优基线: {best['name']} (留出 ARE {best['val_ARE_percent']:.2f}%)")
        print(f"  来源: {best['paper']}")
        print(f"  → 该模型将作为 autoresearch loop 的起点。")
    # 报告占位符状态
    placeholders = [r for r in results if not r.get("rankable", True) and "error" not in r]
    if placeholders:
        print(f"\n[!] 占位符(待补全论文,不计入排行):")
        for r in placeholders:
            print(f"  {r['name']}: ARE={r['val_ARE_percent']:.2f}% — 当前形式为猜测,需找到原始论文后重新评测")
    return best


def save_leaderboard_fig(results, out_path):
    """生成排行榜条形图(中文)。占位符标⚠️、不计入最优。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from pubstyle import PAL
    valid = [r for r in results if "error" not in r]
    if not valid:
        return
    names = [r["name"] + (" [!]" if not r.get("rankable", True) else "") for r in valid]
    vals = [r["val_ARE_percent"] for r in valid]
    rankable = [r.get("rankable", True) for r in valid]
    cols = []
    # 第一个可排行模型标绿(最优);占位符标灰;>4%标红;<2%标蓝;其余灰色
    first_rk_idx = next((i for i, rk in enumerate(rankable) if rk), None)
    for i, (v, rk) in enumerate(zip(vals, rankable)):
        if not rk:
            cols.append("#cccccc")
        elif i == first_rk_idx:
            cols.append(PAL["green"])
        elif v > 4.0:
            cols.append(PAL["red"])
        elif v < 2.0:
            cols.append(PAL["blue"])
        else:
            cols.append(PAL["gray"])

    fig, ax = plt.subplots(figsize=(11, 4.8))
    ypos = range(len(names))
    bars = ax.barh(ypos, vals, color=cols, height=.6)
    ax.set_yticks(ypos); ax.set_yticklabels(names, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("留出能量 ARE (%)", fontsize=11)
    ax.axvline(1.86, ls=(0, (4, 3)), color=PAL["green"], lw=1.5, label="随机搜索地板 ≈1.9%")
    # 标注数值
    for b, v, rk in zip(bars, vals, rankable):
        ax.text(v + .06, b.get_y() + b.get_height() / 2, f"{v:.2f}%",
                va="center", fontsize=8.5, fontweight="bold" if rk and v == min(x for j, x in enumerate(vals) if rankable[j]) else "normal",
                color="#999" if not rk else "#222")
    ax.set_title("无人机能耗模型知识库:文献模型在真实 M100 数据上的留出 ARE", fontsize=11.5)
    ax.legend(fontsize=9, loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"排行榜图存 {out_path}")


def main():
    d = os.path.join(HERE, "experiments")
    os.makedirs(d, exist_ok=True)
    results = benchmark_all()
    best = print_leaderboard(results)

    # 存 JSON
    out_json = os.path.join(d, "model_zoo_leaderboard.json")
    # 存 JSON (去函数引用)
    json_results = [{k: v for k, v in r.items() if k != "_fn"} for r in results]
    json.dump({"generated": time.strftime("%Y-%m-%d %H:%M"),
               "num_models": len(results),
               "best_model": best["code"] if best else None,
               "results": json_results},
              open(out_json, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"\n结果存 {out_json}")

    # 存图
    out_fig = os.path.join(d, "pub", "模型知识库排行榜.png")
    os.makedirs(os.path.dirname(out_fig), exist_ok=True)
    save_leaderboard_fig(results, out_fig)

    # 将可排行最优模型导出为自包含源码写入 best_featurize.py
    if best:
        src_path = os.path.join(HERE, "state", "best_featurize.py")
        assert best.get("rankable", True), f"最优基线 {best['name']} 是占位符!"
        zoo_src, zoo_name, zoo_paper = export_model_src(best["code"])
        header = f"# best_featurize.py — modelo_zoo 排行榜最优({zoo_name}, ARE {best['val_ARE_percent']:.2f}%, {zoo_paper})\n"
        # 仅在 ARE 严格更低时覆盖(或当前 best 不可加载/来自占位符时强制覆盖)
        force = False
        if os.path.exists(src_path):
            import candidate_energy
            try:
                old_fn = candidate_energy.load_featurize(open(src_path).read())
                old_v = np.mean([evaluate(old_fn, seed=k)["energy_ARE"] for k in (5, 6, 7)]) * 100
            except Exception:
                old_v = 9e9; force = True  # 当前 best 损坏/占位符,强制覆盖
            if best["val_ARE_percent"] < old_v or force:
                tag = "强制覆盖(当前best无效)" if force else f"新(ARE {best['val_ARE_percent']:.2f}% < 旧 {old_v:.2f}%)"
                print(f"\n★ {tag} → 更新 best_featurize.py")
                open(src_path, "w", encoding="utf-8").write(header + zoo_src)
            else:
                print(f"\n当前 best_featurize ({old_v:.2f}%) 仍最优,不覆盖")
        else:
            print(f"\n★ 初始化 best_featurize.py → {best['name']}")
            os.makedirs(os.path.dirname(src_path), exist_ok=True)
            open(src_path, "w", encoding="utf-8").write(header + zoo_src)


if __name__ == "__main__":
    main()
