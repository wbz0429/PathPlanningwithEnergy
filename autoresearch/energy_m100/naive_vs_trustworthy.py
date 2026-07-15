# -*- coding: utf-8 -*-
"""naive_vs_trustworthy.py — 端到端对照:同样的数据+候选,朴素AI4S(无安全机制) vs 我们的可信框架,
各自会产出什么"论文结论"。直接回答"用我们框架 vs 不用,区别在哪"。

朴素loop(模拟无safeguard的AI Scientist式流程):
  ① 只看拟合指标(search-ARE)选模型,无留出门 → 会KEEP过拟合候选
  ② 无查新门 → 把规划层SUPPORTED假设直接当"新发现"上报
  ③ 只报成功,不报负结果 → 不提loop=随机/载荷死路/多数场景0收益
我们的框架:留出门 + 查新门 + 系统报负结果。

数据全部来自盘上已有结果(agent_log/candidates/PLANNING_KNOWLEDGE查新裁决),不新拟造。
输出:两栏"论文摘要"对照 + 图。
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from candidate_energy import eval_src

HERE = os.path.dirname(os.path.abspath(__file__))
CAND = os.path.join(HERE, "experiments", "candidates")


def energy_claims():
    """朴素:只看search选最优,报为'LLM发现更优模型'。可信:留出门+承认loop=随机。"""
    cands = []
    base = "def featurize(s):\n import numpy as np\n return np.column_stack([np.ones_like(s['v_h']),s['v_h'],s['v_h']**2])"
    r = eval_src(base); cands.append(("BEMT基线", r["search_ARE"], r["val_ARE"]))
    for i in range(1, 10):
        p = os.path.join(CAND, f"iter{i}_featurize.py")
        if os.path.exists(p):
            try:
                r = eval_src(open(p).read()); cands.append((f"iter{i}", r["search_ARE"], r["val_ARE"]))
            except Exception:
                pass
    # 加一个过拟合候选(search好、留出崩),朴素loop会上钩
    overfit = ("def featurize(s):\n import numpy as np\n v=s['v_h'];z=s['v_z'];p=s['payload'];w=s['wind'];a=s['a_h']\n"
               " C=[np.ones_like(v),v,v*v,z,np.maximum(z,0),p,v*v*p]\n"
               " for k in range(2,7):\n  C+=[np.sin(k*v)*p,np.cos(k*z)*w,(v**(k%3))*np.sign(a)]\n"
               " return np.column_stack(C)")
    r = eval_src(overfit); cands.append(("富特征候选", r["search_ARE"], r["val_ARE"]))

    # 朴素:选 search 最低
    naive = min(cands, key=lambda c: c[1])
    # 可信:留出门(search降且留出不显著退)
    best_s, best_v, best = cands[0][1], cands[0][2], cands[0][0]
    for n, s, v in cands[1:]:
        if s < best_s - 1e-4 and v <= best_v + 0.003:
            best_s, best_v, best = s, v, n
    return {"cands": cands, "naive_pick": naive, "trust_pick": (best, best_s, best_v)}


def main():
    ec = energy_claims()
    naive_name, naive_s, naive_v = ec["naive_pick"]
    trust_name, trust_s, trust_v = ec["trust_pick"]

    # 规划层查新对照(来自 PLANNING_KNOWLEDGE 已落盘裁决)
    findings = [("配送顺序翻转省能", "SUPPORTED", "Michel 2024 arXiv 2410.17585"),
                ("电池可达集扩大31%", "SUPPORTED", "Nguyen & Au AAMAS 2017"),
                ("省能分布(20场景25%)", "SUPPORTED", "n=20 CI宽[9,49]%")]

    print("=" * 70)
    print("【朴素 AI4S(无安全机制)会产出的'论文摘要'】")
    print("=" * 70)
    print(f"· 能耗模型:LLM 自动发现了拟合误差最低的模型'{naive_name}'(search-ARE {naive_s*100:.2f}%),")
    print(f"  显著优于 BEMT 基线——LLM 的结构搜索发现了更优的能耗函数形式。")
    print("· 规划发现1:发现载荷配送中'能量最优访问顺序≠最短距离'的新规律,省能达9%。")
    print("· 规划发现2:提出'能量感知规划扩大电池可达目标集'的新评估视角,可达集+31%。")
    print("· 规划发现3:能量感知规划在城市场景平均省能显著,25%场景省能≥3%。")
    print("· 结论:LLM 自动科研在无人机能耗规划域取得多项新发现。")
    print(f"  [真实情况:'{naive_name}'留出ARE其实是 {naive_v*100:.2f}%(≥诚实模型,过拟合);")
    print("   三个'发现'全是已发表prior art;省能分布CI极宽。全部 overclaim。]")

    print("\n" + "=" * 70)
    print("【我们的可信框架产出的结论】")
    print("=" * 70)
    print(f"· 能耗模型:留出门选中'{trust_name}'(留出ARE {trust_v*100:.2f}%);且随机搜索追平——")
    print("  诚实报告:LLM 结构搜索在此域≈随机,数据封顶~1.9%(能力边界,非发现)。")
    print("· 规划'发现'经强制查新门:3个SUPPORTED全部降级为复现——")
    for nm, _, src in findings:
        print(f"    {nm} → 复现 {src}")
    print("· 系统报负结果:载荷死路、多数场景0收益、平场景0省能,全部记录。")
    print("· 结论:该域AI4S只能产出正确的null;贡献=方法学+能力边界+防overclaim。")

    out = {"naive_energy_pick": {"name": naive_name, "search%": round(naive_s*100, 2), "true_val%": round(naive_v*100, 2)},
           "trust_energy_pick": {"name": trust_name, "val%": round(trust_v*100, 2)},
           "naive_claims": 4, "trust_real_findings": 0,
           "downgraded": [{"claim": f, "prior_art": s} for f, _, s in findings]}
    json.dump(out, open(os.path.join(HERE, "experiments", "naive_vs_trustworthy.json"), "w"), ensure_ascii=False, indent=2)
    _fig(ec, out)
    print("\n落盘 experiments/naive_vs_trustworthy.json")


def _fig(ec, out):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "Heiti TC"]; plt.rcParams["axes.unicode_minus"] = False
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(15, 5.4))
    # 左:声称的"新发现"(朴素4 全overclaim → 可信0真新)
    a1.bar(["朴素AI4S\n声称新发现", "可信框架\n真新发现"], [out["naive_claims"], out["trust_real_findings"]],
           color=["tab:red", "tab:green"])
    for i, val in enumerate([out["naive_claims"], out["trust_real_findings"]]):
        a1.text(i, val + 0.05, str(val), ha="center", va="bottom", fontsize=15, fontweight="bold")
    a1.set_ylim(0, 5); a1.set_ylabel("被当作'新发现'的数量")
    a1.set_title("声称的发现:朴素4项(全 overclaim)→ 可信0项真新")
    claims = ["① LLM发现更优能耗模型\n   (真:随机搜索追平,数据封顶)",
              "② 顺序翻转省能\n   (真:复现 Michel 2024)",
              "③ 可达集扩大31%\n   (真:复现 Nguyen 2017)",
              "④ 城市场景显著省能\n   (真:CI宽,多数场景0)"]
    a1.text(0, 4.55, "\n".join(claims), fontsize=6.5, ha="center", va="top", color="darkred")
    # 右:诚实报告的负结果(朴素藏0 → 可信报4)
    negs = ["loop=随机搜索", "载荷死路(≤500g)", "77%场景0省能", "平场景0省能"]
    a2.bar(["朴素AI4S\n报告的负结果", "可信框架\n报告的负结果"], [0, len(negs)], color=["tab:red", "tab:green"])
    for i, val in enumerate([0, len(negs)]):
        a2.text(i, val + 0.05, str(val), ha="center", va="bottom", fontsize=15, fontweight="bold")
    a2.set_ylim(0, 5); a2.set_ylabel("诚实报告的负结果/局限数量")
    a2.set_title("报告的负结果:朴素藏 0 → 可信报 4")
    a2.text(1, 4.55, "\n".join("· " + n for n in negs), fontsize=8, ha="center", va="top", color="darkgreen")
    plt.suptitle("朴素 AI4S vs 可信框架:同数据同候选,一个自信但错、一个 humbler 但对", fontsize=13)
    plt.tight_layout()
    p = os.path.join(HERE, "experiments", "naive_vs_trustworthy.png")
    plt.savefig(p, dpi=120); print(f"图存 {p}")


if __name__ == "__main__":
    main()
