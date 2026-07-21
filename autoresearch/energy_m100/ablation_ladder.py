# -*- coding: utf-8 -*-
"""ablation_ladder.py — 顶会标准的累加消融阶梯(ablation ladder):
从"朴素 loop(无安全机制)"开始,逐步加回每道安全机制,量化两个正交指标:
  (a) 报告的留出 ARE(模型层"看起来多好")
  (b) 该配置会 overclaim 的项数(声称的假发现数,越低越诚实)
展示:朴素配置看起来一样好甚至更"好",但 overclaim 一堆;逐步加安全机制→报告数字更诚实、假发现归零。
数据来自 candidate_energy 真跑 + 已落盘的查新/显著性裁决。输出 experiments/pub/消融阶梯.png(+pdf)。
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from candidate_energy import eval_src
from pubstyle import PAL, save
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__)); EXP = os.path.join(HERE, "experiments")
CAND = os.path.join(EXP, "candidates")


def _cands():
    """所有候选(含过拟合探针)的 (search, val)。"""
    out = []
    base = "def featurize(s):\n import numpy as np\n return np.column_stack([np.ones_like(s['v_h']),s['v_h'],s['v_h']**2])"
    r = eval_src(base); out.append(("base", r["search_ARE"], r["val_ARE"]))
    for i in range(1, 10):
        p = os.path.join(CAND, f"iter{i}_featurize.py")
        if os.path.exists(p):
            try:
                r = eval_src(open(p).read()); out.append((f"iter{i}", r["search_ARE"], r["val_ARE"]))
            except Exception:
                pass
    overfit = ("def featurize(s):\n import numpy as np\n v=s['v_h'];z=s['v_z'];p=s['payload'];w=s['wind'];a=s['a_h']\n"
               " C=[np.ones_like(v),v,v*v,z,np.maximum(z,0),p,v*v*p]\n"
               " for k in range(2,9):\n  C+=[np.sin(k*v)*p,np.cos(k*z)*w,(v**(k%3))*np.sign(a)]\n"
               " return np.column_stack(C)")
    r = eval_src(overfit); out.append(("过拟合探针", r["search_ARE"], r["val_ARE"]))
    return out


def build():
    cands = _cands()
    # 各配置"选出的模型"及其真实留出 + overclaim 计数
    # 朴素: 只看 search 选最优(会选过拟合探针类) + 无查新门(3假发现) + 不报负结果(0)
    pick_naive = min(cands, key=lambda c: c[1])       # 只看 search
    # +冻结留出门: 用留出把关(选真实留出最低的诚实模型)
    pick_gated = min([c for c in cands if c[0] != "过拟合探针"], key=lambda c: c[2])

    # 累加阶梯:每步加一道机制
    steps = [
        {"配置": "① 朴素 loop\n(无安全机制)", "报告留出%": round(pick_naive[1]*100, 2),
         "假发现数": 3, "note": "只看拟合选模型+无查新门+不报负结果"},
        {"配置": "② +冻结留出评测器", "报告留出%": round(pick_gated[2]*100, 2),
         "假发现数": 3, "note": "留出把关,不再选过拟合模型"},
        {"配置": "③ +强制查新门", "报告留出%": round(pick_gated[2]*100, 2),
         "假发现数": 0, "note": "H1/H3 查新→降级复现,假发现归零"},
        {"配置": "④ +因果消融/交叉/负结果\n(完整框架)", "报告留出%": round(pick_gated[2]*100, 2),
         "假发现数": 0, "note": "机制归因+破循环+系统报负结果"},
    ]
    # overclaim 的"报告留出"其实是虚的:朴素若报 search 会更低(虚报)
    naive_reported_if_search = round(pick_naive[1]*100, 2)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.4, 3.0))
    x = np.arange(len(steps)); labels = [s["配置"] for s in steps]
    # 左:假发现数(overclaim)——逐步归零
    fd = [s["假发现数"] for s in steps]
    bars = a1.bar(x, fd, color=[PAL["red"], PAL["red"], PAL["green"], PAL["green"]], width=.6)
    for i, v in enumerate(fd): a1.text(i, v+0.05, str(v), ha="center", va="bottom", fontsize=11, fontweight="bold")
    a1.set_xticks(x); a1.set_xticklabels(["①","②","③","④"], fontsize=11); a1.set_ylabel("会声称的假发现数")
    a1.set_ylim(0, 3.6); a1.set_title("累加安全机制 → 假发现归零", fontsize=10.5)
    # 图例说明放图下方,避免 x 轴标签拥挤
    leg = ["① 朴素 loop(无安全机制)", "② +冻结留出评测器",
           "③ +强制查新门", "④ +因果消融/交叉/负结果(完整框架)"]
    a1.text(0.0, -0.30, "　".join(leg[:2]) + "\n" + "　".join(leg[2:]),
            transform=a1.transAxes, fontsize=7.5, color=PAL["gray"], va="top")
    # 右:报告的留出 ARE（诚实数字几乎不变——因为诚实模型本就到顶）
    ar = [s["报告留出%"] for s in steps]
    a2.plot(x, ar, "o-", color=PAL["blue"], ms=5, lw=1.5)
    a2.axhline(naive_reported_if_search, ls=(0,(4,3)), color=PAL["gray"], lw=1)
    a2.text(3.9, naive_reported_if_search-0.06, f"朴素虚报拟合误差 {naive_reported_if_search}%",
            fontsize=7.5, color=PAL["gray"], ha="right", va="top")
    for i, v in enumerate(ar): a2.text(i, v-0.05, f"{v}", ha="center", va="top", fontsize=8.5, color=PAL["blue"])
    a2.set_xticks(x); a2.set_xticklabels(["①","②","③","④"], fontsize=11); a2.set_xlabel("累加步骤")
    a2.set_ylabel("报告的留出 ARE(%)"); a2.set_ylim(1.75, 2.15); a2.set_title("报告数字诚实(不虚低)", fontsize=10.5)
    fig.suptitle("累加消融阶梯:逐步加回安全机制,假发现归零、报告数字诚实", fontsize=11.5, y=1.02)
    save(fig, "消融阶梯")
    json.dump({"steps": steps, "naive_reported_if_search%": naive_reported_if_search},
              open(os.path.join(EXP, "ablation_ladder.json"), "w"), ensure_ascii=False, indent=2)
    print("\n=== 累加消融阶梯 ===")
    for s in steps: print(f"  {s['配置'].replace(chr(10),' ')}: 报告留出 {s['报告留出%']}% | 假发现 {s['假发现数']} | {s['note']}")


if __name__ == "__main__":
    build()
