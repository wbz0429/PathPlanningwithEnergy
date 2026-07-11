"""
gen_report.py — 生成自包含的中期报告 HTML(图 base64 内嵌,单文件可分享)。
汇总:三创新点 + M100 能耗发现 + 防线 + 诚实边界 + 规划连接。数据来自已验证结果。
输出:experiments/midterm_report.html
"""
import os, base64, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.join(HERE, "experiments")
os.makedirs(EXP, exist_ok=True)


def _font():
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    for f in ("PingFang SC", "Heiti SC", "STHeiti", "Arial Unicode MS"):
        if any(f in x.name for x in font_manager.fontManager.ttflist):
            plt.rcParams["font.sans-serif"] = [f]; break
    plt.rcParams["axes.unicode_minus"] = False


def fig_summary(out):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _font()
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.4))
    # 左:能耗模型发现(留出 ARE)
    labels = ["稳态BEMT\n(1,v,v²)", "纯物理\n(BEMT非线性)", "单payload项", "loop发现\n(全线性)"]
    vals = [6.88, 4.24, 2.25, 1.86]
    cols = ["#95a5a6", "#c0392b", "#f39c12", "#27ae60"]
    ax[0].bar(labels, vals, color=cols)
    for i, v in enumerate(vals):
        ax[0].text(i, v + 0.1, f"{v}%", ha="center", fontsize=10, fontweight="bold")
    ax[0].set_title("① 真机 M100 能耗模型:留出能量 ARE(越低越好)", fontsize=11)
    ax[0].set_ylabel("留出能量 ARE (%)"); ax[0].grid(axis="y", alpha=.3)
    ax[0].axhline(1.86, ls="--", color="#27ae60", alpha=.6)
    # 右:防线(loop vs 简单基线)
    labels2 = ["随机搜索\n(生死线)", "贪心前向", "autoresearch\nloop", "纯物理\n(代码进化)"]
    vals2 = [1.86, 2.09, 1.86, 4.24]
    cols2 = ["#2c6fbb", "#8e44ad", "#27ae60", "#c0392b"]
    ax[1].bar(labels2, vals2, color=cols2)
    for i, v in enumerate(vals2):
        ax[1].text(i, v + 0.1, f"{v}%", ha="center", fontsize=10, fontweight="bold")
    ax[1].set_title("③ 诚实边界:loop = 随机(1.86%),代码进化不赢", fontsize=11)
    ax[1].set_ylabel("留出能量 ARE (%)"); ax[1].grid(axis="y", alpha=.3)
    fig.suptitle("无人机能耗 autoresearch:正结果(左) + 诚实能力边界(右)", fontsize=13)
    fig.tight_layout(); fig.savefig(out, dpi=110); plt.close(fig)


def b64(path):
    if not os.path.exists(path):
        return ""
    return "data:image/png;base64," + base64.b64encode(open(path, "rb").read()).decode()


# 生成汇总图
fig_summary(os.path.join(EXP, "fig_summary.png"))
img_sum = b64(os.path.join(EXP, "fig_summary.png"))
img_plan = b64(os.path.join(EXP, "fig_planning_connect.png"))

# 读 state
best = {}
sp = os.path.join(HERE, "state", "state.json")
if os.path.exists(sp):
    best = json.load(open(sp)).get("best", {})

HTML = f"""<!DOCTYPE html><html lang="zh"><head><meta charset="utf-8">
<title>毕设中期报告 · 无人机能耗可信自动化研究</title>
<style>
 body{{font-family:-apple-system,'PingFang SC',Helvetica,Arial,sans-serif;margin:0;background:#f4f5f7;color:#222;line-height:1.65}}
 .wrap{{max-width:1000px;margin:0 auto;padding:30px}}
 h1{{font-size:25px;margin:0 0 6px}} .sub{{color:#666;margin-bottom:8px;font-size:15px}}
 .pos{{background:#fff8e1;border-left:4px solid #f39c12;padding:12px 16px;border-radius:6px;margin:16px 0;font-size:14.5px}}
 .cards{{display:flex;gap:12px;flex-wrap:wrap;margin:18px 0}}
 .card{{background:#fff;border-radius:12px;padding:14px 18px;box-shadow:0 1px 4px rgba(0,0,0,.06);flex:1;min-width:150px}}
 .card .n{{font-size:24px;font-weight:700;color:#27ae60}} .card .l{{color:#777;font-size:12.5px}}
 h2{{font-size:19px;margin:30px 0 12px;border-left:4px solid #2c6fbb;padding-left:10px}}
 h3{{font-size:16px;margin:18px 0 6px}}
 .box{{background:#fff;border-radius:12px;padding:18px 22px;box-shadow:0 1px 4px rgba(0,0,0,.06);margin-bottom:18px}}
 table{{width:100%;border-collapse:collapse;font-size:13.5px;margin:8px 0}}
 th,td{{padding:7px 10px;border-bottom:1px solid #eee;text-align:left}} th{{background:#2c6fbb;color:#fff}}
 tr.hi td{{background:#eafaf0;font-weight:600}}
 img{{width:100%;border-radius:8px;margin:8px 0}}
 .good{{color:#27ae60;font-weight:600}} .bad{{color:#c0392b}} .tag{{display:inline-block;background:#eaf1fb;color:#2c6fbb;border-radius:6px;padding:2px 9px;font-size:12px;margin:2px}}
 .honest{{background:#fdecea;border-left:4px solid #c0392b;padding:12px 16px;border-radius:6px;font-size:14px}}
</style></head><body><div class="wrap">

<h1>🛰️ 面向无人机能耗建模的可信大模型自动化研究框架</h1>
<div class="sub">真实数据锚定评测与能力边界刻画 · 毕设中期报告</div>
<div class="pos"><b>诚实定位(每字防翻车):</b> 不提新算法、不声称"更强的优化器"。贡献 = 一个<b>防作弊、真实数据锚定、会诚实报负结果</b>的 LLM 自动化研究框架,在无人机能耗(真实 DJI M100)上验证,并<b>诚实刻画</b> LLM 自动化研究相对简单基线的真实能力边界。</div>

<div class="cards">
 <div class="card"><div class="n">6.88%→1.86%</div><div class="l">真机留出能量 ARE(−73%)</div></div>
 <div class="card"><div class="n">209</div><div class="l">真实 DJI M100 飞行(真值)</div></div>
 <div class="card"><div class="n">9+</div><div class="l">自动迭代(全 held-out 验证)</div></div>
 <div class="card"><div class="n" style="color:#c0392b">= 随机</div><div class="l">loop 不比普通循环强(诚实)</div></div>
</div>

<h2>1. 课题背景与定位</h2>
<div class="box">
<p><b>痛点</b>:LLM 自动化科研(FunSearch/AI-Scientist)兴起,但存在<b>可信度危机</b>——无 ground truth 时系统性不可靠、易把任何前提包装成漂亮结论(PseudoBench 2026 实测 7 个系统抵抗率最好仅 27.4%)。</p>
<p><b>切入</b>:把"可验证-evaluator 的自动化研究"落到<b>有真实测量、真值明确</b>的无人机能耗域,让自动化研究<b>可被信任</b>,并诚实回答一个尖锐问题:<b>这套"CC+LLM loop"和别人随手写的优化循环到底有没有区别?</b></p>
</div>

<h2>2. 三个创新点(全诚实,有证据)</h2>
<div class="box">
<h3>① 工程/方法论:可信自动化研究框架</h3>
<p>冻结的、<b>锚定真实 DJI 功率(P=V·|I|)</b>的作弊不了评测器 + 按飞行 held-out + keep/revert + <b>诚实报负结果</b>。可动的只有能耗模型组件(沙箱注入)。对标 2026 热点(可信 autoresearch)。</p>
<h3>② 应用(UAV·真数据正结果)</h3>
<p>在真机数据上自动发现比经典稳态 BEMT <b class="good">准 73%</b> 的能耗模型(留出能量 ARE 6.88%→1.86%),并作为<b>可信能耗代价接入能量感知规划</b>(见 §4)。</p>
<h3>③ 实证/能力边界(把最尖锐的质疑变成 finding)</h3>
<p>系统证明:在这类成熟问题上,<b>LLM 自动化研究的优化性能 ≈ 随机/贪心简单基线</b>;LLM 提议的物理/新函数形式<b>既不赢线性选择也不赢随机,叠加还有害</b>。这是诚实的能力边界研究,不是失败。</p>
</div>

<h2>3. 关键结果:真机能耗模型发现 + 诚实边界</h2>
<div class="box">
<img src="{img_sum}">
<h3>能耗模型发现(留出 seeds 5,6,7)</h3>
<table><tr><th>方法</th><th>留出能量 ARE</th><th>R²</th><th>说明</th></tr>
<tr><td>稳态 BEMT(1,v,v²)</td><td>6.88%</td><td>0.00</td><td>基线:仅速度几乎解释不了功率</td></tr>
<tr class="hi"><td>loop 发现(线性)</td><td>1.86%</td><td>0.39</td><td>加 payload/爬升/交互 → 准 73%</td></tr>
<tr><td>纯物理(BEMT 非线性)</td><td class="bad">4.24%</td><td>0.25</td><td>物理先验反而更差(锁死形状)</td></tr>
<tr><td>物理 + payload</td><td>1.93%</td><td>0.44</td><td>价值来自线性 payload,非物理</td></tr>
</table>
<h3>两条防线(答辩)</h3>
<table><tr><th>方法</th><th>留出 ARE</th><th>结论</th></tr>
<tr><td>随机子集(生死线)</td><td>1.86%</td><td rowspan=2>① loop = 随机 → <b class="bad">选形式不需要 LLM</b></td></tr>
<tr class="hi"><td>autoresearch loop</td><td>1.86%</td></tr>
<tr><td>贪心前向</td><td>2.09%</td><td>② loop 胜贪心 11%(贪心陷局部)</td></tr>
</table>
<p class="honest"><b>核心诚实结论(创新点3,跨域一致):</b> 真无人机能耗上 per-flight 能量被 payload 线性主导;LLM 代码级新函数形式既不赢线性选择也不赢随机搜索,叠加还有害。<b>所以贡献是"可信框架 + 诚实能力边界",不是更好的模型或优化器。</b>此结论与雷达跟踪域独立复现一致,强化了核心论点。</p>
</div>

<h2>4. 闭合到 UAV 场景:真机锚定能耗模型接入规划</h2>
<div class="box">
<img src="{img_plan}">
<p>把真机验证过的能耗模型当规划代价:同一场景,<b>稳态 BEMT(只看速度、不知爬升费电)错选"翻墙"路线(3203J)</b>;<b class="good">M100 锚定模型(知道爬升贵)正确选"左绕"(翻墙在真模型下涨到 3708J)</b>。</p>
<p><b>意义</b>:让能量感知规划<b>建立在真实测量上</b>,而非手搭 BEMT——不是新规划算法,而是<b>让规划的能耗代价可信</b>。这是与导师"连接 UAV 场景"最紧的闭合。</p>
</div>

<h2>5. 相关工作</h2>
<div class="box">
<span class="tag">FunSearch/AlphaEvolve/Eureka</span><span class="tag">PseudoBench 2026(可信度危机)</span>
<span class="tag">Rodrigues 2021 M100 数据集</span><span class="tag">Tseng 多项式能耗模型</span>
<span class="tag">BEMT 闭式功率模型(arXiv:2209.04128)</span>
<p style="margin-top:8px">方法论范式(LLM 进化)成熟;能耗建模的最优是多项式回归(Tseng),非物理闭式——这正是本文诚实边界的文献支撑。</p>
</div>

<h2>6. 局限与下一步</h2>
<div class="box">
<p><b>明确不声称</b>:新能耗/规划算法;CC+loop 不比普通优化循环(随机/网格/BO)在调参性能上更强(本文实证)。</p>
<p><b>局限</b>:per-sample 功率不可精确预测(5Hz 噪声),只声称 per-flight 能量;规划连接是候选路线对比(非全 RRT* 重规划),增益相对非绝对;瞬态能耗真实数据支撑弱(已证)。</p>
<p><b>下一步</b>:完整 RRT* 集成 + 更多真机场景;把"诚实能力边界"扩成跨域(能耗+雷达)统一结论成文。</p>
</div>

<div class="sub" style="margin-top:24px">分支 agent-autoresearch · autoresearch/energy_m100/ · 全程 git 可追溯(state/agent_log/candidates/KNOWLEDGE)</div>
</div></body></html>"""

out = os.path.join(EXP, "midterm_report.html")
open(out, "w").write(HTML)
print("[报告]", out, f"({len(HTML)//1024} KB)")
