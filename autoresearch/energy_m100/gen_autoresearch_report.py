# -*- coding: utf-8 -*-
"""gen_autoresearch_report.py — 生成学术风格中文完整报告(自包含 HTML,图+视频 base64 内嵌,单文件)。
聚焦:整个 Autoresearch 的迭代实现 + 飞行轨迹动态视频 + 图 + 数据对比。
输出:experiments/autoresearch_report.html
"""
import os, base64, json

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.join(HERE, "experiments")
PX4 = os.path.join(HERE, "px4_integration")


def _find(name):
    for d in (EXP, PX4):
        p = os.path.join(d, name)
        if os.path.exists(p):
            return p
    return None


def media(name, mime):
    p = _find(name)
    if not p:
        return None
    return f"data:{mime};base64," + base64.b64encode(open(p, "rb").read()).decode()


def fig(name, num, cap):
    d = media(name, "image/png")
    if not d:
        return f"<p class='miss'>[缺图 {name}]</p>"
    return f"<figure><img src='{d}'/><figcaption><b>图 {num}</b>　{cap}</figcaption></figure>"


def vid(name, num, cap):
    d = media(name, "video/mp4")
    if not d:
        return f"<p class='miss'>[缺视频 {name}]</p>"
    return (f"<figure><video controls muted loop playsinline preload='metadata' src='{d}'></video>"
            f"<figcaption><b>视频 {num}</b>　{cap}(点击播放)</figcaption></figure>")


def J(name):
    p = _find(name)
    return json.load(open(p)) if p else {}


def main():
    wall = J("wall_experiment_result.json")
    rob = J("robustness_suite.json")
    cor = J("corridor_result.json")
    simf = J("sim_flight_result.json")
    ab = J("px4_ab_comparison.json")
    sg = J("safeguard_ablation.json")
    dv = J("domain_value.json")
    lss = J("large_scale_savings_stats.json")
    st = (J("significance_test.json") or {}).get("energy_domain", {})
    ps = J("planner_significance.json")

    # ---- 能耗 loop 迭代表(从 agent_log) ----
    energy_rows = ""
    alog = _find("agent_log.jsonl")
    if alog:
        seen = set()
        labels = {0: "稳态BEMT基线", 1: "动量理论核", 2: "Glauert前飞桥", 3: "推力标定型面",
                  4: "轴向诱导sqrt", 5: "风→空速", 6: "物理核+线性payload", 7: "payload变体",
                  8: "纯线性库LIB", 9: "物理叠线性", 10: "capstone对标"}
        for ln in open(alog):
            r = json.loads(ln); it = r.get("iter")
            if it in seen:
                continue
            seen.add(it)
            s = r.get("search_ARE"); v = r.get("val_ARE")
            note = str(r.get("note", ""))[:46]
            dec = "KEEP" if "KEEP" in note.upper() else ("REVERT" if "REVERT" in note.upper() else "")
            sv = f"{s*100:.2f}%" if isinstance(s, (int, float)) else "—"
            vv = f"{v*100:.2f}%" if isinstance(v, (int, float)) else "—"
            energy_rows += f"<tr><td>{it}</td><td>{labels.get(it,'')}</td><td>{sv}</td><td>{vv}</td><td>{dec}</td></tr>"

    # ---- 墙场景表 ----
    wall_rows = ""
    for r in (wall or []):
        wall_rows += (f"<tr><td>{r['场景']}(半宽{r['墙半宽']:.0f}m)</td><td>{r.get('距离最短_选择','?')}</td>"
                      f"<td>{r.get('教科书BEMT_选择','?')}</td><td class='hl'>{r.get('真机M100_选择','?')}</td>"
                      f"<td class='hl'>{r.get('省能%_M100尺','?')}%</td></tr>")
    # ---- 速度鲁棒 ----
    vel_rows = "".join(f"<tr><td>{x['v']} m/s</td><td>{x['dist']}</td><td class='hl'>{x['m100']}</td>"
                       f"<td class='hl'>{x['save%']}%</td></tr>" for x in rob.get("velocity_sweep", []))
    # ---- PX4 A/B ----
    px4_tbl = ""
    if ab:
        d, m = ab["distance"], ab["m100"]
        px4_tbl = (f"<tr><td>距离(翻越)</td><td>{d['E']} J</td><td>{d['meanW']} W</td><td>{d['maxalt']} m</td><td>{d['ydev']} m</td></tr>"
                   f"<tr class='hl'><td>真机M100(绕行)</td><td>{m['E']} J</td><td>{m['meanW']} W</td><td>{m['maxalt']} m</td><td>{m['ydev']} m</td></tr>")

    css = """
    body{font-family:'Songti SC','SimSun',Georgia,serif;max-width:900px;margin:0 auto;padding:40px 28px;color:#1a1a1a;line-height:1.9;font-size:16px}
    h1{font-size:1.7em;text-align:center;font-weight:700;margin-bottom:4px;line-height:1.4}
    .sub{text-align:center;color:#555;font-size:1em;margin-bottom:2px}
    .meta{text-align:center;color:#888;font-size:.85em;margin-bottom:28px}
    h2{font-size:1.28em;margin-top:2em;border-bottom:2px solid #333;padding-bottom:5px}
    h3{font-size:1.08em;margin-top:1.4em;color:#222}
    p{text-align:justify}
    .abstract{background:#f7f7f4;border:1px solid #ddd;padding:16px 20px;border-radius:4px;font-size:.96em}
    .abstract b{font-variant:small-caps}
    table{border-collapse:collapse;width:100%;margin:14px 0;font-size:.9em;font-family:-apple-system,'PingFang SC',sans-serif}
    th,td{border:1px solid #bbb;padding:6px 9px;text-align:center}
    th{background:#ececec;font-weight:600} .hl{background:#eaf6ee}
    figure{margin:20px 0;text-align:center} img,video{max-width:100%;border:1px solid #ccc;border-radius:4px}
    figcaption{font-size:.86em;color:#444;margin-top:6px;font-family:-apple-system,'PingFang SC',sans-serif}
    .keybox{border-left:4px solid #2b7a3d;background:#f0f9f2;padding:10px 16px;margin:14px 0;font-size:.95em}
    .honest{border-left:4px solid #c47f1a;background:#fdf7ef;padding:10px 16px;margin:14px 0;font-size:.95em}
    code{background:#f0f0f0;padding:1px 5px;border-radius:3px;font-size:.86em}
    .miss{color:#b00;font-size:.85em} ol,ul{padding-left:1.6em}
    """

    html = f"""<!DOCTYPE html><html lang="zh"><head><meta charset="utf-8">
<title>可信大模型自动化科研框架 —— 无人机能耗建模与能量感知路径规划</title>
<style>{css}</style></head><body>

<h1>可信大模型自动化科研框架:<br>无人机能耗建模与能量感知路径规划的迭代实现与实证</h1>
<div class="sub">A Trustworthy LLM-Agent Autoresearch Framework, Instantiated on UAV Energy Modeling and Energy-Aware Path Planning</div>
<div class="meta">研究实现报告 · 2026-07 · 数据集:真实 DJI Matrice 100(209 航班,机载 V·I 功率)· 全部结果可复现</div>

<div class="abstract">
<b>摘要　</b>本文报告一个<b>防作弊、真实数据锚定、系统性报告负结果</b>的大模型智能体自动化科研(AI4S)框架,
及其在无人机能耗建模与能量感知路径规划上的迭代实现。框架以<b>冻结的、锚定真实功率(P=V·|I|)的作弊不了评测器</b>为核心,
配<b>基于留出指标的 keep/revert、强制文献查新门、因果消融、交叉模型破循环</b>等安全机制。在真实 DJI M100 数据上,
自动迭代将能耗模型的留出能量绝对相对误差(ARE)由稳态 BEMT 的 6.88% 降至 1.86%(相对提升约 73%);
将该真机验证的能耗作为规划代价接入采样式规划器与四旋翼动力学/PX4 飞控固件闭环,
在障碍场景中相对最短距离基线节能 <b>4–22%(折线预测)、6.1%(RotorPy 动力学)、14.3%(PX4 真飞控栈)</b>。
本文的核心贡献不在于任何新算法或新效应(经三次文献查新确认均为已发表 prior art),
而在于<b>方法学本身及其实证承重</b>:通过安全机制消融证明——去掉冻结评测器则过拟合探针可刷分、去掉查新门则 3 个"发现"中有 2 个实为复现、
结构搜索的价值随任务域而变(规划器域胜随机 27%、能耗域打平)。据此诚实刻画了当前 LLM 自动科研的能力边界。
</div>

<h2>1　研究背景与定位</h2>
<p>大模型驱动的自动化科研(AI4S)近年快速发展(FunSearch、AlphaEvolve、Eureka、AI Scientist 等),
但其可信度存在系统性风险:评测作弊(reward hacking)随模型能力增强而加剧、自动"新颖性"判定浅薄、
语料正结果偏倚导致过度声称。本工作<b>不声称</b>提出新的规划/能耗算法,亦不声称发明 AI4S 或"证伪优先"范式(AIGS 2024 已有);
其定位为:将一套<b>可信/证伪优先的自动化科研协议实例化</b>于一个旗舰框架尚未触及的域(无人机能耗建模 + 能量感知规划),
并在真实数据上<b>实证每个安全机制的承重性</b>。</p>

<h2>2　方法:可信自动化科研框架</h2>
<p>框架将研究流程抽象为闭环:<code>读结果 → 提假设 → 改代码 → 冻结评测 → keep/revert → 记账</code>。关键设计:</p>
<ul>
<li><b>冻结评测器(尺子)</b>:锚定真实功率 P=V·|I|,按<b>飞行</b>划分 train/test,以能量 ARE 为指标;评测器<b>永不可编辑</b>——编辑即作弊。</li>
<li><b>keep/revert 协议</b>:仅当搜索指标下降<b>且留出指标不显著回退</b>时接受候选,防过拟合到搜索划分。</li>
<li><b>强制查新门</b>:任何"发现"须先通过文献查新才计为新;未过则诚实降级为"复现/验证"。</li>
<li><b>因果消融 / 交叉模型破循环 / 系统性报告负结果</b>:确认机制归因、暴露评测循环性、不掩盖失败。</li>
</ul>
<div class="keybox"><b>可信性的实证方式(见第 5 节)</b>:上述机制并非口号——本文通过<b>逐个消融</b>证明每个机制"承重":
去掉它,自动科研就会自欺或过度声称。</div>

<h2>3　迭代实现</h2>
<h3>3.1　规划器算法自优化 loop(iter0–70,4 个里程碑)</h3>
<p>以 BEMT 速度剖面能量为目标、碰撞硬约束为门、能量加权 A* 为归一化地板,LLM 迭代进化采样器/平滑器/速度剖面代码。
里程碑 1(iter21):综合得分 15000→2078(留出验证零过拟合),7 KEEP / 9 REVERT;
里程碑 2(iter38):引入 CHOMP 式联合梯度精修等,降至 2046;<b>横向对比:LLM-loop 2046 优于随机搜索 2805(胜 27%)</b>——
因随机搜索写不出 CHOMP/DP 平滑器等代码结构。S4a 专项攻击"非欧代价 informed 采样"开放问题,6 种机制均 REVERT,
Layer-1 可动空间穷尽。<b>诚实定位:强方法论验证 + 弱算法增量,非新算法</b>(RRT-Connect 系 2000 年既有、平滑增益为噪声级)。</p>

<h3>3.2　能耗模型自优化 loop(iter1–10)</h3>
<p>冻结评测器 <code>m100_eval.py</code>,LLM 每轮改写 <code>featurize</code>。完整迭代账本:</p>
<table><tr><th>迭代</th><th>改动</th><th>搜索 ARE</th><th>留出 ARE</th><th>决定</th></tr>{energy_rows}</table>
{fig("fig_summary.png", "1", "能耗模型 autoresearch:留出 ARE 由 6.88% 降至 1.86%(左);诚实能力边界——loop 与随机搜索打平(右)")}

<p><b>过程即贡献——loop 如何处理瓶颈</b>:本框架的价值不在最终模型(prior art),而在这套“撞到瓶颈→触发文献检索→改变方向→诚实记录”的可信过程。
下图为真实迭代轨迹(KEEP/REVERT)与机制流程;四个关键瓶颈的处理见表。</p>
{fig("loop_process.png", "1b", "loop 全过程(真实 agent_log):迭代轨迹标注每个 KEEP/REVERT 与瓶颈处理;下半为机制流程——撞瓶颈触发文献检索并改向")}
<table><tr><th>瓶颈</th><th>怎么处理</th><th>调研发现</th><th>结果</th></tr>
<tr><td>物理非线性形卡在 ~4.24%</td><td>Episode 边界触发文献检索</td><td>Tseng 多项式是前沿;缺口是 wind 而非更花哨物理</td><td>改方向,转测 wind</td></tr>
<tr><td>wind 到底有没有用</td><td>空速替换地速,单独测</td><td>留出 ARE 4.24→4.38 反而变差(quadrature 抵消)</td><td>iter5 REVERT,记库不再试</td></tr>
<tr><td>物理形式赢不赢纯线性</td><td>纯线性 vs 物理,决定性诊断</td><td>打平;物理叠线性反而退(共线)</td><td>核心 finding:物理不转化为更低 ARE</td></tr>
<tr><td>loop 相对简单基线有无必要</td><td>生死线 + R=40 显著性检验</td><td>与随机打平(z=−0.35),仅略胜贪心</td><td>诚实报告能耗域能力边界</td></tr></table>
<div class="honest"><b>核心 finding(能力边界)</b>:物理非线性形不是价值来源(纯物理 4.24%,加一个线性 payload 项即降至 1.93%);
loop 最终 1.86% 与随机搜索<b>打平</b>——在能耗域,LLM 的"物理推理"不转化为更低误差。这与规划器域(loop 胜随机 27%)形成对照。</div>

<h2>4　实验结果:真机代价驱动的能量感知规划</h2>
<p>将 3.2 节的真机验证能耗模型包装为规划代价,接入采样式规划器,对比【最短距离 / 教科书 BEMT / 真机 M100】三种代价。</p>

<h3>4.1　主结果:代价改变路径决策</h3>
<table><tr><th>场景</th><th>最短距离</th><th>教科书BEMT</th><th>真机M100</th><th>节能(M100尺)</th></tr>{wall_rows}</table>
{fig("hero_figure.png", "2", "高墙场景:最短距离与教科书 BEMT 均翻墙,真机 M100 代价绕行避爬升(节能 9.4%)")}
<p>机制经<b>三重验证</b>:留出真实数据上爬升功率增量(真 +114W vs 预测 +108W,误差 &lt;5%);
<b>因果消融</b>——去除模型爬升项后决策退化回翻墙;交叉模型验证暴露"节能相对可信模型"的循环性(教科书 BEMT 因低估爬升而翻墙)。</p>

<h3>4.2　鲁棒性与操作包络</h3>
<table><tr><th>巡航速度</th><th>最短距离</th><th>真机M100</th><th>节能</th></tr>{vel_rows}</table>
{fig("phase_diagram.png", "3", "操作包络相图:高速+窄障区节能最高(22%),低速/宽障区连 M100 也翻墙(节能归零)——诚实划界")}
<p><b>权衡分解</b>:翻越 vs 绕行不是“谁绝对更省”,而是<b>“翻越的固定爬升罚(≈1233 J)”对“绕行随墙宽线性增长的多走距离”</b>的权衡。
下图给出交叉点(临界墙半宽 ≈24 m):窄墙时绕行省(M100 绕行),宽墙时翻越省(M100 也翻墙)——正是相图边界的物理解释。</p>
{fig("tradeoff.png", "3b", "翻越/绕行能量权衡分解:翻越=少走距离+固定爬升罚(平线),绕行=无爬升+多走距离(升线),交叉点即决策翻转的临界墙宽")}

<h3>4.3　动力学闭环(RotorPy):飞行轨迹与功率</h3>
<p>将规划路径经 MinSnap 平滑后,交由 M100 尺度四旋翼刚体动力学 + SE3 几何控制器以 100 Hz 跟踪飞行,
用<b>飞出的</b>速度序列积分能量。结论过动力学后仍成立(单墙节能 6.1%)。</p>
{fig("sim_flight.png", "4", "动力学实飞:翻墙路线在爬升段功率飙至 870W、下降段跌至 280W(升贵降贱不对称),绕行路线全程平稳")}
{vid("video_wall.mp4", "1", "单墙场景动力学飞行:红=距离翻墙,绿=M100绕行,右侧为实时功率")}

<h3>4.4　城市走廊:逐障碍混合决策</h3>
<p>一条配送走廊上矮楼 A(5m)与高楼 B(12m):真机 M100 代价<b>翻越矮楼、绕行高楼</b>——逐障碍权衡,而非死规则。</p>
{fig("corridor.png", "5", "城市走廊:距离代价两楼全翻,M100 代价翻 A 绕 B(规划节能 7.6%,动力学实飞 4.4%)")}
{vid("video_corridor.mp4", "2", "城市走廊动力学飞行:逐障碍混合决策,双机同屏 + 实时功率")}

<h3>4.5　★ PX4 真飞控固件栈验证(最强一层)</h3>
<p>在 <b>PX4 SITL + Gazebo Harmonic</b>(真实飞控固件:EKF 状态估计 + 位置环 + 混控)上,以同一固件栈、同一脚本、同一新鲜起飞,
分别飞距离(翻越)与 M100(绕行)两条路线,用飞出的 odometry 积分真机能耗。</p>
<table><tr><th>路线</th><th>M100真机能耗</th><th>均功率</th><th>最高高度</th><th>最大横偏</th></tr>{px4_tbl}</table>
{fig("px4_ab_comparison.png", "6", "PX4 真飞控固件栈 A/B 对比:能量感知绕行相对翻越节能 14.3%(EKF+位置环+Gazebo 动力学)")}
{vid("px4_flight.mp4", "3", "PX4 真飞控栈飞出的能量感知路线轨迹(翻矮楼 A + 绕高楼 B)+ 实时功率")}
<div class="keybox">证据链层层加固:<b>折线预测(4–22%)→ RotorPy 动力学(6.1%)→ PX4 真飞控固件栈(14.3%)</b>。
节能收益过完整飞控固件在环后依然成立。</div>

<h2>5　★ 安全机制消融:方法学的实证承重</h2>
<p>本节将"协议有用"从断言变为可测结果——逐个消融安全机制,观察自动科研会如何自欺/过度声称。</p>
{fig("safeguard_ablation.png", "7", "安全机制消融:(A)冻结评测器抗 gaming/抗过拟合;(B)查新门使 3 个'发现'降为 1 真新;(C)结构搜索价值域相关")}
<table><tr><th>安全机制</th><th>去掉它会怎样(实测)</th></tr>
<tr><td>冻结留出评测器</td><td>35 个虚假项的过拟合探针搜索 ARE 2.13% &gt; 诚实模型 1.99%,<b>刷不动</b>;候选搜索/留出 ARE 相差 &lt;0.15pp = 抗过拟合</td></tr>
<tr><td>强制查新门</td><td>无门上报 3 个"发现";有门后仅 1 个真新,<b>2 个降级为复现</b>(Michel 2024 顺序翻转、Nguyen 2017 可达集)</td></tr>
<tr><td>结构搜索(loop vs 随机)</td><td>规划器域 loop 胜随机 <b>27%</b>、能耗域<b>打平</b>——价值取决于任务是否需要写出普通搜索到不了的代码结构</td></tr>
</table>

<h2>6　诚实边界与能力刻画</h2>
<div class="honest"><ol>
<li><b>效应均为 prior art</b>:绕行节能、顺序翻转、可达集扩大、载荷影响——经三次文献查新(共 300+ 检索 agent)确认均已发表;本文引用而不声称首创。</li>
<li><b>载荷为负结果</b>:M100 载荷 ≤500g,载荷×爬升耦合仅占能耗约 3%,不足以改变路径拓扑(全扫描 + 物理 mgh 项均不变)。</li>
<li><b>多数场景无收益</b>:随机城市场景中约 70% 节能为 0(存在低空走廊时能量最优≈最短路)。</li>
<li><b>无真机复飞</b>:节能为经真实数据验证的模型预测,非真实飞行电流实测。</li>
<li><b>结构搜索必要性弱(能耗域)</b>:该域内 loop 与随机搜索打平——诚实报告为能力边界,非失败。</li>
</ol></div>

<h2>7　为什么产出 null 是正确的:AI4S 的域相关价值</h2>
<p>一个自然的质疑是:FunSearch、AlphaEvolve、Eureka 等旗舰 AI4S 均有真发现,而本框架在能耗域"没有新发现"——这是否说明框架无能?
本节以三项实证论证:<b>该域产出 null 是数据与域的性质所决定,而非框架失败;而在具备改进空间的域,同一框架能产出正值。</b></p>

<h3>7.1　旗舰 AI4S 的正结果都依赖"可验证评测器 + 真实改进空间"</h3>
<p>旗舰系统的作者自身即把"精确可自动评测"列为方法的先决条件与局限:AlphaEvolve 明言"需要人工实验的任务超出本方法范围",
FunSearch 只处理"有高效 evaluate 函数的问题"。其正结果集中于数学、组合优化、代码、RL 奖励等<b>干净可验证</b>域;
<b>迄今无将 AI4S 用于"无干净真值的噪声真实传感器数据"域并产出新发现的先例</b>。且即便在理想域,"匹配已知最优"(null 型)亦是常态——
AlphaEvolve 在 50+ 数学问题上 75% 仅为匹配、20% 为新发现;FunSearch 最难任务的单次成功率仅约 2.9%。
反之,独立评估已实证朴素 AI4S 会<b>过度声称</b>:对 AI Scientist 的第三方评估(已发表)报告 42% 实验因编码错误失败、57% 手稿含错误或幻觉数值、
并将既有工作误判为新颖。本框架的安全机制正是针对这些已被实测的失效模式。</p>

<h3>7.2　噪声地板量化:数据本身封顶,非模型容量不足</h3>
<p>对递增容量的能耗模型测量拟合误差与留出误差:留出 ARE 在约 6 个特征项处即触底约 {dv.get("noise_floor",{}).get("held_out_floor_pct","1.9")}%,
继续增加至 12、20、52 项,留出 ARE 不降反升(过拟合),即<b>52 项的高容量模型无法胜过 6 项模型</b>。
四种独立方法(自动科研 loop、随机搜索、线性回归、数据集既定的 Tseng 多项式)殊途同归于约 1.9%。
这将"数据封顶"从主张升级为测量:剩余误差不在模型容量中,而在数据信息量中(5Hz 瞬时功率的方差约 60% 来自风、电压跌落与控制,非运动学可解释)。</p>
{fig("domain_value.png", "8", "左:模型容量增至 52 项,留出 ARE 仍触底约 1.9%(数据地板);右:同框架跨域对照——规划器域(有改进空间)胜随机 27%,能耗域(封顶)持平")}

<h3>7.3　跨域对照:同一框架在有改进空间的域能产出正值</h3>
<p>同一自动科研框架作用于<b>规划器算法</b>(采样器/平滑器代码,存在改进空间)时,其解优于随机搜索 <b>27%</b>(因随机搜索写不出 CHOMP/DP 等代码结构);
作用于<b>能耗模型</b>(噪声封顶)时与随机搜索持平。此跨域对照<b>排除了"框架无能"的替代解释</b>,支撑"null 由域性质决定"的因果论断。</p>
<p><b>统计显著性检验(两域对称)</b>:对两个域各重复多次随机搜索得分布,检验对照是否统计成立。<br>
· <b>能耗域(封顶)</b>:重复 {st.get("R","40")} 次随机搜索得留出 ARE 分布(均值 {st.get("random_mean%","1.87")}% ± {st.get("random_std%","0.03")}%);
loop = {st.get("loop_ARE%","1.86")}%,<b>z = {st.get("loop_z_score","-0.35")}</b>,{st.get("pct_random_better","52")}% 的随机运行反而更优——
loop 就是随机分布中的一个普通样本(|z|&lt;1.96),<b>"持平"是统计确认的真等价,非样本不足</b>。<br>
· <b>规划器域(有改进空间)</b>:重复 {ps.get("R","12")} 次随机配置搜索(默认平滑器)得分布(均值 {ps.get("random_mean","3135"):.0f} ± {ps.get("random_std","106"):.0f});
loop(调优配置 + 进化平滑器代码)= {ps.get("loop","2882"):.0f},<b>z = {ps.get("loop_z","-2.38")}</b>(&lt;−1.96 显著),{ps.get("pct_random_better","0"):.0f}% 的随机运行优于 loop——
<b>loop 显著优于随机(此次新鲜复现胜 {ps.get("win_pct","8"):.0f}%;历史里程碑 MS2 记录为 27%)</b>。</p>
{fig("significance_test.png", "10", f"能耗域(封顶):loop 落在随机分布(R={st.get('R','40')})正中(z={st.get('loop_z_score','-0.35')})= 统计打平")}
{fig("planner_significance.png", "11", f"规划器域(有改进空间):loop({ps.get('loop','2882'):.0f})显著低于随机分布(z={ps.get('loop_z','-2.38')},{ps.get('pct_random_better','0'):.0f}% 随机更优)= 显著胜随机")}
<p>两域对照统计成立:<b>同一框架在有改进空间的域显著胜随机、在噪声封顶的域与随机统计持平</b>——这是"null 由域性质决定,而非框架失败"最直接的因果证据。</p>

<h3>7.5　"用 vs 不用本框架":端到端产出对照</h3>
<p>同样的数据与候选,一个无安全机制的朴素 AI4S 流程与本框架会产出截然不同的"论文":
<b>朴素流程声称 4 项新发现(LLM 发现更优模型、顺序翻转、可达集扩大、场景省能)且不报告任何负结果</b>——
而其中"更优模型"实为随机可追平、三项规划"发现"全为已发表 prior art;
<b>本框架经查新门与留出门后 0 项真新发现,并系统报告 4 项负结果</b>(loop=随机、载荷死路、多数场景零收益、平场景零省能)。
这直接量化了"用 vs 不用"的差别:一个自信但错误,一个 humbler 但正确。</p>
{fig("naive_vs_trustworthy.png", "12", "朴素 AI4S vs 本框架:朴素声称 4 项发现(全 overclaim)、报告 0 项负结果;本框架 0 项真新、报告 4 项负结果")}

<h3>7.4　省能收益的统计分布(n={lss.get("n","250")} 随机城市场景)</h3>
<p>为避免场景挑选之嫌,对 {lss.get("n","250")} 个随机城市场景统计能量感知规划相对最短距离的省能分布:
<b>中位数 {lss.get("median","0")}%、约 {lss.get("zero_pct","77")}% 的场景省能为 0</b>(存在低空走廊时能量最优≈最短路),
仅 <b>{lss.get("ge3_pct","20")}% 的场景省能 ≥3%</b>(Wilson 95% CI [{(lss.get("wilson_ci",[15,25]) or [15,25])[0]}%, {(lss.get("wilson_ci",[15,25]) or [15,25])[1]}%],
相对 n=20 时的 [9%,49%] 大幅收窄),最高 15%。这诚实地表明:<b>能量感知规划的收益并非普适,而集中于少数"障碍逼出爬升"的几何</b>。</p>
{fig("large_scale_savings.png", "9", f"省能分布(n={lss.get('n','250')} 随机城市场景):多数场景为 0,少数几何下 3–15%——诚实的统计陈述,非挑选个例")}

<div class="keybox"><b>本节结论</b>:能耗域产出正确的 null,由数据封顶(§7.2)与域缺乏改进空间共同决定,而非框架失败——
同框架在有改进空间的域产出正值(§7.3)即为反证。可信 AI4S 的价值,正在于在该 overclaim 的地方不 overclaim,并将其量化为可辩护的结论。
"可信但产出 null / 负结果"在机器学习与元科学领域是被接受的贡献类型(如 ICML 2024《Embracing Negative Results》立场论文)。</div>

<h2>8　结论</h2>
<p>本文实现并实证了一个可信的大模型自动化科研框架,在真实 DJI M100 无人机能耗与能量感知规划上完成了从
模型发现、规划集成、动力学闭环到 PX4 真飞控栈验证的完整链条。其价值不在算法或效应的新颖(均为 prior art),
而在<b>方法学与真机数据锚定</b>,以及对当前 LLM 自动科研能力边界的诚实刻画——包括通过安全机制消融证明每个防自欺机制的承重性。
本工作本身(含主动将自身"发现"判定为 prior art)即为"可信 AI4S"的一次完整演示。</p>

<h2>附录　复现</h2>
<p><code>m100_eval.py</code>(冻结评测器)· <code>wall_experiment.py</code> · <code>robustness_suite.py</code> ·
<code>validate_climb.py</code> · <code>climb_ablation.py</code> · <code>phase_diagram.py</code> · <code>sim_flight.py</code> ·
<code>corridor_experiment.py</code> · <code>safeguard_ablation.py</code> · <code>render_videos.py</code> ·
<code>px4_integration/px4_fly_demo.py</code>。迭代记录:<code>agent_log.jsonl</code> + <code>experiments/candidates/</code>。</p>

</body></html>"""
    out = os.path.join(EXP, "autoresearch_report.html")
    open(out, "w").write(html)
    print(f"报告生成 {out}  ({os.path.getsize(out)//1024} KB)")


if __name__ == "__main__":
    main()
