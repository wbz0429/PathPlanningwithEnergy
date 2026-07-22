# -*- coding: utf-8 -*-
"""build_slides_html.py — 把答辩内容做成横向翻页 HTML 幻灯(自包含单文件)。
浏览器打开:← → / 空格 翻页,F 全屏,Esc 退全屏。图用顶会 pub 版(base64 内嵌)。
输出 experiments/defense_slides.html
"""
import os, base64, json

HERE = os.path.dirname(os.path.abspath(__file__)); EXP = os.path.join(HERE, "experiments")
PX4 = os.path.join(HERE, "px4_integration"); PUBDIR = os.path.join(EXP, "pub"); GIF = os.path.join(EXP, "gifs")


def b64img(*cands):
    for p in cands:
        if p and os.path.exists(p):
            mime = "image/gif" if p.endswith(".gif") else "image/png"
            return f"data:{mime};base64," + base64.b64encode(open(p, "rb").read()).decode()
    return None


def img(pub=None, raw=None, cls="fig"):
    d = b64img(os.path.join(PUBDIR, pub) if pub else None, raw)
    return f"<img class='{cls}' src='{d}'/>" if d else "<div class='miss'>[缺图]</div>"


def J(n):
    p = os.path.join(EXP, n)
    return json.load(open(p)) if os.path.exists(p) else {}


ab = J("px4_ab_comparison.json") or (json.load(open(os.path.join(PX4, "px4_ab_comparison.json"))) if os.path.exists(os.path.join(PX4, "px4_ab_comparison.json")) else {})


def two_col(fig_html, bullets_html):
    return f"<div class='two'><div class='left'>{fig_html}</div><div class='right'>{bullets_html}</div></div>"


def bl(items):
    return "<ul>" + "".join(f"<li>{x}</li>" for x in items) + "</ul>"


# ---- 22 张幻灯(与 PPT 对齐)----
SLIDES = []


def S(html, cls="content"):
    SLIDES.append((cls, html))


# 封面
_LOGO = b64img(os.path.join(HERE, "assets", "xjtu_logo_white.png"))
_LOGO_IMG = f"<img src='{_LOGO}' style='height:8vh;margin-bottom:4vh'/>" if _LOGO else ""

S(f"""<div class='cover'>
{_LOGO_IMG}
<div class='ctitle'>面向无人机能耗建模与能量感知路径规划的<br><span class='hl'>可信大模型自动化科研框架</span></div>
<div class='csub'>A Trustworthy LLM-Agent Autoresearch Framework, Instantiated on UAV Energy Modeling and Energy-Aware Path Planning</div>
<div class='cmeta'>硕士学位论文 · 中期答辩 · 2026</div></div>""", "cover")

S("<h2><span class='n'>1</span>大模型自动科研正在爆发,但它的“发现”普遍不可信</h2>" +
  two_col("<div class='stat'><div class='big'>57%</div><div class='cap'>朴素 AI4S 手稿含错误或幻觉数值</div></div>",
          bl(["FunSearch / AlphaEvolve / Eureka 在数学、代码、RL 上确有新发现",
              "但都要求“精确可验证评测器”——作者自认局限",
              "<b>朴素流程会过度声称</b>(AI Scientist 独立评估):",
              "&nbsp;&nbsp;42% 实验编码错误失败、57% 手稿含幻觉数值",
              "&nbsp;&nbsp;把已发表 prior art 误判为“新颖”"])) +
  "<div class='cite'>Beel, Kan & Baumgart, ACM SIGIR Forum 2025(arXiv:2502.14297)</div>")

S("""<div class='qbox'><div class='qlab'>研究问题</div>
<div class='qbig'>在一个真实、含噪、无干净真值的工程域(无人机能耗与规划),<br>可信的自动科研框架能“得到什么”、又该“诚实承认什么”?</div>
<div class='qnote'>核心主张:贡献不是新算法或新发现(均为 prior art),而是方法学 + 真机数据锚定 + 诚实能力边界。</div></div>""", "dark")

S("<h2><span class='n'>3</span>整体框架:六个环节一环扣一环,贡献落在“过程”而非“发现”</h2>" +
  "<table class='fw'>" +
  "".join(f"<tr><td class='c1'>{a}</td><td class='c2'>{r}</td><td class='c3'>{d}</td></tr>" for a, r, d in [
      ("① 数据与评测", "地基", "真机功率 P=V·I 当尺子,冻结+留出防作弊"),
      ("② 自动科研框架", "过程(真贡献)", "只改模型、撞瓶颈触发外搜、诚实 keep/revert"),
      ("③ 能耗模型", "产物", "9 项物理特征线性拟合(base 全是别人的)"),
      ("④ 能量感知规划", "应用", "模型当边权插进 A*,改变翻越/绕行决策"),
      ("⑤ 动力学与真飞控", "验证层", "RotorPy→PX4 真固件,结论过动力学仍成立"),
      ("⑥ 统计与防守", "能力边界", "两域显著性+噪声地板,证明 null 是域性质")]) + "</table>")

S("<h2><span class='n'>4</span>我们把“可信性”工程化为五道相互独立的安全机制</h2>" +
  "<div class='mech'>" + "".join(
      f"<div class='mrow'><div class='mnum'>{i+1}</div><div class='mh'>{h}</div><div class='md'>{d}</div></div>"
      for i, (h, d) in enumerate([
          ("冻结评测器", "锚定真机功率 P=V·|I|,按航班留出,永不可编辑"),
          ("keep / revert", "仅当搜索降且留出不显著退才接受,防过拟合"),
          ("强制查新门", "任何“发现”须先过文献查新,否则降级为复现"),
          ("因果消融", "挖掉关键项验证机制归因"),
          ("交叉模型 + 诚实报负结果", "暴露评测循环性,系统记录失败")])) + "</div>")

S("<h2><span class='n'>4b</span>框架架构:五节点闭环 + 外搜逃逸支路</h2>" +
  f"<div class='apxfig'>{img(pub='框架架构图.png')}</div>" +
  "<div class='cite'>LLM 提议 → Harness 实验 → 冻结评测器 → keep/revert 闭环;撞瓶颈触发外搜;知识库持久化</div>")

S("<h2><span class='n'>5</span>过程即贡献:loop 撞瓶颈 → 触发外搜 → 改向 → 诚实记录</h2>" +
  two_col(img(raw=os.path.join(EXP, "loop_process.png")),
          f"<div style='text-align:center'>{img(pub='loop流程图.png')}</div>") +
  "<div class='cite'>左:真实迭代轨迹(卡4.24%→外搜→突破1.9%);右:loop 机制流程。数据 agent_log.jsonl + KNOWLEDGE.md</div>")

S("<h2><span class='n'>5b</span>autoresearch 如何突破阈值:局部搜索卡壳 → 外搜识别方向 → 一步突破</h2>" +
  f"<div class='apxfig'>{img(pub='突破阈值叙事.png')}</div>" +
  "<div class='cite'>6.88% → 卡在 4.2%(7 变体全卡)→ 外搜识别“加线性 payload” → 1.9%。突破来自框架的外搜环节</div>")

S("<h2><span class='n'>6</span>能耗模型自动迭代把留出误差 6.88% → 1.86%——但这不是重点</h2>" +
  two_col(img(pub="噪声地板.png", raw=os.path.join(EXP, "fig_summary.png")),
          bl(["冻结评测器 + LLM 每轮改一次 featurize",
              "iter1–10 全程留出验证 + keep/revert",
              "<b>相对提升约 73%</b>",
              "<b class='red'>但——结构搜索与随机搜索打平</b>",
              "(见能力边界)"])) +
  "<div class='cite'>冻结评测器 m100_eval.py;数据:真实 DJI M100(209 航班)</div>")

S("<h2><span class='n'>7</span>真机验证的能耗代价改变规划决策:避开真实存在的爬升能耗</h2>" +
  two_col(img(pub="真机代价改变决策.png") + img(raw=os.path.join(GIF, "video_corridor.gif"), cls="gif"),
          bl(["距离/教科书BEMT → 翻墙",
              "<b class='grn'>真机M100 → 绕行,省 5–13%</b>",
              "<b>机制三重验证:</b>",
              "留出爬升 +114W vs 预测 +108W(&lt;5%)",
              "因果消融:挖爬升项 → 退回翻墙"])) +
  "<div class='cite'>效应属 prior art(EcoFlight 2025);区别 = 真机验证代价 + 可信评测</div>")

S("<h2><span class='n'>8</span>结论过 RotorPy 动力学与 PX4 真飞控固件栈仍成立(省 14.3%)</h2>" +
  two_col(img(pub="PX4真飞控_AB对比.png") + img(raw=os.path.join(GIF, "px4_flight.gif"), cls="gif"),
          bl(["<b>证据链层层加固:</b>",
              "折线预测 4–22%",
              "RotorPy 动力学 6.1%",
              "<b class='blu'>PX4 真飞控栈 14.3%</b>",
              "(EKF + 位置环 + Gazebo 动力学)"])) +
  "<div class='cite'>PX4 SITL + Gazebo Harmonic;同固件栈同脚本 A/B 对比</div>")

S("<h2><span class='n'>9</span>但这些“效应”经三次文献查新,全部是已发表的 prior art</h2>" +
  "<table class='pa'>" + "".join(f"<tr><td>{a}</td><td class='src'>{b}</td></tr>" for a, b in [
      ("能量感知规划改变路径", "EcoFlight (2025)"),
      ("配送顺序因爬降不对称翻转", "Michel et al. 2024, arXiv 2410.17585"),
      ("电池约束下可达集扩大", "Nguyen & Au, AAMAS 2017"),
      ("载荷影响爬升 vs 绕行", "EcoFlight / 物流 UAV, Drones 2025"),
      ("M100 能耗建模 / v* / 爬降不对称", "Dai;Di Franco 2015;Liu 2017")]) + "</table>" +
  "<div class='concl red'>→ 我们主动引用、不声称首创。查新门拦住了把“复现”当“发现”。</div>")

S("<h2><span class='n'>10</span>框架每个节点都“承重”:去掉它,loop 就退化或自欺</h2>" +
  two_col(img(pub="框架消融_局部最优.png"),
          bl(["<b>去掉每个节点会怎样:</b>",
              "评测器 → 只报训练误差:探针可刷分",
              "<b class='red'>外搜 → 只搜物理形:困 4.2% 盆地</b>",
              "查新门 → H1/H3 当发现(实为 prior art)",
              "因果消融 → 证不了爬升是唯一因",
              "报负结果 → 漏 loop≈随机 / 载荷死路"])) +
  "<div class='cite'>framework_ablation.py / safeguard_ablation.py</div>")

S("<h2><span class='n'>11</span>性能被数据噪声封顶:52 项模型也压不下 6 项——不是方法不足</h2>" +
  two_col(img(pub="噪声地板.png"),
          bl(["held-out 6 项即触底 1.88%",
              "加到 12/20/52 项不降反升",
              "<b class='grn'>4 种方法殊途同归 ~1.9%</b>",
              "运动学只解释 &lt;40% 瞬时功率",
              "<b class='blu'>“数据封顶”从主张变测量</b>"])) +
  "<div class='cite'>domain_value.py;对标 Tseng 多项式(数据集既定最优)</div>")

S("<h2><span class='n'>12</span>同框架:有改进空间的域显著胜随机,噪声封顶的域统计打平</h2>" +
  two_col(img(pub="两域显著性.png"),
          bl(["<b>能耗域(封顶):</b>",
              "loop 1.86% vs 随机 1.87±0.03,z=−0.35 → 打平",
              "<b>规划器域(有 headroom):</b>",
              "<b class='grn'>loop 显著低于随机,z=−2.38 → 显著胜</b>",
              "<b class='blu'>→ null 由域性质决定,非框架失败</b>"])) +
  "<div class='cite'>significance_test.py / planner_significance.py(随机分布 + z 检验)</div>")

S("<h2><span class='n'>12b</span>为什么打平?引导搜索的优势随搜索空间复杂度增长(文献规律)</h2>" +
  two_col(img(pub="复杂度规律.png"),
          bl(["<b>不是框架无能,是规律预测的必然:</b>",
              "Bergstra&Bengio'12:低有效维→随机追平",
              "REMBO/贝叶斯:~15–20 维临界点",
              "FunSearch:程序空间>宇宙原子数→引导才行",
              "<b class='red'>我们能耗域(小)→打平</b>",
              "<b class='grn'>我们规划器域(大)→胜 27%</b>",
              "<b class='blu'>同框架横跨两端点,亲手印证规律</b>"])) +
  "<div class='cite'>规律来自文献综合;两端点是我们实测。CMU《Hidden Pitfalls》(2509.08713)反证可信性为刚需</div>")

S("<h2><span class='n'>13</span>用 vs 不用:朴素 AI4S 声称 4 项假发现且藏负结果,我们相反</h2>" +
  two_col(img(pub="消融阶梯.png", raw=os.path.join(EXP, "naive_vs_trustworthy.png")),
          bl(["<b class='red'>朴素:</b> 4 假发现 + 0 负结果",
              "<b class='grn'>我们:</b> 0 真新 + 4 负结果",
              "一个自信但错,",
              "一个 humbler 但对。"])) +
  "<div class='cite'>naive_vs_trustworthy.py / ablation_ladder.py</div>")

S("<h2><span class='n'>14</span>诚实边界:多数场景无收益,载荷无效,贡献是方法学而非发现</h2>" +
  two_col(img(pub="省能分布_n250.png"),
          bl(["n=250 场景:中位省能 0%",
              "77% 场景零收益(有低空走廊)",
              "仅 20% ≥3%,CI[15.5,25.4]%",
              "载荷 ≤500g:全扫不改变路径",
              "无真机飞行(止于仿真)",
              "<b class='blu'>→ 不挑场景、不硬凑,诚实统计</b>"])) +
  "<div class='cite'>large_scale_savings.py;可信 null 是被接受的贡献(ICML 2024)</div>")

S("""<div class='qbox'><div class='qlab'>结论</div>
<ol class='concl-list'>
<li>不声称新算法/新效应——三次查新确认均为 prior art,主动引用</li>
<li class='hl2'>贡献 = 可信自动科研方法学 + 真机数据锚定 + 诚实能力边界</li>
<li>逐个消融证明每道节点“承重”:去掉即退化/overclaim</li>
<li>价值 = 在该 overclaim 处不 overclaim,并量化能力边界</li>
<li>域相关:有 headroom 显著胜随机、封顶域正确产出 null</li>
</ol></div>""", "dark")

S("<h2><span class='n'>16</span>参考文献</h2><div class='refs'>" + "<br>".join([
    "Romera-Paredes et al. FunSearch. Nature 2023.",
    "Novikov et al. AlphaEvolve. DeepMind 2025.",
    "Ma et al. Eureka. ICLR 2024. arXiv:2310.12931.",
    "Beel, Kan & Baumgart. AI Scientist 独立评估. ACM SIGIR Forum 2025. arXiv:2502.14297.",
    "Michel et al. Energy-Optimal Waypoint UAV Missions. 2024. arXiv:2410.17585.",
    "Nguyen & Au. Extending drone delivery reachable set. AAMAS 2017.",
    "Karl et al. Position: Embracing Negative Results in ML. ICML 2024.",
    "Rodrigues et al. DJI M100 energy dataset. Scientific Data 2021.",
    "Di Franco & Buttazzo 2015;Liu 2017;EcoFlight 2025."]) + "</div>")

# 附录
for t, pub, raw, cap in [
    ("附录 A:累加消融阶梯", "消融阶梯.png", None, "逐步加回安全机制:假发现 3→0,报告数字保持诚实"),
    ("附录 A2:翻越/绕行能量权衡", "翻越绕行权衡.png", None, "翻越固定爬升罚 vs 绕行随墙宽增长,交叉≈半宽 24m"),
    ("附录 B:操作包络相图", "操作包络相图.png", None, "高速+窄障省 22%,低速/宽障归零"),
    ("附录 C:安全机制消融", None, os.path.join(EXP, "safeguard_ablation.png"), "过拟合探针刷不动;查新门降级 2/3"),
    ("附录 D:动力学实飞功率剖面", None, os.path.join(EXP, "sim_flight.png"), "翻墙爬升段飙 870W,绕行平稳")]:
    S(f"<h2 class='apx'>{t}</h2><div class='apxfig'>{img(pub=pub, raw=raw)}</div><div class='cite'>{cap}</div>")


def build():
    body = "".join(f"<section class='slide {cls}'>{html}</section>" for cls, html in SLIDES)
    n = len(SLIDES)
    html = """<!DOCTYPE html><html lang="zh"><head><meta charset="utf-8">
<title>中期答辩 · 横向翻页</title><style>
*{margin:0;padding:0;box-sizing:border-box}
html,body{height:100%;overflow:hidden;font-family:"PingFang SC","Microsoft YaHei",sans-serif;background:#0b1020}
#stage{position:relative;width:100vw;height:100vh}
.slide{position:absolute;top:0;left:0;width:100vw;height:100vh;padding:4.5vh 5vw;
  opacity:0;pointer-events:none;transition:opacity .35s;overflow:hidden;
  display:flex;flex-direction:column;justify-content:flex-start;background:#fff}
.slide.on{opacity:1;pointer-events:auto;z-index:2}
.slide.dark{background:#0A2A66;color:#fff}
.slide.cover{background:#0A2A66;color:#fff;justify-content:center;align-items:flex-start}
h2{font-size:2.1vw;color:#0563C1;font-weight:700;margin-bottom:2.5vh;line-height:1.35;border-left:.5vw solid #4472C4;padding-left:1vw}
h2 .n{display:inline-block;background:#4472C4;color:#fff;width:1.9vw;height:1.9vw;line-height:1.9vw;
  text-align:center;border-radius:.4vw;font-size:1.2vw;margin-right:.8vw;vertical-align:middle}
h2.apx{color:#0563C1}
.two{display:flex;gap:2.5vw;flex:1;align-items:center}
.two .left{flex:1.35;display:flex;flex-direction:column;gap:1.5vh;align-items:center;justify-content:center}
.two .right{flex:1;font-size:1.35vw;line-height:2}
img.fig{max-width:100%;max-height:62vh;border:1px solid #e0e0e0;border-radius:.4vw}
img.gif{max-width:70%;max-height:22vh;border:1px solid #ddd;border-radius:.4vw}
.apxfig{flex:1;display:flex;align-items:center;justify-content:center}.apxfig img{max-width:82%;max-height:70vh}
ul{list-style:none}ul li{margin:.9vh 0;padding-left:1.3vw;position:relative}
ul li:before{content:"·";position:absolute;left:0;color:#4472C4;font-weight:700}
.red{color:#B83A3A}.grn{color:#5B9BD5}.blu{color:#0563C1}
.cite{position:absolute;bottom:2.5vh;left:5vw;font-size:1vw;color:#8899aa}
.cover .ctitle{font-size:3vw;font-weight:700;line-height:1.5}.cover .hl{color:#5B9BD5}
.cover .csub{font-size:1.2vw;color:#D6E4F7;margin-top:3vh;line-height:1.6}
.cover .cmeta{font-size:1.3vw;margin-top:4vh}
.qbox{margin:auto 0}.qlab{font-size:1.6vw;color:#5B9BD5;font-weight:700;margin-bottom:3vh}
.qbig{font-size:2.4vw;font-weight:700;line-height:1.7}.qnote{font-size:1.3vw;color:#D6E4F7;margin-top:4vh;line-height:1.7}
.concl-list{font-size:1.7vw;line-height:2.2;padding-left:2vw}.concl-list li{margin:1.2vh 0}.concl-list .hl2{color:#5B9BD5;font-weight:700}
table.fw{width:100%;border-collapse:collapse;font-size:1.4vw}
table.fw td{border:1px solid #dde3e8;padding:1.4vh 1.2vw}
table.fw .c1{font-weight:700;color:#1a2333;width:22%}.fw .c2{color:#4472C4;font-weight:700;width:18%}.fw .c3{color:#6b7a8d}
table.fw tr:nth-child(odd){background:#eef4f7}
.mech{display:flex;flex-direction:column;gap:1.6vh;margin-top:1vh}
.mrow{display:flex;align-items:center;gap:1.2vw}
.mnum{background:#4472C4;color:#fff;width:2.4vw;height:2.4vw;line-height:2.4vw;text-align:center;border-radius:.5vw;font-size:1.3vw;font-weight:700;flex:none}
.mh{font-size:1.5vw;font-weight:700;color:#1a2333;width:16vw}.md{font-size:1.3vw;color:#6b7a8d}
table.pa{width:100%;border-collapse:collapse;font-size:1.5vw}table.pa td{border:1px solid #dde3e8;padding:1.5vh 1.2vw}
table.pa .src{color:#0563C1;font-weight:700}table.pa tr:nth-child(odd){background:#eef4f7}
.concl{font-size:1.5vw;font-weight:700;margin-top:2.5vh}
.stat{text-align:center}.stat .big{font-size:7vw;color:#0563C1;font-weight:800;line-height:1}.stat .cap{font-size:1.3vw;color:#6b7a8d;margin-top:1vh}
.refs{font-size:1.3vw;line-height:2.1;color:#333}
.miss{color:#b00}
#hud{position:fixed;bottom:1.5vh;right:2vw;font-size:1.1vw;color:#8899aa;z-index:10;font-family:sans-serif}
#bar{position:fixed;top:0;left:0;height:.5vh;background:#4472C4;z-index:10;transition:width .3s}
#hint{position:fixed;bottom:1.5vh;left:2vw;font-size:.95vw;color:#8899aa;z-index:10}
#brand{position:fixed;top:1.5vh;right:2vw;font-size:1vw;color:#9aa8b8;z-index:10;font-family:"PingFang SC",sans-serif}
</style></head><body>
<div id="bar"></div><div id="stage">""" + body + """</div>
<div id="hud"></div><div id="hint">← → / 空格 翻页　·　F 全屏　·　Esc 退出</div>
<div id="brand">西安交通大学</div>
<script>
var s=document.querySelectorAll('.slide'),i=0,N=s.length;
function show(k){i=Math.max(0,Math.min(N-1,k));s.forEach(function(e,j){e.classList.toggle('on',j===i)});
 document.getElementById('hud').textContent=(i+1)+' / '+N;
 document.getElementById('bar').style.width=((i)/(N-1)*100)+'%';}
document.addEventListener('keydown',function(e){
 if(e.key==='ArrowRight'||e.key===' '||e.key==='PageDown'){show(i+1);e.preventDefault();}
 else if(e.key==='ArrowLeft'||e.key==='PageUp'){show(i-1);}
 else if(e.key==='Home'){show(0);}else if(e.key==='End'){show(N-1);}
 else if(e.key==='f'||e.key==='F'){if(!document.fullscreenElement)document.documentElement.requestFullscreen();else document.exitFullscreen();}
});
document.addEventListener('click',function(e){if(e.clientX>window.innerWidth*0.5)show(i+1);else show(i-1);});
show(0);
</script></body></html>"""
    out = os.path.join(EXP, "defense_slides.html")
    open(out, "w").write(html)
    print(f"生成 {out}({n} 页,{os.path.getsize(out)//1024} KB)")


if __name__ == "__main__":
    build()
