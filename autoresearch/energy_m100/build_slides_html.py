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
<div class='cmeta2'>
<table class='info'><tr><td>答辩人</td><td>王彬竹</td><td>学号</td><td>3124354082</td></tr>
<tr><td>导师</td><td>张乐 副教授</td><td></td><td></td></tr>
<tr><td colspan='4' style='border:none;padding-top:1.2vh'>硕士学位论文 · 中期答辩 · 2026</td></tr></table>
</div></div>""", "cover")

# === 目录页 ===
S("""<h2 class='toc-title'>目 录</h2>
<div class='toc'>
<div class='toc-row'><span class='toc-n'>1</span><span class='toc-t'>研究背景与问题</span></div>
<div class='toc-row'><span class='toc-n'>2</span><span class='toc-t'>研究方案:可信自动科研框架</span></div>
<div class='toc-row'><span class='toc-n'>3</span><span class='toc-t'>主要工作与结果</span></div>
<div class='toc-row'><span class='toc-n'>4</span><span class='toc-t'>讨论</span></div>
<div class='toc-row'><span class='toc-n'>5</span><span class="toc-t">结论</span></div>
</div>""", "toc")

# === 引入页1:研究场景(先落地,别一上来就宏大)===
def part(num, title, sub):
    S(f"""<div class='partpage'><div class='partnum'>PART {num}</div>
<div class='parttitle'>{title}</div><div class='partsub'>{sub}</div></div>""", "part")

part("01", "研究背景与问题", "无人机能耗建模 + 能量感知路径规划")

S("<h2>研究场景:无人机能耗建模 + 能量感知路径规划</h2>" +
  two_col(img(pub="真机代价改变决策.png"),
          bl(["<b>无人机能耗建模</b>:预测无人机飞一段路耗多少电",
              "&nbsp;&nbsp;(输入速度/爬升/载荷 → 输出功率/能量)",
              "<b>能量感知路径规划</b>:不只找最短路,而是找最省电的路",
              "&nbsp;&nbsp;(如上图:绕开高墙比翻越更省电)",
              "<b>为什么选这个场景:</b>",
              "&nbsp;&nbsp;· 有真机数据(DJI M100 209 航班真实功率)",
              "&nbsp;&nbsp;· 直接关系无人机续航/配送里程",
              "&nbsp;&nbsp;· 能耗与规划天然耦合,是完整工程闭环"])) +
  "<div class='cite'>场景图:真机能耗代价让规划绕开爬升(省电);数据 DJI M100 (Rodrigues 2021)</div>")

# === 引入页2:在这个场景上要实现什么 ===
S("<h2>在这个场景上,我们要实现什么</h2>" +
  "<div class='mech'>" + "".join(
      f"<div class='mrow'><div class='mnum'>{i+1}</div><div class='mh'>{h}</div><div class='md'>{d}</div></div>"
      for i, (h, d) in enumerate([
          ("建能耗模型", "从真机 DJI M100 数据自动拟合出准确的能耗预测模型"),
          ("接入路径规划", "把能耗模型当规划代价,让无人机飞更省电的路(避爬升)"),
          ("用可信方法做", "全程用一套防作弊、防自欺的自动科研流程,并诚实刻画能力边界"),
      ])) + "</div>" +
  "<div class='cite'>三件事一条链:数据→能耗模型→规划省电,方法学贯穿全程</div>")

S("<h2>大模型自动科研正在爆发,但它的“发现”普遍不可信</h2>" +
  two_col("<div class='stat'><div class='big'>57%</div><div class='cap'>朴素 AI4S 手稿含错误或幻觉数值</div></div>",
          bl(["FunSearch / AlphaEvolve / Eureka 在数学、代码、RL 上确有新发现",
              "但都要求“精确可验证评测器”——作者自认局限",
              "<b>朴素流程会过度声称</b>(AI Scientist 独立评估):",
              "&nbsp;&nbsp;42% 实验编码错误失败、57% 手稿含幻觉数值",
              "&nbsp;&nbsp;把已发表 prior art 误判为“新颖”"])) +
  "<div class='cite'>Beel, Kan & Baumgart, ACM SIGIR Forum 2025(arXiv:2502.14297)</div>")

part("02", "研究方案", "可信自动科研框架的架构与机制")

S("""<div class='qbox'><div class='qlab'>研究问题</div>
<div class='qbig'>在一个真实、含噪、无干净真值的工程域(无人机能耗与规划),<br>可信的自动科研框架能“得到什么”、又该“诚实承认什么”?</div>
<div class='qnote'>核心主张:贡献不是新算法或新发现(均为 prior art),而是方法学 + 真机数据锚定 + 诚实能力边界。</div></div>""", "dark")

S("<h2>框架总体结构:六个部分</h2>" +
  "<table class='fw'><tr class='hd'><td>组成部分</td><td>角色</td><td>说明</td></tr>" +
  "".join(f"<tr><td class='c1'>{a}</td><td class='c2'>{r}</td><td class='c3'>{d}</td></tr>" for a, r, d in [
      ("① 数据与评测", "基础", "以真机功率(P=V·I)为基准的防作弊评测"),
      ("② 自动科研框架", "核心", "大模型闭环搜索:提议 → 评测 → 保留/回滚"),
      ("③ 能耗模型", "产物", "从真机数据拟合的能耗预测模型"),
      ("④ 能量感知规划", "应用", "以能耗为代价,规划更省电的路径"),
      ("⑤ 动力学与真飞控", "验证", "RotorPy 动力学 + PX4 真飞控栈验证"),
      ("⑥ 统计与防守", "边界", "统计检验 + 诚实刻画能力边界")]) + "</table>" +
  "<div class='sub2'>其中②的可信性由五道安全机制保证:"
  "<b>冻结评测器 · keep/revert · 强制查新门 · 因果消融 · 交叉模型+诚实报负结果</b></div>")

S("<h2>框架架构:五节点闭环 + 外搜逃逸支路</h2>" +
  f"<div class='apxfig'>{img(pub='框架架构图.png')}</div>" +
  "<div class='cite'>LLM 提议 → Harness 实验 → 冻结评测器 → keep/revert 闭环;撞瓶颈触发外搜;知识库持久化</div>")

S("<h2>过程即贡献:loop 撞瓶颈 → 触发外搜 → 改向 → 诚实记录</h2>" +
  two_col(img(raw=os.path.join(EXP, "loop_process.png")),
          f"<div style='text-align:center'>{img(pub='loop流程图.png')}</div>") +
  "<div class='cite'>左:真实迭代轨迹(卡4.24%→外搜→突破1.9%);右:loop 机制流程。数据 agent_log.jsonl + KNOWLEDGE.md</div>")

S("<h2>autoresearch 如何突破阈值:局部搜索卡壳 → 外搜识别方向 → 一步突破</h2>" +
  f"<div class='apxfig'>{img(pub='突破阈值叙事.png')}</div>" +
  "<div class='cite'>6.88% → 卡在 4.2%(7 变体全卡)→ 外搜识别“加线性 payload” → 1.9%。突破来自框架的外搜环节</div>")

part("03", "主要工作与结果", "能耗建模 → 规划省电 → 动力学与真飞控验证")

S("<h2>从真机数据自动建模:能耗预测误差降 73%(6.88% → 1.86%)</h2>" +
  two_col(img(pub="噪声地板.png", raw=os.path.join(EXP, "fig_summary.png")),
          bl(["冻结评测器 + LLM 每轮改一次能耗公式",
              "iter1–10 全程留出验证 + 保留/回滚",
              "<b>误差 6.88% → 1.86%,相对提升约 73%</b>",
              "<span style='font-size:1vw;color:#666'>ARE = |预测能量−真实能量| / 真实能量,",
              "<span style='font-size:1vw;color:#666'>在“未参与训练的飞行”上取平均(防背答案)</span>"])) +
  "<div class='cite'>能量真值 = 机载电压×电流积分;数据:真实 DJI M100(209 航班);评测器 m100_eval.py 冻结</div>")

S("<h2>真机验证的能耗代价改变规划决策:避开真实存在的爬升能耗</h2>" +
  two_col(img(pub="真机代价改变决策.png") + img(raw=os.path.join(GIF, "video_corridor.gif"), cls="gif"),
          bl(["距离/教科书BEMT → 翻墙",
              "<b class='grn'>真机M100 → 绕行,省 5–13%</b>",
              "<b>机制三重验证:</b>",
              "留出爬升 +114W vs 预测 +108W(&lt;5%)",
              "因果消融:挖爬升项 → 退回翻墙"])) +
  "<div class='cite'>区别 = 真机验证代价 + 可信评测(EcoFlight 2025 等已研究该效应)</div>")

S("<h2>翻越 vs 绕行是权衡:墙越宽绕行越贵,临界墙宽决定选谁</h2>" +
  two_col(img(pub="翻越绕行权衡.png"),
          bl(["<b>不是“绕行一定省”,而是权衡:</b>",
              "翻越:少走距离,但付固定爬升罚(≈1233J)",
              "绕行:不爬升,但随墙宽多走越多",
              "<b class='blu'>两条线交叉≈半宽 24m:</b>",
              "窄墙 → 绕行省(M100 选绕行)",
              "宽墙 → 翻越省(M100 也翻墙)"])) +
  "<div class='cite'>tradeoff.py;这也是操作包络的物理解释(何时该绕、何时该翻)</div>")

S("<h2>结论过 RotorPy 动力学与 PX4 真飞控固件栈仍成立(省 14.3%)</h2>" +
  two_col(img(pub="PX4真飞控_AB对比.png") + img(raw=os.path.join(GIF, "px4_flight.gif"), cls="gif"),
          bl(["<b>证据链层层加固:</b>",
              "折线预测 4–22%",
              "RotorPy 动力学 6.1%",
              "<b class='blu'>PX4 真飞控栈 14.3%</b>",
              "(EKF + 位置环 + Gazebo 动力学)"])) +
  "<div class='cite'>PX4 SITL + Gazebo Harmonic;同固件栈同脚本 A/B 对比</div>")

S("<h2>框架每个节点都“承重”:去掉它,loop 就退化或自欺</h2>" +
  two_col(img(pub="框架消融_局部最优.png"),
          bl(["<b>去掉每个节点会怎样:</b>",
              "评测器 → 只报训练误差:探针可刷分",
              "<b class='red'>外搜 → 只搜物理形:困 4.2% 盆地</b>",
              "查新门 → H1/H3 当发现(实为 prior art)",
              "因果消融 → 证不了爬升是唯一因",
              "报负结果 → 漏 loop≈随机 / 载荷死路"])) +
  "<div class='cite'>framework_ablation.py / safeguard_ablation.py</div>")

part("04", "讨论与展望", "能力边界的诚实刻画,与未来方向")

S("<h2>性能上限由数据决定:留出误差在 6 项处触底,加到 52 项不再降</h2>" +
  two_col(img(pub="噪声地板.png"),
          bl(["held-out 6 项即触底 1.88%",
              "加到 12/20/52 项不降反升",
              "<b class='grn'>4 种方法殊途同归 ~1.9%</b>",
              "运动学只解释 &lt;40% 瞬时功率",
              "<b class='blu'>“数据封顶”从主张变测量</b>"])) +
  "<div class='cite'>domain_value.py;对标 Tseng 多项式(数据集既定最优)</div>")

S("<h2>同框架:有改进空间的域显著胜随机,噪声封顶的域统计打平</h2>" +
  two_col(img(pub="两域显著性.png"),
          bl(["<b>能耗域(封顶):</b>",
              "loop 1.86% vs 随机 1.87±0.03,z=−0.35 → 打平",
              "<b>规划器域(有改进空间):</b>",
              "<b class='grn'>loop 显著低于随机,z=−2.38 → 显著胜</b>",
              "<b class='blu'>→ 优势取决于搜索空间复杂度:小空间相当,大空间显著胜</b>"])) +
  "<div class='cite'>significance_test.py / planner_significance.py(随机分布 + z 检验)</div>")

S("<h2>为什么打平?引导搜索的优势随搜索空间复杂度增长(文献规律)</h2>" +
  two_col(img(pub="复杂度规律.png"),
          bl(["<b>引导搜索的优势随搜索空间复杂度增长:</b>",
              "Bergstra&Bengio'12:低有效维→随机追平",
              "REMBO/贝叶斯:~15–20 维临界点",
              "FunSearch:程序空间>宇宙原子数→引导才行",
              "<b class='red'>我们能耗域(小)→打平</b>",
              "<b class='grn'>我们规划器域(大)→胜 27%</b>",
              "<b class='blu'>同框架横跨两端点,亲手印证规律</b>"])) +
  "<div class='cite'>规律来自文献综合;两端点是我们实测。CMU《Hidden Pitfalls》(2509.08713)反证可信性为刚需</div>")

S("<h2>可信协议的价值:同数据同候选,防住了朴素流程的过度声称</h2>" +
  two_col(img(pub="消融阶梯.png", raw=os.path.join(EXP, "naive_vs_trustworthy.png")),
          bl(["同样的数据与候选,两种流程产出对比:",
              "<b class='red'>朴素 AI4S:</b> 声称 4 项发现(经查全为已有)、不报局限",
              "<b class='grn'>本框架:</b> 查新门/留出门过滤后如实报告,并记录 4 项边界",
              "<b class='blu'>→ 安全机制把“看似发现”挡在门外,结论更可靠</b>"])) +
  "<div class='cite'>naive_vs_trustworthy.py / ablation_ladder.py(累加消融)</div>")

S("<h2>适用范围:省能收益集中于“障碍逼出爬升”的几何(可达 4–22%)</h2>" +
  two_col(img(pub="省能分布_n250.png"),
          bl(["<b>n=250 随机城市场景统计,划清适用边界:</b>",
              "有障碍逼出爬升的几何:省能 <b class='grn'>3–22%</b>",
              "有低空走廊的场景:能量最优≈最短路(收益小)",
              "→ 结论用统计陈述,不挑单一场景",
              "<b>适用条件(诚实标注):</b>",
              "载荷 ≤500g 时不改变路径;当前止于仿真验证"])) +
  "<div class='cite'>large_scale_savings.py(n=250)</div>")

# === 展望页(向上姿态:边界之外还有机会)===
S("<h2>展望:诚实边界之上,仍有多条可突破的方向</h2>" +
  "<div class='mech'>" + "".join(
      f"<div class='mrow'><div class='mnum'>{i+1}</div><div class='mh'>{h}</div><div class='md'>{d}</div></div>"
      for i, (h, d) in enumerate([
          ("功率残差归因", "运动学只解释 ~40%,其余 ~60% 的来源(风/电压/控制)待定量归因,突破则进一步降误差"),
          ("感知闭环", "已知地图 → 深度相机在线建图 + 能量感知滚动重规划,更贴近真实部署"),
          ("更大搜索空间", "能耗小空间(与随机相当)→ 复杂规划器/程序空间,引导优势更显著"),
          ("外场实飞验证", "仿真所得省能结论,补充真实无人机飞行实测"),
      ])) + "</div>" +
  "<div class='cite'>这些是当前能力边界之外的明确下一步,非宣告上限——中期后继续推进</div>")

part("05", "结论", "主要结论与贡献")

S("""<div class='qbox'><div class='qlab'>结论</div>
<ol class='concl-list'>
<li>贡献 = 可信自动科研方法学 + 真机数据锚定 + 诚实能力边界</li>
<li>真机 M100 能耗建模(误差 6.88%→1.86%)接入规划,动力学/PX4 全栈验证省 14.3%</li>
<li>逐个消融证明框架每道安全机制都“承重”</li>
<li>诚实划出适用边界:搜索空间大则显著胜随机,空间小则与随机相当</li>
<li><b>可复用性</b>:框架域无关,已在规划器/雷达/能耗三类任务验证;换新机型只需该机型的飞行数据即可复用同一流程</li>
</ol></div>""", "dark")

S("<h2>参考文献</h2><div class='refs'>" + "<br>".join([
    "Romera-Paredes et al. FunSearch. Nature 2023.",
    "Novikov et al. AlphaEvolve. DeepMind 2025.",
    "Ma et al. Eureka. ICLR 2024. arXiv:2310.12931.",
    "Beel, Kan & Baumgart. AI Scientist 独立评估. ACM SIGIR Forum 2025. arXiv:2502.14297.",
    "Michel et al. Energy-Optimal Waypoint UAV Missions. 2024. arXiv:2410.17585.",
    "Nguyen & Au. Extending drone delivery reachable set. AAMAS 2017.",
    "Karl et al. Position: Embracing Negative Results in ML. ICML 2024.",
    "Rodrigues et al. DJI M100 energy dataset. Scientific Data 2021.",
    "Di Franco & Buttazzo 2015;Liu 2017;EcoFlight 2025."]) + "</div>")

S("""<div class='endpage'><div class='thanks-big'>恳请各位老师批评指正</div>
<div class="end-sub">谢谢</div></div>""", "dark")

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
.cmeta2{margin-top:3.5vh}.cover .info{border-collapse:collapse;font-size:1.35vw}
.cover .info td{border:1px solid rgba(255,255,255,.35);padding:.8vh 1.4vw;color:#fff}
.slide.toc{background:#fff}.toc-title{text-align:center;color:#0563C1;font-size:2.6vw;letter-spacing:.5vw;margin-bottom:4vh}
.toc{margin:0 14vw}.toc-row{display:flex;align-items:baseline;gap:1.5vw;padding:2.2vh 0;border-bottom:1px solid #e2e8ee;font-size:1.7vw}
.toc-n{color:#0563C1;font-weight:700;font-size:1.4vw}.toc-t{color:#1a2333}
.slide.part{background:#0563C1;color:#fff;justify-content:center}
.partpage .partnum{font-size:1.6vw;letter-spacing:.6vw;color:#9ec3f0}
.partpage .parttitle{font-size:3vw;font-weight:700;margin-top:1.5vh}
.partpage .partsub{font-size:1.4vw;color:#d6e4f7;margin-top:2vh}
.slide.part,.slide.dark{display:flex;flex-direction:column;justify-content:center}
.endpage{margin:auto 0;text-align:center}.endpage .thanks-big{font-size:3vw;font-weight:700;color:#fff;letter-spacing:.4vw}.endpage .end-sub{font-size:1.4vw;color:#9ec3f0;margin-top:3vh}
.qbox{margin:auto 0}.qlab{font-size:1.6vw;color:#5B9BD5;font-weight:700;margin-bottom:3vh}
.thanks{margin-top:5vh;font-size:1.8vw;color:#fff;text-align:center;letter-spacing:.3vw}
.qbig{font-size:2.4vw;font-weight:700;line-height:1.7}.qnote{font-size:1.3vw;color:#D6E4F7;margin-top:4vh;line-height:1.7}
.concl-list{font-size:1.7vw;line-height:2.2;padding-left:2vw}.concl-list li{margin:1.2vh 0}.concl-list .hl2{color:#5B9BD5;font-weight:700}
table.fw{width:100%;border-collapse:collapse;font-size:1.4vw}
table.fw td{border:1px solid #dde3e8;padding:1.4vh 1.2vw}
table.fw .c1{font-weight:700;color:#1a2333;width:22%}.fw .c2{color:#4472C4;font-weight:700;width:18%}.fw .c3{color:#6b7a8d}
table.fw tr:nth-child(odd){background:#eef4f7}
table.fw tr.hd td{background:#0563C1;color:#fff;font-weight:700;font-size:1.15vw}
.sub2{margin-top:2.5vh;font-size:1.25vw;color:#333;line-height:1.7}.sub2 b{color:#0563C1}
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
