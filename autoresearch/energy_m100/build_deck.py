# -*- coding: utf-8 -*-
"""build_deck.py — 中期答辩可编辑 PPTX(academic-pptx 规范:action titles / 一页一论点 / 图左释右 / 必引用)。
进化版:加 整体框架总览 + loop 全过程 + 框架节点消融(提为正文)+ 权衡图(附录)。
"""
import os
from pptx import Presentation
from pptx.util import Inches as In, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE
from pptx.oxml.ns import qn
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.join(HERE, "experiments"); PX4 = os.path.join(HERE, "px4_integration"); GIF = os.path.join(EXP, "gifs")
NAVY = RGBColor(0x1E,0x29,0x61); DEEP = RGBColor(0x06,0x5A,0x82); TEAL = RGBColor(0x02,0x80,0x90)
MINT = RGBColor(0x02,0xC3,0x9A); WHITE = RGBColor(0xFF,0xFF,0xFF); INK = RGBColor(0x1A,0x23,0x33)
MUTED = RGBColor(0x6B,0x7A,0x8D); RED = RGBColor(0xB8,0x3A,0x3A); ICE = RGBColor(0xCA,0xDC,0xFC)
CJK = "微软雅黑"; SW, SH = In(13.333), In(7.5)


def _font(run, name=CJK):
    run.font.name = name; rpr = run._r.get_or_add_rPr()
    for tag in ("a:latin","a:ea","a:cs"):
        e = rpr.find(qn(tag))
        if e is None: e = rpr.makeelement(qn(tag), {}); rpr.append(e)
        e.set("typeface", name)


def bg(s, c): f = s.background.fill; f.solid(); f.fore_color.rgb = c
def content(s):
    bg(s, WHITE); b = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, In(0.12), SH)
    b.fill.solid(); b.fill.fore_color.rgb = TEAL; b.line.fill.background()
def dark(s):
    bg(s, NAVY); b = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, In(0.12), SH)
    b.fill.solid(); b.fill.fore_color.rgb = MINT; b.line.fill.background()


def tb(s, l, t, w, h, lines, size=16, color=INK, bold=False, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP, space=4):
    box = s.shapes.add_textbox(l, t, w, h); tf = box.text_frame
    tf.word_wrap = True; tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = Pt(2); tf.margin_top = tf.margin_bottom = Pt(2)
    if isinstance(lines, str): lines = [lines]
    for i, ln in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align; p.space_after = Pt(space)
        txt, sz, cl, bd = ln if isinstance(ln, tuple) else (ln, size, color, bold)
        r = p.add_run(); r.text = txt; r.font.size = Pt(sz); r.font.bold = bd; r.font.color.rgb = cl; _font(r)
    return box


def chip(s, l, t, txt, color=TEAL):
    sh = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, l, t, In(0.5), In(0.5))
    sh.fill.solid(); sh.fill.fore_color.rgb = color; sh.line.fill.background()
    tf = sh.text_frame; tf.margin_top = tf.margin_bottom = 0
    r = tf.paragraphs[0].add_run(); r.text = txt; r.font.size = Pt(20); r.font.bold = True
    r.font.color.rgb = WHITE; _font(r); tf.paragraphs[0].alignment = PP_ALIGN.CENTER


PUB = {"hero_figure.png": "真机代价改变决策.png", "phase_diagram.png": "操作包络相图.png",
       "tradeoff.png": "翻越绕行权衡.png", "corridor.png": "城市走廊_混合决策.png",
       "px4_ab_comparison.png": "PX4真飞控_AB对比.png", "framework_ablation.png": "框架消融_局部最优.png",
       "domain_value.png": "噪声地板.png", "significance_test.png": "两域显著性.png",
       "large_scale_savings.png": "省能分布_n250.png"}
PUBDIR = os.path.join(EXP, "pub")


def fit(s, path, l, t, w, h):
    base = os.path.basename(path)
    if base in PUB and os.path.exists(os.path.join(PUBDIR, PUB[base])):
        path = os.path.join(PUBDIR, PUB[base])
    if not os.path.exists(path): tb(s, l, t, w, In(0.4), f"[缺图 {os.path.basename(path)}]", 12, MUTED); return
    iw, ih = Image.open(path).size; ar = iw/ih
    if ar > w/h: nw = w; nh = int(w/ar)
    else: nh = h; nw = int(h*ar)
    s.shapes.add_picture(path, l+(w-nw)//2, t+(h-nh)//2, nw, nh)


def title(s, txt, sub=None):
    tb(s, In(1.25), In(0.38), In(11.6), In(1.0), txt, 26, DEEP, True)
    if sub: tb(s, In(1.25), In(1.32), In(11.6), In(0.4), sub, 15, TEAL, False)


def cite(s, txt): tb(s, In(0.55), In(7.0), In(12.2), In(0.35), txt, 11, MUTED, False)


def build():
    prs = Presentation(); prs.slide_width = SW; prs.slide_height = SH
    blank = prs.slide_layouts[6]; _sec = [0]
    def S(): return prs.slides.add_slide(blank)
    def numbered(dk=False):
        s = S(); (dark if dk else content)(s); _sec[0] += 1
        chip(s, In(0.55), In(0.4), str(_sec[0]), MINT if dk else TEAL); return s

    # ---- 封面 ----
    s = S(); dark(s)
    tb(s, In(0.9), In(2.2), In(11.5), In(1.9),
       [("面向无人机能耗建模与能量感知路径规划的", 26, WHITE, False),
        ("可信大模型自动化科研框架", 40, MINT, True)], space=8)
    tb(s, In(0.9), In(4.4), In(11.5), In(1.2),
       [("A Trustworthy LLM-Agent Autoresearch Framework, Instantiated on UAV Energy", 15, ICE, False),
        ("Modeling and Energy-Aware Path Planning", 15, ICE, False)], space=3)
    tb(s, In(0.9), In(6.2), In(11.5), In(0.6), "硕士学位论文·中期答辩　|　2026", 15, WHITE, False)

    # ---- 1 背景 ----
    s = numbered(); title(s, "大模型自动科研正在爆发,但它的“发现”普遍不可信")
    tb(s, In(0.7), In(2.2), In(5.9), In(3.2),
       [("57%", 96, DEEP, True), ("朴素 AI4S 手稿含错误或幻觉数值", 18, MUTED, False)],
       anchor=MSO_ANCHOR.MIDDLE, align=PP_ALIGN.CENTER, space=6)
    tb(s, In(6.9), In(2.0), In(6.0), In(4.2),
       [("· FunSearch/AlphaEvolve/Eureka 在数学、代码、RL 上确有新发现", 17, INK, False),
        ("· 但都要求“精确可验证评测器”——作者自认局限", 17, INK, False),
        ("· 朴素流程会过度声称(AI Scientist 独立评估):", 17, INK, True),
        ("    42% 实验编码错误失败、57% 手稿含幻觉数值", 16, RED, False),
        ("    把已发表 prior art 误判为“新颖”", 16, RED, False)], space=10)
    cite(s, "来源:Beel, Kan & Baumgart, ACM SIGIR Forum 2025(arXiv:2502.14297)")

    # ---- 2 研究问题 ----
    s = numbered(dk=True)
    tb(s, In(0.9), In(1.7), In(11.5), In(0.7), "研究问题", 22, MINT, True)
    tb(s, In(0.9), In(2.7), In(11.5), In(3.0),
       [("在一个真实、含噪、无干净真值的工程域(无人机能耗与规划),", 29, WHITE, True),
        ("可信的自动科研框架能“得到什么”、又该“诚实承认什么”?", 29, WHITE, True)], space=14)
    tb(s, In(0.9), In(5.6), In(11.5), In(0.9),
       "核心主张:贡献不是新算法或新发现(均为 prior art),而是方法学 + 真机数据锚定 + 诚实能力边界。", 17, ICE, False)

    # ---- 3 整体框架总览(NEW,一图讲全局)----
    s = numbered(); title(s, "整体框架:六个环节一环扣一环,贡献落在“过程”而非“发现”")
    rows = [("① 数据与评测", "地基", "真机功率 P=V·I 当尺子,冻结+留出防作弊"),
            ("② 自动科研框架", "过程(真贡献)", "只改模型、撞瓶颈触发外搜、诚实 keep/revert"),
            ("③ 能耗模型", "产物", "9 项物理特征线性拟合(base 全是别人的)"),
            ("④ 能量感知规划", "应用", "模型当边权插进 A*,改变翻越/绕行决策"),
            ("⑤ 动力学与真飞控", "验证层", "RotorPy→PX4 真固件,结论过动力学仍成立"),
            ("⑥ 统计与防守", "能力边界", "两域显著性+噪声地板,证明 null 是域性质")]
    y = In(2.1)
    for i, (a, role, d) in enumerate(rows):
        r = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, In(0.7), y, In(12.0), In(0.72))
        r.fill.solid(); r.fill.fore_color.rgb = RGBColor(0xEE,0xF4,0xF7) if i%2==0 else WHITE
        r.line.color.rgb = RGBColor(0xDD,0xE3,0xE8); r.line.width = Pt(0.5)
        tb(s, In(0.95), y+In(0.14), In(3.1), In(0.5), a, 16, INK, True)
        tb(s, In(4.1), y+In(0.14), In(2.2), In(0.5), role, 15, TEAL, True)
        tb(s, In(6.4), y+In(0.14), In(6.1), In(0.5), d, 14, MUTED, False)
        y += In(0.8)

    # ---- 4 五道安全机制 ----
    s = numbered(); title(s, "我们把“可信性”工程化为五道相互独立的安全机制")
    items = [("冻结评测器", "锚定真机功率 P=V·|I|,按航班留出,永不可编辑"),
             ("keep / revert", "仅当搜索降且留出不显著退才接受,防过拟合"),
             ("强制查新门", "任何“发现”须先过文献查新,否则降级为复现"),
             ("因果消融", "挖掉关键项验证机制归因"),
             ("交叉模型 + 诚实报负结果", "暴露评测循环性,系统记录失败")]
    y = In(2.15)
    for i, (h, d) in enumerate(items):
        c = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, In(0.7), y, In(0.55), In(0.55))
        c.fill.solid(); c.fill.fore_color.rgb = TEAL if i%2==0 else DEEP; c.line.fill.background()
        rr = c.text_frame.paragraphs[0].add_run(); rr.text = str(i+1); rr.font.size = Pt(18); rr.font.bold = True
        rr.font.color.rgb = WHITE; _font(rr); c.text_frame.paragraphs[0].alignment = PP_ALIGN.CENTER
        tb(s, In(1.5), y-In(0.02), In(3.4), In(0.6), h, 18, INK, True)
        tb(s, In(5.0), y+In(0.02), In(7.7), In(0.6), d, 15, MUTED, False)
        y += In(0.92)

    # ---- 5 loop 全过程(NEW,过程即贡献)----
    s = numbered(); title(s, "过程即贡献:loop 撞瓶颈→触发外搜→改向→诚实记录(真实迭代)")
    fit(s, os.path.join(EXP, "loop_process.png"), In(0.55), In(1.95), In(9.0), In(4.7))
    tb(s, In(9.7), In(2.1), In(3.3), In(4.6),
       [("价值不在最终模型", 16, INK, True),
        ("(prior art),", 15, MUTED, False),
        ("在这套过程:", 16, INK, True),
        ("· 卡 4.24% → 查文献", 15, INK, False),
        ("· wind 试→REVERT 记库", 15, INK, False),
        ("· 物理≈线性→诚实报", 15, INK, False),
        ("· loop≈随机→报边界", 15, INK, False)], space=10)
    cite(s, "真实 agent_log.jsonl + KNOWLEDGE.md;下半为 loop 机制流程")

    # ---- 6 能耗迭代 ----
    s = numbered(); title(s, "能耗模型自动迭代把留出误差从 6.88% 降到 1.86%——但这不是重点")
    fit(s, os.path.join(EXP, "fig_summary.png"), In(0.6), In(2.0), In(7.6), In(4.6))
    tb(s, In(8.5), In(2.1), In(4.4), In(4.4),
       [("· 冻结评测器 + LLM 每轮改一次 featurize", 16, INK, False),
        ("· iter1–10 全程留出验证 + keep/revert", 16, INK, False),
        ("· 相对提升约 73%", 16, INK, True),
        ("· 但——结构搜索与随机搜索打平", 16, RED, True),
        ("  (见能力边界)", 15, MUTED, False)], space=12)
    cite(s, "冻结评测器 m100_eval.py;数据:真实 DJI M100(209 航班)")

    # ---- 7 真机代价改规划 ----
    s = numbered(); title(s, "真机验证的能耗代价改变规划决策:避开真实存在的爬升能耗")
    fit(s, os.path.join(EXP, "hero_figure.png"), In(0.55), In(2.0), In(8.5), In(3.4))
    fit(s, os.path.join(GIF, "video_corridor.gif"), In(0.9), In(5.5), In(4.0), In(1.5))
    tb(s, In(9.3), In(2.1), In(3.6), In(4.6),
       [("· 距离/教科书BEMT → 翻墙", 16, INK, False),
        ("· 真机M100 → 绕行,省 5–13%", 16, MINT, True),
        ("机制三重验证:", 16, INK, True),
        ("· 留出爬升 +114W vs 预测 +108W(<5%)", 15, INK, False),
        ("· 因果消融:挖爬升项→退回翻墙", 15, INK, False),
        ("下方 GIF:城市走廊逐障碍混合决策", 13, MUTED, False)], space=10)
    cite(s, "效应属 prior art(EcoFlight 2025);区别 = 真机验证代价 + 可信评测")

    # ---- 8 PX4 ----
    s = numbered(); title(s, "结论过 RotorPy 动力学与 PX4 真飞控固件栈仍成立(省 14.3%)")
    fit(s, os.path.join(PX4, "px4_ab_comparison.png"), In(0.55), In(2.0), In(8.4), In(3.4))
    fit(s, os.path.join(GIF, "px4_flight.gif"), In(0.9), In(5.5), In(4.2), In(1.5))
    tb(s, In(9.2), In(2.1), In(3.7), In(4.6),
       [("证据链层层加固:", 16, INK, True),
        ("· 折线预测 4–22%", 16, INK, False),
        ("· RotorPy 动力学 6.1%", 16, INK, False),
        ("· PX4 真飞控栈 14.3%", 18, DEEP, True),
        ("(EKF + 位置环 + Gazebo 动力学)", 14, MUTED, False),
        ("下方 GIF:PX4 真飞控飞出的轨迹", 13, MUTED, False)], space=10)
    cite(s, "PX4 SITL + Gazebo Harmonic;同固件栈同脚本 A/B 对比")

    # ---- 9 prior art 表 ----
    s = numbered(); title(s, "但这些“效应”经三次文献查新,全部是已发表的 prior art")
    pr = [("能量感知规划改变路径", "EcoFlight (2025)"),
          ("配送顺序因爬降不对称翻转", "Michel et al. 2024, arXiv 2410.17585"),
          ("电池约束下可达集扩大", "Nguyen & Au, AAMAS 2017"),
          ("载荷影响爬升 vs 绕行", "EcoFlight / 物流 UAV, Drones 2025"),
          ("M100 能耗建模 / v* / 爬降不对称", "Dai;Di Franco 2015;Liu 2017")]
    y = In(2.2)
    for i, (a, b) in enumerate(pr):
        r = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, In(0.7), y, In(12.0), In(0.72))
        r.fill.solid(); r.fill.fore_color.rgb = RGBColor(0xEE,0xF4,0xF7) if i%2==0 else WHITE
        r.line.color.rgb = RGBColor(0xDD,0xE3,0xE8); r.line.width = Pt(0.5)
        tb(s, In(0.95), y+In(0.13), In(6.6), In(0.5), a, 17, INK, False)
        tb(s, In(7.7), y+In(0.13), In(4.9), In(0.5), b, 15, DEEP, True)
        y += In(0.8)
    tb(s, In(0.7), In(6.5), In(12.0), In(0.5), "→ 我们主动引用、不声称首创。查新门拦住了把“复现”当“发现”。", 16, RED, True)

    # ---- 10 框架节点消融(NEW,提为正文)----
    s = numbered(); title(s, "框架每个节点都“承重”:去掉它,loop 就退化或自欺")
    fit(s, os.path.join(EXP, "framework_ablation.png"), In(0.5), In(1.95), In(7.2), In(4.7))
    tb(s, In(7.9), In(2.0), In(5.0), In(4.8),
       [("消融每个节点 → 去掉会怎样:", 15, INK, True),
        ("· 评测器→只报训练误差:探针可刷分", 14, INK, False),
        ("· 外搜→只搜物理形:困 4.2% 盆地", 14, RED, True),
        ("· 查新门→H1/H3 当发现(实为 prior art)", 14, INK, False),
        ("· 因果消融→证不了爬升是唯一因", 14, INK, False),
        ("· 交叉模型→循环(BEMT尺下反贵)", 14, INK, False),
        ("· 报负结果→漏 loop≈随机/载荷死路", 14, INK, False),
        ("左图:只搜物理形困局部最优,", 14, MUTED, False),
        ("外搜识别加线性payload才突破1.9%", 14, MUTED, False)], space=8)
    cite(s, "framework_ablation.py / safeguard_ablation.py(每节点一个可复现消融)")

    # ---- 11 噪声地板 ----
    s = numbered(); title(s, "性能被数据噪声封顶:52 项模型也压不下 6 项——不是方法不足")
    fit(s, os.path.join(EXP, "domain_value.png"), In(0.55), In(2.0), In(8.6), In(4.5))
    tb(s, In(9.4), In(2.1), In(3.5), In(4.6),
       [("· held-out 6 项即触底 1.88%", 16, INK, False),
        ("· 加到 12/20/52 项不降反升", 16, INK, False),
        ("· 4 种方法殊途同归 ~1.9%", 16, MINT, True),
        ("· 运动学只解释 <40% 瞬时功率", 15, INK, False),
        ("“数据封顶”从主张变测量", 16, DEEP, True)], space=12)
    cite(s, "domain_value.py;对标 Tseng 多项式(数据集既定最优)")

    # ---- 12 两域显著性 ----
    s = numbered(); title(s, "同框架:有改进空间的域显著胜随机,噪声封顶的域统计打平")
    fit(s, os.path.join(EXP, "significance_test.png"), In(0.5), In(2.2), In(6.3), In(4.4))
    tb(s, In(6.9), In(2.1), In(6.0), In(4.8),
       [("能耗域(封顶):", 17, INK, True),
        ("  loop 1.86% vs 随机 1.87±0.03,z=−0.35", 16, INK, False),
        ("  52% 随机更好 → 统计打平", 16, INK, False),
        ("规划器域(有 headroom):", 17, INK, True),
        ("  loop 显著低于随机,z=−2.38", 16, MINT, True),
        ("  0% 随机更好 → 显著胜", 16, INK, False),
        ("→ null 由域性质决定,非框架失败", 17, DEEP, True)], space=12)
    cite(s, "significance_test.py / planner_significance.py(随机分布 + z 检验)")

    # ---- 13 用vs不用 ----
    s = numbered(); title(s, "用 vs 不用:朴素 AI4S 声称 4 项假发现且藏负结果,我们相反")
    fit(s, os.path.join(EXP, "naive_vs_trustworthy.png"), In(0.6), In(2.0), In(9.2), In(4.4))
    tb(s, In(10.0), In(2.2), In(2.9), In(4.4),
       [("朴素:", 16, RED, True), ("4 假发现 + 0 负结果", 15, INK, False),
        ("我们:", 16, MINT, True), ("0 真新 + 4 负结果", 15, INK, False),
        ("一个自信但错,", 15, INK, False), ("一个 humbler 但对。", 15, INK, True)], space=12)
    cite(s, "naive_vs_trustworthy.py(同数据同候选,端到端产出对照)")

    # ---- 14 诚实边界 ----
    s = numbered(); title(s, "诚实边界:多数场景无收益,载荷无效,贡献是方法学而非发现")
    fit(s, os.path.join(EXP, "large_scale_savings.png"), In(0.55), In(2.0), In(7.6), In(4.5))
    tb(s, In(8.4), In(2.1), In(4.5), In(4.6),
       [("· n=250 场景:中位省能 0%", 16, INK, False),
        ("· 77% 场景零收益(有低空走廊)", 16, INK, False),
        ("· 仅 20% ≥3%,CI[15.5,25.4]%", 16, INK, False),
        ("· 载荷 ≤500g:全扫不改变路径", 16, INK, False),
        ("· 无真机飞行(止于仿真)", 16, INK, False),
        ("→ 不挑场景、不硬凑,诚实统计", 16, DEEP, True)], space=11)
    cite(s, "large_scale_savings.py;可信 null 是被接受的贡献(ICML 2024)")

    # ---- 15 结论 ----
    s = numbered(dk=True)
    tb(s, In(0.9), In(1.3), In(11.5), In(0.7), "结论", 24, MINT, True)
    tb(s, In(0.9), In(2.3), In(11.7), In(4.4),
       [("① 不声称新算法/新效应——三次查新确认均为 prior art,主动引用", 21, WHITE, False),
        ("② 贡献 = 可信自动科研方法学 + 真机数据锚定 + 诚实能力边界", 21, MINT, True),
        ("③ 逐个消融证明每道节点“承重”:去掉即退化/overclaim", 21, WHITE, False),
        ("④ 价值 = 在该 overclaim 处不 overclaim,并量化能力边界", 21, WHITE, False),
        ("⑤ 域相关:有 headroom 显著胜随机、封顶域正确产出 null", 21, WHITE, False)], space=16)

    # ---- 16 参考文献 ----
    s = numbered(); title(s, "参考文献")
    refs = ["Romera-Paredes et al. FunSearch. Nature 2023.",
            "Novikov et al. AlphaEvolve. DeepMind 2025.",
            "Ma et al. Eureka. ICLR 2024. arXiv:2310.12931.",
            "Beel, Kan & Baumgart. AI Scientist 独立评估. ACM SIGIR Forum 2025. arXiv:2502.14297.",
            "Michel et al. Energy-Optimal Waypoint UAV Missions. 2024. arXiv:2410.17585.",
            "Nguyen & Au. Extending drone delivery reachable set. AAMAS 2017.",
            "Karl et al. Position: Embracing Negative Results in ML. ICML 2024.",
            "Rodrigues et al. DJI M100 energy dataset. Scientific Data 2021.",
            "Di Franco & Buttazzo 2015;Liu 2017;EcoFlight 2025."]
    tb(s, In(0.75), In(2.0), In(12.0), In(4.8), [(r, 15, INK, False) for r in refs], space=8)

    # ---- 附录 ----
    for name, fig_path, cap in [
        ("附录 A:累加消融阶梯(框架每步都承重)", os.path.join(EXP, "pub", "消融阶梯.png"), "逐步加回安全机制:假发现 3→0(左),报告数字保持诚实不虚低(右)"),
        ("附录 A2:翻越/绕行能量权衡分解", os.path.join(EXP, "tradeoff.png"), "翻越=固定爬升罚+少走距离,绕行=无爬升+多走距离,交叉≈半宽24m=相图边界物理解释"),
        ("附录 B:操作包络相图", os.path.join(EXP, "phase_diagram.png"), "高速+窄障省 22%,低速/宽障归零"),
        ("附录 C:安全机制消融(评测器抗 gaming)", os.path.join(EXP, "safeguard_ablation.png"), "过拟合探针刷不动;查新门降级 2/3"),
        ("附录 D:动力学实飞功率剖面", os.path.join(EXP, "sim_flight.png"), "翻墙爬升段飙 870W,绕行平稳"),
        ("附录 E:PX4 真飞控轨迹(俯视/侧视)", os.path.join(PX4, "px4_traj.png"), "翻矮楼 A + 绕高楼 B")]:
        s = S(); content(s); title(s, name); fit(s, fig_path, In(0.8), In(1.9), In(11.7), In(4.9)); cite(s, cap)

    out = os.path.join(EXP, "midterm_defense.pptx"); prs.save(out)
    print(f"生成 {out}")


if __name__ == "__main__":
    build()
