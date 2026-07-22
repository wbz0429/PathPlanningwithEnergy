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
# 西安交通大学创新港模板配色(西交蓝系)
NAVY = RGBColor(0x0A,0x2A,0x66); DEEP = RGBColor(0x05,0x63,0xC1); TEAL = RGBColor(0x44,0x72,0xC4)
MINT = RGBColor(0x5B,0x9B,0xD5); WHITE = RGBColor(0xFF,0xFF,0xFF); INK = RGBColor(0x1A,0x23,0x33)
MUTED = RGBColor(0x6B,0x7A,0x8D); RED = RGBColor(0xC0,0x50,0x2D); ICE = RGBColor(0xD6,0xE4,0xF7)
CJK = "微软雅黑"; SW, SH = In(13.333), In(7.5)
ASSETS = os.path.join(HERE, "assets")
LOGO = os.path.join(ASSETS, "xjtu_logo_white.png")       # 白色校徽横版(透明底)
BUILDING = os.path.join(ASSETS, "xjtu_building.jpg")      # 创新港建筑


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
    # 右上角校名(内容页角标)
    tb(s, In(9.9), In(0.32), In(3.1), In(0.4), "西安交通大学", 12, RGBColor(0x9A,0xA8,0xB8), False, align=PP_ALIGN.RIGHT)
def dark(s):
    bg(s, NAVY); b = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, In(0.12), SH)
    b.fill.solid(); b.fill.fore_color.rgb = MINT; b.line.fill.background()
    if os.path.exists(LOGO):   # 深色页右上角白色校徽
        s.shapes.add_picture(LOGO, In(9.5), In(0.4), height=In(0.5))


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
    w = In(0.5) if len(txt) <= 1 else In(0.62 + 0.12 * (len(txt) - 2))   # 多位数加宽
    sh = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, l, t, w, In(0.5))
    sh.fill.solid(); sh.fill.fore_color.rgb = color; sh.line.fill.background()
    tf = sh.text_frame; tf.margin_top = tf.margin_bottom = tf.margin_left = tf.margin_right = 0
    tf.word_wrap = False
    fs = 20 if len(txt) <= 1 else (16 if len(txt) == 2 else 13)          # 多位数缩字号
    r = tf.paragraphs[0].add_run(); r.text = txt; r.font.size = Pt(fs); r.font.bold = True
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

    # ---- 封面(西交蓝 + 建筑背景 + 校徽)----
    s = S(); bg(s, NAVY)
    if os.path.exists(BUILDING):        # 建筑图铺底部(半幅),上方深蓝叠色
        s.shapes.add_picture(BUILDING, 0, In(4.7), SW, In(2.8))
        band = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, In(4.7), SW, In(2.8))
        band.fill.solid(); band.fill.fore_color.rgb = NAVY
        try: band.fill.transparency = 0.45
        except Exception: pass
        band.line.fill.background()
    if os.path.exists(LOGO):            # 顶部白色校徽横版
        s.shapes.add_picture(LOGO, In(0.85), In(0.7), height=In(0.85))
    tb(s, In(0.9), In(2.4), In(11.5), In(1.9),
       [("面向无人机能耗建模与能量感知路径规划的", 26, WHITE, False),
        ("可信大模型自动化科研框架", 40, RGBColor(0x8C,0xC0,0xFF), True)], space=8)
    tb(s, In(0.9), In(4.05), In(11.5), In(0.9),
       [("A Trustworthy LLM-Agent Autoresearch Framework, Instantiated on UAV Energy", 14, ICE, False),
        ("Modeling and Energy-Aware Path Planning", 14, ICE, False)], space=3)
    tb(s, In(0.9), In(6.5), In(11.5), In(0.6), "西安交通大学　硕士学位论文·中期答辩　|　2026", 15, WHITE, False, anchor=MSO_ANCHOR.MIDDLE)

    # ---- 引入1:研究场景(先落地)----
    s = S(); content(s); title(s, "研究场景:无人机能耗建模 + 能量感知路径规划")
    fit(s, os.path.join(EXP, "pub", "真机代价改变决策.png"), In(0.5), In(2.0), In(7.4), In(4.6))
    tb(s, In(8.1), In(2.0), In(4.9), In(4.8),
       [("能耗建模:", 17, DEEP, True), ("预测飞一段路耗多少电(速度/爬升/载荷→功率)", 14, INK, False),
        ("能量感知规划:", 17, DEEP, True), ("找最省电的路,不只最短(如图:绕墙比翻墙省电)", 14, INK, False),
        ("为什么选这个场景:", 17, DEEP, True),
        ("· 有真机数据(DJI M100,209 航班真实功率)", 14, INK, False),
        ("· 直接关系续航/配送里程", 14, INK, False),
        ("· 能耗与规划天然耦合,完整工程闭环", 14, INK, False)], space=9)
    cite(s, "场景图:真机能耗代价让规划绕开爬升;数据 DJI M100 (Rodrigues 2021)")

    # ---- 引入2:要实现什么 ----
    s = S(); content(s); title(s, "在这个场景上,我们要实现什么")
    for i, (h, d) in enumerate([
            ("建能耗模型", "从真机 DJI M100 数据自动拟合出准确的能耗预测模型"),
            ("接入路径规划", "把能耗模型当规划代价,让无人机飞更省电的路(避爬升)"),
            ("用可信方法做", "全程用防作弊、防自欺的自动科研流程,并诚实刻画能力边界")]):
        y = In(2.4 + i * 1.35)
        c = s.shapes.add_shape(MSO_SHAPE.OVAL, In(1.2), y, In(0.75), In(0.75))
        c.fill.solid(); c.fill.fore_color.rgb = [DEEP, TEAL, MINT][i]; c.line.fill.background()
        rr = c.text_frame.paragraphs[0].add_run(); rr.text = str(i+1); rr.font.size = Pt(26); rr.font.bold = True
        rr.font.color.rgb = WHITE; _font(rr); c.text_frame.paragraphs[0].alignment = PP_ALIGN.CENTER
        tb(s, In(2.4), y + In(0.02), In(3.5), In(0.7), h, 22, INK, True, anchor=MSO_ANCHOR.MIDDLE)
        tb(s, In(6.2), y + In(0.05), In(6.7), In(0.7), d, 16, MUTED, False, anchor=MSO_ANCHOR.MIDDLE)
    cite(s, "三件事一条链:数据 → 能耗模型 → 规划省电,方法学贯穿全程")

    # ---- 1 背景(为什么要'可信')----
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

    # ---- 3 框架总体结构(六部分 + 底部五道机制条)----
    s = numbered(); title(s, "框架总体结构:六个部分")
    # 表头
    hd = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, In(0.7), In(1.95), In(12.0), In(0.5))
    hd.fill.solid(); hd.fill.fore_color.rgb = DEEP; hd.line.fill.background()
    tb(s, In(0.95), In(2.03), In(3.1), In(0.4), "组成部分", 14, WHITE, True)
    tb(s, In(4.1), In(2.03), In(2.2), In(0.4), "角色", 14, WHITE, True)
    tb(s, In(6.4), In(2.03), In(6.1), In(0.4), "说明", 14, WHITE, True)
    rows = [("① 数据与评测", "基础", "以真机功率(P=V·I)为基准的防作弊评测"),
            ("② 自动科研框架", "核心", "大模型闭环搜索:提议 → 评测 → 保留/回滚"),
            ("③ 能耗模型", "产物", "从真机数据拟合的能耗预测模型"),
            ("④ 能量感知规划", "应用", "以能耗为代价,规划更省电的路径"),
            ("⑤ 动力学与真飞控", "验证", "RotorPy 动力学 + PX4 真飞控栈验证"),
            ("⑥ 统计与防守", "边界", "统计检验 + 诚实刻画能力边界")]
    y = In(2.5)
    for i, (a, role, d) in enumerate(rows):
        r = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, In(0.7), y, In(12.0), In(0.55))
        r.fill.solid(); r.fill.fore_color.rgb = RGBColor(0xEE,0xF4,0xF7) if i%2==0 else WHITE
        r.line.color.rgb = RGBColor(0xDD,0xE3,0xE8); r.line.width = Pt(0.5)
        tb(s, In(0.95), y+In(0.08), In(3.1), In(0.4), a, 15, INK, True)
        tb(s, In(4.1), y+In(0.08), In(2.2), In(0.4), role, 14, TEAL, True)
        tb(s, In(6.4), y+In(0.08), In(6.1), In(0.4), d, 13, MUTED, False)
        y += In(0.6)
    # 底部:②的五道安全机制(压缩成一条)
    tb(s, In(0.7), y + In(0.15), In(12.0), In(1.0),
       [("其中 ② 的可信性由五道安全机制保证:", 15, INK, True),
        ("冻结评测器 · keep/revert · 强制查新门 · 因果消融 · 交叉模型+诚实报负结果", 15, DEEP, True)], space=4)

    # ---- 架构图(NEW)----
    s = numbered(); title(s, "框架架构:五节点闭环 + 外搜逃逸支路")
    fit(s, os.path.join(EXP, "pub", "框架架构图.png"), In(0.7), In(1.85), In(11.9), In(4.9))
    cite(s, "LLM 提议→Harness 实验→冻结评测器→keep/revert 闭环;撞瓶颈触发外搜;知识库持久化")

    # ---- 5 loop 全过程(过程即贡献)----
    s = numbered(); title(s, "过程即贡献:loop 撞瓶颈→触发外搜→改向→诚实记录(真实迭代)")
    fit(s, os.path.join(EXP, "loop_process.png"), In(0.5), In(2.0), In(7.6), In(4.5))
    fit(s, os.path.join(EXP, "pub", "loop流程图.png"), In(8.3), In(1.9), In(4.6), In(4.9))
    cite(s, "左:真实迭代轨迹(卡4.24%→外搜→突破1.9%);右:loop 机制流程。数据 agent_log.jsonl + KNOWLEDGE.md")

    # ---- 突破阈值叙事(NEW)----
    s = numbered(); title(s, "autoresearch 如何突破阈值:局部搜索卡壳→外搜识别方向→一步突破")
    fit(s, os.path.join(EXP, "pub", "突破阈值叙事.png"), In(0.7), In(1.9), In(11.9), In(4.6))
    cite(s, "6.88% → 卡 4.2%(7变体全卡)→ 外搜识别'加线性payload' → 1.9%。突破来自框架的外搜环节,非更花哨的模型")

    # ---- 6 能耗迭代 ----
    s = numbered(); title(s, "从真机数据自动建模:能耗预测误差降 73%(6.88% → 1.86%)")
    fit(s, os.path.join(EXP, "fig_summary.png"), In(0.6), In(2.0), In(7.6), In(4.6))
    tb(s, In(8.3), In(2.1), In(4.7), In(4.6),
       [("· 冻结评测器 + LLM 每轮改一次能耗公式", 16, INK, False),
        ("· iter1–10 全程留出验证 + 保留/回滚", 16, INK, False),
        ("· 误差 6.88% → 1.86%(相对提升约 73%)", 16, DEEP, True),
        ("", 8, MUTED, False),
        ("ARE = |预测能量 − 真实能量| / 真实能量", 13, MUTED, False),
        ("在“未参与训练的飞行”上取平均(防背答案)", 13, MUTED, False),
        ("能量真值 = 机载电压 × 电流积分", 13, MUTED, False)], space=8)
    cite(s, "数据:真实 DJI M100(209 航班);评测器 m100_eval.py 冻结")

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
    s = numbered(); title(s, "性能上限由数据决定:留出误差 6 项处触底,加到 52 项不再降")
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
        ("规划器域(有改进空间):", 17, INK, True),
        ("  loop 显著低于随机,z=−2.38", 16, MINT, True),
        ("  0% 随机更好 → 显著胜", 16, INK, False),
        ("→ 优势取决于搜索空间复杂度:小空间相当,大空间显著胜", 16, DEEP, True)], space=12)
    cite(s, "significance_test.py / planner_significance.py(随机分布 + z 检验)")

    # ---- 12b 复杂度规律(为什么打平)----
    s = numbered(); title(s, "为什么打平?引导搜索的优势随搜索空间复杂度增长(文献规律)")
    fit(s, os.path.join(EXP, "pub", "复杂度规律.png"), In(0.5), In(2.0), In(7.4), In(4.6))
    tb(s, In(8.1), In(2.1), In(4.9), In(4.8),
       [("引导搜索优势随空间复杂度增长:", 16, INK, True),
        ("· Bergstra'12:低有效维→随机追平", 14, INK, False),
        ("· REMBO:~15–20 维临界点", 14, INK, False),
        ("· FunSearch:巨大程序空间→引导才行", 14, INK, False),
        ("· 我们能耗域(小)→打平", 15, RED, True),
        ("· 我们规划器域(大)→胜 27%", 15, MINT, True),
        ("→ 同框架横跨两端点,亲手印证规律", 15, DEEP, True)], space=9)
    cite(s, "规律=文献综合;两端点=我们实测。CMU《Hidden Pitfalls》2509.08713 反证可信性为刚需")

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
        ("⑤ 搜索空间大则显著胜随机,空间小(已封顶)则与随机相当", 21, WHITE, False)], space=16)

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
