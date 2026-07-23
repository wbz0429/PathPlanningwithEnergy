# -*- coding: utf-8 -*-
"""gen_midterm_docx.py — 按附件4(附件4:硕士研究生中期进展报告.doc)的结构生成中期进展报告 Word。
老 .doc 模板无法直接编辑,故用 python-docx 按模板栏目结构生成新 .docx。
输出:experiments/硕士研究生中期进展报告.docx
"""
import os
from docx import Document
from docx.shared import Pt, RGBColor, Cm
from docx.enum.text import WD_ALIGN_PARAGRAPH

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "experiments", "硕士研究生中期进展报告.docx")

CJK = "宋体"
def set_font(run, size=12, bold=False, name=CJK):
    run.font.name = name; run.font.size = Pt(size); run.font.bold = bold
    from docx.oxml.ns import qn
    run._element.rPr.rFonts.set(qn('w:eastAsia'), name)


def head(doc, txt, size=15):
    p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    r = p.add_run(txt); set_font(r, size, True); return p


def body(doc, txt, size=12, indent=True):
    p = doc.add_paragraph()
    if indent:
        p.paragraph_format.first_line_indent = Cm(0.74)
    p.paragraph_format.line_spacing = 1.5
    for seg in txt.split("**"):
        r = p.add_run(seg); set_font(r, size, bold=(seg == txt and False))
    return p


def cover(doc):
    doc.add_paragraph()
    for txt, sz, al in [("硕士研究生", 26, "C"), ("中期进展报告", 22, "C")]:
        p = doc.add_paragraph(); p.alignment = {"C": WD_ALIGN_PARAGRAPH.CENTER}[al]
        r = p.add_run(txt); set_font(r, sz, True)
    doc.add_paragraph(); doc.add_paragraph()
    for lab in ["学    号：", "研 究 生：", "导   师：", "论文题目：面向无人机能耗建模与能量感知路径规划的可信大模型自动化科研框架",
                "学    科：航空宇航科学与技术（控制科学与工程）", "填写时间：        年    月    日"]:
        p = doc.add_paragraph(); p.paragraph_format.left_indent = Cm(2.2)
        r = p.add_run(lab); set_font(r, 14)


def main():
    doc = Document()
    # 页边距
    for s in doc.sections:
        s.top_margin = Cm(2.54); s.bottom_margin = Cm(2.54)
        s.left_margin = Cm(3.17); s.right_margin = Cm(3.17)

    cover(doc)
    doc.add_page_break()

    # ---- 一、研究内容简介(300-500字)----
    head(doc, "研究题目：面向无人机能耗建模与能量感知路径规划的可信大模型自动化科研框架", 15)
    head(doc, "一、研究内容简介（300～500字）", 14)
    body(doc,
        "本课题面向无人机（UAV）能耗建模与能量感知路径规划，研究一类以真实飞行数据为锚定、以大语言模型（LLM）智能体为驱动的自动化科研框架。"
        "大模型自动科研（AI4S）近年快速发展，但其产出的可靠性存在系统性风险：在缺乏标准答案的真实工程场景中，"
        "自动科研流程容易产生过拟合、把已发表成果误判为新发现、以及选择性报告正结果等问题。"
        "本课题的核心工作是：以真实 DJI Matrice 100 无人机的 209 次飞行机载功率测量数据（电压×电流）为真实基准，"
        "构建一个“防作弊、会诚实报告负结果”的自动科研闭环，实现三个目标："
        "① 自动拟合出准确的无人机能耗预测模型；② 将能耗模型作为规划代价接入路径规划，使无人机飞出更省电的路径；"
        "③ 全程以可信方法完成，并诚实刻画该框架在不同任务上的能力边界。"
        "课题的意义不在于提出新算法（该领域已较成熟），而在于提供一套可信的、真实数据锚定的自动科研方法学，"
        "并系统刻画当前大模型自动科研在真实含噪工程域中的能力边界，为可靠的人机协同科研提供参考。")

    # ---- 二、研究工作进展(主体,详细)----
    head(doc, "二、研究工作进展", 14)
    body(doc,
        "本课题目前已完成从能耗建模、规划集成、动力学与真飞控验证到可信性论证的完整工作链条，"
        "全部实验结果均可由代码与落盘数据复现。主要进展如下。")

    head(doc, "（一）真实数据锚定的能耗模型构建", 13)
    body(doc,
        "以 209 次真实 DJI M100 飞行的机载功率（P=V·|I|）为真值，建立冻结评测器（按航班划分训练/留出集，"
        "以每趟飞行总能量的绝对相对误差 ARE 为指标，评测器全程不可编辑以防作弊）。"
        "LLM 智能体以“提出假设→改写能耗公式（featurize）→冻结评测→保留或回滚”的闭环自动迭代 10 轮，"
        "将能耗预测的留出 ARE 由教科书稳态 BEMT 模型的 6.88% 降至 1.86%（相对提升约 73%）。"
        "测试集上的模型为若干物理特征项（动量诱导功率、前飞诱导、型面与寄生阻力、爬升/下降不对称、载荷）的加权组合，"
        "在 55 趟未参与训练的飞行上泛化良好，可作为后续规划的可信能耗代价。")

    head(doc, "（二）能耗模型接入能量感知路径规划", 13)
    body(doc,
        "将上述真机验证的能耗模型封装为规划器的边权代价函数，接入采样式 A* 规划器。"
        "在高墙等障碍场景中，与“最短距离”“教科书 BEMT”两种代价对比："
        "前两者选择翻越障碍（爬升），真机代价选择绕行以避开爬升能耗，相对省能 5–13%，"
        "且在全速度范围 4–12 m/s 内结论稳定（省能 4–22%）。"
        "机制经三重验证：① 留出真实数据上爬升功率增量预测误差小于 5%（实测 +114W vs 预测 +108W）；"
        "② 因果消融——挖除模型爬升项后规划决策退回翻越，证明爬升标定是分叉的唯一原因；"
        "③ 交叉模型验证暴露评测循环性。进一步在“城市走廊”场景中实现逐障碍混合决策（翻越矮楼、绕行高楼）。")

    head(doc, "（三）动力学与真飞控固件栈闭环验证", 13)
    body(doc,
        "为验证结论是否经受真实飞行动力学，构建两级闭环："
        "① RotorPy 四旋翼刚体动力学（按 M100 尺度配参 2.65kg）+ SE3 几何控制器，规划路径经 MinSnap 平滑后真飞，"
        "绕行省能 6.1%（折线预测 10.7%），并发现“规划安全裕度需给轨迹平滑留余量”的工程规律；"
        "② PX4 SITL + Gazebo Harmonic 真飞控固件在环（EKF 状态估计 + 位置环 + 混控），"
        "以同一固件栈、同一脚本 A/B 对比，绕行相对翻越省能 14.3%。"
        "证据链从折线预测、动力学仿真到真飞控固件栈逐层加固，结论成立。")

    head(doc, "（四）可信自动科研方法学与能力边界刻画", 13)
    body(doc,
        "课题的核心贡献在方法学。构建“冻结评测器 + 保留/回滚 + 强制查新门 + 因果消融 + 交叉模型 + 诚实报负结果”的安全机制，"
        "并通过累加消融逐一证明其承重性：去掉强制查新门，则把配送顺序、可达集等“发现”误报为新结果"
        "（实际为 Michel 2024、Nguyen 2017 等已发表成果，本课题主动引用而不声称首创）；"
        "去掉留出评测，则过拟合探针可刷高分。"
        "同时诚实刻画能力边界：① 能耗预测的留出误差在 6 个特征项处即触底约 1.9%，"
        "增至 52 项仍不降（四种独立方法殊途同归），表明该上界由真实功率中的非运动学噪声决定，"
        "而非模型容量不足；② 统计检验（随机搜索 40 次分布）表明结构搜索在小空间任务上与随机搜索相当（z=−0.35），"
        "而在需要编写代码结构的规划器域显著优于随机（z=−2.38，无一例随机更优），"
        "揭示“自动科研的价值取决于任务是否具备改进空间”的规律（与 Bergstra & Bengio、NAS 等文献一致）。")

    head(doc, "（五）主要成果与阶段性成绩", 13)
    body(doc,
        "1. 建立了冻结的、真实数据锚定的能耗评测与自动科研框架（冻结评测器 m100_eval + 闭环 loop）；"
        "2. 获得留出 ARE 1.86% 的真机验证能耗模型，并接入规划实现省能 4–22%（PX4 真飞控栈省 14.3%）；"
        "3. 形成可信自动科研方法学的安全机制与累加消融证据，以及“数据封顶”与“跨域显著性”两条能力边界量化结论；"
        "4. 产出可复现实验代码 20+ 个、实验结果 JSON 30+ 项、图表与飞行动画若干，全部纳入版本管理可溯源。", indent=False)

    # ---- 三、下一步工作计划 ----
    head(doc, "三、下一步的工作计划", 14)
    body(doc,
        "1. 完善论文写作：将现有方法、实验与能力边界整理为学位论文初稿（方法章、实验章、讨论章）。", indent=False)
    body(doc,
        "2. 功率残差归因（拟进一步研究）：真机功率中运动学仅能解释约 40%，其余约 60% 的来源（风场、电压跌落、控制器修正等）"
        "尚未定量归因，拟通过引入风场通道、分离电压项、分析时间对齐与传感器噪声等手段进一步刻画。", indent=False)
    body(doc,
        "3. 感知闭环扩展：将当前“已知地图 + 全局规划”升级为“深度相机在线建图 + 能量感知滚动重规划”，"
        "并推进 PX4 深度相机（gz_x500_depth）的感知—规划—飞控全闭环验证。", indent=False)
    body(doc,
        "4. 在更大搜索空间验证框架价值：当前能耗任务搜索空间较小（结构搜索与随机相当），"
        "拟在更复杂的规划器/程序级模型空间进一步验证自动科研框架的引导优势。", indent=False)
    body(doc,
        "5. 视情况补充真实无人机外场飞行实验，验证仿真所得省能结论在实机上的有效性。", indent=False)

    # ---- 四、导师评语 ----
    doc.add_page_break()
    head(doc, "四、导师评语", 14)
    for _ in range(8):
        doc.add_paragraph()
    p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    r = p.add_run("签名：　　　　　　　　　　日期：      年    月    日"); set_font(r, 12)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    doc.save(OUT)
    print(f"生成 {OUT}({os.path.getsize(OUT)//1024} KB)")


if __name__ == "__main__":
    main()
