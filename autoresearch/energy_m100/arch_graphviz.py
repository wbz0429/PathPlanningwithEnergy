# -*- coding: utf-8 -*-
"""arch_graphviz.py — 用 Graphviz(自动布局,不重叠,正交箭头)重画框架架构图。
节点按角色配色;主循环左→右;外搜支路在上;知识库在下;冻结评测器标"锁"。
输出 experiments/pub/框架架构图_gv.pdf + .png(矢量)。
"""
import os
import graphviz

HERE = os.path.dirname(os.path.abspath(__file__)); EXP = os.path.join(HERE, "experiments", "pub")
os.makedirs(EXP, exist_ok=True)
FONT = "PingFang SC"
# 角色配色(与 pub 风一致)
LLM, HAR, EVAL, DEC, EXT, KB = "#8172B3", "#4C72B0", "#028090", "#55A868", "#DD8452", "#8C8C8C"

g = graphviz.Digraph("arch", format="pdf")
g.attr(rankdir="LR", splines="spline", nodesep="0.55", ranksep="1.0", bgcolor="white",
       fontname=FONT, labelloc="t", label="可信 AI4S 框架架构:五节点闭环 + 外搜逃逸支路", fontsize="18",
       concentrate="false")
g.attr("node", shape="box", style="rounded,filled", fontname=FONT, fontcolor="white",
       fontsize="13", width="1.9", height="0.95", penwidth="0")
g.attr("edge", fontname=FONT, fontsize="10", color="#444", penwidth="1.6", arrowsize="0.8")

# 主循环四节点(同一 rank,左→右)
def node(nid, title, sub, color):
    g.node(nid, f"{title}\n{sub}", fillcolor=color)

with g.subgraph() as s:
    s.attr(rank="same")
    node("llm", "① LLM 提议", "改写 featurize 代码", LLM)
    node("har", "② Harness 实验", "沙箱执行候选", HAR)
    node("evl", "③ 冻结评测器 🔒", "真机功率·留出·不可改", EVAL)
    node("dec", "④ keep / revert", "留出退则回滚", DEC)

g.edge("llm", "har"); g.edge("har", "evl"); g.edge("evl", "dec")
# 主循环回环(dec→llm),走上方,标签
g.edge("dec", "llm", label="  接受/回滚→下一轮  ", color=DEC, fontcolor=DEC, constraint="false")

# 外搜(顶层,与 llm 同列上方)
node("ext", "★ 外搜 / 文献检索", "撞瓶颈→查新→改方向", EXT)
g.edge("llm", "ext", xlabel="触发:留出停滞", color=EXT, fontcolor=EXT, style="dashed", constraint="false")
g.edge("ext", "llm", xlabel="洞见回灌", color=EXT, fontcolor=EXT, constraint="false")

# 知识库(底层)
node("kb", "⑤ 知识库", "洞见/失败判据/查新裁决 持久化", KB)
g.edge("dec", "kb", color=KB, style="dashed", constraint="false")
g.edge("kb", "llm", color=KB, style="dashed", constraint="false")

# 输入/输出
g.attr("node", fontcolor="#1a2333", fillcolor="#eef4f7", penwidth="1.2", height="0.8")
g.node("data", "真机 M100 数据\n209 航班 · P=V·|I|", color=EVAL)
g.node("out", "输出:模型 + 边界\n+ 诚实负结果", fillcolor="#f0faf4", color=DEC)
g.edge("data", "evl", color=EVAL, style="dashed")
g.edge("dec", "out", color=DEC)

# rank 约束:外搜在顶(与主循环同源但更高),知识库+数据在底
g.body.append('{rank=min; ext}')
g.body.append('{rank=max; kb; data; out}')

out = os.path.join(EXP, "框架架构图_gv")
g.render(out, cleanup=True)   # → .pdf
# 再出 PNG(高清)
g.format = "png"; g.attr(dpi="200"); g.render(out, cleanup=True)
print(f"生成 {out}.pdf + .png")
