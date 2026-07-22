# -*- coding: utf-8 -*-
"""loop_flow_gv.py — 用 Graphviz 重画 loop 机制流程图(替换 loop_process 下半部乱箭头)。
读→提→评→keep/revert→记账 主循环 + 撞瓶颈触发外搜改向。箭头自动布局,不乱。
输出 experiments/pub/loop流程图.pdf + .png
"""
import os, graphviz
HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, "experiments", "pub")
os.makedirs(OUT, exist_ok=True)
FONT = "PingFang SC"
BLUE, TEAL, GRN, EXT = "#0563C1", "#4472C4", "#55A868", "#DD8452"

g = graphviz.Digraph("loopflow", format="pdf")
g.attr(rankdir="LR", bgcolor="white", fontname=FONT, nodesep="0.4", ranksep="0.85",
       labelloc="t", label="loop 机制:读→提→评→keep/revert→记账(循环);撞瓶颈触发外搜改向", fontsize="16")
g.attr("node", shape="box", style="rounded,filled", fontname=FONT, fontcolor="white",
       fontsize="12", width="1.55", height="0.85", penwidth="0")
g.attr("edge", fontname=FONT, fontsize="10", color="#555", penwidth="1.6", arrowsize="0.85")

# 主循环五步(同一 rank)
steps = [("read", "读状态", "冻结评测器/知识库"), ("prop", "提假设", "改一次 featurize"),
         ("eval", "沙箱评测", "search+留出 ARE"), ("dec", "keep / revert", "留出退则回滚"),
         ("log", "记账+提交", "agent_log / git")]
with g.subgraph() as s:
    s.attr(rank="same")
    for nid, t, sub in steps:
        s.node(nid, f"{t}\n{sub}", fillcolor=BLUE)
for a, b in zip([x[0] for x in steps], [x[0] for x in steps][1:]):
    g.edge(a, b)
# 回环:log → read(走上方大弧)
g.edge("log", "read", label="  下一轮  ", color=TEAL, fontcolor=TEAL, constraint="false")

# 外搜(撞瓶颈触发,底部支路)
g.node("ext", "★ 外搜 / 文献检索\n撞瓶颈→查新→改方向", shape="box", style="rounded,filled",
       fillcolor=EXT, fontcolor="white", width="2.6", height="0.9")
g.edge("eval", "ext", label="留出停滞", color=EXT, fontcolor=EXT, style="dashed", constraint="false")
g.edge("ext", "prop", label="带来源洞见回灌", color=EXT, fontcolor=EXT, constraint="false")

out = os.path.join(OUT, "loop流程图")
g.render(out, cleanup=True)
g.format = "png"; g.attr(dpi="200"); g.render(out, cleanup=True)
print(f"生成 {out}.pdf + .png")
