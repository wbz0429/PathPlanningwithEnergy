"""
gen_report_html.py — 把 Milestone 快照生成一个自包含的 HTML 报告(图 base64 内嵌,单文件可分享)。
用法: python gen_report_html.py [里程碑号N]   (默认读 experiments/ 当前图 + agent_log)
输出: experiments/report_ms<N>.html
"""
import os, sys, json, base64, html

_HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.join(_HERE, "experiments")
N = sys.argv[1] if len(sys.argv) > 1 else "1"
SRC = os.path.join(EXP, "milestones", f"ms{N}")
if not os.path.isdir(SRC):
    SRC = EXP   # 回退到当前

FIGS = [
    ("fig_ms1_convergence.png", "收敛曲线", "score 15000 → 2078,21 轮每步 KEEP(绿)/REVERT(灰),关键节点标注"),
    ("fig_ms1_overwall.png", "场景A 翻墙拓扑", "agent 自主发现:BEMT 下降近免费 → 翻墙(爬到 ~15m)胜过横向绕行"),
    ("fig_ms1_trajectory_all.png", "三场景侧视轨迹", "基线 vs 调优;A 翻墙大幅优化,B/C 已近最优"),
    ("fig_ms1_waterfall.png", "每个 KEEP 的贡献", "RRT-Connect 压倒性(−11809),其后为平滑器/步长/翻墙探针"),
    ("fig_ms1_energy.png", "能耗 + 留出验证", "调优(seed0)与留出(seed100)几乎重合 → 无种子过拟合"),
]


def img_tag(fname):
    p = os.path.join(SRC, fname)
    if not os.path.exists(p):
        p = os.path.join(EXP, fname)
    if not os.path.exists(p):
        return "<i>(图缺失)</i>"
    b = base64.b64encode(open(p, "rb").read()).decode()
    return f'<img src="data:image/png;base64,{b}" style="width:100%;border-radius:8px;">'


def load_log():
    p = os.path.join(SRC, "agent_log.jsonl")
    if not os.path.exists(p):
        p = os.path.join(EXP, "agent_log.jsonl")
    return [json.loads(l) for l in open(p) if l.strip()]


rows = load_log()
best = {}
bp_ = os.path.join(SRC, "best.json")
if not os.path.exists(bp_):
    bp_ = os.path.join(_HERE, "state", "best.json")
if os.path.exists(bp_):
    best = json.load(open(bp_))

kn = ""
kp = os.path.join(SRC, "KNOWLEDGE.md")
if not os.path.exists(kp):
    kp = os.path.join(_HERE, "KNOWLEDGE.md")
if os.path.exists(kp):
    kn = open(kp).read()

# 迭代表格
tr = []
for r in rows:
    dec = r.get("decision", "")
    color = {"KEEP": "#1b7f37", "REVERT": "#999", "MILESTONE": "#c0392b", "BASELINE": "#2c6fbb"}.get(dec, "#999")
    bg = {"KEEP": "#eaf7ee", "MILESTONE": "#fdecea"}.get(dec, "#fff")
    hyp = html.escape(r.get("hypothesis", "") or "")
    sc = r.get("score")
    sc_str = f"{sc:.1f}" if isinstance(sc, (int, float)) else html.escape(str(sc))
    tr.append(
        f'<tr style="background:{bg}">'
        f'<td>{r.get("iter","")}</td>'
        f'<td>{html.escape(str(r.get("name","")))}</td>'
        f'<td style="text-align:right">{sc_str}</td>'
        f'<td>{html.escape(str(r.get("min_success","")))}</td>'
        f'<td><b style="color:{color}">{html.escape(dec)}</b></td>'
        f'<td style="font-size:12px;color:#555">{hyp}</td></tr>'
    )

cfg = best.get("config", {})
res = best.get("result", {})
det = best.get("detail", {})

kn_html = html.escape(kn)

HTML = f"""<!DOCTYPE html><html lang="zh"><head><meta charset="utf-8">
<title>Autoresearch · Milestone {N}</title>
<style>
 body{{font-family:-apple-system,'PingFang SC',Helvetica,Arial,sans-serif;margin:0;background:#f5f6f8;color:#222;line-height:1.55}}
 .wrap{{max-width:1100px;margin:0 auto;padding:28px}}
 h1{{font-size:26px;margin:0 0 4px}} .sub{{color:#666;margin-bottom:20px}}
 .cards{{display:flex;gap:14px;flex-wrap:wrap;margin:18px 0}}
 .card{{background:#fff;border-radius:12px;padding:16px 20px;box-shadow:0 1px 4px rgba(0,0,0,.06);flex:1;min-width:150px}}
 .card .n{{font-size:28px;font-weight:700}} .card .l{{color:#777;font-size:13px}}
 .green{{color:#1b7f37}} .blue{{color:#2c6fbb}}
 h2{{font-size:19px;margin:30px 0 12px;border-left:4px solid #2c6fbb;padding-left:10px}}
 .fig{{background:#fff;border-radius:12px;padding:16px;box-shadow:0 1px 4px rgba(0,0,0,.06);margin-bottom:20px}}
 .fig h3{{margin:0 0 4px;font-size:16px}} .fig p{{margin:0 0 10px;color:#666;font-size:13px}}
 table{{width:100%;border-collapse:collapse;background:#fff;border-radius:10px;overflow:hidden;box-shadow:0 1px 4px rgba(0,0,0,.06)}}
 th,td{{padding:8px 10px;border-bottom:1px solid #eee;font-size:13px;text-align:left;vertical-align:top}}
 th{{background:#2c6fbb;color:#fff;position:sticky;top:0}}
 pre{{background:#fff;border-radius:10px;padding:18px;box-shadow:0 1px 4px rgba(0,0,0,.06);white-space:pre-wrap;font-size:12.5px;line-height:1.5;max-height:520px;overflow:auto}}
 .tag{{display:inline-block;background:#eaf1fb;color:#2c6fbb;border-radius:6px;padding:2px 8px;font-size:12px;margin-right:6px}}
</style></head><body><div class="wrap">
<h1>🛰️ LLM 智能体驱动的无人机能量-路径规划自优化</h1>
<div class="sub">Milestone {N} · Phase-A 诚实评测器(接地地图 + 速度剖面能量 + 独立碰撞复核)· LLM 全自动闭环</div>

<div class="cards">
 <div class="card"><div class="n blue">15000→2078</div><div class="l">score(7× ↓)</div></div>
 <div class="card"><div class="n green">2079</div><div class="l">留出 seed100(+0.04%,无过拟合)</div></div>
 <div class="card"><div class="n">21</div><div class="l">全自动迭代(7 KEEP / 9 REVERT)</div></div>
 <div class="card"><div class="n green">100%</div><div class="l">三场景成功率</div></div>
</div>

<h2>最优配置</h2>
<div class="fig">
 {' '.join(f'<span class="tag">{html.escape(str(k))}={html.escape(str(v))}</span>' for k,v in cfg.items())}
 <p style="margin-top:10px">平滑器:{html.escape(str(best.get('smoother','—')))} · 各场景剖面能耗:{' / '.join(f"{k.split('_')[0]}={((v or {}).get('energy_mean') or 0):.0f}J" for k,v in det.items())}</p>
</div>

<h2>过程图(科研留痕)</h2>
{''.join(f'<div class="fig"><h3>{i+1}. {t}</h3><p>{d}</p>{img_tag(f)}</div>' for i,(f,t,d) in enumerate(FIGS))}

<h2>迭代账本(每步:假设 → 评测 → 采纳/回退)</h2>
<table><tr><th>#</th><th>动作</th><th>score</th><th>成功率</th><th>决定</th><th>hypothesis(节选)</th></tr>
{''.join(tr)}
</table>

<h2>研究知识库 / 洞见(KNOWLEDGE.md)</h2>
<pre>{kn_html}</pre>

<div class="sub" style="margin-top:26px">分支 agent-autoresearch · 由 autoresearch-step skill × /loop 自动生成</div>
</div></body></html>"""

out = os.path.join(EXP, f"report_ms{N}.html")
open(out, "w").write(HTML)
print("[saved]", out, f"({len(HTML)//1024} KB)")
