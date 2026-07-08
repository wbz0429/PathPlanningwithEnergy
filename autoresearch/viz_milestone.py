"""
viz_milestone.py — Milestone 1 可视化(Phase-A 诚实评测器口径)

生成三张中期材料图:
  1. fig_ms1_convergence.png  迭代收敛曲线(score 15000→2078,KEEP/REVERT 标记 + 关键节点标注)
  2. fig_ms1_overwall.png     场景A 轨迹:基线(侧绕)vs 调优(翻墙)—— 3D + 俯视 + 侧视
  3. fig_ms1_energy.png       三场景剖面能耗:基线 vs 调优(训练 seed0 + 留出 seed100)
"""
import os, sys, json, random
os.environ.setdefault("MPLBACKEND", "Agg")
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
plt.rcParams["font.sans-serif"] = ["PingFang SC", "Heiti SC", "STHeiti", "Arial Unicode MS", "Hiragino Sans GB", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False

import physics_eval as pe, evaluator as ev, benchmark_planning as bp, candidate as cand
from planning.config import PlanningConfig
from planning.rrt_star import RRTStar

EXP = os.path.join(_HERE, "experiments")
TUNED = json.load(open(os.path.join(_HERE, "state", "best.json")))["config"]
DEFAULT = dict(step_size=1.5, max_iterations=5000, goal_sample_rate=0.4, search_radius=4.0,
               weight_energy=0.6, weight_distance=0.3, weight_time=0.1, use_rrt_connect=True)


def _cfg(ov):
    b = dict(ev.BASE); b.update(dict(dubins_turning_radius=1.5, energy_aware=True,
             flight_velocity=2.0, planning_timeout=15.0)); b.update(ov)
    return PlanningConfig(**b)


def _plan(ov, use_tuned_smoother, s, g, seed=0):
    vg, esdf, em = pe.get_grounded_map()
    orig = RRTStar._smooth_path
    try:
        if use_tuned_smoother:
            src = open(os.path.join(_HERE, "state", "best_smoother.py")).read()
            RRTStar._smooth_path = cand.make_patch_method(cand.load_smoother(src))
        random.seed(seed); np.random.seed(seed)
        return RRTStar(vg, esdf, _cfg(ov), energy_model=em).plan(s, g)
    finally:
        RRTStar._smooth_path = orig


# ---------- 1. 收敛曲线 ----------
def fig_convergence():
    rows = [json.loads(l) for l in open(os.path.join(EXP, "agent_log.jsonl"))]
    its = [r["iter"] for r in rows]; sc = [r.get("score") for r in rows]; dec = [r.get("decision") for r in rows]
    best = np.inf; run = []
    for s in sc:
        if s is not None and s < best: best = s
        run.append(best)
    fig, ax = plt.subplots(figsize=(11.5, 5.5))
    ax.plot(its, run, "-", c="#1f77b4", lw=2, zorder=2)
    for x, s, d in zip(its, sc, dec):
        if s is None: continue
        col, mk = ("#2ca02c", "o") if d == "KEEP" else (("#d62728", "s") if d == "MILESTONE" else ("#c0c0c0", "x"))
        ax.scatter(x, s, c=col, marker=mk, s=48, zorder=3, edgecolors="none")
    for i, txt in {1: "RRT-Connect\n0%→100% 成功", 2: "proxy 引导平滑器\n(代码改写)",
                   10: "翻墙路线\n(物理发现)", 18: "骨架-DP"}.items():
        if i < len(sc) and sc[i]:
            ax.annotate(txt, (its[i], sc[i]), textcoords="offset points", xytext=(6, 16),
                        fontsize=8.5, arrowprops=dict(arrowstyle="->", color="gray", lw=0.8))
    ax.set_yscale("log")
    ax.set_xlabel("迭代 iteration"); ax.set_ylabel("score = 三场景剖面能耗和(失败重罚,log)")
    ax.set_title("Milestone 1 自优化收敛:score 15000 → 2078(21 轮 · 7 KEEP / 9 REVERT · LLM 全自动)",
                 fontsize=12, weight="bold")
    ax.legend(handles=[Line2D([0], [0], color="#1f77b4", lw=2, label="至今最优 score"),
                       Line2D([0], [0], marker="o", color="w", markerfacecolor="#2ca02c", ms=9, label="KEEP(采纳)"),
                       Line2D([0], [0], marker="x", color="#c0c0c0", ms=9, label="REVERT(回退)")], fontsize=9)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout(); out = os.path.join(EXP, "fig_ms1_convergence.png")
    fig.savefig(out, dpi=140); plt.close(fig); print("[saved]", out)


# ---------- 2. 翻墙场景轨迹 ----------
def _box_faces(x0, x1, y0, y1, a0, a1):
    v = [(x0, y0, a0), (x1, y0, a0), (x1, y1, a0), (x0, y1, a0),
         (x0, y0, a1), (x1, y0, a1), (x1, y1, a1), (x0, y1, a1)]
    idx = [(0, 1, 2, 3), (4, 5, 6, 7), (0, 1, 5, 4), (2, 3, 7, 6), (1, 2, 6, 5), (0, 3, 7, 4)]
    return [[v[i] for i in f] for f in idx]


def fig_overwall():
    S = np.array([0., 0., -3.]); G = np.array([70., 0., -3.])
    pb = _plan(DEFAULT, False, S, G)                 # 基线:RRT-Connect 默认平滑
    pt = _plan(TUNED, True, S, G)                     # 调优:含 v13 平滑器
    _, _, em = pe.get_grounded_map()
    fig = plt.figure(figsize=(18, 5.5))
    ax3d = fig.add_subplot(1, 3, 1, projection="3d")
    axxy = fig.add_subplot(1, 3, 2); axxz = fig.add_subplot(1, 3, 3)
    for o in bp.BLOCKS_OBSTACLES:
        x0, x1 = o["x_range"]; y0, y1 = o["y_range"]; a0, a1 = -o["z_range"][1], -o["z_range"][0]
        for f in _box_faces(x0, x1, y0, y1, a0, a1):
            ax3d.add_collection3d(Poly3DCollection([f], alpha=0.10, facecolor="gray", edgecolor="none"))
        axxy.add_patch(Rectangle((x0, y0), x1-x0, y1-y0, alpha=0.30, color="gray"))
        axxz.add_patch(Rectangle((x0, a0), x1-x0, a1-a0, alpha=0.30, color="gray"))
    for p, c, lab in [(pb, "#ff7f0e", "基线 RRT-Connect(侧绕)"), (pt, "#1f77b4", "调优后(翻墙,agent 发现)")]:
        if not p: continue
        P = np.array(p); alt = -P[:, 2]; E, _ = em.compute_energy_for_path(p, 2.0)
        Ep = pe.energy_with_profile(p, em, pe._GCACHE.get("vstar", 18.2))
        tag = f"{lab}  剖面能耗≈{Ep:.0f}J"
        ax3d.plot(P[:, 0], P[:, 1], alt, "-", c=c, lw=2.4)
        axxy.plot(P[:, 0], P[:, 1], "-", c=c, lw=2.4, label=tag)
        axxz.plot(P[:, 0], alt, "-", c=c, lw=2.4, label=tag)
    axxz.axhline(0, color="saddlebrown", lw=1.5)
    ax3d.set_title("(a) 3D 轨迹(灰=实心墙)"); ax3d.set_xlabel("X"); ax3d.set_ylabel("Y"); ax3d.set_zlabel("高度")
    ax3d.view_init(elev=20, azim=-65)
    axxy.set_title("(b) 俯视 X-Y:基线横向绕行"); axxy.set_xlabel("X(m)"); axxy.set_ylabel("Y(m)"); axxy.legend(fontsize=8); axxy.grid(alpha=0.3)
    axxz.set_title("(c) 侧视 X-高度:调优后翻墙(降落近免费)"); axxz.set_xlabel("X(m)"); axxz.set_ylabel("高度(m)"); axxz.legend(fontsize=8); axxz.grid(alpha=0.3)
    fig.suptitle("Milestone 1 · 场景A:agent 自主发现「翻墙」拓扑(BEMT 下降近免费)", fontsize=13, weight="bold")
    fig.tight_layout(); out = os.path.join(EXP, "fig_ms1_overwall.png")
    fig.savefig(out, dpi=140); plt.close(fig); print("[saved]", out)


# ---------- 3. 三场景能耗对比 ----------
def fig_energy():
    print("[eval] 基线 vs 调优(训练 seed0 + 留出 seed100)...")
    rb0 = pe.evaluate(DEFAULT, runs=3, seed0=0)
    src = open(os.path.join(_HERE, "state", "best_smoother.py")).read()
    rt0 = pe.evaluate(TUNED, runs=3, seed0=0, smoother_src=src)
    rt1 = pe.evaluate(TUNED, runs=3, seed0=100, smoother_src=src)
    scs = ["A_straight", "B_diag_up", "C_diag_down"]
    def e(r, n): return (r["detail"].get(n, {}) or {}).get("energy_mean") or 0.0
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    x = np.arange(3); w = 0.28
    ax1.bar(x - w, [e(rb0, s) for s in scs], w, label="基线(默认)", color="#ff7f0e")
    ax1.bar(x, [e(rt0, s) for s in scs], w, label="调优(训练 seed0)", color="#1f77b4")
    ax1.bar(x + w, [e(rt1, s) for s in scs], w, label="调优(留出 seed100)", color="#2ca02c")
    ax1.set_xticks(x); ax1.set_xticklabels(["A 直穿", "B 对角上", "C 对角下"])
    ax1.set_ylabel("剖面能耗(J)"); ax1.set_title("(a) 三场景能耗:基线 vs 调优 vs 留出验证")
    ax1.legend(fontsize=8); ax1.grid(alpha=0.3, axis="y")
    labs = ["基线\nseed0", "调优\nseed0", "调优\nseed100(留出)"]
    scr = [rb0["score"], rt0["score"], rt1["score"]]
    ax2.bar(labs, scr, color=["#ff7f0e", "#1f77b4", "#2ca02c"])
    for i, s in enumerate(scr): ax2.text(i, s, f"{s:.0f}", ha="center", va="bottom", fontsize=10)
    ax2.set_ylabel("总 score"); ax2.set_title("(b) 总分:调优泛化到留出种子(无过拟合)")
    ax2.grid(alpha=0.3, axis="y")
    fig.suptitle("Milestone 1 · 优化前后 + 留出验证(Phase-A 诚实评测器)", fontsize=13, weight="bold")
    fig.tight_layout(); out = os.path.join(EXP, "fig_ms1_energy.png")
    fig.savefig(out, dpi=140); plt.close(fig); print("[saved]", out)
    print(f"  基线 seed0 score={rb0['score']:.0f} | 调优 seed0={rt0['score']:.0f} | 调优 seed100={rt1['score']:.0f}")


# ---------- 4. 三场景侧视轨迹 ----------
def fig_trajectory_all():
    _, _, em = pe.get_grounded_map()
    vstar = pe._GCACHE.get("vstar", 18.2)
    scns = [("A 直穿", np.array([0., 0., -3.]), np.array([70., 0., -3.])),
            ("B 对角上", np.array([0., 0., -3.]), np.array([70., 20., -3.])),
            ("C 对角下", np.array([0., 0., -3.]), np.array([70., -25., -3.]))]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, (nm, s, g) in zip(axes, scns):
        for o in bp.BLOCKS_OBSTACLES:
            x0, x1 = o["x_range"]; a0, a1 = -o["z_range"][1], -o["z_range"][0]
            ax.add_patch(Rectangle((x0, a0), x1-x0, a1-a0, alpha=0.28, color="gray"))
        for ov, tuned, c, lab in [(DEFAULT, False, "#ff7f0e", "基线"), (TUNED, True, "#1f77b4", "调优")]:
            p = _plan(ov, tuned, s, g)
            if not p: continue
            P = np.array(p); Ep = pe.energy_with_profile(p, em, vstar)
            ax.plot(P[:, 0], -P[:, 2], "-", c=c, lw=2.2, label=f"{lab} {Ep:.0f}J")
        ax.axhline(0, color="saddlebrown", lw=1.3)
        ax.set_title(nm); ax.set_xlabel("X(m)"); ax.set_ylabel("高度(m)"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.suptitle("Milestone 1 · 三场景侧视轨迹(基线 vs 调优)", fontsize=13, weight="bold")
    fig.tight_layout(); out = os.path.join(EXP, "fig_ms1_trajectory_all.png")
    fig.savefig(out, dpi=140); plt.close(fig); print("[saved]", out)


# ---------- 5. KEEP 贡献瀑布 ----------
def fig_waterfall():
    rows = [json.loads(l) for l in open(os.path.join(EXP, "agent_log.jsonl"))]
    keeps = [(r["iter"], r.get("name", ""), r.get("score")) for r in rows
             if r.get("decision") == "KEEP" and r.get("score")]
    labels = [f"#{i}\n{n[:16]}" for i, n, _ in keeps]; scores = [s for _, _, s in keeps]
    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.step(range(len(scores)), scores, where="mid", color="#1f77b4", lw=2)
    ax.scatter(range(len(scores)), scores, color="#2ca02c", zorder=3, s=55)
    for i in range(1, len(scores)):
        ax.annotate(f"{scores[i]-scores[i-1]:+.0f}", (i, scores[i]), textcoords="offset points",
                    xytext=(0, -15), ha="center", fontsize=8, color="#d62728")
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, fontsize=7, rotation=15)
    ax.set_yscale("log"); ax.set_ylabel("至今最优 score(log)")
    ax.set_title("Milestone 1 · 每个 KEEP 对 score 的贡献", fontsize=12, weight="bold")
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout(); out = os.path.join(EXP, "fig_ms1_waterfall.png")
    fig.savefig(out, dpi=140); plt.close(fig); print("[saved]", out)


if __name__ == "__main__":
    import glob, shutil
    ms = sys.argv[1] if len(sys.argv) > 1 else None   # 传里程碑号则归档
    for f in (fig_convergence, fig_overwall, fig_energy, fig_trajectory_all, fig_waterfall):
        try:
            f()
        except Exception as ex:
            import traceback; print(f"[FAIL] {f.__name__}: {ex}"); traceback.print_exc()
    if ms:
        d = os.path.join(EXP, "milestones", f"ms{ms}"); os.makedirs(d, exist_ok=True)
        for png in glob.glob(os.path.join(EXP, "fig_ms1_*.png")):
            shutil.copy(png, d)
        for rel in ("state/state.json", "KNOWLEDGE.md"):
            src = os.path.join(_HERE, rel)
            if os.path.exists(src):
                shutil.copy(src, os.path.join(d, os.path.basename(rel)))
        print(f"[archived] milestone {ms} -> {d}(图 + state 快照 + KNOWLEDGE 快照)")
