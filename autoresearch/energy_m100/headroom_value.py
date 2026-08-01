# -*- coding: utf-8 -*-
"""headroom_value.py — 头部空间诊断的部署价值量化:跟着诊断走,省预算 or 投预算。
───────────────────────────────────────────────────────────────────────────
核心主张:诊断不是学术玩具——它是一条"要不要跑 autoresearch"的决策规则。
用已落盘数据量化两种情形:

【能耗域(诊断预测 tie)】知识库起点 = Tseng 1.90%。
  · 若按诊断跳过 loop: 拿到 1.90%(知识库),省下整个 loop 预算。
  · 若照旧跑 loop: 产物 physics+payload = 1.93%(iter6),比起点还差。
  → 跟着诊断走 = 省预算 + 不损失(甚至更好)。loop 在饱和域是净负。

【规划器域(诊断预测 config-tie,但识别可逃逸代码空间)】
  · config 域: TUNED+默认平滑器 = 3264 ≈ 随机 3135 = 打平(诊断正确)。
  · 跨空间: 加进化平滑器代码 → 2882,胜随机 8.1%(z=−2.38)。
  → 诊断的价值:识别"当前空间饱和 + 存在可逃逸代码空间"→ 投 loop 有回报。
───────────────────────────────────────────────────────────────────────────
输出: experiments/headroom_value.json
"""
import os, sys, json

HERE = os.path.dirname(os.path.abspath(__file__)); EXP = os.path.join(HERE, "experiments")
J = lambda n: json.load(open(os.path.join(EXP, n)))

# 能耗域:知识库起点 vs loop 产物
zoo = J("model_zoo_leaderboard.json")["results"]
tseng = next(r for r in zoo if r["code"] == "tseng_2022")["val_ARE_percent"]      # 1.90
loop_prod = next(r for r in zoo if r["code"] == "physics_plus_payload")["val_ARE_percent"]  # 2.04 (zoo重实现)
# 用 agent_log 记录的 loop 实际产物(1.93)做对照
agent_loop_prod = 1.93

# 规划器域
pl = J("planner_significance.json")
config_only_loop = 3264.0        # TUNED+默认平滑器(已实测)
full_loop = pl["loop"]           # 2882
random_mean = pl["random_mean"]  # 3135
win_pct = pl["win_pct"]

out = {
    "diagnostic_rule": "if predict==tie → skip loop (save budget); if detect escapable space → invest",
    "energy_domain": {
        "diagnostic": "tie (space saturated)",
        "knowledge_base_start_ARE%": tseng,
        "loop_product_ARE%_agentlog": agent_loop_prod,
        "loop_product_ARE%_zoo": loop_prod,
        "verdict": "loop from a good start was net-negative (product 1.93-2.04% > start 1.90%); "
                   "following the diagnostic saves the loop budget with no loss",
        "loop_budget_saved": "O(10) loop iterations × O(1) eval each, vs diagnostic O(K) single-term evals",
    },
    "planning_domain": {
        "config_space": {
            "diagnostic": "tie (config space saturated)",
            "config_only_loop": config_only_loop,
            "random_mean": random_mean,
            "verdict": "config-only loop ≈ random → tie, diagnostic correct",
        },
        "cross_space_escape": {
            "detection": "escapable program space exists (smoother code)",
            "full_loop": full_loop,
            "escape_gain": full_loop - config_only_loop,
            "win_pct": win_pct,
            "z": pl["loop_z"],
            "verdict": "investing in the loop across space pays +8.1%",
        },
    },
    "summary": "headroom diagnostic = a deployment rule: skip the loop in saturated spaces, "
               "invest where a higher-complexity space is escapable",
}
json.dump(out, open(os.path.join(EXP, "headroom_value.json"), "w"), ensure_ascii=False, indent=2)
print("=== 头部空间诊断的部署价值 ===")
print(f"[能耗域] 诊断=tie → 跳过 loop: 拿知识库 1.90%,省预算")
print(f"         照跑 loop: 产物 {agent_loop_prod}%(agent_log) / {loop_prod}%(zoo) → 比起点差 → 净负")
print(f"[规划器] config 域打平({config_only_loop:.0f}≈{random_mean:.0f}) → 诊断正确;")
print(f"         跨空间逃逸 +{full_loop-config_only_loop:.0f} → 全 loop={full_loop:.0f},胜 {win_pct}% (z={pl['loop_z']})")
print(f"落盘 {EXP}/headroom_value.json")
