---
name: autoresearch-step
description: Run ONE iteration of the UAV energy+path-planning autoresearch loop (propose → evaluate → keep/revert), driven entirely by persistent state files. Invoke directly for a single step, or via /loop for continuous self-optimization. Each call is stateless — it reads state from disk, does exactly one iteration, writes state back.
---

# autoresearch-step — 一次自优化迭代

你是无人机「能量 + 路径规划最优算法」自动化研究的**研究智能体**。本次调用你做**恰好一次迭代**。你没有记忆——所有状态在文件里,**先读状态**。

工作根目录:`/Users/steven/PathPlanningwithEnergy`(下称 ROOT)。Python 用 `ROOT/.venv/bin/python`。

## 文件
- 状态:`autoresearch/state/state.json`(iteration/best/cadence/bottleneck)
- 知识:`autoresearch/KNOWLEDGE.md`(领域知识+insight;你读它、Episode 边界往里追加)
- 账本:`autoresearch/experiments/agent_log.jsonl`(append-only)
- 最优落盘:`autoresearch/state/best.json`(+ 代码改动时 `autoresearch/state/best_smoother.py`)
- 评测器:`autoresearch/physics_eval.py`(**只调用,绝不编辑**)

## 边界规则(铁律,违反=作弊,整个课题作废)
🧊 **冻结,永不可碰**:BEMT 物理参数、kinodynamic(dubins_turning_radius/max_climb_angle)、评测器/度量/能量模型代码、safety_margin、场景/地图/seed、能量锚点。**绝不编辑 `physics_eval.py`、`evaluator.py`、`drone_sim/energy/`、`drone_sim/planning/config.py` 的冻结字段。**
🔧 **只可动 Layer-1**:
- 参数:`step_size, max_iterations, goal_sample_rate, search_radius, weight_energy, weight_distance, weight_time, use_rrt_connect`
- 代码组件:平滑器 `smooth_path(path, is_collision_free, config)`(只 import numpy/math;禁 os/sys/open/eval/exec)

🔍 **探索无限制(重要)**:你可以**随时、任意多次** WebSearch / WebFetch / Read 代码 / 分析,去查文献、找 idea、看实现——**全开放、鼓励**。边界只管「你能**改**什么」(Layer-1),完全不管「你能**查/想**什么」。下面第 3 步的 Episode 检索只是**保证下限**(至少每 N 轮把一次检索蒸馏进 KNOWLEDGE.md),**不是上限**——任何一步你觉得该查文献,就查。

## 一次迭代的步骤(严格照做)
1. **读状态**:读 `program.md`(**人类的研究指令频道——最高优先,先看它的「研究指令」区**)、`state.json`、`KNOWLEDGE.md`、`tail -8 agent_log.jsonl`。记下 `iteration, best, cadence, bottleneck, plateau_count, episode`。**若 `program.md` 的「研究指令」给了新方向,本轮就照它做**(而非只盯 bottleneck)。
2. **若 `best.score` 为 null(首次)** → 本次只建**基线**:`physics_eval.evaluate(best.config, runs=cadence.runs, seed0=0)`,把结果写进 best(score/min_success),写 best.json,append 一行 baseline 到 agent_log,`iteration=1`,报告基线,**结束**。
3. **否则按 cadence 决定**:
   - **Episode 边界**(`iteration % episode_cap == 0` 或 `plateau_count >= plateau_k`):此时**必须至少做一次**针对 `bottleneck` 的文献检索(轻量 3-5 条,如 "energy optimal UAV path smoothing sampling narrow passage"),把有用的招/公式蒸馏 3-5 行追加到 `KNOWLEDGE.md`,`plateau_count=0`、`episode+=1`。然后继续到第 4 步提一个改动。(注:这只是**保证下限**——你在任何一步都可以自由多搜。)
   - **Milestone**(`episode>0 且 episode % milestone_every_episodes == 0` 且尚未为该 episode 出过 milestone):
     (a) **自动出图+归档**(科研过程留痕):跑 `.venv/bin/python autoresearch/viz_milestone.py <里程碑号N>`——生成收敛曲线/翻墙轨迹/三场景轨迹/能耗留出验证/KEEP瀑布,并**归档到 `experiments/milestones/ms<N>/`**(图+state快照+KNOWLEDGE快照);
     (b) 写一段里程碑小结到 `KNOWLEDGE.md`(结果、KEEP/REVERT 账、剩余瓶颈、留出验证);
     (c) `git add -A && git commit -m "Milestone <N>: ..."`;
     (d) **输出小结 + 图路径,停下等用户 review**(本次不提议)。
   - 否则 → 第 4 步。
4. **提议一个改动**(只一个):读 best + 近期 history + KNOWLEDGE,先写一句 `hypothesis`(当前瓶颈 + 为何这改动可能降 score)。参数改动→给 `{键:值}`(只用 Layer-1 键);代码改动→写完整 `smooth_path` 源码。**别重复已 REVERT 过的相同改动**(查 agent_log)。
5. **评测**:写一小段 python 调 `physics_eval.evaluate(overrides, runs=cadence.runs, seed0=0[, smoother_src=<源码字符串>])`,拿 `score / min_success / detail`。（overrides = best.config 合并你的参数改动。）
6. **keep/revert**:**KEEP 当且仅当** `score < best.score` 且 `min_success >= best.min_success`(硬约束不降)。KEEP→更新 best.json(+代码则写 best_smoother.py),`git add -A && git commit -m "autoresearch iter <N>: <name> score=<X> KEEP"`。否则 REVERT(不动 best,不 commit)。
7. **写回状态**:append 一行到 `agent_log.jsonl`(`{iter, kind, name, hypothesis, score, min_success, decision}`);更新 `state.json`(`iteration+=1`;KEEP→`plateau_count=0` 否则 `+=1`;把 `bottleneck` 更新为 detail 里最差的场景)。
8. **报告一句话**:这次改了什么、score/success、KEEP 还是 REVERT。

## 纪律
- 每次**只改一个东西、只评一次**(归因清晰)。搜索用 seed0=0;**别碰 seed**。
- 代码候选先自查只 import numpy/math;`evaluate` 内部已过沙箱/契约,崩溃即当失败 REVERT。
- 若某步不确定是否属于「冻结层」,**默认不碰**并在报告里说明。
