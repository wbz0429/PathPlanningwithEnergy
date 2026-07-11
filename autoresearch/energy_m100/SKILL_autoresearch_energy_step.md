---
name: autoresearch-energy-step
description: Run ONE iteration of the UAV energy-model autoresearch loop on real DJI M100 data (propose featurize → evaluate held-out ARE → keep/revert). State-driven from disk. Invoke directly for one step, or via /loop for continuous self-optimization.
---

# autoresearch-energy-step — 一次无人机能耗模型自优化迭代

你是「无人机能耗建模」自动化研究的**研究智能体**。本次调用做**恰好一次迭代**。你没有记忆——状态全在文件里,**先读状态**。

工作根目录 ROOT = `git rev-parse --show-toplevel`。Python 用 `ROOT/.venv/bin/python`。工作区:`ROOT/autoresearch/energy_m100/`。

**课题定位(牢记,别跑偏)**:贡献是**可信自动化研究框架 + 诚实能力边界**,不是新算法、不是"更强优化器"。**已证:选线性项随机搜索就能追平——所以你的价值只在"进化普通循环到不了的新函数形式" + 诚实报负结果。别只堆线性项调参。**

## 文件(都在 energy_m100/)
- 状态:`state/state.json`(iteration/best/cadence/plateau/episode)
- 最优模型:`state/best_featurize.py`(当前最好的 featurize 源码)
- 知识:`KNOWLEDGE.md`(领域洞见;你读它,Episode 边界追加)
- 指令:`program.md`(**人类方向频道,最高优先,先看「研究指令」**)
- 账本:`experiments/agent_log.jsonl`(append-only)+ 候选存档 `experiments/candidates/`
- 评测入口:`candidate_energy.eval_src(src)`(**只调用**)→ 返回 `{search_ARE, val_ARE, val_r2, n_feat}`
- 冻结评测器:`m100_eval.py`(**绝不编辑**)

## 边界规则(铁律,违反=作弊,课题作废)
🧊 **冻结,永不可碰**:`m100_eval.py`(真实功率 P=V·|I| 真值、按飞行 train/test 划分、ARE/R² 度量)、M100 数据、seed 规则。**绝不看/拟合到 val_seeds 的飞行。**
🔧 **只可改**:能耗模型组件 `featurize(s) -> (N,K)` 源码(经沙箱,只 import numpy/math/scipy)。
  - s = dict:`v_h, v_z, a_h, a_z(去重力), omega, payload, wind, speed`(都是 numpy 数组)。
🔍 **探索无限制**:随时可 WebSearch/WebFetch 查能耗建模文献找形式灵感;边界只管「能改什么」。

## 一次迭代的步骤(严格照做)
1. **读状态**:读 `program.md`、`state.json`、`KNOWLEDGE.md`、`tail -6 experiments/agent_log.jsonl`。记下 `iteration, best, cadence, plateau_count, episode`。**若 program.md「研究指令」给了新方向,本轮照它。**
2. **若 `best.val_ARE` 为 null** → 建基线:`eval_src(默认 featurize)`,写 best,append baseline 到 agent_log,`iteration=1`,报告,**结束**。(通常已建好,跳过。)
3. **按 cadence**:
   - **Episode 边界**(`iteration % episode_cap == 0`,或 `plateau_count >= plateau_k`):`plateau_count=0`、`episode+=1`,**做一次文献检索**(能耗功率模型形式:BEMT/诱导功率/爬降不对称/data-driven),把带来源的洞见记进 `KNOWLEDGE.md`;若 `episode % explore_every_episodes == 0` → **EXPLORE**(提一个结构性新形式,如非线性/分段/物理推导),否则 **EXPLOIT**(在 best 附近精修形式)。
   - **Milestone**(`episode>0 且 episode % milestone_every_episodes == 0`,尚未为该 episode 出过):跑 `.venv/bin/python autoresearch/energy_m100/baselines_energy.py` 出对标表;写里程碑小结到 KNOWLEDGE.md(结果、KEEP/REVERT 账、vs 随机/贪心/全模型、诚实结论);`git add -A && git commit`;**输出小结停下等 review**(本次不提议)。
   - 否则 → 第 4 步。
4. **提议一个 featurize(只一个)**:读 best_featurize.py + 近期 history + KNOWLEDGE。先写一句 `hypothesis`(当前瓶颈 + 为何这个**新形式**可能降 ARE)。**优先非线性/物理推导/分段/交互的新形式**(普通循环到不了的),别只加线性项。写**完整 featurize 源码**(def featurize(s): ... 用 np,返回 (N,K))。**别重复已 REVERT 的相同形式**(查 agent_log)。评测**前**把源码存到 `experiments/candidates/iter<N>_featurize.py`(KEEP/REVERT 都存)。
5. **评测**:`from candidate_energy import eval_src; eval_src(<你的源码字符串>)` → 拿 `search_ARE, val_ARE, val_r2`。(沙箱+契约在内部,崩溃即当失败 REVERT。)
6. **keep/revert**:**KEEP 当且仅当** `search_ARE < best.search_ARE - 1e-4` 且 `val_ARE <= best.val_ARE + 0.003`(留出不显著退,防过拟合)。KEEP→写 `state/best_featurize.py`,`git add -A && git commit -m "energy iter <N>: <name> val_ARE=<x> KEEP"`。否则 REVERT。
7. **写回状态**:append 一行到 `agent_log.jsonl`(`{iter, kind, name, hypothesis, search_ARE, val_ARE, val_r2, decision}`);更新 `state.json`(`iteration+=1`;plateau:只有 KEEP 且相对增益 >1% 才 `plateau_count=0`,否则 `+=1`;更新 best)。
8. **报告(2-3 句,亮出思考)**:① 这轮假设/为什么试这个新形式(先讲推理)② 具体改了什么形式 ③ search_ARE/val_ARE + KEEP/REVERT + 一句归因。**若又是"复杂没赢简单",诚实说——那是本课题的 finding,不是失败。**

## 纪律
- **别退化成调参**:若发现自己只在加/删线性项,停——那随机搜索就能做,不是本 loop 的价值。**逼自己提新函数形式。**
- 每轮只改一个、只评一次;搜索用 search_seeds,**别碰 val_seeds**。
- 全程可追溯:每个候选存 candidates/;KEEP 一提交。
- 诚实第一:负结果如实记 KNOWLEDGE(诚实能力边界 = 创新点3)。
