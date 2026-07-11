# program.md — 无人机能耗模型 autoresearch 研究策略(M100 真数据)

> **人类给研究智能体的指令频道**。agent 每轮先读本文件。karpathy 式:人写方向,循环自动迭代。

## 🎯 研究指令(← 人在此写方向)

**当前指令:进化能耗模型 `featurize(s)`,在真实 DJI M100 功率上把 held-out 能量 ARE 压到最低。**
- 起点:稳态 BEMT `1,v,v²` = 留出 ARE 6.88%;已知加 payload+爬升+加速度可到 ~1.9%。
- **追求 novel 的函数形式**(不只是从项库里挑线性项——随机搜索已能追平线性选择,见 KNOWLEDGE)。
  比如:BEMT 物理推导的非线性(诱导功率 ∝(m·g)^1.5、爬降不对称、v³ 寄生、m 与 v 的耦合)、
  或数据驱动的分段/交互项。**目标是找到线性项库 + 随机搜索到不了的形式。**
- 诚实:若复杂形式没赢简单,如实记 KNOWLEDGE 并 REVERT——负结果也是产出。

## 目标(单一指标,冻结)
最小化 **per-flight 能量 ARE**(留出飞行,seeds 5,6,7)。搜索目标用 seeds 0,1,2 交叉验证。
(per-sample R² 仅参考:5Hz 瞬时功率噪声大,能量才是 planning 要的量。)

## 🧊 不可改(冻结层 = 尺子,碰=作弊)
`m100_eval.py`(真实功率 P=V·|I| 真值、按飞行 train/test 划分、ARE/R² 度量)、M100 数据、seed 规则。
→ 绝不编辑 `m100_eval.py`;绝不看/拟合到 val_seeds 的飞行。

## 🔧 可改(Layer-1 = 搜索空间)
- **能耗模型组件** `featurize(s) -> (N,K)`(经 `candidate_energy` 沙箱注入,只 import numpy/math/scipy)。
  s = dict:v_h, v_z, a_h, a_z(去重力), omega, payload, wind, speed。

## 已知先验(见 KNOWLEDGE.md)
- payload、爬升(v_z>0)、v² 是真实能耗的主导项;稳态 BEMT 漏了 payload/爬升。
- per-sample 功率大部分方差来自风/电压sag/控制(运动学解释 <40%);能量(积分)可预测。
- 结构/项选择:随机搜索追平 autoresearch → **novelty 只能来自新函数形式,不是选项**。

## 迭代纪律
每轮只改 featurize 一次、只评一次;KEEP 仅当 search_ARE 更低且 val_ARE 不显著退(防过拟合)。
