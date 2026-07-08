# 开放问题 & 甜点问题调研(2023-2026,带引用)

## 摘要
在能量感知UAV路径规划(2023-2026)领域,近年survey与primary论文明确点名的开放问题高度收敛于四条主线:(A) 采样式规划器(RRT*/BIT*/informed)的有限时间收敛速度无理论保证,朴素RRT*被证明对路径长度只做次线性收敛,且在能量/多目标等非欧代价下连渐近最优的假设都可能失效;(B) 窄通道/kinodynamic是公认未解benchmark——最优采样规划器在权威review的6个窄通道场景只在1个找到低代价解、最难的Twister无人能解,动态约束下只有可行性规划器能出解而渐近最优的SST*全军覆没;(C) 部分观测/未知地图下的能量感知规划仍属新兴且欠成熟领域,主流frontier探索用几何/时间代价排序、假设功率-速度恒定,忽略了机动瞬态能耗,且当前SOTA(EAAE/WESPR)仅在仿真验证、风扰动与在线地图重建全部deferred为future work;(D) 存在若干"甜点问题":有明确可测指标+已知SOTA留有可量化差距+存在可利用结构,最具体的是DYNUS在动态未知环境达100%成功率、travel time比SOTA快~25%(说明此前SOTA留有~25%时间差距),以及informed sampling在Riemannian/非欧代价下的启发式不可采纳性(标量特征值界过保守、丢弃方向结构)。这些问题因为客观指标清晰、结构可利用、且被权威文献反复点名为未解,特别适合用自动化搜索/LLM闭环去啃。

## 逐条

### 采样式规划器(RRT*/BIT*/informed)的有限时间收敛速度缺乏理论保证,是被权威survey明确点名的开放方向:朴素RRT*对路径长度只做次线性收敛,informed变体可达线性收敛;不同收敛速率造成有限时间性能上数量级差异。这是能量感知规划中所用经典RRT*的一个具体、可量化的弱点与改进空间。
**置信度**:high
Gammell & Strub 2021 (Annual Review) 明确指出渐近最优采样规划器'converge with infinite samples but have no guarantees on their rate of convergence... orders-of-magnitude differences in finite-time performance',并有形式化定理证明'naive RRT* converges sublinearly... focused variants such as Informed RRT* can have linear convergence'(对应Informed Sampling IEEE T-RO Thm18/Thm20)。
**来源**:https://arxiv.org/pdf/2009.10484

### 最优解集测度为零导致采样规划器对缺乏强δ-clearance(非鲁棒可行)的问题求解概率为零——这是窄通道/近零测度困难的形式化陈述,直接激发了受限与能量感知规划的挑战。
**置信度**:high
survey陈述'Sampling-based motion planners have zero probability of solving problems where all solutions do not have strong δ-clearance... The set of optimal solutions has zero measure and therefore zero probability of being sampled',与Karaman&Frazzoli 2011独立印证(μ(σ*)=0)。
**来源**:https://arxiv.org/pdf/2009.10484

### 能量感知/多目标(能量-时间-风险)代价函数是非欧的,复杂/非长度代价(如最大化最小clearance、cost map、目标组合)可能违反证明渐近最优所用的假设,不被原始/标准保证覆盖。
**置信度**:high
survey §3.4.1明确:'Many planning problems seek to optimize a more complex cost than path length... These objective functions may not meet the assumptions used to first prove asymptotically optimality'。Devaurs et al. TASE2016进一步指出非标准代价下Lipschitz连续性不满足、渐近最优无法保证。注:同段也提到存在针对cost map/Pareto的专用渐近最优算法。
**来源**:https://arxiv.org/pdf/2009.10484

### informed sampling(加速RRT*/BIT*/informed的核心手段)主要为欧氏空间设计,在配置相关的Riemannian度量(非欧代价)下Euclidean启发式变得不可采纳(inadmissible);已有修复(标量特征值界)虽恢复可采纳性,却丢弃度量的方向结构、产生过保守的informed set——这是当前最好informed-sampling启发式的一个具体、可量化弱点与开放改进方向。
**置信度**:high
Kyaw & Kelly 2026摘要:'most existing methods rely on Euclidean heuristics that become inadmissible under configuration-dependent Riemannian metrics';'scalar eigenvalue bounds restore admissibility by uniformly scaling... discard the directional structure... producing overly conservative informed sets'。其matrix-valued Loewner-order启发式在UR5/Franka/PR2上给出可测改进差距。arXiv 2602.00992独立印证。
**来源**:https://arxiv.org/pdf/2606.02879

### 窄通道是采样式规划公认的hard、基本未解的benchmark:权威comparative review自身实验中,最优规划器在6个窄通道场景仅1个找到低代价解,最难的Twister(10-DOF snake)无任何规划器求解;经典APF/repulsive代价在窄通道内赋高代价、人为堵死可通行路径。
**置信度**:high
Orthey/Chamzas/Kavraki 2024 review:'near-zero measure, and near-zero probability to sample states';'optimal planners can only find low-cost solutions in one scenario';'twister scenario was not solved by any planner'(6场景)。arXiv 2605.15999:'classical artificial potential field-based costs typically have a high cost in narrow passages, artificially blocking the navigable path';IJCAS 2024 Adaptive Informed RRT*预先识别可通行窄通道以保证可穿越性。
**来源**:https://www.kavrakilab.org/publications/orthey2024-review-sampling.pdf; https://arxiv.org/pdf/2605.15999; https://link.springer.com/article/10.1007/s12555-022-0834-9

### kinodynamic/微分约束规划是独立的开放弱点:review的动态模型实验中只有Kinodynamic-RRT能出解,渐近最优的SST*一个动态场景都解不了,显示微分约束下巨大的'可行性 vs 最优性'gap;此外对许多非完整系统,RRT*因需要steering函数(两点BVP求解器,常迭代慢且任意两状态间可能无可行轨迹)而不实用。
**置信度**:high
Orthey 2024:'In the dynamic cases, only Kinodynamic-RRT is able to find solutions, while the optimal planner SST* cannot solve any of the scenarios'。PracSys/Li,Littlefield,Bekris IJRR2016:'Numerical BVP solvers are often iterative and slow... For many non-holonomic systems, a feasible trajectory between two arbitrary states may not exist'——整个SST/SST*研究计划以此为动机。
**来源**:https://www.kavrakilab.org/publications/orthey2024-review-sampling.pdf; https://www.pracsyslab.org/motion-planning/asymptotic-kinodynamic-motion-planning/

### 部分观测/未知地图下的能量感知规划仍属欠成熟的新兴领域,存在多个被作者明确点名的开放gap:主流frontier探索用几何/时间代理排序目标、假设功率-速度恒定,因而忽略了机动/加速瞬态导致的能耗差异(相同路径长度能耗可显著不同);基于完整候选轨迹预测能量同时保持在线反应性'remains a significant challenge'。
**置信度**:high
EAAE (arXiv 2603.15604, TU Delft/UZH 2026):'underlying cost functions typically assume a constant power-to-speed relationship... fail to account for the high-power transients';目标排序'remain geometric (or time-proxy)... do not explicitly account for trajectory-dependent energy expenditure that can vary significantly even for similar path lengths';'energy-aware planning for UAVs... remains less established than coverage- or information-driven strategies'。
**来源**:https://arxiv.org/pdf/2603.15604

### 能量感知规划存在明确的sim-to-real与风扰动不确定性gap:当前SOTA(EAAE)仅在仿真验证,风扰动、载荷变化下的动态约束、更大/动态环境、多UAV全部deferred为future work;wind-aware规划的最相关先前工作把风模型离线回归成静态场、对新环境适应性差,而在线自适应风建模+在线几何重建+RL自动代价函数优化被明确列为future work——即部分观测能量感知规划与自动代价调参是开放、结构可利用的问题。
**置信度**:high
EAAE conclusion:'Future work will focus on validating EAAE in real-world flight experiments... handle wind disturbances, dynamic constraints under payload changes, and multi-UAV'(仅仿真验证)。WESPR (arXiv 2603.09194):先前CFD-informed工作'wind model was generated offline and regressed into a static field, limiting adaptability';其future work点名'LiDAR-based geometry reconstruction... online flow estimation'与'Reinforcement Learning for automatic optimization of cost-function'。
**来源**:https://arxiv.org/pdf/2603.15604; https://arxiv.org/pdf/2603.09194

### UAV规划survey给出经典规划器(graph/sampling/optimization)可部署自主性的三条具体、可核对的开放gap:(i)缺乏校准的不确定性/风险模型,(ii)缺乏形式化runtime安全层,(iii)缺乏对compute与energy预算的系统处理;能量消耗当前被under-treated,呼吁把计算成本/能耗/通信作为一等objective纳入规划,并用anytime+有界次优性算法作为结构杠杆。
**置信度**:high
RCSR survey (Drones 2026, 10(5),351):三gap列表逐字'(i) calibrated models of uncertainty and risk, (ii) formal runtime safety layers, and (iii) systematic treatment of compute and energy budgets';'develop resource-aware planning frameworks that explicitly incorporate computational cost, energy consumption, and communication requirements into planning objectives';'Anytime planning algorithms with bounded suboptimality guarantees can provide useful tradeoffs'。同时点名采样式规划器'may struggle under strict latency constraints and complex operational restrictions'。
**来源**:https://www.mdpi.com/2504-446X/10/5/351

### 存在满足三条(明确可测指标+已知SOTA留有可量化差距+结构可利用)的'甜点问题'候选:①动态未知环境下的travel-time/成功率——DYNUS达100%成功率、travel time比SOTA快~25%,说明此前SOTA留有~25%时间差距,且软约束(快但不保证无碰)vs硬约束(安全但慢)的权衡是可攻的结构;②非欧/Riemannian代价下的informed sampling收紧(有UR5/Franka/PR2可测差距);③窄通道穿越成功率(6场景仅1解、Twister 0解);④能量感知探索的轨迹级能量排序(相同长度能耗可显著不同,指标=能量/成功率/时间)。
**置信度**:high
DYNUS (arXiv 2504.16734, MIT):'achieves a success rate of 100% and travel times that are approximately 25.0% faster than state-of-the-art methods';并点名'soft-constraint... do not guarantee collision-free paths even with static obstacles... hard-constraint methods ensure collision-free safety, but typically have longer computation times'。配合claim16-18的可测差距(informed set保守度、6场景1解)构成甜点集。注:DYNUS的25%是travel-time而非能量指标。
**来源**:https://arxiv.org/pdf/2504.16734; https://arxiv.org/pdf/2606.02879; https://www.kavrakilab.org/publications/orthey2024-review-sampling.pdf; https://arxiv.org/pdf/2603.15604

### conventional/informed RRT*被指采样策略低效、搜索树规模不当增加计算负担,且'需要多少棵树'在既往研究中鲜有系统研究——是一个结构可利用的欠研究改进点(多树数量优化)。
**置信度**:medium
Adaptive Informed RRT* (IJCAS 2024)摘要:'inefficient sampling strategies and inadequate scales of searching trees increase the burden of calculation';'the necessity of sufficient trees... is rarely investigated in previous research'。RRdT*/RRF*/Multi-Tree-RRT*等印证树数量欠研究。注:2-1投票,且原文批评是对RRT*家族泛指、非专指Informed RRT*(该论文本身即Informed变体),存在轻微scope误attribution。
**来源**:https://link.springer.com/article/10.1007/s12555-022-0834-9
