# 能量 + 路径规划最优算法的 Agent Autoresearch —— 分层共识底稿

> 把整条链路切成三层:🧊冻结层 / 🔧可优化层 / 🔬方法论层。
> 核心不变量:**Layer-1 每个可动旋钮,都必须被 Layer-0 的尺子正确计价**(cheat-proof 原则)。

---

## 🧊 Layer 0 — 物理规律层 / 冻结层(尺子 + 物理 + 公平,agent 永不可碰)

| 冻结项 | 内容 |
|---|---|
| **物理定律** | 17 个 BEMT 参数(mass、旋翼几何、空气密度、推力/功率系数、诱导功率因子、阻力、电机/电调效率、电池) |
| **车辆动力学极限**(也是物理) | `dubins_turning_radius`、`dubins_max_climb_angle`、`flight_velocity`(评测速度) |
| **裁判机制** | BEMT 能耗度量、硬安全约束(碰撞检测 + `safety_margin` + **独立 min_clearance 复核**)、归一化 refs、场景几何/障碍/起终点、seed 与评测协议 |
| **最优性锚点** | ⚠️ **grid-A\* 出局**(见下)。改为:**能量加权 Dijkstra/A\***(离散能量最优,可达参考)+ 可采纳能量下界(真下界)+ AO 采样器收敛能量(有限样本上界参考) |

### 为什么 grid-A\* 作为"能量最优锚点"不成立(深度研究结论,带引用)

**双重不成立:**
1. **grid-A\* 只在它自己的离散图上最优,不是连续最优**。朝向被限制在栅格角度 → 路径可证更长:2D 8-邻 ~8%、**3D 26-邻 ~13%**、粗栅格 4-邻 ~41% / 6-邻 ~73%,无障碍平均也 ~5%。(Nash & Koenig, AI Magazine 2013;Bailey et al., Artif. Intell. 2021;Lazy Theta\* AAAI 2010)
2. **A\* 最小化的是长度,不是能耗——目标就选错了**。能量最优 ≠ 最短:实测能耗降幅(51.6%/34%/59%/55%)远超长度降幅(8.7%/18.7%/13.4%/4.1%);能量最优路径可比时间最优多飞 ~4min。(CEAS Aero. J. 2023;arXiv 2402.10529;NASA/AIAA 2024;Tsinghua 2025)

**决定能量最优路径的**:爬升/高度策略、转弯/加速、风向对齐、**U 形速度-功率曲线 E(v)**(有明确最优速度 v\*,如 IRIS v\*≈12 m/s)。→ 我们评测固定 2 m/s 本身就不在能量最优速度上,是"为公平冻结"的取舍,需在写作里说明。

**可信锚点应是**(研究建议):能量加权图搜索(同图上的 energy-weighted A\*/Dijkstra)= 离散能量最优;辅以可采纳能量下界;并注意 **AO 采样器给的是有限样本"上界"(向下收敛到最优),不是可采纳下界**。(Karaman & Frazzoli 2011;Gammell Informed/BIT\*)

---

## 🔧 Layer 1 — 可调研优化层(agent 可动;**不限于超参数**)

### 1a 参数(超参数)
- 搜索:`step_size`、`goal_sample_rate`、`search_radius`;`max_iterations`/`planning_timeout`(**设算力上限**)
- 结构:`use_rrt_connect`
- 代价权重:`weight_energy` / `weight_distance` / `weight_time`
- [Problem-2 later] 滚动/跟踪:`local_horizon`、`execution_ratio`、`replan_threshold`(单独评测问题,共用 L2 loop)

### 1b 算法组件(代码,含"公式")
- `sampler`(窄通道瓶颈的杠杆)、`smoother`、`steer`
- **代价项公式**:agent 可重写 cost function(如加转弯罚项),只要不碰冻结的 BEMT 尺子
- **不含**:碰撞检测、rewiring 正确性(安全/正确性核心)

---

## 🔬 Layer 2 — 方法论层(loop 本身 = 贡献所在)

- 闭环:propose → evaluate → keep/revert + 留出验证
- proposer:LLM(我 / subagent)或 `LLMProposer`
- **文献在环 + 知识库 `KNOWLEDGE.md`**:FunSearch/AlphaEvolve/Eureka 都不做 live 检索 → 欠研究的机制,方法论卖点
- **baseline 铁律 + 消融**(生死线):随机/best-of-N + CMA-ES/BO + 组件消融,匹配预算
- **cheat-proof 原则**:Layer-1 每旋钮必须被 Layer-0 尺子正确计价

---

## 分层如何自动回答 grilling 树

| 问题 | 落层 | 答案 |
|---|---|---|
| 能源超参 vs 物理 | 权重→L1a;BEMT→L0 | 自动分开 |
| kinodynamic 旋钮 | 车辆物理→L0 冻结 | 自动 |
| "不限于超参数" | L1b 代码/公式 | 自动 |
| 滚动/跟踪超参 | L1a Problem-2 | 单独问题共用 loop |
| 目标=方法论还是算法 | L2 主张 / L1 产出彩票 | 自动 |
| A\* 是否最优 | L0 锚点已修 | 能量加权 Dijkstra + 下界 |

---

## 战略解锁(最重要的一条)

当前 3 场景几乎同高度(z=−3)→ 能耗 ≈ 长度 → 最短 ≈ 能量最优 → **agent 没 headroom 可超越**(不是 A\* 真最优,是场景"能量退化")。

**修对锚点(能量加权)+ 设计有真实 3D 能量结构的场景(爬升/下降权衡、或用 U 形速度曲线)后**:能量最优 ≠ 最短,grid-A\*-最短可证 ~13–50%+ 次优 → **agent 第一次有真正的 headroom 去发现"比最短更省能"的路径**。这是把第②重天花板(目标封顶)唯一能松动的地方,也是从"方法论验证"够向"算法贡献"的具体入口。
