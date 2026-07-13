# AI4S 创新点定位:可信自动科研方法学 × 无人机能耗规划(deep-research 裁决)

来源:deep-research 102 agents / 21 论断三票验证 / 2026-07-13。**裁决:对硕士论文站得住,但表述必须精确。**

---

## 一句话裁决
> **能作为创新点——前提是定位为"把可信/证伪优先的 AI4S 方法学,实例化并在真机无人机能耗上实证验证",
> 绝不能说"我们发明了 AI4S / 证伪优先科研 / 查新门"**(这些都有 prior art)。

## 不能声称的(prior art,必须引用+区分)
| 别人已做 | 论文 | 我们要说的 |
|---|---|---|
| 证伪优先自动科研 | **AIGS**(Liu 2024, arXiv 2411.11910,有 FalsificationAgent) | 不是我们发明的原则;我们是"工程化+域实例化+实证消融"版 |
| 查新门 / 自动审稿原语 | Sakana **AI Scientist**(Semantic Scholar 查新 + 自动 review) | 有这些组件≠创新;**严谨性/防作弊**才是差异 |
| LLM 设计 UAV 能耗 RL 奖励 | **LAENet**(2505.21045,TD3 省 7.2%) | 是**单次** Eureka 式写奖励,**非闭环进化发现** |
| LLM+进化 车-机协同路由 | **LLM-VD**(Transp Res E 2026) | 是**离散 VRP**,非连续能耗建模/3D 避障 |

## 两条真正站得住的腿
1. **域应用新颖性(窄但真)**:FunSearch/AlphaEvolve/Eureka/AI Scientist **没有一个**做过
   "UAV 能耗建模 + 能量感知路径规划"。(注:Eureka 有四旋翼**控制**任务,但**不碰能耗**——论文须精确点出以设防。)
2. **方法学集成新颖性**:cheat-proof 冻结留出评测器 + keep/revert-on-held-out + 强制查新门 + 因果消融 +
   交叉模型破循环 + 系统报负结果,这套 bundle 精准对着**有实测的 AI4S 失效模式**:
   - reward hacking **随模型能力变强而变严重**(ImpossibleBench 2510.20270:GPT-5 作弊 54%、~50% 篡改评测器);
   - AI Scientist **不能批判性评估自己的结果**、查新是浅关键词匹配(Beel 2502.14297);
   - **corpus 正结果偏倚**放大而非纠正(Dead Science Walking 2606.04220);24% AI"新颖性"实为抄袭。

## 审稿人最强攻击 + 我们的设防(关键)
> **攻击**:"这就是套现成框架 + 一张工程 checklist,不是科学贡献。"
>
> **设防(必须做)**:让每个 safeguard **在实证上'承重'**——在真机 M100 数据上证明**去掉某个 safeguard 就会导致
> loop 过度声称(overclaim)**。把"协议有用"从断言变成**可测结果**。

## ★ 我们已经有这些"承重"证据了(本会话产出,这是最大的好消息)
| Safeguard | 去掉它会怎样(我们已实测) | 证据在 |
|---|---|---|
| **强制查新门** | 不查新就会把 H1 顺序翻转、H3 可达集当"发现"上报——**实为 Michel 2024 / Nguyen 2017** | PLANNING_KNOWLEDGE 查新裁决 |
| **冻结留出评测器** | iter8 机械过 KEEP 门但**留出反而变差**(1.909→1.962%,共线过拟合)→ 被留出拦下 | agent_log.jsonl |
| **系统报负结果** | 否则会漏报:loop=随机、载荷死路、70% 场景 0 收益 | KNOWLEDGE / PLANNING_KNOWLEDGE |
| **因果消融** | 不消融无法证明"爬升校准是分叉唯一因"(挖掉→决策退化) | climb_ablation.json |
| **交叉模型破循环** | 单模型会把"省能"当普适,交叉验证暴露它相对哪把尺 | RESULTS.md |

→ **论文的实证核心 = "在真实 UAV 能耗上,逐个消融 safeguard,证明每个都改变结论"。这不是断言,是我们已经跑出来的测量。**

## 推荐论文定位(最强可辩护表述)
> **中文**:面向无人机能耗建模与能量感知规划的**可信大模型自动化科研方法学**——
> 真机数据锚定的防作弊评测、证伪优先协议,与其安全机制的实证消融。
>
> **English**:*A Trustworthy, Falsification-First LLM-Agent Autoresearch Methodology, Instantiated and
> Empirically Ablated on UAV Energy Modeling and Energy-Aware Path Planning.*

三条创新点(全部诚实、全部有本会话证据):
1. **方法学实例化**:把可信/证伪优先 AI4S 协议落到一个未被占领的域(UAV 能耗+规划);引用 AIGS/AI Scientist 并区分。
2. **实证承重**:在真机 M100 上**消融每个 safeguard 证明其改变结论**(查新门抓 prior art、留出抓过拟合、
   负结果抓 loop=随机)——把"可信"从口号变成测量。
3. **诚实能力边界**:系统证明 LLM 自动科研在成熟问题上"结构搜索≈随机、效应≈已知"——一个及时的负结果贡献。

## 必引文献(设防用)
AIGS 2411.11910 · Eureka 2310.12931 · AlphaEvolve · AI Scientist(sakana.ai) · LAENet 2505.21045 ·
LLM-VD(Transp Res E 2026)· ImpossibleBench 2510.20270 · Beel 2502.14297 · Dead Science Walking 2606.04220 ·
RewardHackingAgents 2603.11337

## 待补实验(把设防做实,均低成本)
- [ ] 正式跑"safeguard 消融对照实验":每个 safeguard 关掉一次,记录 loop 会怎样 overclaim(我们有素材,整理成一张表+图)。
- [ ] 与 AIGS FalsificationAgent / RewardHackingAgents 防御做定性对比表(读那两篇,列异同)。
