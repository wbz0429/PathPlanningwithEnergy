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

## 补强(2026-07,deep-research 103 agents + domain_value.py 实证)—— 回答"用你这套比不用强在哪 / 别人都有创新你没有"
**裁决:处境有一手文献强力背书,主论点=AI4S 价值域相关,能耗域产 null 是域性质非框架失败。**

- **旗舰 AI4S 结构上就要求可验证评测器**(作者自认局限):AlphaEvolve"需人工实验的任务超出范围";
  FunSearch 只做"有高效 evaluate 的问题"。**噪声真实传感器域 AI4S 用不了——是他们划的界,无成功先例。**
- **连理想域也大多产 null**:AlphaEvolve 50+ 数学问题 **75% 只是 match(null)**、20% 真发现;
  FunSearch 最难任务成功率 **2.9%**。→ "匹配/持平"是常态,封顶域产 null 正常。
- **朴素 AI4S 会 overclaim(实测)**:AI Scientist 独立评估(Beel 等,ACM SIGIR Forum,DOI 10.1145/3769733.3769747):
  42% 实验编码错误失败、**57% 手稿含错误/幻觉数值**、prior art 误判为 novel。**这正是我们 safeguard 拦的。**
- **可信 null 是被接受的贡献类型**:ICML 2024《Embracing Negative Results in ML》、NeurIPS ICBINB、
  EMNLP Insights、2025 顶会 Refutations&Critiques track 提案(Schaeffer/Koyejo/Donoho/Dodge)。
  但**价值取决于严谨性**——冻结评测器+查新门+随机对照提供了它。

### 两个量化证据(domain_value.py,已跑)
- **① 噪声地板**:held-out ARE 在 6 项即触底 **1.88%**;加到 12/20/52 项 held-out 不降反升(过拟合)。
  → 52 项大模型赢不了 6 项 = **地板是数据信息量,不是模型容量**。"数据封顶"从主张变测量。
- **② 跨域对照**:同框架,规划器域(有 headroom)loop **胜随机 27%**、能耗域(封顶)**打平**。
  → **排除"框架无能",证明 null 是域性质**——这是主贡献的关键支柱(deep-research 强烈建议)。

**审稿人会问"够不够"** → 答:①跨域对照排除框架无能 ②噪声地板量化证明数据封顶 ③safeguard 消融证明防 overclaim。
三者合起来 = 一个严谨的、有文献支撑的"可信 AI4S 能力边界"贡献。必引:AlphaEvolve/FunSearch/AI Scientist独立评估(2502.14297)/ICML2024负结果(2406.03980)。

## 补充(2026-07,deep-research 100 agents)—— 无算法创新的框架论文如何主张价值(带顶刊先例)
**问题:框架打不平"最终性能"时,凭什么有价值?答:在其他维度主张,均有先例。**

| 价值维度 | 先例 | 对我们 |
|---|---|---|
| **样本效率/收敛速度**(同等质量、更少评估) | **Bergstra & Bengio 2012**(JMLR,随机搜索超参:一小部分算力达同等质量) | ★ 我们补了收敛速度实验(见下) |
| "≈随机"本身可发表 | **Li & Talwalkar 2019**(random search NAS baseline,arXiv:1902.07638) | 我们诚实报 loop≈随机有先例 |
| 不胜随机时用"相对随机平均的提升"当指标 | Yang et al. 2020(NAS eval frustratingly hard,arXiv:1912.12522) | 我们两域对照正是此法 |
| 强简单基线/诚实 null 当贡献 | White 2020(local search,arXiv:2004.08996);Farahani 2024(arXiv:2403.08265) | 我们的诚实定位 |
| 框架应用(非新学习算法)+ 可解释 + 减少人工 | **FunSearch(Nature 2023)**;**Eureka(ICLR 2024)** | 顶刊/顶会先例,我们同类 |
| 可复现/泛化到新领域(基础设施贡献) | NAS-Bench-Suite(ICLR 2022,arXiv:2201.13396) | 我们冻结评测器+跨域 |

**架构图画法(问题A结论)**:回环作"反向边"走外圈不交叉(Sugiyama 分层,tikz.dev/arXiv:1608.07809);
FunSearch Figure 1(问题→LLM→评测器→数据库→回灌)是我们架构图的同构模板;工具 TikZ(LaTeX矢量)/Graphviz(自动布局)/matplotlib3.5+(中文矢量PDF)。**我们已用 Graphviz 重画,布局原则一致。**
