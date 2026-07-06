Grounding confirmed against the actual repo. The agent run log (`autoresearch/experiments/agent_log.jsonl`) matches the OURS narrative exactly: iter0 baseline fails scenario A (min_success 0.0, vs_astar 1.128), iter1-2 tuning plateaus at vs_astar **1.1067** (new best), iter3 smoother rewrite → **REVERT**, iter7 a config drifts to **1.2279**. The evaluator (`evaluator.py`) is genuinely cheat-proof: PENALTY=10000 hard-gates 100% success before energy counts, A* normalizes `vs_astar`, `seed0` param enables held-out seeds. Below is the positioning document.

---

# 战略定位:LLM 智能体闭环 × 作弊不了的物理评测器 × 能量感知窄通道 UAV 规划

## 执行摘要(约200字)

诚实结论先摆:**这是一次强方法论验证 + 弱算法增量,不是新算法。** 我们真正做出来、且经得起内行盘问的东西是——在 karpathy/autoresearch 范式上搭了一个零 API key 可跑的 LLM 智能体闭环,配一个结构上作弊不了的物理评测器(碰撞硬约束 + A* 绝对地板 + BEMT 能耗 + 留出种子),并用它自主复现了人类手动做出的算法级改动(RRT-Connect)、当场抓破了作弊(新采样成功率 100%→50%)、给出了专家级根因诊断(场景 A 是同伦类问题)。但闭环本身是 FunSearch/AlphaEvolve/autoresearch 的既有范式,RRT-Connect 是 2000 年的已知算法,平滑增益仅 ~0.5%(低于噪声)。所以别宣称"补上了 AI Scientist 的 verifiability 短板"——我们本来就站在正确家族这侧。可发表的定位是:**首次把可验证-evaluator 科研闭环诚实落到带物理约束的采样式 UAV 规划域,并给出可复现的能力证据 + 评测器设计**,workshop/短文/系统贡献档。

---

## 1. 痛点

**一句话**:LLM 自动化科研的可信度危机——当没有 ground truth 时,用 LLM 当裁判(LLM-as-judge)系统性不可靠,导致"AI 做出了科研贡献"这类主张普遍无法被信任。

**为什么真、且现有方法解不好(挂文献):**
- 靶子是"论文工厂"家族(Sakana AI Scientist `2504.08066`、Zochi、Agent Laboratory `2501.04227`、Google co-scientist):它们靠 LLM-as-reviewer 或真人 peer review 打分。Sakana 的自动 reviewer 会漏掉自己论文的重大缺陷、却把 4 篇已被人类接收的论文拒掉(`TechCrunch 2025-03` / `arXiv 2502.14297`);Zochi 直接把"自动 reviewer 给自己打 7.67"当卖点——这是循环自评。
- 有系统性证据:无 ground truth 时 LLM-judge 存在自我偏好偏差、家族偏差,评分者一致性甚至为负(`Reliability without Validity 2606.19544`;`ICLR2025 Limits to Scalable Evaluation`)。
- **这个痛点解不好的根因不是"没人想到要评测",而是"目标域根本没有便宜、确定、防幻觉的 ground truth"**。软件/ML 域勉强用 held-out loss;生物医学域只能把验证外包给人类湿实验(co-scientist 的做法)。带物理仿真+可执行约束的机器人/控制域几乎没人把闭环落进去并用物理量当硬指标——这是相对空白。

**注意别夸大**:"要造硬评测器"这件事本身不是我们的洞见,FunSearch(`Nature 2023`)2023 年就是核心特征。我们不是修好了全领域的洞,只是**本来就站在有硬评测器的那一侧**。

---

## 2. 场景(范围别贪大)

我们精准打的是**一个具体、狭窄的交叉格子**,不是"自动化科研"也不是"更好的规划器":

> **用一个作弊不了的物理评测器,驱动 LLM 智能体在设计时(design-time)改写一个能量感知的采样式 UAV 规划器,并验证这个闭环能否可靠地区分"真改进 / 过拟合 / 作弊"。**

三个限定词都必须钉死,否则会被审稿人用相邻工作打死:
- **设计时 vs 运行时**:LLM 改的是规划器源代码(monkeypatch 注入 `smooth_path`),不是像 LASMP / LLM-A* / iLLM-A*(`2510.02716`)/ NAMO-LLM 那样在运行时当采样偏置器或 waypoint 生成器。这是我们对"LLM + motion planning"整条线的第一条区隔。
- **域 = 连续采样式 + 物理能耗**:不是 FunSearch/EoH(`ICML2024`)/irace-evo(`2511.14794`)那种组合优化(bin-packing/TSP/CVRP),也不是 Eureka(`ICLR2024`)那种进化 reward。这是对"LLM 进化算法代码"整条线的第二条区隔。
- **能量感知 + 窄通道点到点**:主流能量感知 UAV 工作几乎全在覆盖路径规划(CAPG 等高线、TEeVTOL 风场 `2403.14877`、ARENA `2502.19401`),**没人把能量感知和杂乱窄通道点到点导航放一起**——这个组合欠研究,但我们的差异化在"物理不可作弊评测器",不在能量算法本身。

**不要**把场景说成"补上 AI Scientist 领域短板"或"提出新规划算法"。两个都会被一眼看穿。

---

## 3. 创新点(诚实分档)

| 档 | 内容 | 挂靠证据 |
|---|---|---|
| **真新颖(有限)** | 把 FunSearch/AlphaEvolve/Eureka 式 LLM-代码进化闭环,首次落到**采样式运动规划子程序**(平滑/采样函数),并用**BEMT 物理能耗**评测 | 5路检索均返回 NO hits:"LLM evolve RRT/sampling planner + evaluator"这个格子未被占;irace-evo/EoH 明确不碰 robotics |
| **已知范式的新应用(增量)** | 域迁移:把 karpathy autoresearch 的 propose→evaluate→keep/revert + 留出验证搬到物理约束规划域 | 闭环本体=Karpathy `2026`;进化算子=FunSearch;evaluator 前置条件=AlphaEvolve 的定义性特征。均非我们发明 |
| **评测器设计(增量,但立得住)** | 刻意做成不可作弊:碰撞硬约束(LLM 改不动)+ A* 绝对地板(刷不了分)+ 留出种子——对"reward hacking / gaming verifiers"这一活跃失败模式的具体机器人域回答 | `evaluator.py:129` PENALTY=10000 硬门;`evaluator.py:114` vs_astar 归一化;`seed0` 留出。每个配料单独看都是标配(硬碰撞检测、A* 归一化、held-out),组合起来是"深思但非新算法" |
| **纯增量/已知** | RRT-Connect(2000)、能量感知 RRT* cost(已有范式)、taut 平滑 ~0.5% | Kuffner & LaValle `ICRA 2000`;能量感知 cost 是既有增量;`agent_best.json` vs_astar 1.1067 |

**最立得住的一句 novelty 声明:**

> "首次把 LLM 智能体的 autoresearch 闭环(propose→evaluate→keep/revert + 留出种子)用于**设计时改进一个物理能耗感知的采样式 UAV 规划器**,并以一个**作弊不了的物理评测器**(碰撞硬约束 + A* 绝对最优地板 + BEMT 能耗 + 留出种子)使这套自动化**可被信任**——闭环自主复现了人类专家的算法级改动、被评测器当场抓破了作弊、并给出了同伦类根因诊断。"

注意这句话的落点是**可信性 + 域首次 + 行为证据**,不是"新算法"或"新方法"。

---

## 4. 这套链路真实实现了什么(挂 OURS)

全部挂到仓库真实产物,不空谈:

- **零 key 可跑的闭环**:`autoresearch/agent_loop.py`(MockProposer 离线 / LLMProposer 真调 Anthropic API,也能用 Claude subagent 当 proposer);代码进化目标=`smooth_path`,类级 monkeypatch,不改 `rrt_star.py` 原文件。纯 CPU、Mac 可跑、无需 AirSim。
- **作弊不了的评测器**:`autoresearch/evaluator.py`——三场景(A 直穿窄通道/B 对角上/C 对角下)BEMT 能耗之和为目标;`PENALTY=10000`(`:129`)硬门保证先 100% 到达再谈能耗,LLM 无法松弛;A* 作绝对基线归一化(`:114` `vs_astar`);`seed0` 参数支持 seed100/200 留出。
- **Agent 自研一轮的真实日志**(`experiments/agent_log.jsonl`,与 OURS 完全对上):
  - **纯调参撞天花板**:iter0 基线场景 A 成功率 0%(`min_success:0.0`),调参把 vs_astar 从 1.128 压到 **1.1067**(`agent_best.json`)后停住;iter7 一个配置漂到 **1.2279**——印证 vs-A* 地板 ~1.11 是种子噪声,提可靠只能更慢。
  - **keep/revert 真在工作**:iter1-2 `KEEP <== NEW BEST`,iter3-7 全 `REVERT / no improvement`。
  - **代码级改动被正确评测与取舍**:iter3 `rewrite_smoother: forward_greedy_shortcut` → REVERT(等价目标、不同实现,闭环没被它骗到)。
  - **作弊被当场抓破**:新采样策略成功率 100%→50%(结构性瓶颈,不是调参能救)。
  - **自主根因诊断**:agent 诊断出场景 A 是**同伦(homotopy)问题**——A* 直穿+下潜,RRT* 横向绕 20m。
- **可视化又抓到一个真 bug**:A* "最优"是靠飞到地下 z=-0.8m 实现的(地图墙底 z=0,无地面约束),**所以 vs-A* 的分母部分不物理**。这是自动化闭环+可视化把评测器自身缺陷暴露出来的正面案例。

---

## 5. 优势(相对 AI Scientist / FunSearch / AutoML / 纯人工)

- **相对 Sakana/Zochi/Agent Lab/co-scientist(论文工厂)**:我们不产论文、不靠 LLM 自评审。verifiability 靠 `evaluator.py` 的 CPU 可复现物理评测器,而非循环自评。这条是结构性优势——但要诚实说明:我们不是修好了他们的洞,是本来就站在 FunSearch 家族这侧。
- **相对 FunSearch/AlphaEvolve(可验证家族祖师爷)**:差异是**域 + grounding**。它们的靶子是数学/kernel/组合优化,disembodied 确定性评测器;我们是**物理具身 + BEMT 能耗 + A* 绝对最优地板 + 可执行碰撞硬约束**。A* 地板给了一个刷不了的绝对上限(`vs_astar` 越接近 1 越好),这是纯 held-out loss 给不了的"最优性锚"。
- **相对 Eureka/DrEureka(最近的机器人先例)**:两条硬区别——(1)它们进化 **reward**,我们进化 **planner 本身**;(2)它们的评测器是**昂贵的 RL-in-sim rollout**,我们是**便宜确定性的 planner rollout + 硬碰撞约束**。这是最锋利的对比点。
- **相对 AutoML/BO(SMAC/Optuna/BO-tuning RRT*)**:我们的 agent 不止调数字,还改结构(平滑/RRT-Connect)。而且我们**自己复现了 AutoML 的铁律**——纯调参在 A* 地板前饱和(`agent_log` iter1-7),这与 tuning-saturates 文献一致,所以我们不在调参上主张新颖,主张点在"结构改写 + 不可作弊物理目标"。
- **相对纯人工**:同一评测器下,闭环零 key 复现了人类手动 Stage1-4 的算法级结论(RRT-Connect / 能耗感知平滑),并额外给出根因诊断和自我证伪(REVERT/抓作弊)。这是"两级仿真 + 可验证 grounding"带来的可复现性,人工笔记本给不出这种可审计的 JSONL 证据链。

**最强的一句优势**:不是"我们的 planner 更好",而是"**在一个作弊不了的物理评测器下,自动化闭环的每一个 KEEP/REVERT 都可被独立复核**"——verifiability + 物理约束 + A* 绝对基线 + 两级(评测器/可视化)交叉验证。

---

## 6. Baseline(公平且对我们有利)

现在只有"评测器铁三角(碰撞硬约束 + A* 地板 + 留出种子)"这半套,缺 2025 年后领域明确要求的另一半——**证明"是闭环在干活,而不是评测器形式化在干活"**。必须补:

1. **相同调用预算下的纯 LLM 随机采样 / best-of-N**(照 `Simple Baselines are Competitive with Code Evolution 2602.16805, NeurIPS2025`)。这是**最大威胁也最公平**:该文在同预算/同 prompt/同 verifier 下,纯随机采样在 AlphaEvolve 9 问题上 2 个持平、8 个持平或超过 ShinkaEvolve,结论是"问题形式化决定天花板,进化管线次要"。不加这条,我们连"闭环胜过随机"都没证。**对我们有利**:如果 agent 的根因诊断/结构改写确实超过随机,这条 baseline 恰好把功劳从"评测器形式化"里剥出来给闭环。
2. **无 LLM 黑箱优化**:同一设计空间跑 random search / CMA-ES / Bayesian opt(AutoML 铁律,`Li & Talwalkar NAS`;`ShinkaEvolve 2509.19349` 消融显示 weighted sampling > random > hill climbing)。把"搜索空间本身好优化"从"LLM 有价值"里剥出来。
3. **闭环组件消融**(照 Eureka:Human + Sparse + 消融):
   - Human = 手动 Stage1-4(已有);
   - Sparse = 直接优化 BEMT 不带闭环;
   - 去掉 keep/revert → 看过拟合回来(我们已有 seed200 漂到 1.22 的轶事,`agent_log` iter7);
   - 去掉 held-out → 看调参过拟合;
   - 去掉 hard-constraint → 看作弊回来(成功率 100%→50% 已是现成消融证据)。
4. **能力主张用匹配预算协议**(照 `RE-Bench/METR 2411.15114, ICML2025`):匹配调用预算 + 对 A*/human 归一化 + 报随预算曲线,而非单点。

**为什么这样对我们既公平又有利**:我们本来就有 Human baseline(Stage1-4)、有 seed 漂移和抓作弊的消融轶事,补齐后就能把主张从弱的"我们找到了 RRT-Connect"升级成强的"我们的评测器+闭环能可靠区分真改进/过拟合/作弊"。**主线 baseline 不是"我的 planner 打赢别的 planner",而是"我的闭环 vs 随机 vs 黑箱优化 vs 人类专家,在匹配预算下"。**

---

## 7. 怎么迭代到可发表/可答辩

按依赖顺序,每步都可执行:

1. **先修不物理的 A* 分母**(阻塞项,必须最先做):给地面加硬约束(z≥0 障碍,或 z<0 判碰),否则任何"vs A* xx%"会被 `2602.16805` 一句话打掉——因为 A* 靠飞地下 -0.8m 取得"最优",分母不物理。修好后 A* 地板才是可引用的物理基线。这是 collision-model / free-space fidelity 问题(`Bialkowski et al. IJRR 2016`,tunneling),不是算法发现,但修好它让 benchmark 本身成为可引用贡献。
2. **补齐第6节三类对照 + 消融**,把当前轶事(seed 漂移、抓作弊)变成受控消融表。
3. **把能力主张写成 RE-Bench 式曲线**:闭环/随机/CMA-ES/人类在匹配预算下的 vs_astar & 成功率随预算曲线。
4. **明确写作归因**:Stage-4 如实说"RRT-Connect 是 2000 年已知算法,破可靠性↔能耗权衡的机制是教科书级双向贪心,能耗不降与其非渐近最优一致";平滑 ~0.5% 是同伦类内可泛化小改进。这些**恰是"闭环能复现人类算法级改动"的正面证据**,不是算法卖点。
5. **投稿定位**:workshop / 短文 / 系统贡献档,框成"可验证-evaluator 科研闭环 + benchmark 设计 + 能力/方法论验证"(rediscovery 范式是 community 认可的方法验证路径,见 MOOSE-Chem)。**别自评 novelty**(文献已证 LLM/人评 novelty 系统性虚高)。

---

## 8. 【最关键判决】方法论验证 vs 算法创新

**明确判决:目前是【强方法论/能力验证 + 弱算法增量】,不是算法创新。你的倾向是对的。**

**理由(全部可量化):**
- **算法轴上没有可主张的新颖性**:RRT-Connect = `Kuffner & LaValle 2000` 已知算法,Stage-4 是复现;能量感知 RRT* cost = 已有范式增量;唯一称得上"新"的 taut 平滑 ~0.5%(1.111→1.106)**低于自身噪声**(vs-A* ratio 噪声 ~0.1,seed 漂移 1.11→1.22)。达不到 FunSearch/Eureka 那种"显著超越领域最强 baseline"的效应量门槛。
- **两个"发现"都是正确识别已知概念,不是新发现**:(a)同伦类瓶颈是成熟概念(`Homotopy-aware RRT* Yi 2016`;`Bhattacharya homology-class`;窄通道近零测度 `2309.13119`);(b)A* 钻地是 collision-model fidelity 缺陷(`Bialkowski 2016`)。
- **连"闭环胜过随机"目前都没证**(缺第6节的 random-sampling baseline),而 `2602.16805` 明确警告"评测器形式化可能才是干活的那个"。所以现在只能主张到"方法论/能力验证"这一档。
- **但这一档完全立得住**:按 AI-Scientist/RE-Bench/MOOSE-Chem 的 community 惯例,"闭环自主复现人类算法级改动 + 专家级根因诊断 + 自我证伪"是被承认的方法论/能力贡献。`agent_log.jsonl` 是可审计证据链。

**要够到真正的算法创新,还差明确的 1-2 步:**
1. **先让评测物理可信**:加地面硬约束,让 A* 地板物理化(第7步1)。否则任何"破 A* 地板"都是形式化 artefact。
2. **再让 agent 自主发现文献中不存在的规划组件**:引导闭环去攻**同伦感知 / bridge-test / informed 采样**这一层(而不是我们目前停留的经典 RRT* 层),在物理 BEMT 目标上、跨 held-out 种子、以**明显高于噪声的效应量(需 >~10% 稳健增益,而非 0.5%)**击败"强专家 baseline"(RRT-Connect + informed/BIT* 采样 + 人工平滑),且能泛化。**在此之前别自评为算法创新。** agent 一轮里"新采样被当场抓破(100%→50%)"恰恰印证这一步的结构性难度——这是特性不是 bug,说明评测器真的在把关。

---

## 9. 下一阶段建议(把定位从"方法论验证"推向"有算法贡献")

**建议 A(必做,低风险,前置):地面硬约束 + 物理化 A* 地板。**
- 产出:一个不可作弊 **且** 物理保真的 benchmark,本身可作为可引用贡献(评测器设计)。
- 风险:低(工程量小);但**不做则整篇的 vs-A* 主张全线崩**,是所有后续步骤的前置。

**建议 B(核心,中高风险,唯一能长出算法贡献的路):引导 agent 攻同伦感知 / bridge-test / informed 采样。**
- 具体:把 `program.md` 的搜索空间从"调参 + 平滑"扩到"采样分布重写"(Gaussian sampler `Boor 1999` / Bridge test `Hsu 2003` / homotopy-aware `Yi 2016` 这一层),让 agent 有机会自主提出跨同伦类的采样组件——正是它当前诊断出但解不了的场景 A 瓶颈。
- 产出(若成功):从"复现已知算法"升级为"agent 自主发现一个针对能量感知窄通道的采样组件",效应量若 >10% 稳健且泛化,则可主张有限的算法贡献。
- 风险:高——窄通道近零测度是结构性难题(`2309.13119`),agent 很可能再次被评测器抓破;但**即使失败也是可发表的负结果**("闭环在结构性难题上如何/为何失败"),仍服务方法论主张。

**建议 C(必做,低风险,决定可信度):补齐 baseline 铁律(第6节)。**
- 具体:相同预算的随机采样/best-of-N(`2602.16805`)+ 无 LLM 黑箱优化(CMA-ES/BO,`ShinkaEvolve`)+ 闭环组件消融(Eureka 式)+ RE-Bench 匹配预算曲线。
- 产出:把"闭环 vs 评测器形式化谁在干活"这个致命问题正面回答,把主张从弱升级到强可防守。
- 风险:低(全是已有基础设施的扩展);**不做则第8节的判决停在"未证明闭环优于随机",连方法论档都会被质疑。**

**优先级**:A → C → B。A、C 决定"方法论验证"这一档能不能站稳(可发表下限);B 决定能不能够到"算法贡献"(上限,高风险)。

---

**相关真实文件路径**:
- `/Users/steven/PathPlanningwithEnergy/autoresearch/evaluator.py`(作弊不了的评测器:`:129` PENALTY 硬门、`:114` vs_astar A* 归一化、`seed0` 留出)
- `/Users/steven/PathPlanningwithEnergy/autoresearch/agent_loop.py`(keep/revert 闭环)
- `/Users/steven/PathPlanningwithEnergy/autoresearch/proposer.py`(MockProposer 离线 / LLMProposer 真调 API)
- `/Users/steven/PathPlanningwithEnergy/autoresearch/program.md`(研究策略;明确"不可改=evaluator,可改=搜索空间")
- `/Users/steven/PathPlanningwithEnergy/autoresearch/experiments/agent_log.jsonl`(真实一轮:iter0 A 失败 0%、iter1-2 调参到 vs_astar 1.1067、iter3 平滑 REVERT、iter7 漂到 1.2279)
- `/Users/steven/PathPlanningwithEnergy/autoresearch/experiments/agent_best.json`(best vs_astar=1.1067,min_success=1.0)