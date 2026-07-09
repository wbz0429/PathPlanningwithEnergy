# 毕设中期报告骨架

> 本骨架把本阶段所有工作串成可直接填充的章节 + 图表清单 + 参考文献。
> 核心基调:**诚实定位——方法论验证 + 换域应用 + LLM 自主做研究行为(消融/负结果),不是算法创新。**
> 所有数字、图、引用均可在仓库 `autoresearch/` 复现(agent_log.jsonl / milestones/ / PRIOR_ART.md / OPEN_PROBLEMS.md)。

---

## 0. 题目(候选)
- **主**:大模型智能体驱动的无人机能量感知路径规划算法自优化研究
- 副(可选):从自动调参到代码级算法改写 —— 一个作弊不了的闭环
- English: *Research on an LLM-Agent-Driven Self-Optimizing System for Energy-Aware UAV Path Planning*

## 1. 课题背景与研究意义
- **应用痛点**:无人机能量感知路径规划中,真正决定性能的是算法结构(采样/平滑/速度剖面),而算法级改进历来依赖专家、慢且不可规模化。
- **方法趋势与危机**:LLM 自动化科研(AI Scientist / FunSearch)兴起,但存在**可信度危机**——无 ground truth 时 LLM 自评审系统性不可靠(Sakana 自评审漏掉自身缺陷、错拒已录用论文)。
- **本课题切入**:把"可验证-evaluator 的自动化科研闭环"落到**有物理仿真、硬安全约束、A\*/能量绝对基线**的能量感知 UAV 规划域——让自动化研究**可被信任**。

## 2. 研究目标与内容
- **目标**:构建 LLM 智能体自优化闭环 + 一个**作弊不了的物理评测器**,让智能体自主改进(调参 + 代码级改写)能量感知 UAV 规划算法,并保证结果可信、可复现、可泛化。
- **内容**:① 作弊不了的评测器设计;② propose→evaluate→keep/revert 闭环 + 三道防御门;③ 分阶段能力扩展(算法族→速度维度→泛化→开放问题);④ 诚实的能力边界刻画。

## 3. 国内外研究现状 / 相关工作(见 `PRIOR_ART.md`, `OPEN_PROBLEMS.md`)
- **LLM 驱动算法自动设计**:FunSearch(Nature 2023)、AlphaEvolve(2025)、Eureka(ICLR2024,进化 RL reward)、ReEvo(组合优化启发式)、TrajEvo(轨迹预测)——**范式成熟,但无一针对连续空间运动/路径规划器组件**。→ 差异化即在此。
- **能量感知 UAV 规划(本课题"发现"其实都是已知)**:U 形功率曲线与最优巡航 v\*(**Di Franco & Buttazzo 2015**);加速代价下最优速度低于稳态 v\*(同上;NASA 2024);爬/降能耗不对称(**Liu 2017**:+9.8%爬/−8.5%降);time-energy-optimal 速度剖面(Bianchi 2024);最小曲率 vs 最短路径(TUMFTM;Xue 2023)。
- **开放问题**:非欧/能量代价下 informed 采样不可采纳/过保守(**Kyaw & Kelly 2026**);部分观测能量感知规划仍不成熟、忽略机动瞬态能耗(**EAAE 2026**);窄通道近零测度(Orthey/Kavraki 2024,6 场景仅解 1)。

## 4. 已完成工作:系统与方法
- **闭环本体**(karpathy/autoresearch 范式):固定评测器 + `program.md` 研究策略 + propose→evaluate→keep/revert + JSONL 账本。驱动器 = Claude Code skill `autoresearch-step` × `/loop`(零 API key,Claude 本身即研究智能体)。
- **作弊不了的评测器** `physics_eval.py`:接地地图(禁钻地)+ **速度剖面能量(v\*,转弯限速,加速计价)** + **独立碰撞复核**;A\* / 能量加权 A\* 绝对基线;留出种子。
- **三层模型**(`LAYERS.md`):🧊冻结层(BEMT 物理/kinodynamic/度量/场景/seed)· 🔧可优化层(参数 + 代码组件:采样器/平滑器/速度剖面)· 🔬方法论层(闭环/文献在环/知识库)。**cheat-proof 原则:每个可动旋钮必须被冻结尺子正确计价。**
- **闭环机制**:EXPLOIT/EXPLORE 分阶段(硬周期外部检索 + 换算法族)、文献在环(带来源/置信度)、知识库 `KNOWLEDGE.md`、反空转升级、代码沙箱(AST 白名单 + 契约测试 + 超时)。
- **三道防御门**:① 留出验证(seed + 场景);② 横向对比(生死线,vs 随机搜索/业界算法);③ 鲁棒性门(扰动物理看增益是否稳健)。
- **全程可追溯**:`agent_log.jsonl`(每轮假设/分数/取舍)+ `candidates/`(每个代码候选存档)+ git(每 KEEP 一提交)+ `milestones/`(每里程碑图+快照)。

## 5. 关键结果与分析(数字均来自 `agent_log.jsonl` / `milestones/`)
| 阶段 | 目标 | 结果 | 验证 |
|---|---|---|---|
| **M1** 算法研究 | 已知地图能耗 | 15000→**2046**(RRT-Connect + CHOMP 平滑器,代码级改写) | 三门全过;胜随机搜索 2805、胜 A\* 2618 |
| **S3a/M3** 解冻速度 | 速度剖面能量 | 2821.7→**2556.5**(速度 DP,自主推导 sub-v\* 巡航) | 留出验证;鲁棒性**限定**(速度增益在 v\*×1.5 翻负,诚实标注) |
| **S3b/M4** 泛化 | 6 场景(含程序化生成) | 6066→**6030** | **双重验证**:留出场景 gen400-402 全 100% + 留出 seed |
| **S4a** 攻开放问题 | 非欧能量 informed 采样收敛 | 1419→**810**(informed focusing)| **自消融证明各向异性不 load-bearing**(对称≈各向异性)→ **诚实负结果** |
- **元发现(跨 5 实例)**:surrogate 引导的算法搜索里,**悲观/粗糙估计器一贯打赢精确估计器**(精确代理在最优附近平坦、信号弱、误差直流进采纳)。可写为方法论级观察(待与文献核实是否已知)。
- **图**:各阶段 convergence/trajectory/energy/waterfall/compare 图 + 飞行动画 GIF + S4a 结果图,见第 9 节。

## 6. 创新点与诚实定位(**最关键、决定不翻车**)
- **诚实判决**:**方法论/能力验证 + 换域应用**,**不是算法或物理创新**(RRT-Connect=2000、CHOMP=2009、v\*/爬降不对称/TOPP 皆已知)。
- **能立住的一句 novelty**:*"LLM-evolution + 作弊不了物理评测器"这套方法论已存在(Eureka/AlphaEvolve/ReEvo),但无人将其用于**连续空间、能量感知 UAV 规划器组件(采样器/平滑器/速度剖面)的代码级设计**——这是差异化。*
- **额外的方法论价值(答辩亮点)**:闭环**自主做消融归因、给出诚实负结果(S4a)、自主重新发现已知物理规律(Di Franco/Liu)并被作弊不了的评测器验证**——这是"LLM 真在做科学"而非"自评自夸"的硬证据。
- **三层控制结构(思想深度)**:人(设 frame/价值判断)→ Claude(架构/边界)→ loop(frame 内优化)。诚实指出**不存在全自主研究员**;autonomy 是分层受限的——很多同类工作在此不诚实。

## 7. 存在问题与不足(诚实边界)
- **已知地图 + 离线规划**:非部分观测/滚动规划,比真实导航简单(A\* 锚点要求已知地图)。
- **工程缝合为主**:被优化的算法组件多为已知件的组合/调优。
- **速度增益模型依赖**:S3a 速度增益在 v\*×1.5 翻负(鲁棒性门标注)。
- **S4a 未兑现开放问题的 novel 部分**:各向异性经自消融证明不 load-bearing;baseline 为均匀(未显式对标 Euclidean-informed SOTA)。
- **静态环境假设**、无动态障碍、无 sim-to-real 验证。

## 8. 下一阶段计划(见 `ROADMAP.md`)
1. **部分观测 / 滚动规划(最高价值,realism 跳跃)**:复用 `drone_sim` 的 `RecedingHorizonPlanner`,让自优化落到真实导航;重设最优性 baseline(相对 oracle 的 regret)。
2. **S3c 更丰富场景**(地形/风,真 3D 能量权衡)。
3. **S4 研究员行为**(诊断实验模式 / 多智能体协作)。
4. 补 **Euclidean-informed baseline** 把 S4a 负结果坐死。

## 9. 图表清单(现成,`autoresearch/experiments/`)
- **闭环架构图**(三方:研究智能体 ↔ evaluate 引擎 ↔ 知识库)【需补画,LAYERS.md 有 ASCII】
- **收敛曲线** `milestones/ms*/fig_ms1_convergence.png`(15000→2046,KEEP/REVERT + 节点标注)
- **翻墙轨迹三视图** `fig_ms1_overwall.png` / `fig_ms1_trajectory_all.png`
- **优化前后 + 留出验证** `fig_ms1_energy.png`;**横向对比(生死线)** `fig_ms*_compare.png`
- **KEEP 贡献瀑布** `fig_ms1_waterfall.png`
- **飞行动画 GIF** `flight_A/B/C.gif`(翻墙飞越,无需 AirSim)
- **S4a 开放问题结果** `milestones/ms5_s4a/fig_s4a_result.png`(含自消融诚实负结果)

## 10. 参考文献(核心,详见 PRIOR_ART.md / OPEN_PROBLEMS.md 带链接)
FunSearch (Nature 2023);AlphaEvolve (DeepMind 2025);Eureka (ICLR 2024, arXiv 2310.12931);ReEvo (NeurIPS 2024);AI Scientist (Sakana, arXiv 2504.08066);Di Franco & Buttazzo 2015(功率-速度模型);Liu 2017(爬降能耗);Kyaw & Kelly 2026(非欧 informed 采样, arXiv 2606.02879);EAAE 2026(部分观测能量规划, arXiv 2603.15604);Gammell et al. 2014/2021(Informed RRT\*/综述);Kuffner & LaValle 2000(RRT-Connect);Ratliff et al. 2009(CHOMP);Orthey/Chamzas/Kavraki 2024(采样规划综述);"Simple Baselines are Competitive with Code Evolution"(2602.16805,生死线依据)。
