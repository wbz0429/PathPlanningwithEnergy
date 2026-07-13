# 详细工作记录(给导师·完整铺开版)

**硬指标**:122 次 git 提交 · 13,074 行 Python · 6,275 行文档 · 23 个候选代码存档 · 28 个结果 JSON · 46 张图/视频 ·
5 次 deep-research(共 400+ 检索 agent,含两份千行报告)· 两条自动科研 loop(规划器 90+ 迭代 / 能耗 10 迭代)· 一条 radar 感知线。

> 本文件是"全部工作"的逐条铺开;`ADVISOR_SUMMARY.md` 是其一页纸提炼,`NARRATIVE_WHY_AI4S.md` 是叙事线。

---

# 阶段 0 —— LLM 自动科研进化 RRT\* 规划器(4 个里程碑,90+ 迭代)

## 0.0 基础设施(从零搭)
- **闭环主体**:`agent_loop.py`(读结果→提假设→改动→评测→keep/revert→记账)、`proposer.py`(LLM 提议)、
  `candidate.py`(候选装载)、`sandbox.py`(代码安全沙箱)、`search.py`、`actions.py`(结构化 JSON Action)。
- **作弊不了的评测器** `evaluator.py`:碰撞硬约束(PENALTY=10000,100% 到达才计能耗)+ BEMT 速度剖面能量 +
  A\* 归一化地板 + 留出种子(seed 0-2 训练 / seed 100/400+ 验证)。
- **三层框架** `LAYERS.md`:🧊Layer-0 冻结(17 BEMT 参数 / 动力学极限 / 裁判 / 场景)/🔧Layer-1 可动 /🔬Layer-2 方法学。
- **前置文献深查** `PRIOR_ART.md`(1363 行)+ `OPEN_PROBLEMS.md`(61 行,2023-26 开放问题)+ `POSITIONING.md`(157 行)。

## 0.1 Milestone 1(Ep3,iter21):15000 → 2078.4(留出 2079,零过拟合)
**KEEP 7 条**(每条带能耗降幅):RRT-Connect −79% · 悲观 proxy 平滑器 −15.6% · split-bend −5.7% ·
double-bend −0.1% · step 3.0 −8.2% · step 4.5 −1.5% · 翻墙宏路线探针 −9.2% · 细网格 −0.1% · 骨架 DP −0.2%。
**REVERT/no-op 9 条**(每条关闭一条搜索方向+留判据):
- step 6.0 反弹 +2.3%(峰值在 4.5m);weight_\*/search_radius 在 RRT-Connect 分支=死旋钮;goal_sample_rate 被硬编码取代=死旋钮;
- 全局 DP 骨架仅比贪心捷径好 ~4J(B 场景);平滑器 sweep 6 趟收敛,12 趟 no-op。
**三条硬结论**:①度量 phantom-R 特性→"少顶点长腿"优于圆弧离散;②BEMT 下降近免费→翻墙拓扑占优但须 polish 后再比;
③悲观 proxy 局部搜索优于精确 proxy。

## 0.2 Milestone 2(Ep6,iter38):2078 → 2046.4(留出 2043.8,比训练还低)
**KEEP 4 条**:binormal 顶点搬移 −2.9(C) · basin hopping −7.1(B) ·
**CHOMP 式联合梯度精修 −17.8(A/B/C 全降,EXPLORE 换算法族的直接产出)** · 网格标定 3D 代理 −3.4。
**REVERT 11 条 + 1 EXHAUSTED 判决**,判据:RNG 流敏感(basin-hop 不可扩量)、悲观代理优于精确、驼峰=v\* 保全交易、
free-dive 被 Row4 几何锁死。
**★ 横向对比(关键)**:默认RRT\* 11700 > RRT-Connect 3191 > 能量A\* 2618 > **随机搜索 2805 > LLM-loop 2046**——
**loop 比随机搜索低 27%**(随机搜索写不出 CHOMP/probe/DP 平滑器)。**鲁棒性门:v\* 扰动 0.6×–1.5× 下优势 +26%~+33% 全稳**。

## 0.3 Milestone 3–4(S2/S3b):采样器注入 + 骨架
- 开放采样器注入点 `sample(ctx)`(可逃 z-clamp,让 RRT-Connect 原生翻墙);**4 个候选全 REVERT**(z 噪声污染 B/C)。
- MS4(S3b):6029.7 训练,留出 gen400-402 100% @3337.5。

## 0.4 S4a 专项 —— 攻文献点名的开放问题:非欧能量代价下的 informed sampling(iter57–63)
被 Gammell 2021 / Kyaw&Kelly 2026 点名为开放:informed sampling 的欧氏启发式在非欧代价下不可采纳。我们做各向异性 informed 采样器:
- iter57 aniso-informed v1 **KEEP**(score 810.3);iter58 各向异性消融(关键证据)+ ratio-tuning **REVERT**;
- iter59 headroom+schedule scan **REVERT**;iter60 sub-level-set EXPLORE;iter61 goal-cone **REVERT**;
- iter62 low-dispersion **REVERT**;iter63 annealed goal-bias **REVERT**。
→ **静态采样器空间 6 机制全部 REVERT,S4a 耗尽**;自消融显示各向异性**不承重**(诚实负结果)。

## 0.5 创新期(iter65–70)—— 两个"元发现"都被自己证伪
- iter65-68 **悲观 surrogate 元发现**:局部精修用悲观代理更好→看似新规律;
  **iter67 在程序化场景上证伪:留出反转,不泛化**;iter68 归档为 KNOWN + 非鲁棒。
- iter69 **非欧 informed SOTA 门**:用 Kyaw&Kelly 矩阵值度量构 informed set → **比均匀采样更差**
  (对称度量丢了爬降不对称)。S4a 正式关闭。iter70 里程碑:**Layer-1 创新空间耗尽**。

## 0.6 阶段 0 诚实结论
Layer-1 可动旋钮穷尽;真正杠杆在冻结层(采样器 z 解锁/kinodynamic,动=作弊)。
**定位:强方法论验证 + 弱算法增量,不是新算法**(RRT-Connect=2000 年,平滑增益噪声级)。
**但 loop 在此域比随机低 27%——证明"代码级研究能力"有价值**(与能耗域相反,见阶段2)。

---

# 阶段 1 —— 雷达多目标跟踪(域无关性验证,8 提交)

- **搭建**:OSPA/MOTA 冻结评测器 + DBSCAN→Kalman→关联 pipeline(`ospa.py/mota.py/pipeline.py`);合成烟测全过。
- **真数据**:RadarScenes 适配器(streaming,与 gen_scene 同契约)+ MTI 杂波过滤,真 API 校验;
- **手调**:sequence_1 上 MOTA −24→−1.6→**+0.49**,OSPA 1.97→0.93;
- **loop**:代码级 cluster/associate 沙箱注入,**held-out MOTA −19.4→−0.58,OSPA −43.5%(泛化)**;
- **两条防线**:生死线 PASS(joint>random),**必要性 MARGINAL(+4.3% vs 贪心)**;
- **放弃**:导师要求连 UAV,RadarScenes 是车载→转回 M100。代码全保留为方法学**域无关性**证据。

---

# 阶段 2 —— LLM 自动科研进化能耗模型(真机 DJI M100,10 迭代)

**评测器** `m100_eval.py`(按航班 train/test,能量 ARE,真功率 P=V·|I|,冻结)。每轮账本(`agent_log.jsonl`):
| iter | 改动 | 搜索 ARE | 留出 ARE | 决定 + 判据 |
|---|---|---|---|---|
| base | 稳态 BEMT (1,v,v²) | 7.04% | 6.88% | 基线 |
| 1 | 动量核 T^1.5+T²/V+v³+爬降不对称 | 4.56% | 4.25% | KEEP |
| 2 | Glauert 前飞桥 | 4.93% | 4.91% | **REVERT**(单桥去掉主导项,更差) |
| 3 | 推力标定型面 T·v² | 4.54% | 4.24% | KEEP(勉强,0.24% 相对) |
| 4 | 轴向诱导 sqrt | 4.54% | 4.23% | **REVERT**(增益 0.096pp<1e-4 门) |
| 5 | 风→空速 v_air | 4.54% | 4.38% | **REVERT**(留出反差,quadrature 风无益) |
| 6 | **物理核 + 裸线性 payload** | **2.01%** | **1.93%** | **KEEP(决定性)** |
| 7 | payload 变体 | 2.01% | 1.93% | REVERT(与 iter6 相同) |
| 8 | 纯线性库 LIB | 1.99% | 1.96% | 诊断:纯线性 1.909 略优于物理+payload |
| 9 | 物理形叠线性库 | 2.00% | 1.96% | **REVERT**(共线过拟合,留出退) |
| 10 | capstone 对标 | — | **1.86%** | 里程碑 |

**核心 finding(能力边界)**:①纯物理 4.24% → 加一个裸 payload → 1.93%,**物理非线性形不是价值来源**;
②loop 1.86% **与随机搜索打平**(1.86%),仅略胜贪心(2.09%)——**能耗域:LLM 物理推理不转化为更低 ARE**;
③物理叠线性有害(1.909→1.962%)。**注:这与阶段0规划器域相反(那里 loop 胜随机 27%)——域相关的诚实对照。**

---

# 阶段 3 —— 瞬态能耗证伪
机动加速的瞬态能耗惩罚**不被真数据支持**:横向加速项≈null,瞬态特征仅贡献 ΔR²~1.3%。
→ 不在不存在的效应上建模(干净负结果,记忆 `m100-transient-falsified`)。

---

# 阶段 4 —— 真机代价接入规划 + 规划层研究 loop(H1–H5)

## 4.1 主实验(真规划器 energy_astar,破了 3 个 bug 才分叉)
- **bug1 v_z 符号**:NED 爬升 dz<0,但模型正 v_z=爬升→漏负号→爬升被当下降低估 152W(探针 vz=+3→639W 抓出);
- **bug2 velocity=2 太慢**:爬升罚被巡航时间稀释(×1.14);真实巡航 v=8 时爬升×1.97;
- **bug3 Blocks 原障碍污染绕行**→ `clean_wall_map` 隔离。
- **主结果**:墙场景距离/BEMT 翻墙、M100 绕行,省 13.3/9.4/5.4%;速度 4–12 m/s 全程分叉省 4.0→21.9%。
- **三重机制验证**:留出爬升 premium 真+114W vs 预测+108W(<5%)+ **因果消融**(挖爬升项→决策退化回翻墙)+ BEMT 质量公平对照。
- **四项审计落盘** `robustness_suite.json`(速度/起终点/消融/质量)。

## 4.2 规划层 loop 5 假设(每个预注册→实验→裁决,`pl_iter*.json`)
| H | 假设 | 实验规模 | 裁决 | 对标 |
|---|---|---|---|---|
| H1 | 配送顺序爬降不对称致翻转 | 210 配置全扫+规划器 3 速度 | SUPPORTED(open 3–9%/closed 12–20%) | **TAKEN**:Michel 2024 arXiv 2410.17585 |
| H2 | 风扰鲁棒 | 12 飞行(2 路×6 风)+侵入量化 | **REFUTED**(安全门:2.0m 裕度被风推入墙 0.96m→规则≥3.5m) | WA-LPA\*/CFD-Theta\* |
| H3 | 电池可达集扩大 | 154 次 A\*(77 目标) | SUPPORTED(峰值+31%) | **大部分 TAKEN**:Nguyen 2017 |
| H4 | 借重力下降塑形 | P 曲面 11×17 + 156 剖面 | **REFUTED**(v_z\*≡0,省能只来自避爬升) | — |
| H5 | 省能分布 | 20 随机城市块 | SUPPORTED(压线,70% 场景 0%) | — |

## 4.3 载荷专项(负结果)
0/250/500g × 墙宽 × 加不加物理 mgh，**crossover 全程不动**;机制=M100 载荷项裸线性、无载荷×爬升耦合;
500g 爬 13m mgh≈127J 仅占 ~3%。→ 需 >650g 载荷比+数据(不存在)。对标 EcoFlight/物流UAV，**角度已做+我方负**。

---

# 阶段 5 —— 动力学闭环 + 仿真环境

- **RotorPy 动力学闭环**：M100 尺度 2.65kg/650mm/悬停 550rad/s + SE3 控制器 100Hz；规划路径→MinSnap 平滑→真飞→积分能量。
  单墙实飞省 6.1%(折线 10.7%)、走廊 4.4%；**工程发现:规划裕度须≥2.0m(0.6m 平滑切角蹭墙 0.00m)**。
- **城市走廊旗舰**：逐障碍混合决策(翻矮楼A+绕高楼B);**飞行视频 ×2**(MP4，双机同屏+实时功率)。
- **仿真环境 deep-research**(106 agents/24 源)`SIM_ENV_RESEARCH.md`(1997 行):RotorPy=Mac 能耗研究 tier-1(独立验证);
  PX4+Gazebo Harmonic(2026-02 起 Apple Silicon 原生)=感知升级路径;AirSim/Colosseum 已死、Isaac 系 Mac 不可用。
- **PX4 SITL 实装**:克隆+子模块 2.6G、工具链、Gazebo Harmonic 8.14、**px4_sitl 构建成功**；
  **踩 4 个坑全记录**(①上游 macos.sh trust-gate bug 静默失败 ②xquartz sudo ③gz 子模块克隆失败 ④OpenCV5 不兼容光流插件→本地补丁禁用)。起飞验证暂停待恢复。

---

# 阶段 6 —— 查新 + AI4S 定位（5 次 deep-research）

| # | 主题 | 规模 | 结论 |
|---|---|---|---|
| 1 | 方向调研(能量感知规划有无空位) | — | 域饱和,无开放维度 |
| 2 | 仿真环境全景 | 106 agents | RotorPy tier-1；PX4 升级路径 |
| 3 | 规划效应查新(H1/H3) | 101 agents | **顺序=Michel2024、可达集=Nguyen2017,全 TAKEN** |
| 4 | 载荷角度查新(自查) | WebSearch | EcoFlight/物流UAV 已做+我方负 |
| 5 | **AI4S 方向裁决** | 102 agents | **站得住**(见下) |

**AI4S 裁决**(`AI4S_POSITIONING.md`)：定位为"可信/证伪优先 AI4S 方法学**实例化+实证消融**在 UAV 能耗+规划"，
**不**声称发明 AI4S/证伪(AIGS 2411.11910)/查新门(AI Scientist)。两条腿=域应用(旗舰框架未占)+ 方法学 bundle
对着实测失效模式(reward hacking 随能力越强越严重、GPT-5 作弊 54%)。**设防=safeguard 实证承重**——而**阶段 0–4 的全部失败就是现成承重实验**(查新门抓 Michel/Nguyen、留出抓 iter8/9 过拟合、报负结果抓 loop=随机)。

---

# 全局诚实对照表（一眼看清能力边界）

| 维度 | 规划器域(阶段0) | 能耗域(阶段2) |
|---|---|---|
| loop vs 随机搜索 | **loop 胜 27%**(写得出 CHOMP/DP) | **打平**(结构搜索无益) |
| loop vs 贪心 | 胜 | 略胜(+7%) |
| 新算法? | 否(RRT-Connect 复现) | 否(线性回归追平) |
| 价值所在 | 代码级研究能力 + 可信评测 | 可信评测 + 诚实边界 |

→ **这个"域相关"对照本身是核心 finding**:LLM 自动科研的优化价值**取决于任务是否需要写出普通搜索到不了的代码结构**。

---
*索引:阶段0 `autoresearch/`(agent_loop/evaluator/PRIOR_ART/LAYERS/POSITIONING/milestones/ms*);
阶段1 `autoresearch/radar/`;阶段2-6 `autoresearch/energy_m100/`(全部脚本 + `*_result.json` + `midterm_report_v2.html`
+ `SIM_ENV_RESEARCH.md`/`AI4S_RESEARCH_FULL.md` 两份完整调研 + `px4_integration/`)。*
