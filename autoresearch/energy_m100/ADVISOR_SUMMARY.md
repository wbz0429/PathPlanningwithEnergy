# 完整工作总结:LLM 自动化科研 × 无人机能耗建模 × 能量感知路径规划(给导师)

日期 2026-07-13 · 119 次 git 提交 · 6 个研究阶段 · 两条自动科研 loop + 一条感知线 · 所有数字可由盘上 JSON/log 复现

> **贯穿主线**:一个"防作弊评测器 + LLM 闭环改代码 + keep/revert + 诚实报负结果"的自动化科研框架,
> 先后作用于 ①RRT\* 规划器算法 ②雷达多目标跟踪 ③无人机能耗模型 ④能量感知规划,并做动力学闭环验证。

---

## 阶段总览(时间线)

| 阶段 | 内容 | 迭代/提交 | 核心结果 | 诚实定位 |
|---|---|---|---|---|
| **0** | LLM autoresearch 进化 **RRT\* 规划器**(采样器/平滑器/速度剖面) | iter0–70 | score 15000→2078;S4a 非欧informed采样 | 方法学验证 + 弱算法增量,Layer-1 耗尽 |
| **1** | 同框架迁移到**雷达多目标跟踪**(RadarScenes 真数据) | 8 提交 | held-out MOTA −19.4→−0.58,OSPA −43.5% | 域无关性证据;必要性弱;因非UAV放弃 |
| **2** | LLM autoresearch 进化**能耗模型**(真机 DJI M100) | iter1–10 | 留出能量 ARE **6.88%→1.86%** | 正结果;但结构搜索=随机(能力边界) |
| **3** | **瞬态能耗证伪**(机动加速惩罚) | — | 不支持(ΔR²~1.3%) | 干净负结果,防了错误建模 |
| **4** | 真机代价接入规划 + **规划层研究 loop**(H1–H5) | 6 提交 | 障碍省 4–22%;5 假设 3证2否 | 效应真实但都是 prior art |
| **5** | **动力学闭环**(RotorPy)+ 仿真环境调研 + PX4 SITL | 多提交 | 实飞省 4–6%;PX4 构建成功 | 结论过动力学仍成立 |
| **6** | **查新**(3 次 deep-research)+ AI4S 方向 | 进行中 | 效应都被 prior art 占;AI4S 查证中 | 唯一站得住=方法学+真机锚定 |

---

## 阶段 0:LLM 自动化科研进化 RRT\* 规划器(原始课题,最大工作量)

### 做了什么
- 搭了**零 API key 可跑的 LLM 智能体闭环**(读结果→提假设→改代码→评测→keep/revert→记录):
  `agent_loop.py / proposer.py / candidate.py / evaluator.py / sandbox.py / search.py`;
- **结构上作弊不了的物理评测器** `evaluator.py`:碰撞硬约束(PENALTY=10000 硬门,100% 到达才计能耗)+
  BEMT 速度剖面能量 + A\* 归一化地板 + 留出种子;
- **三层框架**(`LAYERS.md`):🧊Layer-0 冻结(17 个 BEMT 参数、动力学极限、裁判机制、场景)/
  🔧Layer-1 可优化(step_size/采样器/平滑器代码)/🔬Layer-2 方法学;
- 进化对象:`use_rrt_connect`、平滑器 `smooth_path` 代码、采样器代码、速度剖面。

### 关键结果(真实迭代账本)
- **Milestone 1(iter21)**:score 15000 → **2078**(留出 seed100 → 2079,三场景 100% 到达,增益全泛化)。
  配置:RRT-Connect + 悲观 proxy 平滑器 v13(多趟视线捷径 + 顶点删除/搬移/split-bend/double-bend + 翻墙宏路线探针)。
  **21 轮账本:7 KEEP(RRT-Connect −79%、proxy 平滑器 −15.6% 等)/ 9 REVERT**(每个关闭一条搜索方向)。
- **Milestone 4(S3b)**:6029.7 训练,留出 gen400-402 100% @3337.5。
- **S4a(iter57–63)**:攻击被文献点名的开放问题——**非欧能量代价下的 informed sampling**;
  做了各向异性 informed 采样器 v1(score 810.3 KEEP),然后 **6 种机制全部 REVERT**(ratio-tuning/goal-cone/
  low-dispersion/sub-level-set/annealed-goal-bias),S4a 静态采样器空间**耗尽**。
- **创新期(iter65–70)**:①悲观 surrogate 元发现——**证伪**(局部精修的悲观增益不泛化,留出反转);
  ②非欧 informed SOTA 门——用 Kyaw&Kelly 矩阵值度量的 informed set **比均匀采样更差**(对称度量丢了爬降不对称);
  **Layer-1 创新空间耗尽**。

### 尝试的创新 → 对标(`PRIOR_ART.md` 1363 行深查)→ 结果
| 我们试的 | 对标已存在 | 结果 |
|---|---|---|
| LLM 闭环改规划器代码 | FunSearch/Eureka/AlphaEvolve/ReEvo/EoH/TrajEvo | 范式非首创;**但没人做连续空间运动/路径 PLANNER 组件**(差异化缝) |
| 自动发现 RRT-Connect | RRT-Connect(Kuffner & LaValle **2000**) | 复现了人类已知算法,非新 |
| 能量最优巡航速度 v\* | Di Franco & Buttazzo 2015(U 形功率曲线) | prior art |
| 有限段最优速度 < v\* | Di Franco & Buttazzo;NASA lift+cruise 2024 | prior art |
| 爬降能量不对称 | Liu 2017(+9.8% 爬 / −8.5% 降) | prior art |
| 非欧 informed sampling | Kyaw & Kelly 2026(矩阵值 Loewner 启发式) | prior art 且我们试了更差 |

### 负结果/参数不理想(诚实)
- **平滑增益仅 ~0.5%,低于种子噪声**;vs-A\* 地板 ~1.11 是噪声,纯参数调优撞天花板;
- **Layer-1 耗尽**:突破需动 Layer-0(采样器 z 解锁/kinodynamic 松绑),但那是冻结层(动了=作弊);
- 诚实定位(`POSITIONING.md`):**强方法论验证 + 弱算法增量,不是新算法**——RRT-Connect 是 2000 年的,
  平滑增益噪声级。可发表档 = workshop/系统贡献,不是"新规划算法"。

---

## 阶段 1:雷达多目标跟踪(域无关性验证,已放弃但有正结果)

### 做了什么
- 同一套 cheat-proof 范式迁移到感知/跟踪:**OSPA/MOTA 冻结评测器** + DBSCAN→Kalman→关联 pipeline;
- **RadarScenes 真数据**适配器(streaming + MTI 杂波过滤)+ 代码级组件进化(cluster/associate 沙箱注入)。

### 关键结果
- 手调:sequence_1 上 MOTA −24→**+0.49**,OSPA 1.97→0.93;
- **autoresearch loop:held-out MOTA −19.4→−0.58,OSPA −43.5%(泛化)**——这是**真数据上的正结果**。

### 诚实 & 为何放弃
- **生死线 PASS**(joint>random),但**必要性 MARGINAL**(联合调参仅 +4.3% vs 贪心)——同 radar 域"结构搜索必要性弱"结论;
- **放弃原因**:导师要求必须连 UAV 场景,radar 是车载(RadarScenes)——故转回 M100 能耗。代码保留为方法学域无关证据。

---

## 阶段 2:LLM 自动化科研进化能耗模型(真机 DJI M100)

### 做了什么
- 冻结评测器 `m100_eval.py`(按航班 train/test、能量 ARE、真功率 P=V·|I|,绝不可编辑);
- LLM 每轮改一次 `featurize` 源码,keep/revert;iter1–10 全落盘(`candidates/` + `agent_log.jsonl` + git)。

### 迭代真实轨迹
| iter | 改动 | 留出 ARE | 决定 |
|---|---|---|---|
| baseline | 稳态 BEMT (1,v,v²) | 6.88% | — |
| 1 | 动量理论核(T^1.5+T²/V+v³+爬降不对称) | 4.25% | KEEP |
| 2 | Glauert 前飞桥 | 4.91% | REVERT |
| 3 | 推力标定型面 T·v² | 4.24% | KEEP(勉强) |
| 5 | 风→空速 | 4.38% | REVERT(反差) |
| 6 | 物理核 + 裸线性 payload | **1.93%** | KEEP(决定性) |
| 8 | 纯线性库 LIB | 1.91% | 诊断:线性≈物理+payload |
| 10 | capstone | **1.86%** | 里程碑 |

### 尝试的创新 → 对标 → 结果
| 我们试的 | 对标 | 结果 |
|---|---|---|
| M100 能耗建模 | **Dai 已做 M100**;Tseng(该数据集既定最优=多项式回归) | 模型非新 |
| LLM 进化物理函数形式 | FunSearch/Eureka | 框架非新,**且没跑赢线性回归** |
| 动量理论物理核 | 经典 BEMT | 纯物理 4.24%,加一个 payload→1.93%:**物理形不是价值来源** |

### 负结果(核心 finding)
- **结构搜索必要性弱**:loop 1.86% **与随机搜索打平**,只略胜贪心(2.09%)——"LLM 物理推理不转化为更低 ARE";
- per-sample R² 仅 0.44(瞬时功率大半是风/电压 sag,运动学解释 <40%);
- 物理形叠加到线性库上**有害**(1.909→1.962%,共线过拟合)。

---

## 阶段 3:瞬态能耗证伪
机动加速的瞬态能耗惩罚 **不被真数据支持**(横向加速 ≈null,瞬态 ΔR²~1.3%)——干净负结果,
避免了在不存在的效应上建模。(记忆 `m100-transient-falsified`)

---

## 阶段 4:真机代价接入规划 + 规划层研究 loop

### 主结果(真规划器 energy_astar)
| 实验 | 结果 |
|---|---|
| 墙场景三代价 | 距离/BEMT 翻墙,M100 绕行,省 **13.3/9.4/5.4%** |
| 速度鲁棒 4–12 m/s | 全程分叉,省 **4.0→21.9%** |
| 机制验证 | 留出爬升 premium 真 +114W vs 预测 +108W(**<5%**);**因果消融**挖爬升项→决策退化回翻墙 |
| 交叉验证 | 省能是"相对可信模型"(BEMT 尺下反而贵——BEMT 爬升盲) |

### 规划 loop 5 假设 → 裁决 → 查新对标
| 假设 | 裁决 | 对标论文 | 状态 |
|---|---|---|---|
| H1 配送顺序(爬降不对称致翻转) | SUPPORTED | **Michel 2024 arXiv 2410.17585**(几乎同题,up-to-95%,gap 14.9%) | **TAKEN** |
| H3 电池可达集扩大 +31% | SUPPORTED | **Nguyen & Au AAMAS 2017**(可达集框架) | **大部分 TAKEN** |
| H2 风扰鲁棒 | REFUTED | WA-LPA\*/CFD-Theta\* | 符号不翻但 2.0m 裕度被风推入墙→需 ≥3.5m |
| H4 借重力下降塑形 | REFUTED | — | v_z\*≡0,省能只来自避爬升(干净负结果) |
| H5 省能分布(20 随机场景) | SUPPORTED(压线) | — | **70% 场景省 0%**、25% ≥3%,中位 0% |

### 载荷负结果(重点)
载荷 0/250/500g 全扫 **crossover 不动**,加物理 mgh 也不动。原因:M100 载荷项裸线性、**无载荷×爬升耦合**;
500g 爬 13m 的 mgh≈127J 仅占能耗 ~3%。→ "重载绕行"在 ≤500g 包络内**不成立**,需 >650g 载荷比+数据(不存在)。
对标 EcoFlight 2025 / 物流UAV规划 Drones 2025——**角度已被做 + 我方结果为负**,不能当创新点。

---

## 阶段 5:动力学闭环 + 仿真环境

- **RotorPy 动力学闭环**(M100 尺度 2.65kg,SE3 控制器,100Hz):规划路径真飞→积分能量。
  单墙实飞省 **6.1%**(折线 10.7%),走廊 **4.4%**;工程发现:规划裕度须 ≥2.0m(0.6m 平滑切角蹭墙)。
- **城市走廊旗舰**:逐障碍混合决策(翻矮楼 A + 绕高楼 B),含飞行视频。
- **仿真环境 deep-research**(106 agents):RotorPy=Mac 上能耗研究 tier-1(独立验证我们的选择);
  **PX4 SITL+Gazebo Harmonic**(2026-02 起 Apple Silicon 原生)= 感知闭环升级路径。
- **PX4 工具链已装通、px4_sitl 构建成功**(踩 4 个坑含 1 个上游 trust-gate bug,全记录);起飞验证暂停待恢复。

---

## 阶段 6:查新(3 次 deep-research)+ AI4S 方向

- 查新裁决:规划层效应(顺序翻转/可达集/绕行省能/载荷)**全部 prior art**。查新门兑现价值——
  拦住了把"复现 Michel 2024"当创新去写。
- **AI4S 方向 novelty check 完成(102 agents)——裁决:站得住**(详见 `AI4S_POSITIONING.md`):
  - 定位为"把可信/证伪优先 AI4S 方法学**实例化+实证消融**在 UAV 能耗+规划",不声称发明 AI4S/证伪/查新门(AIGS 2024/AI Scientist 已有);
  - 两条腿:①域应用(旗舰框架没做过 UAV 能耗规划)②方法学 bundle 对着实测失效模式(reward hacking 随能力变严重 54%);
  - **关键设防=让 safeguard 实证承重**:证明去掉某 safeguard→loop overclaim。**本会话已产出全部承重证据**
    (查新门抓 H1/H3 prior art、留出抓 iter8 过拟合、负结果抓 loop=随机)。

---

## 诚实总账:哪些新?哪些 prior art?

| 项 | 状态 | 对标 |
|---|---|---|
| RRT\* 平滑/采样进化、RRT-Connect | ❌ | Kuffner2000 / FunSearch |
| v\*、有限段速度、爬降不对称 | ❌ | Di Franco2015 / Liu2017 |
| 非欧 informed sampling | ❌(且我方更差) | Kyaw&Kelly2026 |
| M100 能耗建模 | ❌ | Dai / Tseng |
| 顺序翻转 / 可达集 / 载荷规划 | ❌ | Michel2024 / Nguyen2017 / EcoFlight |
| LLM 自动科研框架 | ❌ 非首创 | FunSearch/Eureka/AlphaEvolve |
| **真机数据锚定的可信代价**(vs 文献第一性原理/手搭模型) | ⭕ 区别 | — |
| **可信/证伪优先自动科研协议**(防作弊+查新门+因果消融+破循环+系统报负结果) | ⭕ 可能真空白 | AI Scientist 反被批 overclaim(查证中) |
| **域无关性证据**(同框架 UAV 规划 + 雷达跟踪 + 能耗都跑通) | ⭕ 支撑方法学论点 | — |

## 唯一站得住的贡献表述(全程一致)
> 不声称任何**算法/模型/效应**首创(都能对上已发表论文)。贡献 = **方法学 + 真机数据锚定 + 诚实能力边界**:
> 一套防自欺的可信自动科研框架(冻结评测器、keep/revert、强制查新门、因果消融、交叉模型破循环、系统报负结果),
> 在真实 DJI M100 上验证,并**诚实刻画** LLM 自动科研相对简单基线的能力边界(结构搜索≈随机;效应≈已知;载荷无效)。
> **本会话本身——包括主动把自己的发现判成 prior art——就是"可信 AI4S"的活体演示。**

---
*产物索引:阶段0 `autoresearch/`(agent_loop/evaluator/PRIOR_ART.md/LAYERS.md/POSITIONING.md/milestones);
阶段1 `autoresearch/radar/`;阶段2-6 `autoresearch/energy_m100/`(m100_eval/wall_experiment/robustness_suite/
climb_ablation/sim_flight/corridor_experiment/render_videos + RESULTS.md + midterm_report_v2.html)。*
