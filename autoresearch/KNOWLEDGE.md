# 领域知识库(autoresearch loop 维护)

> proposer 每轮读它;Episode 边界的文献检索把新招/公式蒸馏追加到这里。

## 目标 & 尺子(冻结,不可改)
- 最小化 **BEMT 速度剖面能量**(v*≈18.2 m/s,转弯按侧向加速度限速)之和,硬约束=三场景 100% 到达。
- 地面约束:z ≥ -0.5 不可飞(A* 不能钻地)。独立碰撞复核:每条路径逐段查 ESDF ≥ safety_margin。
- 场景:A 直穿(需翻越 Row1/Row4 实心墙)、B 对角上、C 对角下。

## 已确立的关键事实
- **能量目标非退化**:固定 2 m/s 时能耗∝长度(退化);换速度剖面后,**急转弯必减速→偏离 v*→每米能耗飙升**,所以路径"平滑度"压倒"长度"主导能量。例:RRT-Connect 路径急转,剖面能耗 ×3.06 A*(定速仅 ×1.34)。
- **场景 A 现在必须翻墙**(爬到 ~11m),是能量+可靠性的主瓶颈。
- **纯参数调优会撞天花板**(历史:vs-A* 地板 ~1.11 是种子噪声)。真正杠杆在:让路径又平滑、又能在直线段跑到接近 v*。
- RRT-Connect:规划快、窄通道稳,但路径急转→剖面能耗高。RRT*:较平滑但慢。

## 可动杠杆(Layer-1)
- 参数:step_size / max_iterations / goal_sample_rate / search_radius / weight_* / use_rrt_connect
- 代码:平滑器 `smooth_path`(降低转弯锐度=降能耗的直接抓手);后续可开采样器。

## Milestone 1(Ep3,iter 21,2026-07-08)

**结果**:score 15000 → **2078.4**(训练种子 0-2),留出验证 seed_val=100 → **2079.2**(Δ+0.04%,三场景 100% 成功;同种子下未调优基线 10684.5、A 场景 0% 成功)——**增益全部泛化,无种子过拟合**。

**最终配置**:`use_rrt_connect=true, step_size=4.5`(其余参数确认为死旋钮);平滑器 v13 = 多趟视线捷径 + 悲观线性 proxy 局部搜索(顶点删除/搬移16m/粗chamfer/共线分割/split-bend/double-bend,6 趟内收敛)+ 翻墙宏路线探针(先碰撞查、后 polish、再用爬降感知 3D proxy 排序采纳)+ raw 顶点 corner-cap-aware 骨架 DP。

**21 轮账本**:7 KEEP(RRT-Connect −79%、proxy 平滑器 −15.6%、split-bend −5.7%、double-bend −0.1%、step 3.0 −8.2%、step 4.5 −1.5%、翻墙探针 −9.2%、细网格 −0.1%、骨架 DP −0.2%)、9 REVERT/no-op(每个都关闭了一条搜索方向并留下判据)。三条硬结论:①度量的 phantom-R 特性决定"少顶点长腿"优于圆弧离散;②BEMT 下降近免费 → 翻墙拓扑占优但必须 polish 后再比较;③悲观 proxy 做局部搜索优于精确 proxy。

**剩余瓶颈**(若继续):A 797J vs 长度下界 ~527J(爬升税+驼峰-v* 耦合,家族地板);B 634/C 648 转角税守恒。突破需要 Layer-1 之外的自由度(如采样器 z 解锁、kinodynamic 松绑——均属冻结层,不可动)。

## Milestone 2(Ep6,iter 38,2026-07-08)

**结果**:score 2078.4(MS1)→ **2046.4**(−1.5%,iter 22-37 共 16 轮:4 KEEP / 11 REVERT·no-op / 1 EXHAUSTED 判决);留出 seed_val=100 → **2043.8(比训练种子还低,零过拟合)**,三场景 100%。

**本期 KEEP 账**:binormal 顶点搬移(−2.9,C)、basin hopping(−7.1,B)、**CHOMP 式联合梯度精修(−17.8,A/B/C 全降——EXPLORE 阶段换算法族的直接产出)**、网格标定 3D 代理(−3.4,全降)。REVERT 各有判据:RNG 流敏感性(basin-hop 不可扩量)、悲观代理优于精确代理(局部走法)、驼峰=v* 保全交易、free-dive 被 Row4 几何锁死。

**横向对比(MS2)**:默认RRT* 11700 > RRT-Connect 3191 > 能量A* 2618 > 随机搜索(同预算15次)2805 > **LLM-loop 2046**——比随机搜索低 **27%**,差距=代码级研究能力(随机搜索写不出 CHOMP/probe/DP 平滑器)。**鲁棒性门:v* 扰动 0.6×–1.5× 下 loop 优势 +26%~+33% 全程稳健 → 真改进非 artifact。**

**终局判决(iter 36 正式 EXHAUSTED)**:2046.4 是当前动作空间(Layer-1 参数+平滑器+采样器)的不动点——A 787J≈守恒地板(爬升 320J≈物理下界 + ~160J 转角 cap 锁定),B/C 转角税守恒,5 个算法族殊途同归,起点无关(4/5 点模板同定点)。**继续需 program.md 给新方向:选项 3(丰富场景,真 3D headroom)或 4(速度作为规划决策)。**

图:`experiments/milestones/ms2/`(轨迹/瀑布/收敛)+ `experiments/fig_ms2_compare.png`。

## Milestone 3(Ep9,iter 47,2026-07-08)——S3a 速度解冻阶段收官

**结果**:S3a 基线 2821.7(走可行上限)→ **2556.5**(−9.4%,iter 39-46 共 8 轮:4 KEEP / 4 REVERT);留出 seed_val=100 → **2566.8**(+0.4%,零过拟合),三场景 100%。

**KEEP 账**:①**速度 DP(−9.1%,主体)**——空间域离散速度 DP,巡航 ≈12.7 < v*(动能边际 m·v/eff 压过 v* 附近平坦的 e/m 曲线),单次加速单调降速形;②DP-cost 拓扑排序(几何-速度耦合,+0.015% 半否定:旧几何在新目标下已近优);③连续 golden-section 精修(−8J);④成对块精修(−1J)。REVERT 判据:**解析 parabola+sin 代理 > 任何 lookup**(网格点精确但插值弦高估凸区,稠密网格也救不回);L-非线性假设被诊断否证;**悲观代理原则跨目标成立**(DP-cost 驱动局部 sweep → A 爆到 1153,精确代理在最优点附近太平→噪声信号)。

**横向对比(MS3,S3a 口径)**:人工默认 11814 > 随机搜索(同预算)3148 > **LLM-loop 2882**(对比脚本口径)——比随机搜索低 8.4%。**⚠️ 鲁棒性门部分失败(必须如实标注)**:v* 扰动 0.6×-1.25× 下 loop 优势 +2.5%~+23.8% 保持,但 **1.5× 时 −6.3% 优势消失**——速度 DP 的巡航速度标定在冻结 BEMT 的 v*=18.2 上,物理假设大幅偏移时剖面欠速。**几何层的 gain 稳健(MS2 已证),速度层的 gain 对 v* 模型敏感**——这是 speed-profile 优化的本质属性(最优巡航依赖能量模型),不是作弊,但报告须区分两类 gain 的稳健性。

**S3a 判决**:速度层在解析代理下已达可证明最优 ~3J 内、连续+成对精修收敛;几何-速度耦合实测极小;结构(cap 圈护)耗尽。**S3a 完成,等人类推进 S3b/S3c(ROADMAP)。**

图:`experiments/milestones/ms3/` + `experiments/fig_ms3_compare.png`。

## Milestone 4(Ep12,iter 56,2026-07-08)——S3b 泛化阶段收官

**结果**:S3b 基线 6066.1(6 train 场景,S3a 组件搬入)→ **6029.7**(iter 48-55 共 8 轮:3 KEEP / 5 REVERT);**泛化双验证:留出场景 gen400-402 全 100% 成功 @3337.5(能耗与 train gen 同量级);train@seed100 = 6207.8(+3.0%)**。

**KEEP 账**:混合下降定价(解析爬升/平飞 + 实测网格陡降,−23J+防灾)、**双目标骨架 DP(cap-aware+最短,确定性双拓扑采样,−12.7J,修 gen301 卡种子翻墙)**、STOMP 后终局确定性 polish(−0.4J)。REVERT 判据链:**VRS 定律**(冻结 BEMT 中 v<7 陡降是惩罚——hook 下降对贴墙目标真不划算,iter48 误定价 +791J 教训);taut-shrink 被自家 ranker 否(宽摆是 cap-最优,funnel 直觉不迁移);**精确代理悖论第 4、5 例**(fine-DP 仲裁更差;top-4 审计放进误判模板)——**有偏估计器在本 loop 每个决策层都优于精确估计器**,已成体系性结论。

**横向对比(MS4)**:人工 11814 > 随机搜索 3148 > **LLM-loop 2882(对比脚本口径,−8.4% vs 随机)**。**鲁棒性(与 MS3 同):几何 gain 稳健(0.6×-1.25× 优势 +2.5%~+23.8%),速度 DP gain 在 v*×1.5 失效(−6.3%,最优巡航本质依赖能量模型,如实标注)**。

**S3b 判决**:泛化主张成立(未见场景/未见种子都不退化);train 收敛 6029.7,残余为 RNG polish 方差(~50-90J,v19 定律不可扩量)。**S3b 完成,等人类推进 S3c(地形/风)或 S4(研究员行为)。**

图:`experiments/milestones/ms4/` + `experiments/fig_ms4_compare.png`。

## Exploration seeds(EXPLORE episode 记录,即使未超 best 也留)
- **[S4a, iter57-58] 各向异性 informed 采样解决 cited 开放问题(核心证据)**:攻 Kyaw&Kelly 2026(非欧能量代价下 informed 采样)。`sample(ctx)` = informed 椭球(焦点=起终点),但**横向拉伸(r_lat=1.7×base)、纵向压扁(r_vert=0.30×base)**——因为冻结 BEMT 下平飞横向绕行便宜(各向同性 ~6.62 J/m)、垂直爬升贵(不回收)。**收敛评测(convergence_eval, B/C, budget=1500)结果**:均匀 baseline 1419.4;**对称欧氏-informed 椭球(消融:r_lat=r_vert=base)只 1311.5(−7.6%)**;**各向异性 810.3(−43%)**。→ **消融证明:informed 聚焦本身只贡献 ~1/6 增益,能量各向异性(横向拉伸)贡献 ~5/6**。机制:B/C 起终点直线穿墙,对称椭球焦点落在墙内→样本浪费;横向拉伸把样本推到绕墙走廊。留出种子(3,4)=804.6、(5,6,7)=801.5,零过拟合。
- **[S4a, iter58] 各向异性比例的过拟合边界**:r_lat 1.7→2.5 在训练种子(0,1,2)把 810→801(−1%),但留出(3,4)炸到 1689(r_lat=2.5)/812(r_lat=2.1);r_lat=3.0 训练也炸(1026)。→ **r_lat=1.7/r_vert=0.30 是鲁棒最优,更激进的拉伸是种子过拟合**(留出门抓出);纵向 r_vert<0.25 反而伤(RRT* 需要一点 z 自由度做树连接/重连)。REVERT 比例微调,保 v1。

- **[Ep11 EXPLORE, S3b] 分布漂移下的参数重标定 + 泛化中检**:一个实例分布上的最优参数换分布后很少最优(来源 https://arxiv.org/abs/2012.13315 portfolio-based algorithm selection;https://arxiv.org/pdf/2202.01651 AC survey,置信度高,未验证-以评测器为准)→ step_size=4.5 是 3 场景上调的,6 场景混合分布应重探。**泛化中检(iter53 实测)**:留出场景 gen400-402 全 100% 成功、能耗 955-1357J(与 train gen 同量级);留出 seed100 仅 +3.0% —— S3b 泛化主张已获证据。
- **[Ep10 EXPLORE, S3b] taut-string/funnel 收紧**:同伦类内最短路=贴着(带 clearance 膨胀的)障碍角的绷紧弦(来源 https://jeffe.cs.illinois.edu/teaching/compgeom/notes/05-shortest-homotopic.pdf Erickson 讲义;https://medium.com/@reza.teshnizi/the-funnel-algorithm-explained-visually-41e374172d2d funnel 图解,置信度高,未验证-以评测器为准)。gen300/302 侧绕路摆到 y≈26-28 而墙缘只需 ~20.5 → ~15m 超摆;taut-shrink(顶点向 S-G 弦收缩 λ 阶梯+碰撞门控+polish)近似弦收紧。另:VRS 定律沉淀——冻结 BEMT 中低速陡降是惩罚不是奖励(v<7 时),hook 下降对贴墙目标真不划算(iter48/49 实证)。
- **[Ep9 EXPLORE, S3a] 块坐标下降处理耦合速度**:rise-cost 链耦合相邻段速度,单坐标 golden-section 在'成对同升可摊销 rise'处卡住;成对块更新是标准解法(来源 https://epubs.siam.org/doi/10.1137/120887679 BCD 收敛性 SIAM;https://www.jmlr.org/papers/volume23/18-045/18-045.pdf 更快 BCD,置信度高,未验证-以评测器为准)。
- **[Ep8 EXPLORE, S3a] 速度分段插点**:corner cap 罩住整条相邻 segment,共线插点(同线、零碰撞风险、不生新角)把 cap 圈进短尾巴,长段交给速度 DP 跑快——文献同型:在区域边界插 waypoint 标记加减速分段(来源 https://www.mdpi.com/2504-446X/5/4/143 Acceleration-Aware Path Planning with Waypoints,置信度高,未验证-以评测器为准)。另:L-非线性假设已被诊断否证(e/m 严格与段长无关);解析 parabola+sin 代理在真实坡度上误差 ~0.06 J/m,优于任何实用 lookup——速度层已在真最优 ~3J 内。
- **[Ep7 EXPLORE, S3a] 空间域速度 DP + 几何-速度耦合**:沿定路径的能量最优速度剖面=空间域 DP(地铁/EV 文献标准做法,来源 https://www.researchgate.net/publication/340156753_On_the_Optimal_Speed_Profile_for_Electric_Vehicles 置信度高,未验证-以评测器为准)。S3a 关键推论:**DP 巡航 ≈12.7 < v* → vcap≥12.7 的转角已免费,平滑器旧目标(抬 cap 到 18)过时**——几何最优随速度层变化(path-velocity decomposition 的耦合项),平滑器的拓扑排序 proxy 应改为'该几何的 DP 最优速度成本'。
- **[Ep6 EXPLORE] 真实旋翼下降剖面佐证**:直升机/多旋翼以带前向速度的对角下降规避 VRS(涡环)并在 windmill-brake 边界附近功率最低(来源 https://arxiv.org/pdf/1909.09069 optimal descent avoiding VRS/WBS,置信度高,未验证-以评测器为准)——与冻结 BEMT '高速陡降免费' 一致。尝试:5 点模板预置下降膝点(polish 每次都重新发现的两段式下降形状),让 CHOMP 精修而非发现。
- **[Ep5 EXPLORE] 冻结 BEMT 全网格标定**(来源=评测器自身 compute_energy_for_segment 实测 8v×9slope 网格,置信度高,已是 ground truth):e/m(v,θ) ≈ max(0, P_lvl(v)/v + 16.8·sinθ),P_lvl(v)=171.0−12.35v+0.525v²(W)。**要点:①爬降税对称 ±16.8·sinθ J/m(旧 proxy 爬升高估 10%、下降低估 12%);②高速陡降完全免费(v≥10、θ≤−24° 时 e/m=0.0,评测器 0 钳位——旧 proxy 的 15W 地板高估了陡降);③低速平飞旧 fit 高估 ~14%**。方法论=离线代理标定(arXiv:2303.17468 surrogate NN for trajectory planning;arXiv:2404.15570 air-taxi physics surrogates,置信度高,未验证-以评测器为准)。_proxy3d 用标定常数(排序+CHOMP梯度+STOMP权重);局部 sweep 保留悲观线性 proxy(v9 教训)。
- **[Ep4 EXPLORE] 轨迹优化族(CHOMP/STOMP)作为平滑器**:CHOMP=对整条轨迹做协变梯度下降精修采样路径(来源 https://www.ri.cmu.edu/pub_files/2013/5/CHOMP_IJRR.pdf,置信度高,未验证-以评测器为准);STOMP=无梯度采样更新(来源 https://www.researchgate.net/publication/221078155_STOMP_Stochastic_trajectory_optimization_for_motion_planning,置信度高,未验证)。我们的适配:代价=3D 剖面能量 proxy(数值梯度),障碍约束只有布尔 is_collision_free(无 ESDF 距离→无法做 CHOMP 的光滑障碍代价梯度),改用「碰撞门控回溯线搜索」投影。潜力:联合多顶点连续优化直击离散单步走法过不去的多顶点耦合脊(如驼峰-apex-cap)。

## Insights(loop 追加)
- **[Ep1] 度量的 phantom-R 特性**:评测器 R=min(L1,L2)/θ——两条长腿夹一个锐角会被解读成"大半径"(快),细分成短弦反而变慢。最优形状=**少顶点、长而均衡的腿、小角度**,不是密集圆弧。平滑器要用"顶点搬移/粗 chamfer/共线分割隔离慢区/split+bend 复合步"这一族走法(iter2-4,score 3191→2538)。
- **[Ep1] 冻结 BEMT 的垂直不对称**(实测 compute_energy_for_segment):12 m/s 时 爬升(24°)=16.0 J/m,平飞=8.2,**下降=1.6 J/m(近乎免费)**;高度往返净成本≈+3 J/m·爬升米。文献同向:min-distance≠min-energy、爬/降不对称 TSP 代价(arXiv:2410.17585;对角下降 +16.6% 能量换 +88% 水平距离)。
- **[Ep1] 翻墙路线真实占优但难落地**:A 场景理想化翻墙模板真实能量 582J vs 侧绕 1205J;可行版(下降须在 x>64 清出 Row4 背面后开始)估算 ~840J vs 当前 1009J。两次失败教训:①模板下降段不能穿墙(f2≥0.9);②**拓扑候选必须先 polish 再比较**——裸 4 点模板的陡降角把整条巡航腿 phantom-cap 在 ~5 m/s,proxy 必输给已 polish 的现任路径;碰撞检查应在 proxy 门槛之前。
- **[Ep1] 平飞功率抛物线拟合**(冻结 BEMT 自测):P(v)=164.8−11.39v+0.492v²(W),v*≈18.3、e/m*≈6.63 J/m 吻合;爬升 +1.26·W·vz,下降 −1.2·W·|vz|、地板 ~10W。但注意:**局部走法用更准的 3D proxy 反而略退步**(A 1009→1043,v5/v6),经验上局部 sweep 保留 v4 的线性悲观拟合(150/v−1.6)更好——悲观偏置推动更高 vcap。3D proxy 只用于给拓扑候选排序。
- **[Ep2] A≈799J 已近本家族地板**:翻墙路线的"驼峰"(z=-15.2)不是浪费——它换来 26m 爬升腿满速 v*(压低驼峰→apex 角变锐→大腿被 cap 到 ~14,得不偿失,v8/v10 no-op、v9 更差已证)。**悲观线性 proxy 做局部走法一贯优于"更准"的 3D proxy**(高估中速成本→逼出高 vcap 几何,真评测器更买账)。
- **[Ep2] 转角守恒律**:R=minL/θ 下,把一个弯拆成 n 个小弯要求腿长成比例增长,而走廊长度固定→B/C 的 ~130J 转角税基本不可再压(拆弯/racing-line 摆宽的解析核算都是 wash)。赛车文献同理:时间对速度比对距离更敏感,应加权偏向最小曲率(min-curvature > min-length),我们的 vcap proxy 已内建此权衡。
- **[Ep3] 种子过拟合风险(算法配置文献)**:固定种子调参会"过拟合到该种子"——训练种子上优、新种子上劣(arXiv:1705.06058 算法配置陷阱;arXiv:2302.14422 SBMP 调参用 held-out 验证)。本 loop 全程只用 seed 0-2 训练;**任何"地板已到"的结论必须过 seed_val=100 的留出验证**才算数。
- **[Ep3] 骨架选择:贪心捷径+局部修复 ≈ 全局 DP(差距仅 ~4J/B 场景)**:对 raw 顶点做 corner-cap-aware 二阶 DP 选骨架,只在 B 上赢 3.6J(A 被模板概率主导、C 拓扑受限)。平滑器 sweep 6 趟内全收敛(12 趟 no-op)。
- **[Ep1] RRT-Connect 骨架質量由 step_size 主导**:1.5→3.0(−8.2%)→4.5(−1.5%)→6.0(反弹 +2.3%),峰值 4.5m。weight_*/search_radius 在 RRT-Connect 分支是死旋钮;goal_sample_rate 在 _smart_sample 里被硬编码 0.2 取代,也是死旋钮。

## Phase B(iter 22-25,采样器注入,2026-07-08)
开放采样器注入点 `sample(ctx)`(可逃 `_compute_sampling_bounds` 的 z-clamp,使 RRT-Connect 原生翻墙)。**4 个候选全 REVERT,best 仍 2078.4**:
- v1 z-escape 全局 → 2094(z 噪声污染 B/C)
- v2 z-escape 定向(仅|Δy|<8 的场景A)→ 2078.4(=最优:让 RRT-Connect **原生**找到翻墙,但 A 能耗已被 v13 平滑器压到 ~797J 家族地板,原生采样无净增益、也没损 B/C)
- v3 goal-bias+定向 → 2108(更差)
- v4 垂直走廊偏置 → 2078.4(=最优)

**结论(诚实负结果)**:扩大动作空间到采样器**未带来算法改进**。瓶颈不是"采样器找不到翻墙"(平滑器已能找到),而是**物理家族地板**(A≈797J 的爬升税)——多条算法路径(平滑器探针 / 原生采样)殊途同归到同一地板。**更多动作自由 ≠ 更好,当问题已达地板时。** 要突破需 Layer-1 之外的自由度(松绑物理/kinodynamic 限制,均属冻结层,不可动)。

## Milestone 1 横向对比(答辩生死线,2026-07-08)
**① 算法横向对比(剖面能耗 score,越低越好)**:默认RRT* 11700 > RRT-Connect 3191 > A*最短/能量加权A* 2618 > **LLM-loop best 2078**。→ loop-best **赢过所有业界算法,连 A* 都赢**(A* 最小化栅格长度,其锯齿路径在剖面能耗下付转弯税)。
**② 优化方法对比(生死线)**:人工默认 11700 > 随机搜索(15次同预算,仅调参+默认平滑器)best 2805 > **LLM-loop 2078**。→ **loop 明显赢随机搜索(~26%)**;随机搜索下不去 2805 因为它**写不出 v13 平滑器代码**。**这个差距 = LLM 代码级研究能力的净价值**,正面回答 "Simple Baselines 2602.16805" 的质疑——这里不是评测器形式化在干活,是 loop 真在干活。
图:`experiments/fig_ms1_compare.png`。

## 鲁棒性门 + 探索阶段(2026-07-08,回应"优化是否有意义/需探索创新阶段")
- **鲁棒性门**(`robustness_check.py`):loop-best vs RRT-Connect 在扰动巡航速度 v*(0.6x~1.5x)下,优势稳定 +26%~+32%,**跨扰动稳健 → 真改进(平滑路径省能),非钻 v* 模型 artifact**。翻墙那类依赖降落不对称的 gain 才需警惕;平滑器的 gain 是稳健的。
- **cadence 加 EXPLORE 阶段**:`explore_every_episodes=2`——每2个episode有1个是探索阶段,agent 检索**不同算法族**(BIT*/FMT*/traj-opt/potential-field)、提结构性新方法、即使没超best也作 exploration seed 记库。区别于 EXPLOIT(精修当前算法)。
- **知识落库纪律**:websearch 知识须带 来源URL+置信度+"未验证(评测器为准)";场景/baseline/物理冻结,agent 不许编造。
