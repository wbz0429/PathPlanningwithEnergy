# program.md — 无人机能量-路径规划 autoresearch 研究策略

> **这是你(人类)给研究智能体的"指令频道"**。agent 每轮迭代先读本文件。
> 改这里的「研究指令」= 给 agent 下达新方向。karpathy 式:人写方向,循环自动迭代。

---

## 🎯 研究指令(← 人类在此写方向,agent 每轮遵循)

**当前指令(S4a 攻 cited 开放问题:非欧能量代价下的 informed 采样,见 OPEN_PROBLEMS.md / Kyaw & Kelly 2026 arXiv 2606.02879)**:
**问题**:informed 采样(加速 RRT\*/BIT\* 收敛的核心)多为欧氏空间设计,在**能量这种非欧/各向异性代价**下,欧氏启发式**不可采纳/过保守**(标量界丢弃方向结构)。攻它 = **进化一个"能量各向异性感知"的 `sample(ctx)`**,利用能量代价的方向性(降落便宜、直线便宜、转弯/爬升贵)加速收敛。
**评测(用这个,不是 evaluate)**:`physics_eval.convergence_eval(sampler_src=<你的sample源码>, seeds=(0,1,2), budget=1500)`——指标=**固定样本预算 1500 下 anytime RRT\* 收敛到的剖面能耗**(越低=收敛越快;成功率<100%重罚)。场景=B/C(RRT\*可靠求解、排除翻墙A)。
**采样器可查** `ctx.edge_cost(a, b)`=冻结 BEMT 能量 oracle(只能查、不能改),据此做非欧 informed 采样。
**目标**:击败**均匀 baseline 1419.4**(证明各向异性采样加速收敛)。**已验证**:朴素 cost-greedy(只挑起点到候选最便宜的)会**过度集中→0%成功**,必须平衡探索(如 informed 椭球 + 各向异性加权、桥测试、方向偏置)。留出 seed(如 3,4)验证不过拟合。真正 SOTA 门槛=Euclidean-informed(待我加)。榨干后回报。

---

## 目标(单一指标,冻结)
在「三场景 100% 到达 + 独立碰撞复核」硬约束下,**最小化 BEMT 速度剖面能耗(J) 之和**。
- 评测器 `physics_eval.evaluate`:接地地图(禁钻地)+ 速度剖面能量(v*≈18.2 m/s,转弯限速)。
- 当前最优 score = **2078**(Milestone 1);各场景 A≈797 / B≈630 / C≈648 J。

## 🧊 不可改(冻结层 = 尺子,碰=作弊)
BEMT 物理参数、kinodynamic 极限(turning_radius/climb_angle)、能量度量、A* 锚点、
safety_margin、**三场景/地图/障碍布局**、seed 规则。→ 不编辑 `physics_eval.py`/`evaluator.py`/`energy/`。

## 🔧 可改(Layer-1 = 搜索空间)
- 参数:`step_size, max_iterations, goal_sample_rate, search_radius, weight_*, use_rrt_connect`
- 代码组件(经沙箱注入,不改 drone_sim 源码):
  - 平滑器 `smooth_path(path, is_collision_free, config)`
  - 采样器 `sample(ctx)`(可逃 z-clamp,ctx 给全网格 bounds)
  - **[S3a] 速度剖面 `speed_profile(path, v_star, vcap)`**(每段选速,裁到 [0.5,vcap];评测经 `speed_src` 注入)

## 已知先验(来自 KNOWLEDGE.md,避免重复踩坑)
- 能量非退化:剖面能耗由**路径平滑度**主导(急转弯→减速→偏离 v*→费能),非单纯长度。
- 场景 A 需**翻墙**(下降近免费);A≈797J 已近家族地板,平滑器/采样器殊途同归到此。
- 纯参数是死旋钮(除 step_size);详见 KNOWLEDGE.md 各 Episode insight。

## 可选新方向(人类挑一条填进上面「研究指令」)
1. **换更强算法族**:让 agent 写 informed-RRT*/BIT*-风格采样器,或基于梯度/clearance 的平滑器。
2. **新代价公式**:让 agent 重写 cost 项(如显式转弯惩罚),看能否改善 B/C 的转角税。
3. **[需人先扩问题]** 更丰富场景(地形起伏/风场/多航点)——有真 3D 能量 headroom,agent 才有得发挥。
4. **[需人先解冻]** 速度作为规划器决策(每段选速),U 形 E(v) 下能量最优≠最短 → 新优化维度。

## 迭代纪律
每次只改一个东西、只评一次;KEEP 仅当 score 更低且 min_success 不降;seed0 训练、seed100 留出复核。
