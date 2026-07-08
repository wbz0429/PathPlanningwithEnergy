# program.md — 无人机能量-路径规划 autoresearch 研究策略

> **这是你(人类)给研究智能体的"指令频道"**。agent 每轮迭代先读本文件。
> 改这里的「研究指令」= 给 agent 下达新方向。karpathy 式:人写方向,循环自动迭代。

---

## 🎯 研究指令(← 人类在此写方向,agent 每轮遵循)

**当前指令(S3b 泛化,见 ROADMAP.md)**:场景从 3 个手工场景扩到 **6 个 train 场景**(A/B/C + 程序化生成 gen300-302,生成器固定、你改不了)。
**评测必须**:`physics_eval.evaluate(overrides, ..., scenarios=physics_eval.get_train(), smoother_src=<state/best_smoother.py>, speed_src=<state/best_speed.py>)`——带上 S3a 累积的平滑器+速度剖面,只改本轮提议的那个。
**目标**:让算法在**更多样的查询**上都省能(生成场景 gen300-302 现 1000-1277J 有空间)。**泛化验证**:留出 `physics_eval.get_test()`(gen400-402)+ seed100,best 必须在留出场景上也不退化。这测的是"解泛化到没见过的场景",比留出 seed 更强。榨干后我推进 S3c(地形/风)或 S4(研究员行为)。

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
