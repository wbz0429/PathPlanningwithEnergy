# program.md — 无人机能耗 autoresearch 研究策略

> 对标 karpathy/autoresearch 的 `program.md`：人类在此写"研究方向"，
> autoresearch 循环按此自动迭代。可改。

## 目标（单一指标）
在「三场景全部 100% 到达」的硬约束下，**最小化 BEMT 物理模型能耗(J) 之和**。
参考：A* 最优 baseline 总能耗 ≈ 16410 J；默认 RRT* ≈ +25.6%，且场景 A 仅 67% 成功。
→ 第一目标：把场景 A 成功率推到 100%；第二目标：把 vs-A* 从 1.26 压向 1.0。

## 不可改（评测器 evaluator.py = prepare.py 角色）
- 地图/障碍物布局、ESDF、A* baseline、能耗模型、三场景定义、随机种子规则。

## 可改（搜索空间 = train.py 角色）
- RRT*：`step_size`, `max_iterations`, `goal_sample_rate`, `search_radius`
- 几何：`dubins_turning_radius`
- 代价权重：`weight_energy / weight_distance / weight_time`（水平飞行下影响小，低优先）
- 算法结构（stage-2/3 代码改写）：平滑、采样、RRT-Connect/informed

## 已知先验（来自仓库 benchmark_report.md + 标定）
- 水平等高飞行时能耗 ≈ 正比于路径长度 → 降能耗 ≈ 让 RRT* 收敛到更短路径。
- 场景 A 是窄通道，采样式规划器经典弱点 → 需要更强的目标偏置/更多迭代/更大重连半径。
- RRT* 有随机性 → 每配置多次取均值；最终用新种子做留出验证防过拟合。

## 迭代纪律
1. 先随机探索找到"100% 成功"区域；2. 再在最优点附近局部扰动压能耗；
3. 每次只接受 score 更低者（keep/revert）；4. 全程记 JSONL 日志；
5. 收敛后用 held-out 种子 + 更多 runs 复核 best。
