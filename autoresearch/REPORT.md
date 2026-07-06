# 无人机能耗 autoresearch — 实验报告

对标 karpathy/autoresearch：固定评测器 + 单指标 + 提议→跑→keep/revert。
全程**无需 AirSim**，纯 CPU 跑在 `drone_sim/` 的算法模块上。

## 评测设置（固定，= prepare.py 角色）
- 评测器 `evaluator.py`：缓存 Blocks 地图 + ESDF + A* 最优 baseline；三场景。
- 指标：**三场景总能耗(J)**（BEMT 物理模型算），硬约束=全部 100% 到达。
- A* 最优 baseline 总能耗 = **16409.6J**（A 5138.9 / B 5547.9 / C 5722.8）。
- 抗噪：每配置多次取均值；**留出种子(seed0=100)** 复核，搜索用 seed0=0。

## Stage 1 — 参数搜索（search.py，40 次实验，20.9 min）
- 搜索 `step_size / max_iterations / goal_sample_rate / search_radius / dubins_R / 代价权重`。
- 训练种子上找到 best：vs A* **1.167**、100% 成功 —— 看起来很好。
- **留出验证 → 过拟合**：换种子后 best 退化到 vs A* **1.264**、A/C 掉到 90%，
  反而不如默认(1.199)。runs=3 太少，搜索在优化噪声。
- 结论：参数搜索撞到天花板；窄通道场景 A 的瓶颈是**算法本身**，非调参可解。
  （方法论上：留出纪律成功抓出了过拟合。）

## Stage 2a — 代码改写（_smooth_path 多趟视线捷径）→ 保留
- 假设：旧 `_smooth_path` 有 10m 跳点上限 + 易碰撞回退的 B-spline，路径偏长 → 偏耗能。
- 改动：重写为「多趟视线捷径 (iterative shortcutting)」，去掉上限、多趟收敛；
  每段仍过 `_is_collision_free`，故长度单调不增、安全不降。
- 留出种子严格评测（默认参数，只改代码）：

  | 场景 | 旧代码 | 新代码 | 降幅 |
  |---|---|---|---|
  | A 直穿 | 7229J | 6763J | −6.4% |
  | B 对角上 | 6124J | 5643J | −7.9% |
  | C 对角下 | 6209J | 5751J | −7.4% |
  | **vs A\*** | **1.199** | **1.113** | **−0.086** |

- **判定：KEEP**。能泛化的真改进（留出种子 + 确定性平滑），与 stage-1 过拟合形成对照。

## Stage 2b — 代码改写（前沿定向采样）→ 否决
- 假设：从最接近目标的前沿节点朝目标定向采样可钻过窄通道。
- 留出评测：场景 A 80%→**0%**、B 100%→70%、C 100%→30%。
- 诊断：贪心目标牵引把树困在实心墙根，饿死探索。**判定 REVERT**。
- 价值：展示了 autoresearch 的负反馈——坏点子被留出验证当场打回。

## Stage 3 — 稳健再搜索（runs=5，在含平滑改进的代码上，24 次实验 18.9 min）
- runs 从 3 提到 5 抗噪；利用期一大片配置稳定落在 vs A* 1.108–1.122（非孤立幸运点）。
- **留出验证（seed0=100, runs=10）→ 这次泛化住了**：

  | 配置 | 场景A成功率 | 规划耗时(A) | vs A\* 能耗 | 可靠性 |
  |---|---|---|---|---|
  | 默认+平滑 | 80% | 8.0s | 1.113 | ✗ 20% 到不了 |
  | **再搜索 best** | **100%** | **2.4s** | 1.181 | ✓ 全部可靠 |

  best 配置：step_size=2.42, max_iter=7124, goal_sample_rate=0.50, search_radius=3.93, dubins_R=2.31。

## 当前结论 & 关键发现
- **平滑重写(stage2a)是无条件的赢**：任何配置下都降能耗 ~7%、可泛化、不损安全。
- **再搜索 best 用可靠性+速度换了一点能耗**：A 场景 80%→100%、规划快 3–4×，但 vs A* 1.113→1.181。
- **暴露真问题：可靠性 vs 能耗的权衡**。autoresearch 忠实优化了"100% 成功下最小能耗"，
  于是优先保证了可靠性（默认的低能耗是"有 20% 到不了"换来的假象）。

## Stage 4 — 算法改写：RRT-Connect 双向树 → 保留（打破权衡）
- 目标：打破可靠性↔能耗权衡——同时拿到 100% 可靠 + 低能耗 + 快。
- 手段：`config.use_rrt_connect` 开关；start/goal 两棵树相向贪心连接（`_rrt_connect/_extend_tree/
  _connect_tree/_trace_tree`），连接段经 `_is_collision_free` 校验；末端复用保留的捷径平滑压能耗。
- **留出验证（seed0=100, runs=15）**：

  | 方法 | 场景A成功率 | vs A\* 能耗 | 规划耗时 | 可靠 |
  |---|---|---|---|---|
  | RRT\* 默认+平滑 | 80% | 1.113* | 8.0s | ✗ (*仅成功时) |
  | RRT\* 再搜索best | 100% | 1.181 | 2.4s | ✓ |
  | **RRT-Connect** | **100% (15/15)** | **1.140** | **0.13s** | ✓ |

- **判定：KEEP**。在所有 100% 可靠方案中能耗最低（1.140 < 1.181）、且规划快 ~18×。
  唯一更低的默认 1.113 是"20% 到不了终点"换来的假象。**权衡被打破**。
- 推荐配置：默认参数 + `use_rrt_connect=True`。

## 开放前沿
- 多目标/Pareto：可靠性达标后最小能耗，或约束 step_size 上限，寻求兼得。
- 方法论：runs=5 + 留出验证这套协议已被证明能区分"真改进"与"过拟合"。
- 诚实边界：能耗是 BEMT 物理模型值，非真机实测；捷径平滑优化的是该指标，
  极端拉直可能牺牲可飞性（曲率），与 DC-RRT 的 Dubins 平滑互补。

## 复现
```
cd drone_sim && python ../autoresearch/evaluator.py      # 标定 baseline
python ../autoresearch/search.py 40 3                    # stage-1 搜索
python ../autoresearch/validate.py 10                    # 留出验证
python ../autoresearch/stage2_eval.py 10                 # 代码改写评测
```
文件：`evaluator.py`(评测器) `program.md`(策略) `search.py`(循环) `validate.py`(留出) `stage2_eval.py`(代码改写评测)
代码改动：`drone_sim/planning/rrt_star.py`（`_smooth_path` 捷径 + `_rrt_connect` 双向树；备份 `.bak`/`.stage2a`）
