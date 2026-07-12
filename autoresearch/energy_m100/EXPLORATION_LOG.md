# 自主探索日志(用户休息期间)

**目标**:填好工程缝合框架 → 跑出真机代价 vs 距离 baseline 的对比结果 → 探索场景性工程创新。
**定位**:工程/系统级缝合,非算法创新(6轮 research 已确认域饱和)。

## 状态
- [x] M100 能耗模型包成 em 接口,插进 energy_astar(planning_experiment.py)——通
- [x] 修 tuple 解包 bug(energy_astar 返回 (path,cost))
- [x] **卡点破了**:找到三个 bug 导致"不分叉"——见下
- [x] **三代价出真结果**:距离&BEMT翻墙,M100绕行,省 5–13%(wall_experiment.py)
- [x] 双模型交叉验证暴露诚实核心:省能是"相对可信模型"的
- [ ] 探索:载荷敏感(physics_climb + payload sweep)、鲁棒性、写入论文

## 里程碑:工程缝合出真结果(2026-07-13)
**发现**:距离最短 & 教科书BEMT → 翻墙(爬升);真机M100(实测爬升+20%功率)→ 绕行,省 5–13%。
真规划器(energy_astar),非手画。图 experiments/wall_experiment.png,数据 wall_experiment_result.json。

**破卡点的三个 bug**:
1. **v_z 符号**:模型 正v_z=爬升(639W),但 NED 里爬升 dz<0 → 我喂了负 v_z → 模型当"下降"低估爬升152W。修:v_z=-dz/t。
2. **velocity=2 太慢**:energy_astar 默认 v=2,爬升罚只×1.14(被巡航时间稀释);真实巡航 v=8 时爬升×1.97。用 v=8。
3. **Blocks 原障碍污染**:get_grounded_map 带其他障碍挡绕行→被迫翻。用 clean_wall_map 隔离(只地面+墙)。

**诚实核心(交叉验证)**:M100选的绕行只在M100尺下省(13.3%);BEMT尺下反而贵(-1.5%/-5.9%)。
不是bug——BEMT低估爬升(没见过真功率),M100从真数据学到爬升贵。省能是"相对真机验证模型",非模型无关。
锚点三重:held-out ARE 1.93% + 原始数据 vz-功率相关(599 vs 499W) + 拟合模型一致。
局限:无真机飞验路径→省能是模型预测非实飞测量(写进论文)。

## 关键诚实约束(别忘)
- M100 能耗 distance-dominated → 省能幅度可能小(~6%),诚实报;
- 用真规划器出真路径(不用手画,用户批过 mock);
- 破循环:双模型交叉验证。
