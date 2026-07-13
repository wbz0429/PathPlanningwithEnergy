# 规划层研究知识库(loop 读写)

## 已证(带证据,勿重复实验)
- 真机代价改变决策:墙 5-13% / 走廊混合决策 7.6%;动力学后 6.1% / 4.4%(wall_experiment / corridor_experiment / sim_flight JSON)
- 机制=爬升盲:held-out 爬升 premium 真+114W vs 预测+108W;消融挖爬升项→决策退化(climb_ablation)
- 操作包络:高速+窄障省 22%,低速/宽障归零(phase_diagram.json)
- 规划裕度须 ≥2.0m 给平滑留余量,否则蹭墙(sim_flight 工程发现)

## 已否(死路,勿重试)
- 载荷≤500g 改变拓扑(0/250/500g×宽度×mgh 全扫不动)
- 风的能量项(iter5:quadrature 风使留出 ARE 4.24→4.38 变差)
- 平场景省能(Blocks 原生 12/12 全 0%)
- 模型层 featurize 进化(iter1-9 收官:物理形不赢灵活线性)

## 探针速记(未到裁决标准)
- 闭合巡回·等高双塔夹低点 = 平局(总爬升与顺序无关);H1 需不等高链构型(高-谷-中),预估 2-4%

## 迭代记录
(loop 追加)
