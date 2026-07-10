"""
autoresearch_energy_real.py — 在【真 M100】上跑 UAV 能耗模型 autoresearch(keep/revert)。
冻结评测器 = m100_eval(真实功率的 held-out 能量ARE);loop 搜索能耗模型结构(哪些特征项),
propose(加/删一个项)→ evaluate → keep/revert。随机结构搜索演示机制,LLM proposer(进化 featurize 代码)后接。

诚实:搜索目标用 seeds(0,1,2)交叉验证的 ARE(防过拟合到单一划分);最终在【留出 seeds(5,6,7)】上复核。
与 radar autoresearch_real 完全平行,只是域=UAV能耗、组件=能耗模型。
"""
import numpy as np
from m100_eval import evaluate

# 候选特征项库(物理可解释)——loop 从中增删构造能耗模型
LIB = {
    "v":       lambda s: s["v_h"],
    "v2":      lambda s: s["v_h"] ** 2,
    "v3":      lambda s: s["v_h"] ** 3,
    "vz":      lambda s: s["v_z"],
    "climb+":  lambda s: np.maximum(0.0, s["v_z"]),    # 爬升分量
    "|az|":    lambda s: np.abs(s["a_z"]),             # 垂直加速度(去重力)
    "ah":      lambda s: s["a_h"],                     # 水平机动加速度
    "omega":   lambda s: s["omega"],                   # 转弯角速度
    "payload": lambda s: s["payload"],                 # 载荷(重→费电)
    "wind":    lambda s: s["wind"],                    # 风速
    "v*pay":   lambda s: s["v_h"] * s["payload"],      # 速度×载荷 交互
    "v2*pay":  lambda s: s["v_h"] ** 2 * s["payload"],
}
BASELINE = ["v", "v2"]        # 稳态 BEMT


def feat(terms):
    return lambda s: np.column_stack([np.ones_like(s["v_h"])] + [LIB[t](s) for t in terms])


def obj(terms, seeds=(0, 1, 2)):        # 搜索目标:交叉验证平均能量ARE
    return float(np.mean([evaluate(feat(terms), seed=k)["energy_ARE"] for k in seeds]))


def holdout(terms, seeds=(5, 6, 7)):    # 留出复核(不同飞行划分)
    rs = [evaluate(feat(terms), seed=k) for k in seeds]
    return float(np.mean([r["energy_ARE"] for r in rs])), float(np.mean([r["r2"] for r in rs]))


def search(iters=60, seed=0):
    rng = np.random.default_rng(seed)
    best = list(BASELINE); best_o = obj(best); keeps = 0
    for _ in range(iters):
        cand = list(best)
        t = str(rng.choice(list(LIB)))
        if t in cand:
            cand.remove(t)                 # 删项
        else:
            cand.append(t)                 # 加项
        if not cand:
            continue
        o = obj(cand)
        if o < best_o - 1e-5:              # KEEP:交叉验证 ARE 更低
            best, best_o = cand, o; keeps += 1
    return best, keeps


if __name__ == "__main__":
    print("真 M100 · UAV 能耗模型 autoresearch(结构搜索,60轮)\n")
    b_are, b_r2 = holdout(BASELINE)
    best, keeps = search(iters=60)
    o_are, o_r2 = holdout(best)
    print(f"基线(稳态BEMT 1,v,v²)   留出: 能量ARE={b_are*100:.2f}%  R²={b_r2:+.3f}")
    print(f"loop发现的模型 [{'+'.join(best)}]")
    print(f"  ({keeps}次KEEP)         留出: 能量ARE={o_are*100:.2f}%  R²={o_r2:+.3f}   (ARE {100*(b_are-o_are)/b_are:+.1f}%)")
    print("\n" + ("✅ loop 在真 M100 留出飞行上把能耗模型误差压低 —— UAV能耗+真值+正结果(与radar同范式)。"
                  if o_are < b_are - 1e-4 else "⚠️ 未改进"))
