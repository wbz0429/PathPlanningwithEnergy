"""
ospa.py — 作弊不了的跟踪评测度量:OSPA(Optimal Sub-Pattern Assignment)。

设计原则(内存安全):只吃"某一帧"的两个有限点集(估计目标位置 vs 真值目标位置),
返回标量。逐帧调用、算完即弃——绝不囤全序列轨迹。整条评测的内存 = 一帧点数 × 常数。

OSPA(X, Y; c, p)(Schuhmacher et al. 2008,MTT 标准度量):
  设 |X|=m(估计)、|Y|=n(真值),WLOG m<=n:
    d_p^c(X,Y) = ( 1/n [ min_{π} Σ_i min(c, ||x_i - y_{π(i)}||)^p  +  c^p·(n-m) ] )^{1/p}
  - 定位误差项(被截断到 c) + 基数误差项(漏检/虚警按 c 计罚)。
  - c = 截断距离(米),p = 阶(2)。越小越好;完全匹配=0;完全不匹配=c。
分配用匈牙利算法(scipy linear_sum_assignment)在 min(c,d)^p 代价上求最优。
这是冻结层(尺子),autoresearch 只能调管线、不能碰这里。
"""
import numpy as np
from scipy.optimize import linear_sum_assignment


def ospa_distance(X, Y, c=2.0, p=2.0):
    """
    X: (m,2) 估计目标位置; Y: (n,2) 真值目标位置(单位米)。返回标量 OSPA(米)。
    约定:两个都空 -> 0;一个空一个非空 -> c(全基数误差)。
    """
    X = np.asarray(X, dtype=float).reshape(-1, 2)
    Y = np.asarray(Y, dtype=float).reshape(-1, 2)
    m, n = len(X), len(Y)
    if m == 0 and n == 0:
        return 0.0
    if m == 0 or n == 0:
        return float(c)
    # 对称:令 m<=n
    if m > n:
        X, Y, m, n = Y, X, n, m
    # 截断距离矩阵 D[i,j] = min(c, ||x_i - y_j||)
    diff = X[:, None, :] - Y[None, :, :]          # (m,n,2)
    D = np.minimum(c, np.linalg.norm(diff, axis=2))  # (m,n)
    ri, cj = linear_sum_assignment(D ** p)         # 最优分配(最小化 Σ min(c,d)^p)
    loc = np.sum(D[ri, cj] ** p)                    # 定位项
    card = (c ** p) * (n - m)                       # 基数项(未匹配的 n-m 个按 c 计)
    return float(((loc + card) / n) ** (1.0 / p))


def ospa_components(X, Y, c=2.0, p=2.0):
    """返回 (ospa, 定位分量, 基数分量),便于诊断增益来自定位还是漏检/虚警。"""
    X = np.asarray(X, dtype=float).reshape(-1, 2)
    Y = np.asarray(Y, dtype=float).reshape(-1, 2)
    m, n = len(X), len(Y)
    if m == 0 and n == 0:
        return 0.0, 0.0, 0.0
    if m == 0 or n == 0:
        return float(c), 0.0, float(c)
    if m > n:
        X, Y, m, n = Y, X, n, m
    diff = X[:, None, :] - Y[None, :, :]
    D = np.minimum(c, np.linalg.norm(diff, axis=2))
    ri, cj = linear_sum_assignment(D ** p)
    loc = np.sum(D[ri, cj] ** p)
    card = (c ** p) * (n - m)
    e_loc = (loc / n) ** (1.0 / p)
    e_card = (card / n) ** (1.0 / p)
    e = ((loc + card) / n) ** (1.0 / p)
    return float(e), float(e_loc), float(e_card)


if __name__ == "__main__":
    # 合成自测(不需要 RadarScenes 数据)——验证尺子行为正确
    c = 2.0
    # 1) 完全相同 -> 0
    A = np.array([[0., 0.], [10., 5.], [-3., 7.]])
    assert abs(ospa_distance(A, A, c=c)) < 1e-9, "相同集应=0"
    # 2) 两个都空 -> 0;一空一非空 -> c
    assert ospa_distance(np.empty((0, 2)), np.empty((0, 2)), c=c) == 0.0
    assert ospa_distance(A, np.empty((0, 2)), c=c) == c, "全漏检应=c"
    # 3) 已知平移:每个目标偏 0.5m(< c),p=2 -> OSPA=0.5
    B = A + np.array([0.5, 0.0])
    assert abs(ospa_distance(A, B, c=c) - 0.5) < 1e-9, "整体平移0.5应=0.5"
    # 4) 定位误差被截断到 c:偏 100m -> 定位项截到 c -> OSPA=c
    Cfar = A + np.array([100., 0.])
    assert abs(ospa_distance(A, Cfar, c=c) - c) < 1e-9, "超远应截到c"
    # 5) 基数误差:3 真值 vs 2 完美估计 -> 纯漏检 1 个
    #    loc=0, card=c^p*1, n=3 -> OSPA=(c^p/3)^(1/p)
    est2 = A[:2]
    exp = (c ** 2 * 1 / 3) ** 0.5
    got = ospa_distance(est2, A, c=c)
    assert abs(got - exp) < 1e-9, f"漏检1个: 期望{exp:.4f} 得{got:.4f}"
    # 6) 分量分解一致性
    e, el, ec = ospa_components(est2, A, c=c)
    assert abs((el ** 2 + ec ** 2) ** 0.5 - e) < 1e-9, "分量应满足 e^2=el^2+ec^2 (p=2)"
    print("OSPA 自测全过 ✓")
    print(f"  相同集={ospa_distance(A, A, c=c):.3f}  平移0.5={ospa_distance(A, B, c=c):.3f}  "
          f"漏检1={got:.3f}  全漏={ospa_distance(A, np.empty((0,2)), c=c):.3f}")
    print(f"  漏检1分解: 总={e:.3f} 定位={el:.3f} 基数={ec:.3f}")
