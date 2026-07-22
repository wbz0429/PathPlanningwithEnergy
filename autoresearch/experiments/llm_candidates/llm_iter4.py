
def smooth_path(path, is_collision_free, config):
    import numpy as np
    if len(path) <= 2:
        return path
    pts = [p.copy() for p in path]
    # 视线捷径
    for _ in range(10):
        out = [pts[0]]; i = 0; sh = False
        while i < len(pts) - 1:
            bj = i + 1
            for j in range(len(pts) - 1, i + 1, -1):
                if is_collision_free(pts[i], pts[j]):
                    bj = j; break
            if bj > i + 1: sh = True
            out.append(pts[bj]); i = bj
        pts = out
        if not sh or len(pts) <= 2: break
    if len(pts) <= 2:
        return pts
    # 在捷径后的骨架上按弧长重新密采样(给梯度平滑提供可动的中间点)
    seg = [np.linalg.norm(pts[i+1]-pts[i]) for i in range(len(pts)-1)]
    total = sum(seg)
    if total < 1e-6:
        return pts
    N = min(40, max(8, int(total / 3.0)))    # 每~3m 一个点
    cum = np.concatenate([[0], np.cumsum(seg)])
    ss = np.linspace(0, total, N)
    dense = []
    for s in ss:
        k = np.searchsorted(cum, s) - 1; k = max(0, min(k, len(pts)-2))
        t = (s - cum[k]) / max(seg[k], 1e-6)
        dense.append(pts[k] + t * (pts[k+1] - pts[k]))
    # CHOMP式平滑:最小化二阶差分(曲率),端点固定,梯度下降,每步查碰撞
    D = [p.copy() for p in dense]
    for _ in range(40):
        newD = [D[0]]
        for i in range(1, len(D)-1):
            lap = D[i-1] + D[i+1] - 2*D[i]        # 曲率梯度
            cand = D[i] + 0.3 * lap               # 朝平滑方向走
            if is_collision_free(D[i-1], cand) and is_collision_free(cand, D[i+1]):
                newD.append(cand)
            else:
                newD.append(D[i])
        newD.append(D[-1]); D = newD
    return D
