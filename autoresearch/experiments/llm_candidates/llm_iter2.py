
def smooth_path(path, is_collision_free, config):
    import numpy as np
    if len(path) <= 2:
        return path
    pts = [p.copy() for p in path]
    # 第1趟:多趟视线捷径
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
    # 第2趟:转角圆弧化——每个转角用前后段各切一小段插两个过渡点,把尖角变钝(增大转弯半径→少减速)
    for _ in range(3):
        out = [pts[0]]; changed = False
        for i in range(1, len(pts) - 1):
            a, b, c = pts[i-1], pts[i], pts[i+1]
            d1 = np.linalg.norm(b - a); d2 = np.linalg.norm(c - b)
            if d1 < 1e-6 or d2 < 1e-6:
                out.append(b); continue
            # 转角越尖(夹角越小)切得越多;切段长度取相邻段的 25%
            cut = 0.25 * min(d1, d2)
            p1 = b - (b - a) / d1 * cut     # 进入点
            p2 = b + (c - b) / d2 * cut     # 离开点
            if is_collision_free(a, p1) and is_collision_free(p1, p2) and is_collision_free(p2, c):
                out.append(p1); out.append(p2); changed = True   # 用两个钝角过渡点替代尖角
            else:
                out.append(b)
        out.append(pts[-1]); pts = out
        if not changed: break
    return pts
