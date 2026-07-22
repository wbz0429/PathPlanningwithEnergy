
def smooth_path(path, is_collision_free, config):
    import numpy as np
    if len(path) <= 2:
        return path
    pts = [p.copy() for p in path]
    # 第1趟:多趟视线捷径(同基线,删冗余点)
    for _ in range(10):
        out = [pts[0]]; i = 0; shortened = False
        while i < len(pts) - 1:
            best_j = i + 1
            for j in range(len(pts) - 1, i + 1, -1):
                if is_collision_free(pts[i], pts[j]):
                    best_j = j; break
            if best_j > i + 1: shortened = True
            out.append(pts[best_j]); i = best_j
        pts = out
        if not shortened or len(pts) <= 2: break
    # 第2趟:顶点回拉——把每个中间点朝"前后点连线"方向移,拉直急转角(降剖面能耗)
    for _ in range(6):
        moved = False
        for i in range(1, len(pts) - 1):
            mid = 0.5 * (pts[i-1] + pts[i+1])          # 前后点中点
            for a in (0.7, 0.4, 0.2):                   # 试不同回拉强度
                cand = (1-a) * pts[i] + a * mid
                if is_collision_free(pts[i-1], cand) and is_collision_free(cand, pts[i+1]):
                    if np.linalg.norm(cand - pts[i]) > 1e-3:
                        pts[i] = cand; moved = True
                    break
        if not moved: break
    return pts
