
def smooth_path(path, is_collision_free, config):
    import numpy as np
    if len(path) <= 2:
        return path
    pts = [p.copy() for p in path]
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
    # 只对"急转角"(夹角<130度)做小步长回拉,缓解转弯减速;其余不动
    for _ in range(8):
        moved = False
        for i in range(1, len(pts) - 1):
            u = pts[i] - pts[i-1]; v = pts[i+1] - pts[i]
            cu, cv = np.linalg.norm(u), np.linalg.norm(v)
            if cu < 1e-6 or cv < 1e-6: continue
            ang = np.degrees(np.arccos(np.clip(u @ v / (cu*cv), -1, 1)))
            if ang <= 50:   # 转角平缓(方向变化<50度),不用管
                continue
            mid = 0.5 * (pts[i-1] + pts[i+1])
            cand = 0.75 * pts[i] + 0.25 * mid   # 小步长回拉
            if is_collision_free(pts[i-1], cand) and is_collision_free(cand, pts[i+1]):
                pts[i] = cand; moved = True
        if not moved: break
    return pts
