
def smooth_path(path, is_collision_free, config):
    """多趟视线捷径 (iterative shortcutting)。每段直线且过碰撞检查,长度单调不增。"""
    if len(path) <= 2:
        return path
    pts = [p.copy() for p in path]
    for _ in range(10):
        shortened = False
        out = [pts[0]]
        i = 0
        while i < len(pts) - 1:
            best_j = i + 1
            for j in range(len(pts) - 1, i + 1, -1):
                if is_collision_free(pts[i], pts[j]):
                    best_j = j
                    break
            if best_j > i + 1:
                shortened = True
            out.append(pts[best_j])
            i = best_j
        pts = out
        if not shortened or len(pts) <= 2:
            break
    return pts
