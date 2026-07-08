import math


def speed_profile(path, v_star, vcap):
    """Exact DP over discretized speeds on a calibrated BEMT surrogate.

    Cost = sum L_i * e/m(v_i, slope_i) + sum acceleration cost over rises
    (0.5*m*(v^2 - v_prev^2)/eff, starting from rest; deceleration free).
    Riding vcap re-pays kinetic energy after every slow corner; the optimum
    rises once toward a cruise below v* (kinetic marginal ~ m*v/eff beats
    the flat e/m curve near the top) and only re-rises when the remaining
    distance amortizes it. n<=15 segments x ~40 speed levels -> trivial DP.
    """
    n = len(vcap)
    if n == 0:
        return []
    P = [np.asarray(p, dtype=float) for p in path]
    segs = [P[i + 1] - P[i] for i in range(n)]
    L = [max(1e-9, float(np.linalg.norm(s))) for s in segs]
    sin_th = [max(-1.0, min(1.0, float(-segs[i][2]) / L[i])) for i in range(n)]

    # calibrated frozen-BEMT surrogate (measured 8x9 grid):
    # e/m(v, th) = max(0, P_lvl(v)/v + 16.8*sin(th)), P_lvl = 171 - 12.35v + 0.525v^2
    def em_per_m(v, s):
        p = 171.0 - 12.35 * v + 0.525 * v * v
        return max(0.0, p / v + 16.8 * s)

    M_EFF = 1.5 / (0.85 * 0.95)     # mass / (motor_eff * esc_eff)

    def rise_cost(v0, v1):
        if v1 <= v0:
            return 0.0
        return 0.5 * M_EFF * (v1 * v1 - v0 * v0)

    # speed grid: shared levels + each segment's own cap value
    base = [0.5 + k * 0.45 for k in range(40)]
    levels = sorted(set(round(min(v, 18.2), 3) for v in base) |
                    set(round(max(0.5, min(c, 18.2)), 3) for c in vcap))
    NL = len(levels)

    INF = 1e18
    # dp[j] = min cost ending segment i at speed levels[j]
    dp = [INF] * NL
    par = [[-1] * NL for _ in range(n)]
    for j, v in enumerate(levels):
        if v <= vcap[0] + 1e-9:
            dp[j] = rise_cost(0.0, v) + L[0] * em_per_m(v, sin_th[0])
    for i in range(1, n):
        nd = [INF] * NL
        for j, v in enumerate(levels):
            if v > vcap[i] + 1e-9:
                continue
            seg_e = L[i] * em_per_m(v, sin_th[i])
            best = INF
            bk = -1
            for k, u in enumerate(levels):
                if dp[k] >= INF:
                    continue
                c = dp[k] + rise_cost(u, v) + seg_e
                if c < best:
                    best = c
                    bk = k
            nd[j] = best
            par[i][j] = bk
        dp = nd
    jbest = min(range(NL), key=lambda j: dp[j])
    out = [0.0] * n
    j = jbest
    for i in range(n - 1, -1, -1):
        out[i] = levels[j]
        j = par[i][j] if i > 0 else j

    # continuous polish: coordinate descent on speeds between DP levels
    # (recovers discretization error; rise-cost coupling handled by
    # evaluating the full profile cost at each trial value)
    def total_cost(sp):
        c = 0.0
        vp = 0.0
        for i in range(n):
            v = sp[i]
            c += L[i] * em_of(v, sin_th[i])
            if v > vp:
                c += 0.5 * M_EFF * (v * v - vp * vp)
            vp = v
        return c

    def em_of(v, s):
        p = 171.0 - 12.35 * v + 0.525 * v * v
        return max(0.0, p / max(0.5, v) + 16.8 * s)

    for _sweep in range(4):
        moved = False
        for i in range(n):
            lo = max(0.5, out[i] - 0.5)
            hi = min(vcap[i], out[i] + 0.5)
            if hi <= lo + 1e-6:
                continue
            base_c = total_cost(out)
            best_v = out[i]
            best_c = base_c
            for _it in range(18):
                m1 = lo + 0.382 * (hi - lo)
                m2 = lo + 0.618 * (hi - lo)
                out[i] = m1
                c1 = total_cost(out)
                out[i] = m2
                c2 = total_cost(out)
                if c1 < c2:
                    hi = m2
                    if c1 < best_c:
                        best_c = c1
                        best_v = m1
                else:
                    lo = m1
                    if c2 < best_c:
                        best_c = c2
                        best_v = m2
            out[i] = best_v
            if best_c < base_c - 1e-6:
                moved = True
        if not moved:
            break
    return out
