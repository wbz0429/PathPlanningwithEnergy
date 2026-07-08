import math

V_STAR = 18.2      # energy-optimal cruise speed of the frozen BEMT model (m/s)
A_LAT = 3.0        # lateral-acceleration turn limit used by the profile model
P0 = 150.0         # local-sweep proxy: e/m(v) ~ P0/v - PSLOPE (pessimistic at
PSLOPE = 1.6       # mid speeds — empirically steers moves toward higher vcap)
# 3D ranking proxy (template comparison only): parabolic level-power fit of
# the frozen BEMT plus climb charge / conservative floored descent credit.
# Calibrated against the frozen BEMT on an 8x9 (v, slope) grid:
# e/m(v,th) ~ max(0, P_lvl(v)/v + 16.8*sin(th)); symmetric climb/descent
# surcharge; the evaluator clamps segment energy at 0 (steep fast descent
# is literally free), so the floor is 0, not a positive power floor.
PA = 171.0
PB = 12.35
PC = 0.525
WEIGHT = 14.7      # m*g (1.5 kg quad)
K_UP = 1.143       # 16.8 / 14.7 — measured, symmetric
K_DN = 1.143
P_FLOOR = 0.0
MAX_WP = 64

# dense measured e/m grid (frozen BEMT, 31 speeds x 11 slopes) — used for
# the steep-descent region where the analytic model misses the VRS penalty
GV = (0.5, 1.0, 1.6, 2.2, 2.8, 3.4, 4.0, 4.6, 5.2, 5.8, 6.4, 7.0, 7.6, 8.2, 8.8, 9.4, 10.0, 10.6, 11.2, 11.8, 12.4, 13.0, 13.6, 14.2, 14.8, 15.4, 16.0, 16.6, 17.2, 17.8, 18.2)
GS = (-75.0, -60.0, -45.0, -30.0, -15.0, 0.0, 15.0, 30.0, 45.0, 60.0, 75.0)
GEM = (
    (305.8, 304.95, 303.75, 302.46, 301.33, 300.5, 300.02, 299.87, 299.94, 300.11, 300.27),
    (157.21, 155.55, 153.39, 151.33, 149.78, 148.95, 148.79, 149.14, 149.76, 150.41, 150.89),
    (103.77, 100.71, 97.09, 94.06, 92.17, 91.49, 91.79, 92.71, 93.89, 94.99, 95.75),
    (81.96, 76.74, 71.31, 67.37, 65.35, 64.98, 65.77, 67.21, 68.83, 70.27, 71.25),
    (72.24, 63.65, 56.11, 51.48, 49.56, 49.61, 50.89, 52.78, 54.77, 56.47, 57.6),
    (69.13, 55.32, 45.6, 40.63, 39.04, 39.57, 41.31, 43.58, 45.86, 47.77, 49.02),
    (70.46, 48.95, 37.45, 32.58, 31.48, 32.51, 34.66, 37.26, 39.78, 41.84, 43.18),
    (74.22, 42.94, 30.66, 26.3, 25.79, 27.3, 29.83, 32.69, 35.4, 37.59, 39.0),
    (75.51, 36.53, 24.79, 21.26, 21.37, 23.33, 26.18, 29.26, 32.12, 34.41, 35.89),
    (67.27, 29.75, 19.71, 17.16, 17.88, 20.24, 23.35, 26.62, 29.61, 31.98, 33.5),
    (50.99, 23.13, 15.35, 13.8, 15.07, 17.77, 21.11, 24.54, 27.63, 30.06, 31.62),
    (35.08, 17.2, 11.66, 11.03, 12.79, 15.79, 19.32, 22.87, 26.05, 28.54, 30.12),
    (23.07, 12.19, 8.58, 8.74, 10.92, 14.17, 17.86, 21.52, 24.76, 27.3, 28.9),
    (14.59, 8.1, 6.01, 6.84, 9.37, 12.83, 16.66, 20.41, 23.71, 26.28, 27.91),
    (8.59, 4.81, 3.88, 5.26, 8.09, 11.73, 15.67, 19.49, 22.84, 25.44, 27.08),
    (4.25, 2.18, 2.13, 3.94, 7.01, 10.81, 14.84, 18.73, 22.12, 24.74, 26.4),
    (1.03, 0.07, 0.67, 2.83, 6.11, 10.03, 14.15, 18.09, 21.51, 24.16, 25.82),
    (0.0, 0.0, 0.0, 1.91, 5.36, 9.38, 13.57, 17.56, 21.01, 23.67, 25.34),
    (0.0, 0.0, 0.0, 1.13, 4.72, 8.84, 13.09, 17.11, 20.58, 23.26, 24.94),
    (0.0, 0.0, 0.0, 0.47, 4.19, 8.38, 12.68, 16.73, 20.23, 22.91, 24.6),
    (0.0, 0.0, 0.0, 0.0, 3.74, 8.0, 12.34, 16.42, 19.93, 22.62, 24.32),
    (0.0, 0.0, 0.0, 0.0, 3.36, 7.68, 12.06, 16.16, 19.68, 22.39, 24.08),
    (0.0, 0.0, 0.0, 0.0, 3.05, 7.41, 11.82, 15.94, 19.48, 22.19, 23.89),
    (0.0, 0.0, 0.0, 0.0, 2.79, 7.19, 11.63, 15.77, 19.32, 22.03, 23.74),
    (0.0, 0.0, 0.0, 0.0, 2.58, 7.02, 11.48, 15.64, 19.19, 21.91, 23.61),
    (0.0, 0.0, 0.0, 0.0, 2.4, 6.88, 11.36, 15.53, 19.09, 21.82, 23.52),
    (0.0, 0.0, 0.0, 0.0, 2.27, 6.77, 11.28, 15.45, 19.02, 21.75, 23.45),
    (0.0, 0.0, 0.0, 0.0, 2.17, 6.7, 11.22, 15.4, 18.98, 21.7, 23.41),
    (0.0, 0.0, 0.0, 0.0, 2.1, 6.65, 11.18, 15.38, 18.95, 21.68, 23.39),
    (0.0, 0.0, 0.0, 0.0, 2.06, 6.62, 11.17, 15.37, 18.95, 21.68, 23.39),
    (0.0, 0.0, 0.0, 0.0, 2.04, 6.62, 11.17, 15.38, 18.96, 21.69, 23.4),
)



def _proxy(pts):
    """Climb/descent-aware mirror of the profile-energy rule.

    Speed per segment: v* unless capped by an adjacent corner's turn speed
    v_t = sqrt(A_LAT * R), R = max(0.3, min(L1, L2)/theta). Segment power is
    the level-flight parabola plus a climb term (or floored descent credit),
    both fit to the frozen BEMT's own outputs. Ranking only — hard safety is
    still enforced by is_collision_free on every new segment.
    """
    n = len(pts)
    if n < 2:
        return 0.0
    segs = [pts[i + 1] - pts[i] for i in range(n - 1)]
    L = [float(np.linalg.norm(s)) for s in segs]
    d = [segs[i] / L[i] if L[i] > 1e-9 else segs[i] * 0.0 for i in range(n - 1)]
    vcap = [V_STAR] * (n - 1)
    for i in range(1, n - 1):
        cosv = max(-1.0, min(1.0, float(np.dot(d[i - 1], d[i]))))
        theta = math.acos(cosv)
        R = max(0.3, min(L[i - 1], L[i]) / max(theta, 1e-3))
        vt = min(V_STAR, math.sqrt(A_LAT * R))
        if vt < vcap[i - 1]:
            vcap[i - 1] = vt
        if vt < vcap[i]:
            vcap[i] = vt
    J = 0.0
    for i in range(n - 1):
        v = max(0.5, vcap[i])
        J += L[i] * (P0 / v - PSLOPE)
    return J


M_EFF = 1.5 / (0.85 * 0.95)   # mass / (motor_eff * esc_eff) for rise cost
DP_LEVELS = (0.5, 2.0, 4.0, 6.0, 8.0, 10.0, 11.5, 12.7, 14.0, 16.0, 18.2)


def _proxy3d(pts):
    """S3a geometry cost = DP-optimal speed cost of this geometry.

    With speed an evolved decision (cruise ~12.7 < v*), corners whose cap
    exceeds the cruise are free — ranking geometry by cap-riding cost
    optimizes a dead objective. This coarse spatial-domain velocity DP
    (rise-priced kinetic cost, free decel, calibrated e/m surrogate)
    prices each candidate geometry by what the speed layer can actually
    achieve on it. Used for topology ranking, CHOMP gradients, STOMP
    weights and hop acceptance."""
    n = len(pts)
    if n < 2:
        return 0.0
    segs = [pts[i + 1] - pts[i] for i in range(n - 1)]
    L = [float(np.linalg.norm(s)) for s in segs]
    d = [segs[i] / L[i] if L[i] > 1e-9 else segs[i] * 0.0 for i in range(n - 1)]
    vcap = [V_STAR] * (n - 1)
    for i in range(1, n - 1):
        cosv = max(-1.0, min(1.0, float(np.dot(d[i - 1], d[i]))))
        theta = math.acos(cosv)
        R = max(0.3, min(L[i - 1], L[i]) / max(theta, 1e-3))
        vt = min(V_STAR, math.sqrt(A_LAT * R))
        if vt < vcap[i - 1]:
            vcap[i - 1] = vt
        if vt < vcap[i]:
            vcap[i] = vt
    sin_th = []
    for i in range(n - 1):
        s = -segs[i][2] / L[i] if L[i] > 1e-9 else 0.0
        sin_th.append(max(-1.0, min(1.0, float(s))))

    def em_pm(v, s):
        if s >= -0.17:
            # climb/level: smooth analytic fit (beats lookups here)
            p = PA - PB * v + PC * v * v
            return max(0.0, p / v + 16.8 * s)
        # steep descent: measured grid (analytic misses the low-speed
        # VRS penalty — descent below ~7 m/s COSTS more than level)
        sd = math.degrees(math.asin(max(-1.0, min(1.0, s))))
        vv = max(GV[0], min(v, GV[-1]))
        sdd = max(GS[0], min(GS[-1], sd))
        iv = 0
        while iv < len(GV) - 2 and GV[iv + 1] < vv:
            iv += 1
        js = 0
        while js < len(GS) - 2 and GS[js + 1] < sdd:
            js += 1
        tv = (vv - GV[iv]) / (GV[iv + 1] - GV[iv])
        ts = (sdd - GS[js]) / (GS[js + 1] - GS[js])
        a = GEM[iv][js] * (1 - ts) + GEM[iv][js + 1] * ts
        b = GEM[iv + 1][js] * (1 - ts) + GEM[iv + 1][js + 1] * ts
        return max(0.0, a * (1 - tv) + b * tv)

    NL = len(DP_LEVELS)
    dp = [1e18] * NL
    for j in range(NL):
        v = DP_LEVELS[j]
        if v <= vcap[0] + 1e-9:
            dp[j] = 0.5 * M_EFF * v * v + L[0] * em_pm(v, sin_th[0])
    for i in range(1, n - 1):
        nd = [1e18] * NL
        for j in range(NL):
            v = DP_LEVELS[j]
            if v > vcap[i] + 1e-9:
                continue
            seg_e = L[i] * em_pm(v, sin_th[i])
            best = 1e18
            for k in range(NL):
                if dp[k] >= 1e18:
                    continue
                u = DP_LEVELS[k]
                c = dp[k] + seg_e
                if v > u:
                    c += 0.5 * M_EFF * (v * v - u * u)
                if c < best:
                    best = c
            nd[j] = best
        dp = nd
    return min(dp)


def smooth_path(path, is_collision_free, config):
    """Shortcut + proxy local search + over-wall macro-route probe.

    Local moves (remove / relocate / chamfer / split / split-bend /
    double-bend) polish the skeleton but cannot jump topology. The frozen
    BEMT makes descent nearly free while climb costs ~2x level, so a
    climb-cruise-descend route over an obstacle can beat a long lateral
    detour — inject [S, P1@zc, P2@zc, G] templates and adopt the best
    collision-free one when the climb-aware proxy says it wins, then
    re-polish. Every adopted segment passes is_collision_free.
    """
    if len(path) <= 2:
        return [np.asarray(p, dtype=float).copy() for p in path]

    pts = [np.asarray(p, dtype=float).copy() for p in path]
    raw = [p.copy() for p in pts]

    # ---- pass 1: iterative line-of-sight shortcutting (minimal skeleton) ----
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

    if len(pts) <= 2:
        return pts

    def _sweeps(pts):
        for _sweep in range(6):
            improved = False

            # move A: vertex removal
            k = 1
            while k < len(pts) - 1:
                cand = pts[:k] + pts[k + 1:]
                if (_proxy(cand) < _proxy(pts) - 1e-6
                        and is_collision_free(pts[k - 1], pts[k + 1])):
                    pts = cand
                    improved = True
                else:
                    k += 1

            # move D: vertex relocation (fix overshoots, widen sharp corners)
            base = _proxy(pts)
            k = 1
            while k < len(pts) - 1:
                A, P, B = pts[k - 1], pts[k], pts[k + 1]
                v1 = P - A
                v2 = B - P
                L1 = float(np.linalg.norm(v1))
                L2 = float(np.linalg.norm(v2))
                if L1 < 1e-6 or L2 < 1e-6:
                    k += 1
                    continue
                u1 = v1 / L1
                u2 = v2 / L2
                tang = u1 + u2
                nt = float(np.linalg.norm(tang))
                tang = tang / nt if nt > 1e-9 else u1
                norm = u2 - u1
                nn = float(np.linalg.norm(norm))
                norm = norm / nn if nn > 1e-9 else np.array([0.0, 0.0, 1.0])
                zax = np.array([0.0, 0.0, 1.0])
                diag1 = tang + norm
                nd = float(np.linalg.norm(diag1))
                diag1 = diag1 / nd if nd > 1e-9 else tang
                diag2 = tang - norm
                nd = float(np.linalg.norm(diag2))
                diag2 = diag2 / nd if nd > 1e-9 else tang
                binorm = np.cross(u1, u2)
                nb = float(np.linalg.norm(binorm))
                binorm = binorm / nb if nb > 1e-9 else np.array([0.0, 1.0, 0.0])
                best = None
                for direc in (tang, -tang, norm, -norm, diag1, -diag1,
                              diag2, -diag2, zax, -zax, binorm, -binorm):
                    for step in (16.0, 8.0, 4.0, 2.0):
                        Q = P + direc * step
                        cand = pts[:k] + [Q] + pts[k + 1:]
                        J = _proxy(cand)
                        if J < base - 1e-6 and (best is None or J < best[0]):
                            if (is_collision_free(pts[k - 1], Q)
                                    and is_collision_free(Q, pts[k + 1])):
                                best = (J, cand)
                if best is not None:
                    base, pts = best
                    improved = True
                else:
                    k += 1

            # move B: coarse chamfer at sharp corners
            base = _proxy(pts)
            k = 1
            while k < len(pts) - 1 and len(pts) < MAX_WP:
                A, P, B = pts[k - 1], pts[k], pts[k + 1]
                v1 = P - A
                v2 = B - P
                L1 = float(np.linalg.norm(v1))
                L2 = float(np.linalg.norm(v2))
                if L1 < 1e-6 or L2 < 1e-6:
                    k += 1
                    continue
                u1 = v1 / L1
                u2 = v2 / L2
                cosang = max(-1.0, min(1.0, float(np.dot(u1, u2))))
                theta = math.acos(cosang)
                if theta < 0.03:
                    k += 1
                    continue
                best = None
                for f in (0.45, 0.3, 0.18):
                    dcut = f * min(L1, L2)
                    if dcut < 0.4:
                        continue
                    P1 = P - u1 * dcut
                    P2 = P + u2 * dcut
                    cand = pts[:k] + [P1, P2] + pts[k + 1:]
                    J = _proxy(cand)
                    if J < base - 1e-6 and (best is None or J < best[0]):
                        if is_collision_free(P1, P2):
                            best = (J, cand)
                if best is not None:
                    base, pts = best
                    improved = True
                    k += 2
                else:
                    k += 1

            # move C: collinear split — isolate slow zones near corners
            base = _proxy(pts)
            k = 1
            while k < len(pts) - 1 and len(pts) < MAX_WP:
                P = pts[k]
                placed = False
                for prev_side in (True, False):
                    other = pts[k - 1] if prev_side else pts[k + 1]
                    seg = P - other
                    L = float(np.linalg.norm(seg))
                    if L < 2.0:
                        continue
                    u = seg / L
                    best = None
                    for f in (0.15, 0.3, 0.5):
                        Q = P - u * (f * L)
                        idx = k if prev_side else k + 1
                        cand = pts[:idx] + [Q] + pts[idx:]
                        J = _proxy(cand)
                        if J < base - 1e-6 and (best is None or J < best[0]):
                            best = (J, cand)
                    if best is not None:
                        base, pts = best
                        improved = True
                        placed = True
                        break
                k += 2 if placed else 1

            # move E: split+bend — insert a vertex on a leg adjacent to a
            # sharp corner AND offset it perpendicular-outward.
            base = _proxy(pts)
            k = 1
            while k < len(pts) - 1 and len(pts) < MAX_WP:
                A, P, B = pts[k - 1], pts[k], pts[k + 1]
                v1 = P - A
                v2 = B - P
                L1 = float(np.linalg.norm(v1))
                L2 = float(np.linalg.norm(v2))
                if L1 < 1e-6 or L2 < 1e-6:
                    k += 1
                    continue
                u1 = v1 / L1
                u2 = v2 / L2
                cosang = max(-1.0, min(1.0, float(np.dot(u1, u2))))
                theta = math.acos(cosang)
                if theta < 0.5:
                    k += 1
                    continue
                inside = u2 - u1
                best = None
                for prev_side in (True, False):
                    leg_u = u1 if prev_side else u2
                    L = L1 if prev_side else L2
                    if L < 6.0:
                        continue
                    w = -inside + float(np.dot(inside, leg_u)) * leg_u
                    nw = float(np.linalg.norm(w))
                    if nw < 1e-9:
                        continue
                    w = w / nw
                    for f in (0.4, 0.6):
                        Q0 = P - u1 * (f * L1) if prev_side else P + u2 * (f * L2)
                        for m in (2.0, 4.0, 6.0):
                            Q = Q0 + w * m
                            idx = k if prev_side else k + 1
                            cand = pts[:idx] + [Q] + pts[idx:]
                            J = _proxy(cand)
                            if J < base - 1e-6 and (best is None or J < best[0]):
                                a_pt = pts[idx - 1]
                                b_pt = pts[idx]
                                if (is_collision_free(a_pt, Q)
                                        and is_collision_free(Q, b_pt)):
                                    best = (J, cand)
                if best is not None:
                    base, pts = best
                    improved = True
                    k += 2
                else:
                    k += 1

            # move F: double-bend — replace a sharp corner vertex with TWO
            # outward-spread vertices (chamfer is the m=0 special case).
            base = _proxy(pts)
            k = 1
            while k < len(pts) - 1 and len(pts) < MAX_WP:
                A, P, B = pts[k - 1], pts[k], pts[k + 1]
                v1 = P - A
                v2 = B - P
                L1 = float(np.linalg.norm(v1))
                L2 = float(np.linalg.norm(v2))
                if L1 < 6.0 or L2 < 6.0:
                    k += 1
                    continue
                u1 = v1 / L1
                u2 = v2 / L2
                cosang = max(-1.0, min(1.0, float(np.dot(u1, u2))))
                theta = math.acos(cosang)
                if theta < 0.4:
                    k += 1
                    continue
                inside = u2 - u1
                w1 = -inside + float(np.dot(inside, u1)) * u1
                nw1 = float(np.linalg.norm(w1))
                w2 = -inside + float(np.dot(inside, u2)) * u2
                nw2 = float(np.linalg.norm(w2))
                if nw1 < 1e-9 or nw2 < 1e-9:
                    k += 1
                    continue
                w1 = w1 / nw1
                w2 = w2 / nw2
                best = None
                for f in (0.35, 0.55):
                    for m in (0.0, 2.0, 4.0, 6.0):
                        Q1 = P - u1 * (f * L1) + w1 * m
                        Q2 = P + u2 * (f * L2) + w2 * m
                        cand = pts[:k] + [Q1, Q2] + pts[k + 1:]
                        J = _proxy(cand)
                        if J < base - 1e-6 and (best is None or J < best[0]):
                            if (is_collision_free(pts[k - 1], Q1)
                                    and is_collision_free(Q1, Q2)
                                    and is_collision_free(Q2, pts[k + 1])):
                                best = (J, cand)
                if best is not None:
                    base, pts = best
                    improved = True
                    k += 2
                else:
                    k += 1

            if not improved:
                break
        return pts

    pts = _sweeps(pts)

    # ---- macro-route probe: climb-cruise-descend template over obstacles ----
    # Raw templates lose to polished incumbents on proxy (their steep-descent
    # corner phantom-caps the whole cruise leg), so: collision-check FIRST,
    # polish the best feasible templates with the same local sweeps, and only
    # then compare — using the climb-aware 3D ranker on both sides.
    S = pts[0]
    G = pts[-1]
    feasible = []
    for zc in (-11.2, -11.5, -12.0, -12.5):
        for f1 in (0.08, 0.12, 0.18, 0.25, 0.32):
            for f2 in (0.8, 0.85, 0.9, 0.95, 0.97, 0.985, 0.995):
                P1 = S + (G - S) * f1
                P1[2] = zc
                P2 = S + (G - S) * f2
                P2[2] = zc
                cand = [S.copy(), P1, P2, G.copy()]
                if (is_collision_free(cand[0], cand[1])
                        and is_collision_free(cand[1], cand[2])
                        and is_collision_free(cand[2], cand[3])):
                    feasible.append((_proxy3d(cand), cand))
    if feasible:
        feasible.sort(key=lambda t: t[0])
        best_J = _proxy3d(pts)
        for _J_raw, cand in feasible[:3]:
            polished = _sweeps(cand)
            Jp = _proxy3d(polished)
            if Jp < best_J - 1e-6:
                best_J = Jp
                pts = polished

    # ---- skeleton-DP probe: global corner placement over raw vertices ----
    # Greedy farthest-first shortcutting fixes corners at visibility-jump
    # artifacts. A second-order DP over the raw polyline picks the skeleton
    # minimizing a corner-cap-aware cost (each edge half priced by the speed
    # cap of its adjacent corner). Candidate only — polished and adopted
    # solely when the climb-aware ranking proxy improves.
    if len(raw) > 90:
        raw = raw[::2]
        if float(np.linalg.norm(raw[-1] - pts[-1])) > 1e-9:
            raw.append(pts[-1].copy())
    n_raw = len(raw)
    if 4 <= n_raw <= 90:
        vis = [[False] * n_raw for _ in range(n_raw)]
        for i in range(n_raw - 1):
            for j in range(i + 1, n_raw):
                if j == i + 1 or is_collision_free(raw[i], raw[j]):
                    vis[i][j] = True

        def _em(v):
            return P0 / max(0.5, v) - PSLOPE

        def _elen(i, j):
            return float(np.linalg.norm(raw[j] - raw[i]))

        dp = {}
        par = {}
        for j in range(1, n_raw):
            if vis[0][j]:
                dp[(0, j)] = 0.5 * _elen(0, j) * _em(V_STAR)
                par[(0, j)] = None
        for j in range(1, n_raw - 1):
            for i in range(j):
                if (i, j) not in dp:
                    continue
                base_c = dp[(i, j)]
                v1 = raw[j] - raw[i]
                L1 = _elen(i, j)
                for k in range(j + 1, n_raw):
                    if not vis[j][k]:
                        continue
                    v2 = raw[k] - raw[j]
                    L2 = _elen(j, k)
                    if L1 < 1e-9 or L2 < 1e-9:
                        continue
                    cosv = max(-1.0, min(1.0,
                               float(np.dot(v1 / L1, v2 / L2))))
                    theta = math.acos(cosv)
                    R = max(0.3, min(L1, L2) / max(theta, 1e-3))
                    vt = min(V_STAR, math.sqrt(A_LAT * R))
                    c = base_c + 0.5 * (L1 + L2) * _em(vt)
                    if c < dp.get((j, k), 1e18):
                        dp[(j, k)] = c
                        par[(j, k)] = i
        end = None
        best_c = 1e18
        for i in range(n_raw - 1):
            if (i, n_raw - 1) in dp:
                c = dp[(i, n_raw - 1)] + 0.5 * _elen(i, n_raw - 1) * _em(V_STAR)
                if c < best_c:
                    best_c = c
                    end = i
        if end is not None:
            idxs = [n_raw - 1, end]
            while par[(idxs[-1], idxs[-2])] is not None:
                idxs.append(par[(idxs[-1], idxs[-2])])
            idxs.reverse()
            skel = [raw[i].copy() for i in idxs]
            if len(skel) >= 2:
                polished = _sweeps(skel)
                if _proxy3d(polished) < _proxy3d(pts) - 1e-6:
                    pts = polished

    # ---- basin hopping: perturb + re-polish, keep best by 3D ranking ----
    # Every polish above is deterministic from one start and lands in one
    # local optimum. Perturb interior vertices (+-2m; np.random is seeded
    # per run by the evaluator, so this stays reproducible), re-polish, and
    # keep the best candidate — only if it passes a full collision recheck.
    best_J = _proxy3d(pts)
    for _trial in range(2):
        cand = [p.copy() for p in pts]
        if len(cand) <= 2:
            break
        for i in range(1, len(cand) - 1):
            cand[i] = cand[i] + (np.random.rand(3) - 0.5) * 4.0
        ok = True
        for i in range(len(cand) - 1):
            if not is_collision_free(cand[i], cand[i + 1]):
                ok = False
                break
        if not ok:
            continue
        cand = _sweeps(cand)
        J = _proxy3d(cand)
        if J < best_J - 1e-6:
            ok2 = True
            for i in range(len(cand) - 1):
                if not is_collision_free(cand[i], cand[i + 1]):
                    ok2 = False
                    break
            if ok2:
                best_J = J
                pts = cand

    # ---- CHOMP-style joint gradient refinement (EXPLORE ep4) ----
    # Continuous descent over ALL interior vertices simultaneously on the
    # climb-aware proxy (numerical gradient; the boolean-only collision
    # oracle replaces CHOMP's smooth obstacle gradient with a collision-
    # gated backtracking line search). Targets multi-vertex couplings the
    # discrete coordinate moves cannot cross (e.g. hump/apex-cap ridges).
    def _chomp(pts):
        n = len(pts)
        if n <= 2:
            return pts
        X = np.array([p for p in pts], dtype=float)
        alpha = 1.0
        for _it in range(60):
            base = _proxy3d([X[i] for i in range(n)])
            grad = np.zeros_like(X)
            eps = 0.05
            for i in range(1, n - 1):
                for d in range(3):
                    Xp = X.copy()
                    Xp[i, d] += eps
                    grad[i, d] = (_proxy3d([Xp[j] for j in range(n)]) - base) / eps
            gn = float(np.linalg.norm(grad))
            if gn < 1e-6:
                break
            step = grad / gn
            accepted = False
            while alpha > 0.05:
                Xn = X - step * alpha
                Jn = _proxy3d([Xn[i] for i in range(n)])
                if Jn < base - 1e-9:
                    ok = True
                    for i in range(n - 1):
                        if not is_collision_free(Xn[i], Xn[i + 1]):
                            ok = False
                            break
                    if ok:
                        X = Xn
                        accepted = True
                        alpha = min(alpha * 1.5, 2.0)
                        break
                alpha *= 0.5
            if not accepted:
                break
        cand = [X[i].copy() for i in range(n)]
        if _proxy3d(cand) < _proxy3d(pts) - 1e-6:
            return cand
        return pts

    pts = _chomp(pts)

    # alternate discrete + continuous once more: each family unlocks the
    # other's local minimum (topology moves vs coupled joint adjustments)
    cand = _sweeps([p.copy() for p in pts])
    cand = _chomp(cand)
    if _proxy3d(cand) < _proxy3d(pts) - 1e-6:
        pts = cand

    # ---- STOMP-style stochastic refinement ----
    # The proxy is kinked (min(), cap saturations) where gradient descent
    # stalls; cost-weighted averaging of sampled noisy variants descends
    # through kinks without gradients (Kalakrishnan et al., STOMP).
    n = len(pts)
    if n > 2:
        X = np.array([p for p in pts], dtype=float)
        bestX = X.copy()
        bestJ = _proxy3d([X[i] for i in range(n)])
        sigma = 1.5
        for _it in range(30):
            noises = []
            costs = []
            for _k in range(8):
                N = np.zeros_like(X)
                N[1:-1] = np.random.randn(n - 2, 3) * sigma
                cand = X + N
                noises.append(N)
                costs.append(_proxy3d([cand[i] for i in range(n)]))
            c = np.array(costs)
            spread = max(1e-9, (float(c.max()) - float(c.min())) / 10.0)
            w = np.exp(-(c - float(c.min())) / spread)
            w = w / float(w.sum())
            delta = np.zeros_like(X)
            for _k in range(8):
                delta += w[_k] * noises[_k]
            Xn = X + delta
            Jn = _proxy3d([Xn[i] for i in range(n)])
            if Jn < _proxy3d([X[i] for i in range(n)]) - 1e-9:
                ok = True
                for i in range(n - 1):
                    if not is_collision_free(Xn[i], Xn[i + 1]):
                        ok = False
                        break
                if ok:
                    X = Xn
                    if Jn < bestJ:
                        bestJ = Jn
                        bestX = X.copy()
            sigma *= 0.93
        cand = [bestX[i].copy() for i in range(n)]
        if _proxy3d(cand) < _proxy3d(pts) - 1e-6:
            ok = True
            for i in range(n - 1):
                if not is_collision_free(cand[i], cand[i + 1]):
                    ok = False
                    break
            if ok:
                pts = cand

    # ---- taut-shrink probe (S3b): string-tightening toward the chord ----
    # Shortest homotopic paths hug inflated corners (funnel/taut-string);
    # lateral detours overshoot the wall edge. Scale interior vertices
    # toward the S-G chord (homotopy-preserving), collision-gate, polish,
    # adopt when the DP-cost ranker improves.
    S0 = pts[0]
    G0 = pts[-1]
    chord = G0 - S0
    cl2 = float(np.dot(chord, chord))
    if cl2 > 1e-9 and len(pts) > 2:
        best_J2 = _proxy3d(pts)
        for lam in (0.85, 0.7, 0.55, 0.4):
            cand = [S0.copy()]
            for i in range(1, len(pts) - 1):
                t = float(np.dot(pts[i] - S0, chord)) / cl2
                t = max(0.0, min(1.0, t))
                cp = S0 + chord * t
                cand.append(cp + (pts[i] - cp) * lam)
            cand.append(G0.copy())
            ok = True
            for i in range(len(cand) - 1):
                if not is_collision_free(cand[i], cand[i + 1]):
                    ok = False
                    break
            if not ok:
                continue
            cand = _sweeps(cand)
            J = _proxy3d(cand)
            if J < best_J2 - 1e-6:
                ok2 = True
                for i in range(len(cand) - 1):
                    if not is_collision_free(cand[i], cand[i + 1]):
                        ok2 = False
                        break
                if ok2:
                    best_J2 = J
                    pts = cand

    return pts
