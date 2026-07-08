import math

V_STAR = 18.2      # energy-optimal cruise speed of the frozen BEMT model (m/s)
A_LAT = 3.0        # lateral-acceleration turn limit used by the profile model
P0 = 150.0         # local-sweep proxy: e/m(v) ~ P0/v - PSLOPE (pessimistic at
PSLOPE = 1.6       # mid speeds — empirically steers moves toward higher vcap)
# 3D ranking proxy (template comparison only): parabolic level-power fit of
# the frozen BEMT plus climb charge / conservative floored descent credit.
PA = 164.8
PB = 11.39
PC = 0.492
WEIGHT = 14.7      # m*g (1.5 kg quad)
K_UP = 1.26
K_DN = 1.00
P_FLOOR = 15.0
MAX_WP = 64


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


def _proxy3d(pts):
    """Climb/descent-aware ranking proxy — used ONLY to compare polished
    topology candidates (over-wall template vs incumbent), never to accept
    local moves (the pessimistic linear proxy performs better there)."""
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
        if L[i] < 1e-9:
            continue
        v = max(0.5, vcap[i])
        vz = -(segs[i][2] / L[i]) * v      # NED: dz<0 means climbing
        P = PA - PB * v + PC * v * v
        if vz > 0.0:
            P += K_UP * WEIGHT * vz
        else:
            P = max(P_FLOOR, P + K_DN * WEIGHT * vz)
        J += L[i] * P / v
    return J


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
            for f2 in (0.8, 0.85, 0.9, 0.95):
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

    return pts
