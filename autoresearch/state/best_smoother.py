import math

V_STAR = 18.2     # energy-optimal cruise speed of the frozen BEMT model (m/s)
A_LAT = 3.0       # lateral-acceleration turn limit used by the profile model
P0 = 150.0        # e/m(v) ~ P0/v - PSLOPE : linear-fall fit of BEMT power on [0.5, v*]
PSLOPE = 1.6
MAX_WP = 64


def _proxy(pts):
    """Mirror of the profile-energy speed-cap rule with an analytic e/m fit.

    Speed per segment: v* unless capped by an adjacent corner's turn speed
    v_t = sqrt(A_LAT * R), R = max(0.3, min(L1, L2)/theta). Energy proxy is
    sum L_i * e/m(v_i). Used only to rank candidate moves; hard safety is
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


def smooth_path(path, is_collision_free, config):
    """Shortcut + proxy-guided local search (collinear split / chamfer / remove).

    A sharp corner caps BOTH adjacent segments at its turn speed for their
    entire length, so the dominant waste is long segments dragged down by one
    corner. Moves:
      - collinear split: isolate the slow zone to a short sub-segment near the
        corner (same line, so inherently collision-free);
      - coarse chamfer: one long cut segment halves the corner angle and keeps
        min(L1, L2) large (the metric's effective turn radius);
      - vertex removal: drop vertices the shortcut left behind.
    Every move is accepted only if the profile-energy proxy improves; every
    new off-line segment must pass is_collision_free.
    """
    if len(path) <= 2:
        return [np.asarray(p, dtype=float).copy() for p in path]

    pts = [np.asarray(p, dtype=float).copy() for p in path]

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

    # ---- pass 2: proxy-guided local improvement sweeps ----
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
            best = None
            for direc in (tang, -tang, norm, -norm, diag1, -diag1, diag2, -diag2,
                          zax, -zax):
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
                    Q = P - u * (f * L) if prev_side else P + (-u) * (f * L)
                    # Q lies on the existing (already safe) straight segment
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

        # move E: split+bend — escape the local minimum where a collinear
        # split alone shrinks min(L1, L2) at a sharp corner (rejected) but
        # split-then-bend-outward would divide the turn into two gentle
        # corners with long legs. Insert a vertex on a leg adjacent to a
        # sharp corner AND offset it perpendicular-outward in one candidate.
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
            inside = u2 - u1          # points into the turn
            best = None
            for prev_side in (True, False):
                leg_u = u1 if prev_side else u2
                L = L1 if prev_side else L2
                if L < 6.0:
                    continue
                # outward = -(inside) component perpendicular to this leg
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
        # outward-spread vertices in one candidate (chamfer is the m=0
        # special case). Directly reaches the two-gentle-corners-with-
        # long-legs geometry that single insertions approach only slowly.
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
