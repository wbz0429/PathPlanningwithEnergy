def sample(ctx):
    """Energy-anisotropy-aware informed sampler (v1).

    Attacks the cited open problem (Kyaw&Kelly 2026): the Euclidean informed
    set (symmetric prolate spheroid, Gammell 2014) is over-conservative under
    energy cost because it treats every perpendicular direction equally. But
    the frozen BEMT metric is anisotropic: in-plane lateral detours are cheap
    (level flight ~const J/m in any horizontal heading) while vertical
    excursions are expensive (climb energy is not recovered). So the true
    energy-informed set is an ellipsoid STRETCHED laterally and SQUASHED
    vertically. c_best is not provided by the harness, so the transverse size
    is annealed broad->tight over the sample budget (explore first, focus
    later) instead of keyed on the current solution cost.
    """
    rng = ctx.rng
    s = np.asarray(ctx.start, dtype=float).reshape(3)
    g = np.asarray(ctx.goal, dtype=float).reshape(3)
    lo = np.asarray(ctx.local_bounds_min, dtype=float).reshape(3)
    hi = np.asarray(ctx.local_bounds_max, dtype=float).reshape(3)

    try:
        budget = float(ctx.config.max_iterations)
    except Exception:
        budget = 1500.0
    if budget <= 0.0:
        budget = 1500.0
    try:
        it = float(ctx.iteration)
    except Exception:
        it = 0.0
    frac = min(1.0, max(0.0, it / max(1.0, 0.9 * budget)))

    r = float(rng.random())
    # (1) goal bias — RRT* must connect; keep modest so it doesn't starve refine
    if r < 0.08:
        return g.copy()
    # (2) uniform local exploration — coverage/robustness, annealed high->low so
    #     the first solution is found before focus tightens (avoid greedy 0%)
    p_unif = 0.12 + 0.38 * (1.0 - frac)
    if r < 0.08 + p_unif:
        return rng.uniform(lo, hi)

    # (3) anisotropic informed ellipsoid, foci = start/goal
    d = g - s
    L = float(np.linalg.norm(d))
    if L < 1e-6:
        return rng.uniform(lo, hi)
    a1 = d / L
    # transverse (major) half-axis: annealed margin m: 1.75L(early)->1.12L(late)
    m = 0.10 + 0.65 * (1.0 - frac)
    c_max = L * (1.0 + m)
    rt = 0.5 * c_max
    base = 0.5 * np.sqrt(max(1e-9, c_max * c_max - L * L))
    r_lat = 1.0 * base       # ABLATION: symmetric (Euclidean-informed)
    r_vert = 1.0 * base      # ABLATION: symmetric

    # orthonormal frame: a1, a horizontal-perp (lateral), the remaining (vertical-ish)
    ez = np.array([0.0, 0.0, 1.0])
    lat = np.cross(a1, ez)
    nlat = float(np.linalg.norm(lat))
    if nlat < 1e-6:                     # a1 nearly vertical (not the B/C case)
        lat = np.cross(a1, np.array([1.0, 0.0, 0.0]))
        nlat = float(np.linalg.norm(lat))
    lat = lat / max(nlat, 1e-9)
    vert = np.cross(a1, lat)
    vert = vert / max(float(np.linalg.norm(vert)), 1e-9)

    # uniform point in unit 3-ball
    u = rng.normal(size=3)
    u = u / max(float(np.linalg.norm(u)), 1e-9)
    u = u * (float(rng.random()) ** (1.0 / 3.0))

    center = 0.5 * (s + g)
    p = center + a1 * (rt * u[0]) + lat * (r_lat * u[1]) + vert * (r_vert * u[2])
    return p
