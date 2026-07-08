def sample(ctx):
    """Sub-level-set energy-informed rejection sampler (EXPLORE, iter60).

    Literature-grounded (HRS / MCMC family for non-Euclidean informed sets,
    Kyaw&Kelly 2026 arXiv:2606.02879): when the cost is non-Euclidean the
    informed set is NOT an ellipsoid; sample it as the sub-level set
    {x : edge_cost(s,x) + edge_cost(x,g) <= c_thresh} directly via the frozen
    BEMT energy oracle. This PRINCIPLED version needs no hand-tuned axes: the
    vertical squash emerges automatically because any altitude excursion
    inflates edge_cost (climb energy unrecovered) and gets rejected. c_best is
    not provided, so c_thresh anneals broad->tight over the sample budget.
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
    if r < 0.08:
        return g.copy()
    # annealed uniform fallback (find first solution before tightening)
    p_unif = 0.12 + 0.38 * (1.0 - frac)
    if r < 0.08 + p_unif:
        return rng.uniform(lo, hi)

    # energy informed sub-level set via rejection on the oracle
    c_min = float(ctx.edge_cost(s, g))          # straight-line energy (ignores collision)
    m = 0.10 + 0.65 * (1.0 - frac)
    c_thresh = c_min * (1.0 + m)
    for _try in range(16):
        x = rng.uniform(lo, hi)
        f = float(ctx.edge_cost(s, x)) + float(ctx.edge_cost(x, g))
        if f <= c_thresh:
            return x
    # rejection exhausted -> uniform (keeps success up)
    return rng.uniform(lo, hi)
