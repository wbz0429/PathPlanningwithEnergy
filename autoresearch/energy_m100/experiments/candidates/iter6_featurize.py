def featurize(s):
    """iter6 — iter3 physics core + ONE bare linear payload term (diagnostic-driven).

    Decisive diagnostic (same evaluator, held-out seeds 5,6,7): a single bare `payload`
    term scores 2.25% ARE, vs my 8-term physics model at 4.24%. Hypothesis: the physics
    model underperforms because it forces payload into a rigid T^1.5 shape (base 2.4 kg,
    payload a small perturbation) instead of a flexible linear term. Add bare `payload`
    on top of iter3 and see if ARE collapses toward the linear frontier — isolating whether
    physics priors *hurt* by over-constraining the payload->power map.
    """
    g = 9.81
    m0 = 2.4                                   # DJI M100 base mass (kg)
    pay = s["payload"]                         # payload (grams)
    m = m0 + pay / 1000.0                       # payload -> kg; effective mass
    az_eff = np.maximum(g + s["a_z"], 0.0)     # gravity-removed a_z -> total vertical accel; clamp >=0
    vh = s["v_h"]
    vz = s["v_z"]
    T = m * az_eff                             # instantaneous thrust load (N)
    Veff = np.sqrt(vh * vh + 1.0)              # regularized airspeed (hover -> 1)

    P_hover = T ** 1.5                         # hover induced power (momentum theory) — NON-poly
    P_fwd = (T * T) / Veff                     # forward-flight induced relief (Glauert) — reciprocal
    P_prof = vh * vh                           # profile power base ~ v²
    P_prof_load = T * (vh * vh)                # thrust-scaled profile (v²·payload interaction)
    P_para = vh ** 3                           # parasite body drag ~ v³
    climb = np.maximum(vz, 0.0)                # climb work (asymmetric)
    desc = np.minimum(vz, 0.0)                # descent (asymmetric coeff)

    return np.column_stack([np.ones_like(vh), P_hover, P_fwd, P_prof, P_prof_load, P_para, climb, desc, pay])
