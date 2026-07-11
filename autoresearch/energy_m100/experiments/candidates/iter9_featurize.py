def featurize(s):
    """iter9 — iter6 (best) + electrical I²R efficiency-loss term T³ (non-aerodynamic mechanism).

    All prior novel forms were aerodynamic (induced/parasite/air-relative) or interactions and
    failed to beat linear. The distinct untried mechanism is ELECTRICAL: motor+ESC copper losses
    scale as I²R ∝ current² ∝ (mechanical power)² ∝ (T^1.5)² = T³. Add T³ to iter6 — a non-LIB,
    non-aerodynamic form. Completes the honest sweep (aero + electrical). Expected collinear with
    T^1.5 over the narrow M100 mass range -> likely REVERT.
    """
    g = 9.81
    m0 = 2.4
    pay = s["payload"]
    m = m0 + pay / 1000.0
    az_eff = np.maximum(g + s["a_z"], 0.0)
    vh = s["v_h"]
    vz = s["v_z"]
    T = m * az_eff
    Veff = np.sqrt(vh * vh + 1.0)

    P_hover = T ** 1.5                         # hover induced power
    P_fwd = (T * T) / Veff                     # forward induced relief
    P_prof = vh * vh                           # profile ~ v²
    P_prof_load = T * (vh * vh)                # thrust-scaled profile
    P_para = vh ** 3                           # parasite ~ v³
    climb = np.maximum(vz, 0.0)                # climb work
    desc = np.minimum(vz, 0.0)                # descent
    P_esc = T ** 3                             # NEW: electrical copper loss ~ I² ~ (mech power)² ~ (T^1.5)² = T³

    return np.column_stack([np.ones_like(vh), P_hover, P_fwd, P_prof, P_prof_load, P_para, climb, desc, pay, P_esc])
