def featurize(s):
    """iter3 — iter1 (kept best) + ONE physics interaction: thrust-scaled profile power.

    iter1's winning core is unchanged (two-term induced split T^1.5 + T²/V, profile v²,
    parasite v³, asymmetric climb/descent). Single added term:
      P_prof_load = T · v_h²   (rotor loading × advance-ratio) — the v²·payload interaction
    the ~1.9% frontier model used, expressed through thrust T = m·(g+a_z). Heavier/accelerating
    craft spin rotors faster, so profile power at a given airspeed scales with thrust.
    Interaction terms are an explicitly-preferred novel form (a bare linear v² + payload
    cannot form their product without this cross column).
    """
    g = 9.81
    m0 = 2.4                                   # DJI M100 base mass (kg)
    m = m0 + s["payload"] / 1000.0             # payload grams -> kg; effective mass
    az_eff = np.maximum(g + s["a_z"], 0.0)     # gravity-removed a_z -> total vertical accel; clamp >=0
    vh = s["v_h"]
    vz = s["v_z"]
    T = m * az_eff                             # instantaneous thrust load (N)
    Veff = np.sqrt(vh * vh + 1.0)              # regularized airspeed (hover -> 1)

    P_hover = T ** 1.5                         # hover induced power (momentum theory) — NON-poly
    P_fwd = (T * T) / Veff                     # forward-flight induced relief (Glauert) — reciprocal
    P_prof = vh * vh                           # profile power base ~ v²
    P_prof_load = T * (vh * vh)                # NEW: thrust-scaled profile (v²·payload interaction)
    P_para = vh ** 3                           # parasite body drag ~ v³
    climb = np.maximum(vz, 0.0)                # climb work
    desc = np.minimum(vz, 0.0)                # descent (asymmetric coeff)

    return np.column_stack([np.ones_like(vh), P_hover, P_fwd, P_prof, P_prof_load, P_para, climb, desc])
