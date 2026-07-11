def featurize(s):
    """iter1 — BEMT momentum-theory power form (novel, non-polynomial; fit linearly).

    Hypothesis: baseline 1,v,v² is a crude parabola for the true U-shaped rotor
    power curve. Momentum/blade-element theory gives an induced branch that is
    NON-polynomial (a linear term-library can't reach it):
      - hover induced power  P_i ∝ T^1.5            (T = thrust load)
      - forward induced relief P_i ∝ T² / V          (Glauert, high-speed approx)
    with thrust T = m·(g+a_z), m = base + payload (payload dominates on M100),
    plus parasite body drag ∝ v³ and an asymmetric climb/descent split.
    """
    g = 9.81
    m0 = 2.4                                  # DJI M100 base mass (kg)
    m = m0 + s["payload"] / 1000.0            # payload grams -> kg; effective mass
    az_eff = np.maximum(g + s["a_z"], 0.0)    # gravity-removed a_z (~0 hover) -> total vertical accel; clamp >=0
    vh = s["v_h"]
    vz = s["v_z"]
    T = m * az_eff                            # instantaneous thrust load (N)
    Veff = np.sqrt(vh * vh + 1.0)             # regularized airspeed (hover -> 1)

    P_hover = T ** 1.5                        # hover induced power (momentum theory) — NON-poly
    P_fwd = (T * T) / Veff                    # forward-flight induced relief (Glauert) — reciprocal, NON-poly
    P_prof = vh * vh                          # profile power (advance-ratio ~ v²)
    P_para = vh ** 3                          # parasite body drag ~ v³
    climb = np.maximum(vz, 0.0)               # climb work
    desc = np.minimum(vz, 0.0)               # descent (separate coeff -> asymmetry)

    return np.column_stack([np.ones_like(vh), P_hover, P_fwd, P_prof, P_para, climb, desc])
