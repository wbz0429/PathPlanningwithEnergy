def featurize(s):
    """iter4 — iter3 (kept best) + ONE non-polynomial form: momentum-theory axial induced modulation.

    Beyond a polynomial/interaction library (sqrt-of-sum). In axial climb the rotor ingests
    more mass flow, so induced velocity to make thrust T DECREASES:
        v_i = -v_z/2 + sqrt((v_z/2)² + v_i0²),   v_i0² ∝ T  (hover induced velocity²)
    The induced-power change from vertical motion is T·(v_i - v_i0); subtracting v_i0 isolates
    the vertical-motion signal and removes collinearity with the existing T^1.5 hover term.
    Everything from iter3 is unchanged (add, don't replace — iter2's lesson).
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
    P_prof_load = T * (vh * vh)                # thrust-scaled profile (v²·payload interaction)
    P_para = vh ** 3                           # parasite body drag ~ v³
    climb = np.maximum(vz, 0.0)                # climb work (asymmetric)
    desc = np.minimum(vz, 0.0)                # descent (asymmetric coeff)

    vi0_sq = 0.83 * T                          # hover induced velocity² ~ T/(2·rho·A)
    vi_mod = -0.5 * vz + np.sqrt((0.5 * vz) ** 2 + vi0_sq) - np.sqrt(vi0_sq)
    P_ind_axial = T * vi_mod                   # NEW: axial induced-power modulation — NON-poly sqrt-of-sum

    return np.column_stack([np.ones_like(vh), P_hover, P_fwd, P_prof, P_prof_load, P_para, climb, desc, P_ind_axial])
