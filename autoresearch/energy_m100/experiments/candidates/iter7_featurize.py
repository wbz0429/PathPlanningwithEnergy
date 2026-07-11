def featurize(s):
    """iter7 — iter6 (best) + payload×climb interaction (weight-scaled climb work; beyond LIB).

    Best is at the linear frontier (1.93% ~ LIB 1.91%). To beat it needs a form the polynomial
    LIB lacks. Climb work = weight × climb-rate = (m0+payload)·g·v_z, so heavier craft pay
    more per unit climb. The LIB has bare climb+ and bare payload but NOT their product; add
    payload×climb+ (interaction = preferred novel form). Tests whether a beyond-LIB interaction
    pushes below ~1.9%.
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
    P_climb_load = pay * climb                 # NEW: payload×climb (weight-scaled climb work) — beyond LIB

    return np.column_stack([np.ones_like(vh), P_hover, P_fwd, P_prof, P_prof_load, P_para, climb, desc, pay, P_climb_load])
