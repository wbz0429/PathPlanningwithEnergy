def featurize(s):
    """iter2 — full-Glauert induced bridge + air-relative-speed (wind) + thrust-coupled climb.

    Fixes three physics gaps in iter1 (all forms a linear term-library cannot reach):
      1. Induced power as ONE Glauert bridge  P_i = T² / sqrt(v_h² + T), which
         smoothly connects hover (-> T^1.5) and forward flight (-> T²/v), replacing
         iter1's two collinear high-speed-limit terms.
      2. Aero drag acts on AIR-relative speed v_air = sqrt(v_h² + wind²) (wind in
         quadrature, direction unknown): profile ~ v_air², parasite ~ v_air³, plus a
         thrust-scaled profile T·v_air² (advance-ratio × rotor loading = the v²·payload
         interaction, expressed through physics).
      3. Climb power = T·max(v_z,0) (heavier craft pay more to climb) + a separate
         asymmetric descent coeff (descent recovers little).
    """
    g = 9.81
    m0 = 2.4                                   # DJI M100 base mass (kg)
    m = m0 + s["payload"] / 1000.0             # payload grams -> kg; effective mass
    az_eff = np.maximum(g + s["a_z"], 0.0)     # gravity-removed a_z -> total vertical accel; clamp >=0
    vh = s["v_h"]
    vz = s["v_z"]
    wind = s["wind"]
    T = m * az_eff                             # instantaneous thrust load (N)

    v_air2 = vh * vh + wind * wind             # air-relative speed² (wind in quadrature)
    v_air = np.sqrt(v_air2)

    P_ind = (T * T) / np.sqrt(vh * vh + T)     # full Glauert induced bridge  (hover T^1.5 <-> fwd T²/v)  NON-poly
    P_prof = v_air2                            # profile power base ~ v_air²
    P_prof_load = T * v_air2                   # thrust-scaled profile (advance-ratio × loading)  INTERACTION
    P_para = v_air2 * v_air                    # parasite body drag ~ v_air³  NON-poly (wind-coupled)
    P_climb = T * np.maximum(vz, 0.0)          # thrust-coupled climb  INTERACTION
    P_desc = np.maximum(-vz, 0.0)              # asymmetric descent

    return np.column_stack([np.ones_like(vh), P_ind, P_prof, P_prof_load, P_para, P_climb, P_desc])
