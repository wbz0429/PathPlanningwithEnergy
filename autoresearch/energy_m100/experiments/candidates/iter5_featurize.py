def featurize(s):
    """iter5 EXPLORE (episode 1) — air-relative speed for aero drag (uses anemometer wind).

    Literature (Rodrigues 2021 dataset; Tseng model; arXiv:2209.04128) says wind is a key
    predictor and this dataset uniquely measured it. Structural hypothesis: aero power depends
    on AIR-relative speed v_air = sqrt(v_h² + wind²), NOT ground speed. Swap ground->air-relative
    throughout iter3's drag terms (profile, thrust-scaled profile, parasite, forward-relief);
    keep hover induced T^1.5 and asymmetric climb/descent. Single clean structural test of "is
    the gap to the ~1.9% polynomial frontier just wind?".
    """
    g = 9.81
    m0 = 2.4                                   # DJI M100 base mass (kg)
    m = m0 + s["payload"] / 1000.0             # payload grams -> kg; effective mass
    az_eff = np.maximum(g + s["a_z"], 0.0)     # gravity-removed a_z -> total vertical accel; clamp >=0
    vh = s["v_h"]
    vz = s["v_z"]
    wind = s["wind"]
    T = m * az_eff                             # instantaneous thrust load (N)

    v_air2 = vh * vh + wind * wind             # AIR-relative airspeed² (anemometer wind, quadrature)
    v_air = np.sqrt(v_air2)
    Veff = np.sqrt(v_air2 + 1.0)               # regularized air-relative airspeed

    P_hover = T ** 1.5                         # hover induced power (momentum theory) — NON-poly
    P_fwd = (T * T) / Veff                     # forward induced relief on AIR-relative speed — reciprocal
    P_prof = v_air2                            # profile power ~ v_air²
    P_prof_load = T * v_air2                   # thrust-scaled profile, air-relative (interaction)
    P_para = v_air2 * v_air                    # parasite body drag ~ v_air³
    climb = np.maximum(vz, 0.0)                # climb work (asymmetric)
    desc = np.minimum(vz, 0.0)                # descent (asymmetric coeff)

    return np.column_stack([np.ones_like(vh), P_hover, P_fwd, P_prof, P_prof_load, P_para, climb, desc])
