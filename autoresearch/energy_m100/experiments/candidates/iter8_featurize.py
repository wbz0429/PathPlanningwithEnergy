def featurize(s):
    """iter8 — full linear LIB + physics non-poly (T^1.5, T²/V). Decisive 'does physics beat linear?' test.

    Full polynomial LIB reaches 1.91% held-out. Add the two physics non-polynomial forms the LIB
    structurally cannot express (hover induced T^1.5, forward relief T²/V). If held-out ARE drops
    below 1.91%, the LLM's novel forms add value over ordinary linear selection; if it stays ~1.91%,
    physics adds nothing beyond flexible linear regression (strong honest boundary).
    """
    v = s["v_h"]
    vz = s["v_z"]
    pay = s["payload"]
    wind = s["wind"]
    omega = s["omega"]
    absaz = np.abs(s["a_z"])
    ah = s["a_h"]
    g = 9.81
    m = 2.4 + pay / 1000.0
    az_eff = np.maximum(g + s["a_z"], 0.0)
    T = m * az_eff
    Veff = np.sqrt(v * v + 1.0)

    return np.column_stack([
        np.ones_like(v),
        # ---- full linear LIB (what ordinary term-selection reaches: 1.91%) ----
        v, v * v, v ** 3, vz, np.maximum(0.0, vz), absaz, ah, omega, pay, wind, v * pay, v * v * pay,
        # ---- physics non-polynomial (beyond the LIB) ----
        T ** 1.5, (T * T) / Veff,
    ])
