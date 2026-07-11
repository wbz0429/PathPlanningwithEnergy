def featurize(s):
    """默认 = 稳态 BEMT U 形:P ~ 1 + v + v²(仅水平速度)。"""
    return np.column_stack([np.ones_like(s["v_h"]), s["v_h"], s["v_h"] ** 2])