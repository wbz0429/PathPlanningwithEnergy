
def featurize(s):
    """Tseng 基线 + 爬升/下降不对称的独立非线性标定(v_z>0 与 v_z<0 分开乘幂,非对称多项式)。"""
    vh, vz = s["v_h"], s["v_z"]
    pay = s["payload"]; m = 2.4 + pay/1000.0
    az = s["a_z"]
    vh2, vh3 = vh**2, vh**3
    climb = np.maximum(vz, 0.0)
    desc  = np.minimum(vz, 0.0)
    # 非对称:爬升段给 v_z 更高阶(3次),下降段低阶(2次)——普通线性项库到不了的对称破缺
    climb3 = climb**3
    desc2  = desc**2
    return np.column_stack([np.ones_like(vh), vh, vh2, vh3, climb, climb3, desc, desc2, m, az])
