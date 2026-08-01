
def featurize(s):
    """Tseng 起点 + 分段:低速(悬停/慢速)与巡航分段不同律(v_crit=2.5m/s)。
    线性多项式无法表达的 piecewise 结构——每个区间独立缩放。"""
    vh = s["v_h"]; vz = s["v_z"]
    pay = s["payload"]; m = 2.4 + pay/1000.0
    az = s["a_z"]
    vc = 2.5
    hover = vh < vc                     # 低速区
    cruise = ~hover                     # 巡航区
    vh2, vh3 = vh**2, vh**3
    climb = np.maximum(vz, 0.0); desc = np.minimum(vz, 0.0)
    # 分段门控:同一变量在不同区间给不同列(系数独立) → 非线性分段
    vh_h = np.where(hover, vh, 0.0); vh2_h = np.where(hover, vh2, 0.0)
    vh_c = np.where(cruise, vh, 0.0); vh2_c = np.where(cruise, vh2, 0.0); vh3_c = np.where(cruise, vh3, 0.0)
    hover_flag = hover.astype(float)
    return np.column_stack([np.ones_like(vh), vh_h, vh2_h, hover_flag,
                            vh_c, vh2_c, vh3_c, climb, desc, pay, m])
