# best_featurize.py — modelo_zoo 排行榜最优(Tseng 2022(9项多项式), ARE 1.90%, arXiv:2206.01609)
# -*- coding: utf-8 -*-
# 自包含 featurize 源码,由 energy_model_zoo.export_model_src("tseng_2022") 生成
# 模型: Tseng 2022(9项多项式)  来源: arXiv:2206.01609
import numpy as np
G = 9.81
M0 = 2.4

def _kinematics(s):
    """从原始 state 提取运动学派生量,所有模型共享。"""
    vh  = s["v_h"]
    vz  = s["v_z"]
    az  = s["a_z"]           # 已含重力(来自 load_m100: a_z+9.81)
    pay = s["payload"]       # grams
    m   = M0 + pay / 1000.0
    T   = m * np.maximum(az, 0.01)   # 推力负载 (N),令 >= 微小正值避免除零
    Veff = np.sqrt(vh * vh + 1.0)    # 正则化空速(hover=1)
    return dict(vh=vh, vz=vz, az=az, pay=pay, m=m, T=T, Veff=Veff,
                omega=s.get("omega", np.zeros_like(vh)),
                wind=s.get("wind", np.zeros_like(vh)))


def featurize_tseng2022(s):
    k = _kinematics(s)
    vh, vz = k["vh"], k["vz"]
    a_h = np.abs(k.get("az", -G) - G)  # 水平加速度幅值(近似)
    m = k["m"]
    wind = k["wind"]
    return np.column_stack([np.ones_like(vh),
                            vh, vh**2, vh**3,
                            vz, vz**2, vz**3,
                            a_h, m, wind])


# sandbox 要求函数名为 featurize
featurize = featurize_tseng2022
