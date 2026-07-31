# -*- coding: utf-8 -*-
"""as_published_models.py — 文献模型的 as-published(原样、零调整)版本。
───────────────────────────────────────────────────────────────────────────
判断标准:模型能否不依赖我们的数据、仅凭论文公式+平台物理常数直接预测 M100 功率。
  ✅ 可 as-published(零自由系数 / 论文报告了系数):
     - 动量理论悬停功率 P=W^1.5/√(2ρA)           [纯物理]
     - 动量理论前飞诱导 P=W·v_i                    [纯物理]
     - Tseng 简化回归 P=−2.595·va+0.197·m3+251.7  [论文报告系数]
  ❌ 只有形式、论文未公开平台标定系数(需 refit,已入 zoo):
     - Abeywardena VRS / Morbidi / Stolaroff / Dorling 的校准项
     → 这些的"原样搬"在论文上不成立,其正确前置评测= refit 版本(zoo 已做)
───────────────────────────────────────────────────────────────────────────
"""
import numpy as np

G = 9.81
RHO = 1.225                     # kg/m³,海平面空气密度
M0 = 2.4                        # DJI M100 空机质量 (kg)
ROTOR_R = 0.17                  # m, M100 桨半径≈0.17m(340mm 桨)
A_DISK = 4 * np.pi * ROTOR_R**2 # 4 旋翼总桨盘面积 ≈0.363 m²

AS_PUBLISHED = [
    # (code, name, predictor(payload_kg, state)->P_watts, 来源)
    ("momentum_hover",
     "动量理论悬停(as-published,零拟合)",
     lambda pay, s: (M0 + pay) * G * np.sqrt((M0 + pay) * G / (2 * RHO * A_DISK)),
     "动量理论 P=W^1.5/√(2ρA),代入 M100 质量+桨盘面积,无任何自由系数"),

    ("momentum_forward",
     "动量理论前飞诱导(as-published,零拟合)",
     lambda pay, s: _fwd_power(M0 + pay, s["v_h"]),
     "诱导速度 v_i 满足 v_i·√(v_h²+v_i²)=W/(2ρA);P=W·v_i。仅诱导功率(下限)"),

    ("tseng_simplified",
     "Tseng 简化回归(论文报告系数)",
     lambda pay, s: -2.595 * np.sqrt(s["v_h"]**2 + s["v_z"]**2) + 0.197 * pay + 251.7,
     "Muli/Park/Liu arXiv:2206.01609 报告的简化式 P=−2.595·va+0.197·m3+251.7。注:可能为归一化尺度,直接套用于展示跨平台迁移偏差"),
]


def _fwd_power(mass, vh):
    """动量理论前飞诱导功率。v_i = sqrt((−v²+sqrt(v⁴+4w²))/2), P = W·v_i。"""
    W = mass * G
    w = W / (2 * RHO * A_DISK)
    v2 = vh**2
    v4 = v2**2
    y = (-v2 + np.sqrt(v4 + 4 * w**2)) / 2.0
    y = np.maximum(y, 0.0)
    vi = np.sqrt(y)
    return W * vi


def predict_published(code, d, mask):
    """对 state dict 的 mask 子集给出 as-published 功率预测 (W)。"""
    s = {k: d[k][mask] for k in ("v_h", "v_z", "a_h", "a_z", "omega", "payload", "wind", "speed")}
    pay_kg = d["payload"][mask] / 1000.0
    for c, name, pred, src in AS_PUBLISHED:
        if c == code:
            return np.asarray(pred(pay_kg, s), float)
    raise KeyError(code)
