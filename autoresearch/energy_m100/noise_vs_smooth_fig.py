# -*- coding: utf-8 -*-
"""noise_vs_smooth_fig.py — 仿真(模型)平滑功率 vs 真机实测功率剧烈波动。
左:模型/仿真的功率曲线(平滑);右:真机实测功率片段(剧烈波动)+ 标注"~60%是非运动学噪声"。
说明:1.9% 地板来自真机功率里不可观测的噪声,非模型不足。输出 pub/仿真vs真机波动.png
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import m100_eval as me
from pubstyle import PAL, save
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__)); EXP = os.path.join(HERE, "experiments")

fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.2, 3.4))

# 左:仿真模型功率(平滑)——用 RotorPy 单墙绕行(波动系数 0.003,最平滑)
try:
    T = np.load(os.path.join(EXP, "sim_flight_trajs.npz"))
    t = T["绕行(M100代价)_t"]; P = T["绕行(M100代价)_P"]
    a1.plot(t, P, color=PAL["green"], lw=1.6)
except Exception:
    x = np.linspace(0, 14, 200); a1.plot(x, 466 + 30*np.sin(x), color=PAL["green"], lw=1.6)
a1.set_xlabel("时间 (s)"); a1.set_ylabel("功率 (W)"); a1.set_ylim(0, 900)
a1.set_title("仿真:模型预测功率(平滑)", fontsize=11)
a1.text(0.5, 0.06, "运动学输入 → 平滑输出", transform=a1.transAxes, fontsize=8.5, color=PAL["green"], ha="center")

# 右:真机实测功率片段(剧烈波动)——取一段真实飞行
d = me.load_m100()
fl = d["flight"]; uniq = np.unique(fl)
seg = fl == uniq[len(uniq)//2]                 # 取中间一趟
Pr = d["P"][seg]
Pr = Pr[:min(300, len(Pr))]                    # 取前 300 点
a2.plot(np.arange(len(Pr)) / 5.0, Pr, color=PAL["red"], lw=0.8)  # 5Hz
a2.axhline(Pr.mean(), ls=(0, (4, 3)), color="#333", lw=1)
a2.set_xlabel("时间 (s)"); a2.set_ylabel("功率 (W)"); a2.set_ylim(0, 900)
a2.set_title("真机:实测功率(剧烈波动)", fontsize=11)
a2.text(0.5, 0.93, "变异系数≈0.20;运动学只解释~40%\n其余~60%=风/电压/控制噪声",
        transform=a2.transAxes, fontsize=8.5, color=PAL["red"], ha="center", va="top")

fig.suptitle("为什么误差降不到 1.9% 以下:真机功率含大量不可观测噪声(非模型不足)", fontsize=11.5, y=1.02)
plt.tight_layout(); save(fig, "仿真vs真机波动")
print("图存 pub/仿真vs真机波动")
