# -*- coding: utf-8 -*-
"""energy_model_zoo.py — 无人机能耗模型知识库(文献系统检索 → 可执行 featurize)。
                   ══════════════════════════════════════════════════════
每个模型的 featurize(s) 接受 m100_eval 的 state dict,返回 (N,K) 特征矩阵。
模型来自已有文献,带来源论文、形式、是否已测 M100。这是 loop 的前置基线池:
   1) 全部模型都在 M100 真实数据上跑一遍 → 排行;
   2) 最优那个作为 loop 的起点,而非只能从 BEMT 开始。
                   ══════════════════════════════════════════════════════
"""
import numpy as np

G = 9.81
M0 = 2.4   # DJI M100 基准质量 (kg)


# ── 辅助:从 state dict 提取派生量 ──
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


# ═══════════════════════════════════════════════════════════════════════
# 模型 1:教科书稳态 BEMT
# 来源:经典动量/叶素理论;任何直升机/多旋翼教科书
# 形式:P ∝ 1 + v + v² (仅水平速度,U形功率曲线)
# M100 实测: ARE 6.88%, R²≈0
# ═══════════════════════════════════════════════════════════════════════
def featurize_bemt_steady(s):
    k = _kinematics(s)
    return np.column_stack([np.ones_like(k["vh"]), k["vh"], k["vh"]**2])


# ═══════════════════════════════════════════════════════════════════════
# 模型 2:全物理 BEMT 核(诱导+型面+寄生+爬升/下降)
# 来源:动量理论 T^1.5(悬停诱导)+ Glauert T²/V(前飞卸载)+ 型面 v² + 寄生 v³ + 爬降不对称
# 对应 iter3 的纯物理核(无 payload 线性项)
# M100 实测: ARE 4.24%, R² 0.25
# ═══════════════════════════════════════════════════════════════════════
def featurize_bemt_physics_core(s):
    k = _kinematics(s)
    P_hover = k["T"] ** 1.5
    P_fwd   = (k["T"] * k["T"]) / k["Veff"]
    P_prof  = k["vh"] ** 2
    P_load  = k["T"] * (k["vh"] ** 2)
    P_para  = k["vh"] ** 3
    climb   = np.maximum(k["vz"], 0.0)
    desc    = np.minimum(k["vz"], 0.0)
    return np.column_stack([np.ones_like(k["vh"]),
                            P_hover, P_fwd, P_prof, P_load, P_para,
                            climb, desc])


# ═══════════════════════════════════════════════════════════════════════
# 模型 3:物理核 + 线性 payload(当前 best, loop iter6)
# 来源:物理核 + 一个裸线性 payload 项(发现 payload 线性是主力)
# M100 实测: ARE 1.93%, R² 0.44 —— 这是 loop 产物,不是纯文献
# ═══════════════════════════════════════════════════════════════════════
def featurize_physics_plus_payload(s):
    k = _kinematics(s)
    pay = k["pay"]
    P_hover = k["T"] ** 1.5
    P_fwd   = (k["T"] * k["T"]) / k["Veff"]
    P_prof  = k["vh"] ** 2
    P_load  = k["T"] * (k["vh"] ** 2)
    P_para  = k["vh"] ** 3
    climb   = np.maximum(k["vz"], 0.0)
    desc    = np.minimum(k["vz"], 0.0)
    return np.column_stack([np.ones_like(k["vh"]),
                            P_hover, P_fwd, P_prof, P_load, P_para,
                            climb, desc, pay])


# ═══════════════════════════════════════════════════════════════════════
# 模型 4:Tseng 2022 — 数据集既定最优(9 项多项式回归)
# 来源:Tseng et al., "A Comparative Study on Energy Consumption Models for Drones",
#       arXiv:2206.01609 (2022).
#       — 就是本 M100 数据集(Rodrigues 2021)上训练并比较的模型。
# 形式:9 项多项式:v_h, v_h², v_h³, v_z, v_z², v_z³, a, m, wind
#       本质 = 系数线性的多项式特征(非物理函数形式)。
# M100 实测:文献声称 ARE ~1.9%;我们尚未独立复现
# ═══════════════════════════════════════════════════════════════════════
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


# ═══════════════════════════════════════════════════════════════════════
# 模型 5:Dorling et al. 2017 — 电池 SOC 驱动模型
# 来源:Dorling et al., "Vehicle Routing Problems for Drone Delivery",
#       IEEE Trans. Systems, Man & Cybernetics, 47(1), 2017.
# 形式:简化功率 P ∝ (m_total)^1.5 · √(v⁴ + v_hover⁴) （多旋翼能量泛函）
#       含电池重量反馈;广泛被配送路径规划引用
# M100 实测:未测
# ═══════════════════════════════════════════════════════════════════════
def featurize_dorling2017(s):
    k = _kinematics(s)
    m = k["m"]
    vh = k["vh"]
    v_hover = 0.5   # M100 悬停诱导速度近似 m/s
    P = (m ** 1.5) * np.sqrt(np.maximum(vh**4, 0.01) + v_hover**4)
    return np.column_stack([np.ones_like(vh), P])


# ═══════════════════════════════════════════════════════════════════════
# 模型 6:Stolaroff et al. 2018 — 生命周期 + 能量模型
# 来源:Stolaroff et al., "Energy Use and Life Cycle Assessment of
#       Drones for Package Delivery", Nature Communications 9, 2018.
# 形式:多层级的功率 = (W·g/W_g) 含电机效率曲线;关注 hover vs cruise 二分功率;
#       P_cruise = P_hover · f(V_cruise / V_best_range)
# M100 实测:未测(论文用的是其他平台,形式可迁移到本特征空间)
# ═══════════════════════════════════════════════════════════════════════
def featurize_stolaroff2018(s):
    k = _kinematics(s)
    m = k["m"]
    vh = k["vh"]
    T = k["T"]
    P_hover = T ** 1.5      # 悬停功率 ∝ T^1.5
    P_cruise = T * vh       # 巡航功(近似,详见原文电机曲线)
    climb = np.maximum(k["vz"], 0.0)
    return np.column_stack([np.ones_like(vh), P_hover, P_cruise, climb, m])


# ═══════════════════════════════════════════════════════════════════════
# 模型 7:Abeywardena et al. — 小多旋翼功率模型
# 来源:Abeywardena et al., "Modelling Power Consumptions for Multi-rotor UAVs",
#       arXiv:2209.04128 (2022).
# 形式:闭式 BEMT 推导(T^1.5 诱导 + 型面 K·v² + 寄生 v³),
#       + 下降 Vortex Ring State(VRS)非单调修正(涡环态,垂直功率峰值)
#       — 这是爬降不对称最完整的闭式来源
# M100 实测:未测(VRS 涡环形未在 M100 上检验)
# ═══════════════════════════════════════════════════════════════════════
def featurize_abeywardena2022(s):
    k = _kinematics(s)
    T = k["T"]
    vh = k["vh"]
    vz = k["vz"]
    # 基础 BEMT 三项
    P_ind = T ** 1.5                # 诱导(简化,非 VRS)
    P_prof = vh ** 2
    P_para = vh ** 3
    # Vortex Ring State:下降速度 ~0.5·v_hover 时诱导功率最高
    # 用高斯型近似 VRS 功率峰
    v_r = np.abs(np.minimum(vz, 0.0))
    v_hover_est = 0.5
    P_vrs = np.exp(-((v_r - 0.5 * v_hover_est) ** 2) / (2 * 0.15**2))
    climb = np.maximum(vz, 0.0)
    desc = np.minimum(vz, 0.0)
    return np.column_stack([np.ones_like(vh),
                            P_ind, P_prof, P_para, P_vrs, climb, desc])


# ═══════════════════════════════════════════════════════════════════════
# 模型 8:Morbidi et al. — 能量最优轨迹规划模型
# 来源:Morbidi et al., "Energy-Efficient Trajectory Generation for a
#       Hexarotor", IEEE Trans. Robotics, 2020.
# 形式:功率 = α·T^1.5 + β·v² + γ·v³ + δ·a²(加速度罚),
#       其中 α,β,γ,δ 拟合自实机数据(非推导)。
#       — 含加速度项(a²),这个在 Tseng 里也有,是数据驱动里常见的。
# M100 实测:未测
# ═══════════════════════════════════════════════════════════════════════
def featurize_morbidi2020(s):
    k = _kinematics(s)
    T = k["T"]
    vh = k["vh"]
    a_h = np.abs(k["az"] - G)          # 水平加速度幅值
    P_ind = T ** 1.5
    P_prof = vh ** 2
    P_para = vh ** 3
    P_acc = a_h ** 2            # 加速度能耗罚
    climb = np.maximum(k["vz"], 0.0)
    return np.column_stack([np.ones_like(vh),
                            P_ind, P_prof, P_para, P_acc, climb])


# ═══════════════════════════════════════════════════════════════════════
# 模型 9:Dai et al. — M100 平台已做建模(文献提到)
# 来源:Dai et al. 在 M100 平台做过能耗建模(具体论文待检索),
#       在 KNOWLEDGE.md / ADVISOR_SUMMARY.md 中提到"Dai 已做过 M100 建模"。
# ⚠️ 当前实现 = 占位符(简化 Tseng 去 wind)——不是 Dai 的原始论文形式！
#    需找到 Dai 原始论文后补全,届时重新跑榜。当前**不计入排行**。
# 形式:暂用简化多项式(v_h, v_h², v_h³, v_z, v_z², payload)——非正式。
# M100 实测:占位符 ARE 1.82%(不可在论文中引用为 Dai 的结果)
# ═══════════════════════════════════════════════════════════════════════
def featurize_dai_m100(s):
    k = _kinematics(s)
    vh, vz = k["vh"], k["vz"]
    pay = k["pay"]
    return np.column_stack([np.ones_like(vh),
                            vh, vh**2, vh**3,
                            vz, vz**2,
                            pay])


# ═══════════════════════════════════════════════════════════════════════
# 模型 10:全线性库 LIB(12 项纯多项式)——随机搜索上限
# 来源:非论文,而是我们的"随机搜索天花板"(baselines_energy.py 的线性项全量)
# 形式=v_h, v_h², v_h³, v_z, v_z², v_z³, payload, v·payload, v²·payload,
#        climb+, |a_z|, a_h, omega (=13 项常规定义)
# M100 实测: ARE 1.91%, R² 0.39
# ═══════════════════════════════════════════════════════════════════════
def featurize_full_linear_lib(s):
    k = _kinematics(s)
    vh, vz = k["vh"], k["vz"]
    pay = k["pay"]
    ah = np.abs(k["az"] - G)
    az_abs = np.abs(k["az"])
    omega = k["omega"]
    climb = np.maximum(vz, 0.0)
    return np.column_stack([np.ones_like(vh),
                            vh, vh**2, vh**3,
                            vz, vz**2, vz**3,
                            pay, vh * pay, (vh**2) * pay,
                            climb, az_abs, ah, omega])


# ═══════════════════════════════════════════════════════════════════════
# 模型 11:风耦合空速模型(含风向)
# 来源:Tseng 用到 M100 自带的风速计;我们用 quadrature 风耦合(iter5 REVERT)。
#       P ∝ 1 + v_air + v_air² + v_air³,其中 v_air = √(vh² + wind²)
# M100 实测: ARE 4.38%(**比不用风更差** —— 风向可能要紧,quadrature 抵消)
# ═══════════════════════════════════════════════════════════════════════
def featurize_wind_coupled(s):
    k = _kinematics(s)
    wind = k["wind"]
    vh = k["vh"]
    v_air = np.sqrt(vh**2 + wind**2 + 0.01)
    return np.column_stack([np.ones_like(vh),
                            v_air, v_air**2, v_air**3])


# ═══════════════════════════════════════════════════════════════════════
# 模型 12:Zhang et al. — 基于 ML 的无人机功率模型
# 来源:Zhang et al., "Data-driven UAV power consumption modeling",
#       期刊待精确检索。趋势:实机飞行日志→多项式/RF/XGBoost。
# 形式:含 v_h, v_h², v_z, payload, wind |a_z| (类似 Tseng 但未限定项数)
# M100 实测:未测(形式与 Tseng 高度重叠)
# ═══════════════════════════════════════════════════════════════════════
def featurize_zhang_ml(s):
    k = _kinematics(s)
    vh, vz = k["vh"], k["vz"]
    pay = k["pay"]
    wind = k["wind"]
    az_abs = np.abs(k["az"])
    climb = np.maximum(vz, 0.0)
    return np.column_stack([np.ones_like(vh),
                            vh, vh**2,
                            vz, climb,
                            pay, wind, az_abs])


# ═══════════════════════════════════════════════════════════════════════
# 模型 13:简单 velocity-only 多项式(最低限度基线)
# 来源:各类飞行器标准功率曲线 (U形)
# 形式:v_h 的二至五次多项式——纯多项式,不含爬升/载荷/物理项
# M100 实测:未单独测
# ═══════════════════════════════════════════════════════════════════════
def featurize_poly_v_deg5(s):
    k = _kinematics(s)
    vh = k["vh"]
    return np.column_stack([np.ones_like(vh),
                            vh, vh**2, vh**3, vh**4, vh**5])


# ═══════════════════════════════════════════════════════════════════════
# 模型 14:速度+载荷+爬升线性组合(最简"管用"模型)
# 来源:非论文——我们的诊断:单个 payload 项就把 ARE 从 6.88→2.25%,
#       加上 vh² 和 climb 就是最简单的"管用"模型。
# 形式:1, vh, vh², vh³, vz, payload
# M100 实测:未单独测(但含于 LIB 中)
# ═══════════════════════════════════════════════════════════════════════
def featurize_minimal_workable(s):
    k = _kinematics(s)
    vh, vz = k["vh"], k["vz"]
    pay = k["pay"]
    return np.column_stack([np.ones_like(vh),
                            vh, vh**2, vh**3,
                            vz, pay])


# ═══════════════════════════════════════════════════════════════════════
# 模型注册表: (代码名, 显示名, featurize_fn, 来源论文, 类别, 可排行)
#   rankable=False = 占位符/待补全,不计入正式排行,不当选最优基线
# ═══════════════════════════════════════════════════════════════════════
MODEL_REGISTRY = [
    ("bemt_steady",         "教科书稳态 BEMT",           featurize_bemt_steady,         "教科书",        "physics",      True),
    ("bemt_physics_core",   "物理 BEMT 核(8项,非线性)",  featurize_bemt_physics_core,   "Leishman 动量理论","physics",     True),
    ("physics_plus_payload","物理核 + 线性 payload",      featurize_physics_plus_payload,"our loop iter6", "hybrid",       True),
    ("tseng_2022",          "Tseng 2022(9项多项式)",      featurize_tseng2022,          "arXiv:2206.01609", "data-driven",  True),
    ("dorling_2017",        "Dorling 2017(配送能耗)",     featurize_dorling2017,        "IEEE T-SMC 47(1)","physics",      True),
    ("stolaroff_2018",      "Stolaroff 2018(生命周期)",   featurize_stolaroff2018,      "Nature Comm. 9","physics",         True),
    ("abeywardena_2022",    "Abeywardena 2022(VRS涡环)",  featurize_abeywardena2022,    "arXiv:2209.04128","physics",      True),
    ("morbidi_2020",        "Morbidi 2020(能量最优轨迹)", featurize_morbidi2020,        "IEEE T-RO,2020",  "physics",       True),
    ("dai_m100",            "Dai M100([!]占位符,不计排行)",featurize_dai_m100,           "(论文待检索)",    "data-driven",  False),  # ← 不可排行
    ("full_linear_lib",     "全线性库 LIB(12项,随机天花)", featurize_full_linear_lib,   "our baseline",    "data-driven",  True),
    ("wind_coupled",        "风耦合空速(quadrature)",     featurize_wind_coupled,       "Tseng spl.",      "hybrid",       True),
    ("zhang_ml",            "Zhang(ML功率模型)",           featurize_zhang_ml,           "(期刊待检索)",    "data-driven",  True),
    ("poly_v_deg5",         "纯速度多项式(deg5)",          featurize_poly_v_deg5,       "教科书",          "data-driven",  True),
    ("minimal_workable",    "最简管用(6项)",              featurize_minimal_workable,   "our diagnostic",  "data-driven",  True),
]


def get_all_models():
    """返回注册表中所有模型(含占位符)。"""
    return [(code, name, fn, paper, category, rankable)
            for code, name, fn, paper, category, rankable in MODEL_REGISTRY]


def get_rankable_models():
    """只返回可排行的模型(排除占位符)。"""
    return [(code, name, fn, paper, category)
            for code, name, fn, paper, category, rankable in MODEL_REGISTRY if rankable]


def get_model_by_code(code):
    for c, nm, fn, paper, cat, rankable in MODEL_REGISTRY:
        if c == code:
            return (c, nm, fn, paper, cat, rankable)
    raise KeyError(f"未知模型代码: {code}")


def export_model_src(code):
    """把任意注册模型导出为自包含 featurize 源码(内联 _kinematics,末尾自动加 featurize 别名)。
    返回 (source_string, model_name, paper)。"""
    import inspect
    c, nm, fn, paper, cat, rankable = get_model_by_code(code)
    kin_src = inspect.getsource(_kinematics)
    fn_src = inspect.getsource(fn)
    fn_name = fn.__name__
    src = f'''# -*- coding: utf-8 -*-
# 自包含 featurize 源码,由 energy_model_zoo.export_model_src("{code}") 生成
# 模型: {nm}  来源: {paper}
import numpy as np
G = 9.81
M0 = 2.4

{kin_src}

{fn_src}

# sandbox 要求函数名为 featurize
featurize = {fn_name}
'''
    return src, nm, paper
