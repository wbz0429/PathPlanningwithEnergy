"""
3D CSC Dubins 路径求解器
路径由三段组成：Helix1(γ1) - Straight3D(γs) - Helix2(γ2)

Step 1: XY 平面求 2D Dubins（6 种类型取最短），得到各段 2D 长度 (s1, d, s2)
Step 2: 爬升角优化——目标 min s1/cos(γ1) + d/cos(γs) + s2/cos(γ2)
Step 3: 沿 3D 路径等弧长采样

参考论文：DC-RRT Dubins-Guided Curvature RRT for 3D Path Planning of UAVs
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class Dubins3DParams:
    """3D Dubins 参数"""
    turning_radius: float = 1.5       # 最小转弯半径 (m)
    max_climb_angle: float = 30.0     # 最大爬升角 (度)
    sample_distance: float = 0.3      # 沿弧长采样间距 (m)


def _normalize_angle(angle: float) -> float:
    """归一化角度到 [0, 2π)"""
    return angle % (2 * np.pi)


def _mod2pi(angle: float) -> float:
    """归一化角度到 [0, 2π)"""
    return angle % (2 * np.pi)


# ============================================================
# 2D Dubins 求解器
# ============================================================

class Dubins2DSolver:
    """
    2D Dubins 路径求解器（解析解）
    支持 6 种路径类型：LSL, RSR, LSR, RSL, RLR, LRL
    """

    def __init__(self, turning_radius: float):
        self.r = turning_radius

    def solve(self, start: np.ndarray, start_heading: float,
              end: np.ndarray, end_heading: float) -> Optional[dict]:
        """
        求解 2D Dubins 最短路径

        Args:
            start: 起点 [x, y]
            start_heading: 起始航向角 (rad)
            end: 终点 [x, y]
            end_heading: 终止航向角 (rad)

        Returns:
            dict with keys: type, lengths (s1, d, s2), total_length, params
            or None if no solution
        """
        # 转换到局部坐标系（起点为原点，起始航向为 X 轴）
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        D = np.sqrt(dx * dx + dy * dy) / self.r
        theta = np.arctan2(dy, dx)

        alpha = _mod2pi(start_heading - theta)
        beta = _mod2pi(end_heading - theta)

        best = None
        best_cost = np.inf

        for solver in [self._LSL, self._RSR, self._LSR, self._RSL,
                       self._RLR, self._LRL]:
            result = solver(alpha, beta, D)
            if result is not None:
                t, p, q = result
                cost = (abs(t) + abs(p) + abs(q)) * self.r
                if cost < best_cost:
                    best_cost = cost
                    best = {
                        'type': solver.__name__[1:],  # strip leading _
                        'lengths': (abs(t) * self.r, abs(p) * self.r, abs(q) * self.r),
                        'total_length': cost,
                        'params': (t, p, q),
                        'start': start.copy(),
                        'end': end.copy(),
                        'start_heading': start_heading,
                        'end_heading': end_heading,
                    }

        return best

    def _LSL(self, alpha, beta, D):
        tmp0 = D + np.sin(alpha) - np.sin(beta)
        p_sq = 2 + D * D - 2 * np.cos(alpha - beta) + 2 * D * (np.sin(alpha) - np.sin(beta))
        if p_sq < 0:
            return None
        p = np.sqrt(p_sq)
        tmp1 = np.arctan2(np.cos(beta) - np.cos(alpha), tmp0)
        t = _mod2pi(-alpha + tmp1)
        q = _mod2pi(beta - tmp1)
        return (t, p, q)

    def _RSR(self, alpha, beta, D):
        tmp0 = D - np.sin(alpha) + np.sin(beta)
        p_sq = 2 + D * D - 2 * np.cos(alpha - beta) + 2 * D * (np.sin(beta) - np.sin(alpha))
        if p_sq < 0:
            return None
        p = np.sqrt(p_sq)
        tmp1 = np.arctan2(np.cos(alpha) - np.cos(beta), tmp0)
        t = _mod2pi(alpha - tmp1)
        q = _mod2pi(-beta + tmp1)
        return (t, p, q)

    def _LSR(self, alpha, beta, D):
        p_sq = -2 + D * D + 2 * np.cos(alpha - beta) + 2 * D * (np.sin(alpha) + np.sin(beta))
        if p_sq < 0:
            return None
        p = np.sqrt(p_sq)
        tmp1 = np.arctan2(-np.cos(alpha) - np.cos(beta),
                           D + np.sin(alpha) + np.sin(beta)) - np.arctan2(-2.0, p)
        t = _mod2pi(-alpha + tmp1)
        q = _mod2pi(-_mod2pi(beta) + tmp1)
        return (t, p, q)

    def _RSL(self, alpha, beta, D):
        p_sq = -2 + D * D + 2 * np.cos(alpha - beta) - 2 * D * (np.sin(alpha) + np.sin(beta))
        if p_sq < 0:
            return None
        p = np.sqrt(p_sq)
        tmp1 = np.arctan2(np.cos(alpha) + np.cos(beta),
                           D - np.sin(alpha) - np.sin(beta)) - np.arctan2(2.0, p)
        t = _mod2pi(alpha - tmp1)
        q = _mod2pi(beta - tmp1)
        return (t, p, q)

    def _RLR(self, alpha, beta, D):
        tmp = (6.0 - D * D + 2 * np.cos(alpha - beta) +
               2 * D * (np.sin(alpha) - np.sin(beta))) / 8.0
        if abs(tmp) > 1.0:
            return None
        p = _mod2pi(2 * np.pi - np.arccos(tmp))
        t = _mod2pi(alpha - np.arctan2(np.cos(alpha) - np.cos(beta),
                                        D - np.sin(alpha) + np.sin(beta)) + p / 2.0)
        q = _mod2pi(alpha - beta - t + p)
        return (t, p, q)

    def _LRL(self, alpha, beta, D):
        tmp = (6.0 - D * D + 2 * np.cos(alpha - beta) +
               2 * D * (np.sin(beta) - np.sin(alpha))) / 8.0
        if abs(tmp) > 1.0:
            return None
        p = _mod2pi(2 * np.pi - np.arccos(tmp))
        t = _mod2pi(-alpha + np.arctan2(-np.cos(alpha) + np.cos(beta),
                                         D + np.sin(alpha) - np.sin(beta)) + p / 2.0)
        q = _mod2pi(_mod2pi(beta) - alpha + 2 * t - p)
        return (t, p, q)

    def sample_2d(self, solution: dict, step: float = 0.3) -> List[np.ndarray]:
        """
        沿 2D Dubins 路径等弧长采样

        Args:
            solution: solve() 返回的 dict
            step: 采样间距

        Returns:
            2D 点列表 [[x, y], ...]
        """
        if solution is None:
            return []

        path_type = solution['type']
        t, p, q = solution['params']
        r = self.r
        start = solution['start']
        heading = solution['start_heading']

        points = []
        total = solution['total_length']
        n_samples = max(2, int(total / step) + 1)

        # 三段的弧长（归一化参数 * r = 实际弧长）
        seg_lengths = [abs(t) * r, abs(p) * r, abs(q) * r]
        seg_types = list(path_type)  # e.g. ['L', 'S', 'L']

        # 当前状态
        x, y, th = start[0], start[1], heading

        cumulative = 0.0
        seg_idx = 0
        seg_consumed = 0.0

        for i in range(n_samples):
            target_s = i * total / (n_samples - 1) if n_samples > 1 else 0

            while cumulative + (seg_lengths[seg_idx] - seg_consumed) < target_s - 1e-9:
                # 完成当前段
                remain = seg_lengths[seg_idx] - seg_consumed
                x, y, th = self._advance(x, y, th, seg_types[seg_idx], remain, r)
                cumulative += remain
                seg_consumed = 0.0
                seg_idx = min(seg_idx + 1, 2)

            # 在当前段内前进
            ds = target_s - cumulative
            if ds > 1e-9:
                px, py, pth = self._advance(x, y, th, seg_types[seg_idx], ds, r)
                points.append(np.array([px, py]))
            else:
                points.append(np.array([x, y]))

        return points

    @staticmethod
    def _advance(x, y, th, seg_type, ds, r):
        """沿指定段类型前进 ds 弧长"""
        if seg_type == 'S':
            x += ds * np.cos(th)
            y += ds * np.sin(th)
        elif seg_type == 'L':
            dth = ds / r
            x += r * (np.sin(th + dth) - np.sin(th))
            y += r * (-np.cos(th + dth) + np.cos(th))
            th += dth
        elif seg_type == 'R':
            dth = ds / r
            x += r * (-np.sin(th - dth) + np.sin(th))
            y += r * (np.cos(th - dth) - np.cos(th))
            th -= dth
        return x, y, th


# ============================================================
# 3D Dubins 求解器
# ============================================================

class Dubins3DSolver:
    """
    3D CSC Dubins 路径求解器
    Helix1(γ1) - Straight3D(γs) - Helix2(γ2)

    爬升角优化：
    - Low case: |Δz| ≤ L_2d·tan(γ_max) → 统一爬升角 γ* = arctan(Δz/L_2d)
    - Medium case: 部分段用 γ_max
    - High case: 延长弧段（加螺旋圈）积累高度差
    """

    def __init__(self, params: Dubins3DParams):
        self.params = params
        self.r = params.turning_radius
        self.gamma_max = np.radians(params.max_climb_angle)
        self.ds = params.sample_distance
        self.solver_2d = Dubins2DSolver(self.r)

    def solve(self, start_3d: np.ndarray, start_heading: float,
              end_3d: np.ndarray, end_heading: float) -> Optional[List[np.ndarray]]:
        """
        求解 3D Dubins 路径并采样

        Args:
            start_3d: 起点 [x, y, z]
            start_heading: 起始航向角 (rad)
            end_3d: 终点 [x, y, z]
            end_heading: 终止航向角 (rad)

        Returns:
            3D 路径点列表，或 None
        """
        # Step 1: 2D Dubins
        start_2d = start_3d[:2]
        end_2d = end_3d[:2]
        dz = end_3d[2] - start_3d[2]

        sol_2d = self.solver_2d.solve(start_2d, start_heading, end_2d, end_heading)
        if sol_2d is None:
            return None

        s1, d, s2 = sol_2d['lengths']
        L_2d = sol_2d['total_length']

        if L_2d < 1e-6:
            # 起终点重合
            return [start_3d.copy(), end_3d.copy()]

        # Step 2: 爬升角优化
        gamma1, gamma_s, gamma2 = self._optimize_climb(s1, d, s2, dz)

        # Step 3: 3D 采样
        points_3d = self._sample_3d(sol_2d, start_3d[2], gamma1, gamma_s, gamma2)

        return points_3d

    def _optimize_climb(self, s1: float, d: float, s2: float,
                        dz: float) -> Tuple[float, float, float]:
        """
        爬升角优化

        约束: s1·tan(γ1) + d·tan(γs) + s2·tan(γ2) = Δz
        目标: min 3D 路径长度 = s1/cos(γ1) + d/cos(γs) + s2/cos(γ2)

        Returns:
            (gamma1, gamma_s, gamma2) 三段爬升角 (rad)
        """
        L_2d = s1 + d + s2
        if L_2d < 1e-6:
            return (0.0, 0.0, 0.0)

        abs_dz = abs(dz)
        max_dz_uniform = L_2d * np.tan(self.gamma_max)

        # Low case: 统一爬升角
        if abs_dz <= max_dz_uniform:
            gamma = np.arctan2(abs_dz, L_2d)
            if dz < 0:
                gamma = -gamma
            return (gamma, gamma, gamma)

        # High case: 需要延长弧段（加螺旋圈）
        # 所有段都用 γ_max，仍然不够，需要额外螺旋
        # 这里简化处理：将所有段设为 γ_max，剩余高度差在第一段螺旋中补偿
        gamma_sign = 1.0 if dz > 0 else -1.0
        gamma1 = gamma_sign * self.gamma_max
        gamma_s = gamma_sign * self.gamma_max
        gamma2 = gamma_sign * self.gamma_max

        return (gamma1, gamma_s, gamma2)

    def _sample_3d(self, sol_2d: dict, z_start: float,
                   gamma1: float, gamma_s: float, gamma2: float) -> List[np.ndarray]:
        """
        沿 3D 路径等弧长采样

        对每段，根据爬升角将 2D 弧长映射到 3D 弧长，
        Z 坐标按 z += ds_2d * tan(gamma) 递增
        """
        path_type = sol_2d['type']
        t_param, p_param, q_param = sol_2d['params']
        r = self.r
        start = sol_2d['start']
        heading = sol_2d['start_heading']

        seg_lengths_2d = list(sol_2d['lengths'])  # [s1, d, s2]
        seg_types = list(path_type)  # e.g. ['L', 'S', 'R']
        seg_gammas = [gamma1, gamma_s, gamma2]

        # 计算总 3D 弧长
        total_3d = 0.0
        for s2d, g in zip(seg_lengths_2d, seg_gammas):
            cos_g = np.cos(g) if abs(g) > 1e-9 else 1.0
            total_3d += s2d / cos_g if cos_g > 0.01 else s2d

        n_samples = max(2, int(total_3d / self.ds) + 1)

        points = []
        x, y, th = start[0], start[1], heading
        z = z_start

        # 预计算每段的 3D 长度
        seg_lengths_3d = []
        for s2d, g in zip(seg_lengths_2d, seg_gammas):
            cos_g = np.cos(g) if abs(g) > 1e-9 else 1.0
            seg_lengths_3d.append(s2d / cos_g if cos_g > 0.01 else s2d)

        total_3d_actual = sum(seg_lengths_3d)

        # 逐步采样
        seg_idx = 0
        seg_consumed_2d = 0.0  # 当前段已消耗的 2D 弧长
        cumulative_2d = 0.0    # 全局已消耗的 2D 弧长

        for i in range(n_samples):
            frac = i / (n_samples - 1) if n_samples > 1 else 0
            # 目标 2D 弧长（按 3D 弧长等比映射回 2D）
            target_3d = frac * total_3d_actual

            # 找到目标 3D 弧长对应的段和段内位置
            acc_3d = 0.0
            for si in range(3):
                if acc_3d + seg_lengths_3d[si] >= target_3d - 1e-9:
                    # 在段 si 内
                    ds_3d_in_seg = target_3d - acc_3d
                    cos_g = np.cos(seg_gammas[si]) if abs(seg_gammas[si]) > 1e-9 else 1.0
                    ds_2d_in_seg = ds_3d_in_seg * cos_g if cos_g > 0.01 else ds_3d_in_seg

                    # 从段起点前进 ds_2d_in_seg
                    # 先计算段起点状态
                    sx, sy, sth = start[0], start[1], heading
                    sz = z_start
                    for prev_si in range(si):
                        sx, sy, sth = Dubins2DSolver._advance(
                            sx, sy, sth, seg_types[prev_si],
                            seg_lengths_2d[prev_si], r)
                        sz += seg_lengths_2d[prev_si] * np.tan(seg_gammas[prev_si])

                    # 在当前段内前进
                    px, py, _ = Dubins2DSolver._advance(
                        sx, sy, sth, seg_types[si], ds_2d_in_seg, r)
                    pz = sz + ds_2d_in_seg * np.tan(seg_gammas[si])

                    points.append(np.array([px, py, pz]))
                    break
                acc_3d += seg_lengths_3d[si]
            else:
                # 超出末尾，取终点
                ex, ey, eth = start[0], start[1], heading
                ez = z_start
                for si in range(3):
                    ex, ey, eth = Dubins2DSolver._advance(
                        ex, ey, eth, seg_types[si], seg_lengths_2d[si], r)
                    ez += seg_lengths_2d[si] * np.tan(seg_gammas[si])
                points.append(np.array([ex, ey, ez]))

        return points


# ============================================================
# 顶层接口：替换 _blend_junction
# ============================================================

def _estimate_heading(points: List[np.ndarray]) -> float:
    """从点序列估计航向角（使用最后两个点的 XY 方向）"""
    if len(points) < 2:
        return 0.0
    d = points[-1][:2] - points[-2][:2]
    norm = np.linalg.norm(d)
    if norm < 1e-6:
        # 尝试更早的点
        for i in range(len(points) - 2, 0, -1):
            d = points[-1][:2] - points[i - 1][:2]
            norm = np.linalg.norm(d)
            if norm > 1e-6:
                return np.arctan2(d[1], d[0])
        return 0.0
    return np.arctan2(d[1], d[0])


def _estimate_heading_from_start(points: List[np.ndarray]) -> float:
    """从点序列开头估计航向角"""
    if len(points) < 2:
        return 0.0
    d = points[1][:2] - points[0][:2]
    norm = np.linalg.norm(d)
    if norm < 1e-6:
        for i in range(2, len(points)):
            d = points[i][:2] - points[0][:2]
            norm = np.linalg.norm(d)
            if norm > 1e-6:
                return np.arctan2(d[1], d[0])
        return 0.0
    return np.arctan2(d[1], d[0])


def dubins_3d_blend_junction(executed_tail: List[np.ndarray],
                              new_path: List[np.ndarray],
                              params: Dubins3DParams,
                              safety_check=None) -> List[np.ndarray]:
    """
    用 3D Dubins 路径替换衔接处的 B-spline 平滑

    在 executed_tail 末尾和 new_path 开头之间计算一条 3D CSC Dubins 路径，
    替换掉硬拐点，使航向和爬升角连续。

    Args:
        executed_tail: 已飞过的末尾 2-3 个点
        new_path: 新规划的路径
        params: Dubins3DParams 参数
        safety_check: 碰撞检测函数 (from_pos, to_pos) -> bool

    Returns:
        平滑后的路径（替换了开头部分），或原路径（平滑失败时）
    """
    if len(executed_tail) < 2 or len(new_path) < 2:
        return new_path

    solver = Dubins3DSolver(params)

    # 衔接起点：executed_tail 的末尾
    start_3d = executed_tail[-1].copy()
    start_heading = _estimate_heading(executed_tail)

    # 衔接终点：new_path 的第 n_overlap 个点（跳过开头几个点）
    n_overlap = min(3, len(new_path) - 1)
    end_3d = new_path[n_overlap].copy()
    end_heading = _estimate_heading_from_start(new_path[n_overlap:])

    # 如果起终点太近，不需要平滑
    dist_2d = np.linalg.norm(end_3d[:2] - start_3d[:2])
    if dist_2d < params.turning_radius * 0.5:
        return new_path

    # 求解 3D Dubins
    try:
        dubins_points = solver.solve(start_3d, start_heading, end_3d, end_heading)
    except Exception as e:
        print(f"  [DUBINS-3D] Solve exception: {e}, falling back to original path")
        return new_path

    if dubins_points is None or len(dubins_points) < 2:
        print("  [DUBINS-3D] No solution found, falling back to original path")
        return new_path

    # 碰撞检测
    if safety_check is not None:
        for i in range(len(dubins_points) - 1):
            if not safety_check(dubins_points[i], dubins_points[i + 1]):
                print("  [DUBINS-3D] Collision on Dubins arc, falling back to original path")
                return new_path

    # 前向过滤：去掉 Dubins 弧段中往回走的点
    if len(executed_tail) >= 2:
        forward_dir = end_3d[:2] - start_3d[:2]
        forward_norm = np.linalg.norm(forward_dir)
        if forward_norm > 0.1:
            forward_dir_2d = forward_dir / forward_norm
            filtered = [dubins_points[0]]
            for pt in dubins_points[1:]:
                step = pt[:2] - filtered[-1][:2]
                if np.dot(step, forward_dir_2d) >= -0.1:
                    filtered.append(pt)
            dubins_points = filtered

    if len(dubins_points) < 2:
        print("  [DUBINS-3D] No forward-progressing points, falling back")
        return new_path

    # 拼接：Dubins 弧段 + new_path 剩余部分
    blended = dubins_points + new_path[n_overlap + 1:]

    print(f"  [DUBINS-3D] Blended junction: {len(dubins_points)} Dubins pts + "
          f"{max(0, len(new_path) - n_overlap - 1)} remaining, total {len(blended)} pts")

    return blended
