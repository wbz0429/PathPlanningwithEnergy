"""
candidate.py — 代码进化目标:路径平滑函数

The evolvable code target for the closed loop.
`_smooth_path` 是 autoresearch 的代码进化对象(stage-2a 已手动改过一次)。
这里把它抽成一个自由函数签名,供 LLM 重写;通过类级 monkeypatch 注入
`RRTStar._smooth_path`,从而在**完全不修改 drone_sim/planning/rrt_star.py**
的前提下评测候选实现。

候选函数契约(LLM 必须实现):
    def smooth_path(path, is_collision_free, config) -> list[np.ndarray]
      - path: 原始路径点列表(np.ndarray)
      - is_collision_free(a, b) -> bool: 两点间直线是否无碰撞(硬安全约束,不可绕过)
      - config: PlanningConfig(只读)
      - 返回:平滑后的路径点列表;每段必须过 is_collision_free 检查
"""
import numpy as np


# 默认候选 = 当前 rrt_star.py 的 stage-2a 多趟视线捷径(作为进化起点/基线)
DEFAULT_SMOOTHER_SRC = '''
def smooth_path(path, is_collision_free, config):
    """多趟视线捷径 (iterative shortcutting)。每段直线且过碰撞检查,长度单调不增。"""
    if len(path) <= 2:
        return path
    pts = [p.copy() for p in path]
    for _ in range(10):
        shortened = False
        out = [pts[0]]
        i = 0
        while i < len(pts) - 1:
            best_j = i + 1
            for j in range(len(pts) - 1, i + 1, -1):
                if is_collision_free(pts[i], pts[j]):
                    best_j = j
                    break
            if best_j > i + 1:
                shortened = True
            out.append(pts[best_j])
            i = best_j
        pts = out
        if not shortened or len(pts) <= 2:
            break
    return pts
'''


def load_smoother(src: str):
    """
    从源码字符串加载 smooth_path 函数(受限命名空间)。
    调用方应先用 sandbox.check_code 校验。

    Returns:
        callable smooth_path(path, is_collision_free, config)
    """
    ns = {"np": np, "numpy": np}
    exec(compile(src, "<candidate_smoother>", "exec"), ns)
    fn = ns.get("smooth_path")
    if fn is None or not callable(fn):
        raise ValueError("candidate code must define a callable `smooth_path`")
    return fn


def make_patch_method(smoother_fn):
    """把自由函数 smoother_fn 包成 RRTStar._smooth_path 的绑定方法形态。"""
    def _patched(self, path):
        return smoother_fn(path, self._is_collision_free, self.config)
    return _patched


def contract_test(smoother_fn) -> None:
    """注册前契约测试:返回类型/长度合理/不改入参/终点保持。异常即视为不合格。"""
    path = [np.array([0., 0., -3.]), np.array([2., 1., -3.]),
            np.array([4., 0., -3.]), np.array([6., 0., -3.])]
    snapshot = [p.copy() for p in path]
    out = smoother_fn(path, lambda a, b: True, _DummyConfig())
    assert isinstance(out, (list, tuple)) and len(out) >= 2, "must return >=2 waypoints"
    for p in out:
        assert isinstance(p, np.ndarray) and p.shape == (3,), "waypoints must be 3D np.ndarray"
        assert np.all(np.isfinite(p)), "waypoints must be finite"
    # 全通行时应能压到起点+终点两点(视线捷径的最基本性质),但不强制;至少终点不变
    assert np.allclose(out[-1], snapshot[-1], atol=1e-6), "endpoint must be preserved"
    assert np.allclose(out[0], snapshot[0], atol=1e-6), "startpoint must be preserved"
    # 入参不被破坏
    for a, b in zip(path, snapshot):
        assert np.allclose(a, b), "input path must not be mutated"


class _DummyConfig:
    safety_margin = 1.0
    dubins_turning_radius = 1.5


# ============================================================
# Phase B: 采样器进化目标 sample(ctx)
# ============================================================
# 契约:def sample(ctx) -> np.ndarray[3]
#   ctx.rng               : np.random(已按 seed 播种,保持确定性)
#   ctx.bounds_min/max    : 全网格边界(可用来逃出 z-clamp 实现翻墙采样)
#   ctx.local_bounds_min/max : 现有局部采样框(z 被夹在起终点±3m —— 禁翻墙的根源)
#   ctx.start/goal/config/iteration
# 返回的点会被裁剪到全网格 bounds 内(防越界),但不裁到 local(允许翻墙)。
class SampleCtx:
    __slots__ = ("rng", "bounds_min", "bounds_max", "local_bounds_min",
                 "local_bounds_max", "start", "goal", "config", "iteration",
                 "edge_cost", "c_best")
    # edge_cost(a, b) -> float:冻结的 BEMT 能量代价 oracle(agent 只能查、不能改)
    #   → 让采样器能利用能量代价的**各向异性/方向结构**(降落便宜、直线便宜)做非欧 informed 采样。
    # c_best: 当前最优解代价(informed set 用;无解时 inf)。

    def __init__(self, **kw):
        for k in self.__slots__:
            setattr(self, k, kw.get(k))


DEFAULT_SAMPLER_SRC = '''
def sample(ctx):
    """默认:局部框内均匀采样(= 现有 _random_sample,含 z-clamp)。"""
    return ctx.rng.uniform(ctx.local_bounds_min, ctx.local_bounds_max)
'''


def load_sampler(src: str):
    ns = {"np": np, "numpy": np}
    exec(compile(src, "<candidate_sampler>", "exec"), ns)
    fn = ns.get("sample")
    if fn is None or not callable(fn):
        raise ValueError("candidate code must define a callable `sample(ctx)`")
    return fn


def sampler_random_patch(fn):
    """把 sample(ctx) 包成 RRTStar._random_sample(self)。"""
    def _patched(self):
        ctx = SampleCtx(
            rng=np.random,
            bounds_min=self.bounds_min, bounds_max=self.bounds_max,
            local_bounds_min=self.local_bounds_min, local_bounds_max=self.local_bounds_max,
            start=getattr(self, "_samp_start", None), goal=getattr(self, "_samp_goal", None),
            config=self.config, iteration=getattr(self, "_samp_iter", 0),
        )
        p = np.asarray(fn(ctx), dtype=float).reshape(3)
        return np.clip(p, self.bounds_min, self.bounds_max)   # 只裁全网格,不裁 local
    return _patched


def contract_test_sampler(fn) -> None:
    """契约测试:返回 3D 有限点。"""
    ctx = SampleCtx(rng=np.random,
                    bounds_min=np.array([-10., -30., -15.]), bounds_max=np.array([80., 30., 5.]),
                    local_bounds_min=np.array([0., -5., -6.]), local_bounds_max=np.array([70., 5., 0.]),
                    start=np.array([0., 0., -3.]), goal=np.array([70., 0., -3.]),
                    config=_DummyConfig(), iteration=1,
                    edge_cost=lambda a, b: float(np.linalg.norm(np.asarray(b) - np.asarray(a))), c_best=None)
    for _ in range(5):
        p = np.asarray(fn(ctx), dtype=float).reshape(3)
        assert p.shape == (3,) and np.all(np.isfinite(p)), "sample must return finite 3D point"


# ============================================================
# S3a: 速度剖面进化目标 speed_profile(path, v_star, vcap)
# ============================================================
# 契约:def speed_profile(path, v_star, vcap) -> list[float](每段目标速度)
#   path: list[np.ndarray(3,)];v_star: 能量最优巡航速度;vcap: 每段转弯可行速度上限(冻结约束)
#   返回值会被裁到 [0.5, vcap](不许超上限);目标=让加速平滑、贴近 v*,压总能耗(含提速动能代价)。
DEFAULT_SPEED_SRC = '''
def speed_profile(path, v_star, vcap):
    """默认:每段走可行上限(最快)。"""
    return list(vcap)
'''


def load_speed(src: str):
    ns = {"np": np, "numpy": np}
    exec(compile(src, "<candidate_speed>", "exec"), ns)
    fn = ns.get("speed_profile")
    if fn is None or not callable(fn):
        raise ValueError("candidate code must define a callable `speed_profile(path, v_star, vcap)`")
    return fn


def contract_test_speed(fn) -> None:
    path = [np.array([0., 0., -3.]), np.array([5., 0., -3.]), np.array([10., 2., -3.])]
    vcap = [18.0, 9.0]
    out = fn([p.copy() for p in path], 18.2, list(vcap))
    assert len(out) == len(vcap), "speed_profile must return one speed per segment"
    for v in out:
        assert np.isfinite(float(v)), "speeds must be finite"
