"""
actions.py — 研究智能体的结构化动作空间

Structured action space for the closed loop.
LLM(或 MockProposer)每轮只能产出一个结构化 Action,两类:
  - set_params:  调 PlanningConfig 参数(零风险、最高频)
  - rewrite_smoother: 重写路径平滑函数(代码级改动,走沙箱)
统一表达 + 校验/裁剪,便于 apply 与 accept/reject 归因、JSONL 落盘。
"""
from dataclasses import dataclass, field, asdict
from typing import Optional
import hashlib


# 参数搜索空间(与 search.py 的 SPACE 对齐,附类型与范围)
PARAM_SPACE = {
    "step_size":             (1.0, 3.0,  "float"),
    "max_iterations":        (3000, 8000, "int"),
    "goal_sample_rate":      (0.1, 0.6,  "float"),
    "search_radius":         (3.0, 7.0,  "float"),
    "weight_energy":         (0.0, 1.0,  "float"),
    "weight_distance":       (0.0, 1.0,  "float"),
    "weight_time":           (0.0, 1.0,  "float"),
}
# 安全/物理/评测相关字段:锁定,agent 不可改(防 reward hacking / 破坏可比性)
# [Phase A] kinodynamic 旋钮(转弯半径/爬升角)是车辆物理极限,且 BEMT 不给转弯计价 →
#           它们是隐藏作弊向量,必须冻结(见 LAYERS.md Layer-0 / Q3 决策)。
LOCKED_FIELDS = {
    "voxel_size", "grid_size", "origin", "safety_margin", "max_depth",
    "energy_ref", "distance_ref", "time_ref", "flight_velocity", "planning_timeout",
    "dubins_turning_radius", "dubins_max_climb_angle", "dubins_sample_distance",
}


@dataclass
class Action:
    kind: str                      # "set_params" | "rewrite_smoother"
    name: str                      # 简短标签
    hypothesis: str                # 提出该动作的假设/理由
    params: Optional[dict] = field(default=None)   # set_params 时生效
    code: Optional[str] = field(default=None)      # rewrite_smoother 时:smooth_path 源码

    def code_hash(self) -> Optional[str]:
        if self.code is None:
            return None
        return hashlib.sha1(self.code.encode("utf-8")).hexdigest()[:12]

    def to_log(self) -> dict:
        d = {"kind": self.kind, "name": self.name, "hypothesis": self.hypothesis}
        if self.kind == "set_params":
            d["params"] = self.params
        else:
            d["code_hash"] = self.code_hash()
            d["code_len"] = len(self.code or "")
        return d


def clip_params(params: dict):
    """按 PARAM_SPACE 裁剪 + 类型转换 + 剔除锁定/未知字段。返回 (clean, rejected_keys)。"""
    clean, rejected = {}, []
    for k, v in (params or {}).items():
        if k in LOCKED_FIELDS:
            rejected.append(k)
            continue
        if k not in PARAM_SPACE:
            rejected.append(k)
            continue
        lo, hi, typ = PARAM_SPACE[k]
        try:
            v = float(v)
        except (TypeError, ValueError):
            rejected.append(k)
            continue
        v = max(lo, min(hi, v))
        clean[k] = int(round(v)) if typ == "int" else round(v, 4)
    return clean, rejected


def normalize_weights(cfg: dict) -> dict:
    """三权重存在时归一化(和为1),与 evaluator 口径一致。"""
    ws = [cfg.get("weight_energy"), cfg.get("weight_distance"), cfg.get("weight_time")]
    if all(w is not None for w in ws):
        s = sum(ws)
        if s > 0:
            cfg["weight_energy"] = round(ws[0] / s, 4)
            cfg["weight_distance"] = round(ws[1] / s, 4)
            cfg["weight_time"] = round(ws[2] / s, 4)
    return cfg


def action_from_dict(d: dict) -> Action:
    return Action(
        kind=d["kind"], name=d.get("name", ""), hypothesis=d.get("hypothesis", ""),
        params=d.get("params"), code=d.get("code"),
    )
