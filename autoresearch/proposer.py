"""
proposer.py — 研究智能体的"提议者"(闭环里唯一动脑的一环)

Two implementations of the same Proposer interface:
  - MockProposer: 确定性启发式,无需 API/网络,用于把闭环端到端跑通、离线 demo、CI。
  - LLMProposer:  真调 Anthropic API(tool-use 结构化输出),读实验历史→提假设→
                  产出参数改动或代码级改动。需要环境变量 ANTHROPIC_API_KEY。

两者都实现 propose(ctx) -> Action。ctx 由 agent_loop 组装。
"""
import os
import random
from typing import Optional

from actions import Action, PARAM_SPACE, clip_params


class Proposer:
    def propose(self, ctx: dict) -> Action:
        raise NotImplementedError


# ----------------------------------------------------------------------------
# MockProposer:确定性,离线可跑
# ----------------------------------------------------------------------------
_ALT_SMOOTHER_SRC = '''
def smooth_path(path, is_collision_free, config):
    """前向贪心视线捷径:从每个保留点尽量远地直达。等价目标、实现不同,用于验证代码进化路径。"""
    if len(path) <= 2:
        return path
    pts = [p.copy() for p in path]
    for _ in range(12):
        out = [pts[0]]
        i = 0
        changed = False
        while i < len(pts) - 1:
            j = len(pts) - 1
            reached = i + 1
            while j > i + 1:
                if is_collision_free(pts[i], pts[j]):
                    reached = j
                    break
                j -= 1
            if reached > i + 1:
                changed = True
            out.append(pts[reached])
            i = reached
        pts = out
        if not changed or len(pts) <= 2:
            break
    return pts
'''


class MockProposer(Proposer):
    """确定性提议者:多数轮扰动最优参数,少数轮提一个代码改写(演示代码进化路径)。"""

    def __init__(self, seed: int = 12345):
        self.rng = random.Random(seed)

    def _perturb(self, base: dict, scale: float = 0.25) -> dict:
        c = dict(base)
        for k, (lo, hi, typ) in PARAM_SPACE.items():
            if k not in base:
                continue
            span = (hi - lo) * scale
            v = base[k] + self.rng.gauss(0, span)
            v = max(lo, min(hi, v))
            c[k] = int(round(v)) if typ == "int" else round(v, 3)
        return c

    def propose(self, ctx: dict) -> Action:
        it = ctx["iteration"]
        base = ctx["best"]["config"]
        # 在第 3 轮(若允许代码改动)提一个代码改写,其余轮扰动参数
        if ctx.get("allow_code") and it == 3:
            return Action(
                kind="rewrite_smoother",
                name="forward_greedy_shortcut",
                hypothesis="用前向贪心视线捷径替代反向扫描,验证代码级改动路径能否被闭环正确评测与取舍。",
                code=_ALT_SMOOTHER_SRC,
            )
        params, _ = clip_params(self._perturb(base))
        return Action(
            kind="set_params",
            name=f"perturb#{it}",
            hypothesis="在当前最优附近做高斯扰动,局部搜索更低能耗配置。",
            params=params,
        )


# ----------------------------------------------------------------------------
# LLMProposer:真调 Anthropic API
# ----------------------------------------------------------------------------
_SYSTEM = """你是一个无人机路径规划算法的**自动化研究智能体**。你在一个闭环里工作:
读实验结果 → 提假设 → 提出一个改动 → 系统自动评测 → keep/revert → 你看到结果后继续。

## 固定评测器(你无法修改,不要试图绕过)
- 三个场景(A 直穿窄通道 / B 对角向上 / C 对角向下),硬约束=全部 100% 到达。
- 单一指标 score(越低越好):每个 100% 成功的场景贡献其 BEMT 物理模型能耗(J);
  任一场景 <100% 成功则被重罚(PENALTY=10000×(2-成功率))。
- 因此策略永远是:先保证 100% 成功,再压能耗。参考:A* 最优基线总能耗≈16410J。
- 安全(ESDF 碰撞)是硬约束,平滑函数每段都会被 is_collision_free 校验,改不动。

## 你每轮只能产出一个结构化动作(通过 propose_action 工具),两类:
1) set_params:调这些参数(范围内):
   step_size[1.0-3.0], max_iterations[3000-8000], goal_sample_rate[0.1-0.6],
   search_radius[3.0-7.0], dubins_turning_radius[1.0-2.5],
   weight_energy/weight_distance/weight_time[0-1,内部归一化]。
2) rewrite_smoother:重写路径平滑函数,签名必须是:
   def smooth_path(path, is_collision_free, config):
       # path: list[np.ndarray(3,)];is_collision_free(a,b)->bool 两点直线是否无碰撞
       # config: 只读;返回平滑后的 list[np.ndarray],每段须过 is_collision_free
   只能 import numpy/math/scipy/typing;禁 os/sys/open/eval/exec。np 已注入。

## 要求
- 先在 hypothesis 里说清:当前瓶颈是什么、你的改动为什么可能降低 score。
- 参数搜索若已撞天花板(窄通道 A 靠调参上不去),优先考虑代码级改动(更强的捷径/曲率感知平滑等)。
- 不要重复已被 revert 的相同改动。"""

_TOOL = {
    "name": "propose_action",
    "description": "提出下一个改动(参数或代码级)。",
    "input_schema": {
        "type": "object",
        "properties": {
            "kind": {"type": "string", "enum": ["set_params", "rewrite_smoother"]},
            "name": {"type": "string", "description": "简短标签,如 perturb_goalbias / curvature_aware_smooth"},
            "hypothesis": {"type": "string", "description": "瓶颈分析 + 为何这个改动可能降 score"},
            "params": {
                "type": "object",
                "description": "kind=set_params 时给出;只放要改的参数键值",
                "properties": {k: {"type": "number"} for k in PARAM_SPACE},
                "additionalProperties": False,
            },
            "code": {"type": "string", "description": "kind=rewrite_smoother 时:smooth_path 完整源码"},
        },
        "required": ["kind", "name", "hypothesis"],
    },
}


class LLMProposer(Proposer):
    def __init__(self, model: str = "claude-opus-4-8", max_tokens: int = 4000):
        try:
            import anthropic
        except ImportError as e:
            raise RuntimeError("需要 anthropic 包:.venv/bin/pip install anthropic") from e
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError(
                "未设置 ANTHROPIC_API_KEY。LLMProposer 需要 API 访问;"
                "无 key 时请用 --proposer mock 先跑通闭环。")
        self.client = anthropic.Anthropic()
        self.model = model
        self.max_tokens = max_tokens

    def _user_msg(self, ctx: dict) -> str:
        b = ctx["best"]
        lines = [
            f"# 研究策略\n{ctx.get('strategy','(见 program.md)')}",
            f"\n# 进度: 第 {ctx['iteration']}/{ctx['n_iters']} 轮 | A* 基线总能耗={ctx['astar_total']:.0f}J",
            f"\n# 当前最优 (incumbent)",
            f"config = {b['config']}",
            f"score={b['score']:.0f}  vs_astar={b.get('vs_astar')}  "
            f"min_success={b.get('min_success')}  energy_total={b.get('energy_total')}  "
            f"平滑器={b.get('smoother_name','default')}",
            f"\n# 最近实验轨迹 (从差到好, 供你看'改动→结果'梯度)",
        ]
        for h in ctx.get("history", [])[-12:]:
            lines.append(
                f"iter={h['iter']:>2} {h['kind']:15s} {h['name'][:24]:24s} "
                f"score={h['score']:.0f} vsA*={h.get('vs_astar')} succ={h.get('min_success')} "
                f"[{h['decision']}{': '+h['reason'] if h.get('reason') else ''}]")
        lines.append(f"\n# 当前生效的平滑函数源码\n```python\n{ctx.get('current_smoother_src','')}\n```")
        lines.append("\n# 任务: 提出使 score 更低的下一个改动。先在 hypothesis 说明瓶颈, 再调用 propose_action。")
        return "\n".join(lines)

    def propose(self, ctx: dict) -> Action:
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=_SYSTEM,
            tools=[_TOOL],
            tool_choice={"type": "tool", "name": "propose_action"},
            messages=[{"role": "user", "content": self._user_msg(ctx)}],
        )
        block = next((b for b in resp.content if getattr(b, "type", None) == "tool_use"), None)
        if block is None:
            raise RuntimeError("LLM 未返回 tool_use 动作")
        a = block.input
        return Action(
            kind=a["kind"], name=a.get("name", "llm_action"),
            hypothesis=a.get("hypothesis", ""),
            params=a.get("params"), code=a.get("code"),
        )
