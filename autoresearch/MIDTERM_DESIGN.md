# LLM 研究智能体驱动的无人机路径规划算法自优化系统 —— 中期实现方案

> 本文档由 5 份并行设计小节综合而成,统一术语:**研究智能体**(读结果→提假设→改动→评测→判定→记录的 LLM 闭环主体)、**evaluate 实验引擎**(确定性、无副作用的评测台入口 `evaluate()`)、**策略插件**(可注入替换的采样/代价/扩展组件)、**知识库**(持久化实验数据库)、**结构化动作**(LLM 唯一允许产出的 JSON Action)、**代码安全沙箱**。统一新增包为 `drone_sim/agentopt/`(整合原方案中的 `autotune/`、`eval/`、`selfopt/`)。

---

## 0. 执行摘要

本方案将毕设从"能量感知 RRT\*"重定位为"**研究智能体驱动的无人机三维路径规划算法自优化系统**"。研究智能体按"读结果→提假设→改参数/改代码→跑评测→统计判定→接受或回滚→记录→迭代"的闭环自动运行,优化对象同时覆盖 `PlanningConfig` 调参与算法组件(采样/代价/扩展)的**代码级替换**。全部迭代在无 AirSim 的 evaluate 实验引擎上秒级完成,少量优胜配置回 AirSim 高保真验证。现有 `benchmark_planning.py`、`test_planning_simulation.py` 复用为评测引擎,`rrt_star.py` 一次性插件化。中期交付:可运行的自优化闭环原型 + 至少一个被接受的代码级改动 + 优化前后对比图/表与收敛曲线。

---

## 1. 一句话课题定位 + 候选题目

**一句话定位**:把"人工读实验→提假设→改参数或改算法组件→跑评测→分析→接受/回滚"的慢科研循环,自动化为一个由 LLM 研究智能体驱动、跑在物理仿真+硬安全约束+A\* 最优基线之上的路径规划算法自优化闭环。

**候选题目**(从保守到激进,推荐主用 T1,副标题借 T2 后半句):

| 编号 | 中文题目 | English Title | 侧重 |
|------|---------|---------------|------|
| **T1** | 大模型智能体驱动的无人机三维路径规划算法自优化系统研究 | Research on an LLM-Agent-Driven Self-Optimizing System for UAV 3D Path Planning Algorithms | 系统/闭环(最稳妥) |
| **T2** | 面向能量感知路径规划的 LLM 闭环自动科研框架:从自动调参到算法组件生成 | An LLM Closed-Loop Automated-Research Framework for Energy-Aware Path Planning: From Auto-Tuning to Algorithm-Component Generation | 承接已有能量感知资产,突出双范围 |
| T3 | 基于大语言模型的采样式路径规划器自动设计与迭代优化 | Automatic Design and Iterative Optimization of Sampling-Based Path Planners via Large Language Models | 算法生成(偏理论) |
| T4 | LLM as Researcher:无人机路径规划的假设-实验-验证自动化闭环 | LLM as Researcher: An Automated Hypothesis-Experiment-Validation Loop for UAV Path Planning | 叙事亮眼,适合答辩 |
| T5 | 仿真在环的大模型驱动路径规划自优化:快速评测台迭代与 AirSim 高保真验证 | Simulation-in-the-Loop LLM-Driven Self-Optimization of Path Planning | 突出两级回路 |

**推荐**:主标题 T1 + 副标题"从自动调参到算法组件生成"。

---

## 2. 系统总览

### 2.1 高层数据流

```
        ┌───────────────────────── 研究智能体 (LLM 闭环) ─────────────────────────┐
        │                                                                          │
        │   知识库(SQLite)──top-K 轨迹+insight+已试清单──►  IDEATION/PROPOSE       │
        │        ▲                                              │ 结构化 Action     │
        │        │ 记录 trial/metrics/hypothesis                ▼                   │
        │   ┌────┴────┐                                    代码安全沙箱             │
        │   │  LOG    │◄───accept/reject/insight────┐      (AST白名单+子进程+git)   │
        │   └─────────┘                             │           │ apply             │
        └──────────────────────────────────────────┼───────────┼───────────────────┘
                                                    │           ▼
                              统计接受准则     ┌──────────────────────────┐
                            (配对Wilcoxon +    │  evaluate 实验引擎        │
                             train/val分离)◄───┤  (无AirSim, 确定性, 秒级) │
                                               │  策略插件 + PlanningConfig│
                                               │  Blocks/墙/U/随机场景     │
                                               │  A*基线 + BEMT能量模型    │
                                               └──────────────────────────┘
                                                            │ 最终少量优胜配置
                                                            ▼
                                               ┌──────────────────────────┐
                                               │  AirSim 高保真验证        │
                                               │  (fly_planned_path.py)    │
                                               └──────────────────────────┘
```

**核心隔离原则**:LLM 只能产出结构化 Action;指标计算代码(evaluate 实验引擎)与 LLM 生成的代码**物理隔离**,LLM 无法触碰评测台本身 —— 这是防 reward hacking 的第一道也是最根本的一道防线。

### 2.2 研究智能体循环状态机

```
                 ┌──────────────┐
                 │  INIT        │ 用 evaluate 跑基线, 建立 current_best (config+plugins+score)
                 └──────┬───────┘
                        ▼
       ┌───────────►┌──────────────┐
       │            │  IDEATION    │ LLM 读知识库(top-K轨迹+insight+已试清单)+当前最优 → 产出 Hypothesis
       │            └──────┬───────┘
       │                   ▼
       │            ┌──────────────┐
       │            │  PROPOSE     │ LLM 把假设具体化为 1 个结构化 Action(strict 工具)
       │            └──────┬───────┘
       │                   ▼
       │            ┌──────────────┐  语法/AST白名单/去重 校验失败(最多重试2次)
       │            │  APPLY       │──────────────────────────┐
       │            │ git_snapshot │                          │
       │            └──────┬───────┘                          ▼
       │                   ▼                            ┌──────────────┐
       │            ┌──────────────┐  子进程崩溃/超时/NaN │  REJECT      │
       │            │  EVALUATE    │───────────────────►│ git_rollback │
       │            │ 子进程沙箱     │                     └──────┬───────┘
       │            │ 跑 train 场景  │                            │
       │            └──────┬───────┘                            │
       │                   ▼                                    │
       │            ┌──────────────┐  配对统计+硬约束守卫+过拟合检验 │
       │            │  ANALYZE     │                            │
       │            └──────┬───────┘                            │
       │             accept│reject                             │
       │         ┌─────────┴─────────┐                         │
       │  accept ▼                   ▼ reject                  │
       │  ┌──────────────┐    ┌──────────────┐                 │
       │  │  ACCEPT      │    │  REJECT      │◄────────────────┘
       │  │ 在 val 复核   │    │ git_rollback │
       │  │ 过则更新best  │    └──────┬───────┘
       │  │ +git commit   │           │
       │  └──────┬───────┘           │
       │         ▼                    ▼
       │      ┌───────────────────────────────────┐
       └──────┤  LOG  KB.record_trial + 蒸馏 insight │
              └───────────────┬───────────────────┘
                              ▼
                       预算耗尽? ── 否 ──► 回到 IDEATION
                              │是
                              ▼
                       ┌──────────────┐
                       │  REPORT      │ 导出最优 config/plugins + 优化前后对比图/表 + 收敛曲线
                       └──────────────┘
```

**关键状态语义**:
- **IDEATION 与 PROPOSE 刻意分离**:IDEATION 先用自然语言推理瓶颈与方向(产出 `Hypothesis` 独立入库),PROPOSE 再压成可执行 Action,便于在 IDEATION 阶段做 explore/exploit 温度控制、后续复盘。
- **APPLY 失败**(语法/AST/去重)不进 EVALUATE,带具体报错回 PROPOSE 重试(≤2 次);仍失败记 `INVALID` trial 回 IDEATION。
- **EVALUATE 崩溃/超时/返回 NaN** → 直接 REJECT + git 回滚,失败原因写入知识库作负样本。
- **每轮只评一个 Action**,A/B 对照对象始终是 current_best,保证归因清晰。

---

## 3. 与现有代码的复用映射表

**总原则**:MVP 不重写任何规划/能量代码。只做两件事 —— (1) 把 benchmark 单体函数抽成纯函数 `evaluate()`;(2) 在外面套研究智能体闭环。`rrt_star.py` 仅做一次性插件化重构(约 25 行,向后兼容)。

| 现有资产 | 可复用组件(函数级) | 在新系统中的角色 | 需要的改造 |
|---|---|---|---|
| `benchmark_planning.py` | `build_known_map()`、`AStarPlanner.plan()`、`smooth_path()`、`compute_path_length()`、`BLOCKS_OBSTACLES`、`generate_benchmark_figures` | evaluate 引擎的**Blocks 场景源 + A\* 最优基线 + 出图**;A\* 基线缓存后跨迭代复用 | `run_benchmark()` 单体函数(硬编码 config/场景/权重/重复次数、直接 print+savefig、无返回)改为**薄封装调用 `evaluate()`**;障碍物/场景定义迁往 `scenarios.py`;旧命令行仍可用 |
| `test_planning_simulation.py` | `MockDrone`、`build_wall_map()`、`build_u_shape_map()`、`run_simulation()`、`analyze_trajectory()`(path_length/efficiency/y_oscillations/sharp_turns/x_retreats) | evaluate 引擎的**第二类无 AirSim 场景源 + 轨迹质量指标**(已 `np.random.seed(42)`) | 直接并入 `scenarios.py`;`analyze_trajectory` 的 sharp_turns 并入指标体系 |
| `planning/config.py` `PlanningConfig` | ~20 个可调字段(step_size/max_iterations/goal_sample_rate/search_radius/safety_margin、w_e/w_d/w_t、ref 类、dubins_*) | **参数搜索空间来源** | 新增字段 `sampling_strategy/cost_strategy/steer_strategy: str`、`rng_seed: Optional[int]`、`deterministic: bool=False`(均带默认值,向后兼容) |
| `planning/rrt_star.py` `RRTStar` | 已支持注入 `cost_function`;`_smart_sample()`、`_steer()`、`_nearest_node()`、`_random_sample()`、`plan()` 主循环 | 采样/代价/扩展三点抽成**策略插件**;碰撞检测 `_is_valid_point`/`_is_collision_free` 保留内部(硬安全约束,**不开放**) | **确定性修复**:加 `self.rng = np.random.default_rng(seed)`,替换 6 处全局 `np.random.*`;`plan()` 内 2 行改调策略;新增 `_make_ctx()`;`deterministic` 时跳过墙钟超时。默认行为完全不变 |
| `planning/rrt_star.py` `EnergyAwareCostFunction` | `compute_cost(from,to)`(已是可注入对象) | 直接复用为默认 `cost_term="energy_aware"` 策略插件 | 无需改;另加 `cost_term="distance"` 纯距离版供消融 |
| `energy/physics_model.py` | `compute_energy_for_segment()`、`compute_energy_for_path()`(确定性纯函数) | evaluate 引擎的**能耗指标计算** | 无需改,直接调用 |
| `fly_planned_path.py`、`fly_with_energy_visualization.py` | 现有 AirSim 飞行与能量可视化 | **AirSim 高保真验证**(不进内层循环) | 接收最优 `best_config.json` 注入运行;中期可选/降级为预录视频 |

**验证顺序(防回归)**:先只改 `config.py` + `rrt_star.py`(RNG/策略)→ 跑现有 `benchmark_planning.py` 确认默认策略数值不回归 → 再叠加 `evaluate()` 与 `agentopt/` 包。

---

## 4. 核心组件设计

### 4.0 `agentopt/` 包模块清单

| 模块 | 职责 | 关键接口 |
|---|---|---|
| `orchestrator.py` | 状态机主控,驱动闭环,管理预算(迭代/wall-clock/token) | `run(max_trials, time_budget)` |
| `state_machine.py` | 状态枚举与合法转移表 | `State`, `TRANSITIONS` |
| `schemas.py` | 核心数据结构 | `Trial`, `Action`, `EvalResult`, `Hypothesis` |
| `actions.py` | Action JSON schema、反序列化、静态校验 | `parse_action`, `validate_action` |
| `llm_client.py` | 封装 Anthropic SDK、prompt 组装、strict 工具、prompt caching | `ideate()`, `propose()` |
| `prompts.py` | System/User prompt 模板(OPRO 轨迹) | `build_system_prompt`, `build_user_prompt` |
| `knowledge_base.py` | SQLite 实验库,去重、检索 top-K、蒸馏 insight | `record_trial`, `top_k_trials`, `already_tried`, `insights` |
| `evaluator.py` | **evaluate 实验引擎**:无副作用可调用评测台 + `compute_score()` + 缓存 | `evaluate(config, scenarios, seeds, ...) -> EvalResult` |
| `param_space.py` | `PARAM_SPACE` 声明式参数表 + apply/validate/clip/sample/canonical + `config_hash` | — |
| `scenarios.py` | `ScenarioSpec` + 随机场景生成器 + train/val/test 划分 | `SCENARIO_REGISTRY`, `generate_random_scenario` |
| `metrics.py` | 全指标计算(含曲率/平滑度)+ fitness + Pareto/HV | `compute_metrics`, `fitness` |
| `stats.py` | bootstrap CI、配对 Wilcoxon、Cliff's delta | — |
| `acceptance.py` | 接受准则:显著性 + 硬约束守卫 + 过拟合检验 | `decide(cand, best) -> Decision` |
| `applier.py` | Action 落地:参数写 config、策略插件写 `candidates/` 并注册 | `apply(action) -> AppliedState` |
| `strategies/` | 策略插件包:`base.py`(Protocol)、`registry.py`、`builtin.py`、`generated/` | `register_strategy`, `build_strategy` |
| `sandbox.py` | 子进程执行 + setrlimit + AST 白名单 + git 快照/回滚 | `run_in_subprocess`, `ast_guard`, `git_snapshot/rollback` |
| `report.py` | 收敛曲线 + 优化前后对比图/表 | — |

### 4.1 智能体循环状态机

见 §2.2。状态枚举 `INIT / IDEATION / PROPOSE / APPLY / EVALUATE / ANALYZE / (ACCEPT|REJECT) / LOG / REPORT`,合法转移由 `state_machine.TRANSITIONS` 固定。`orchestrator.run()` 驱动;失败降级路径见 §2.2 关键语义。

### 4.2 结构化动作空间(Action Space)

**参数编辑与代码级改动用同一个 tagged-union JSON schema**,LLM 通过 Anthropic **strict 工具 `propose_action`** 产出,保证输出永远 schema-valid。每个 Action 强制带 `hypothesis_id / rationale / predicted_effect`,把"改动—假设—预期"绑定。

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "AgentOptAction",
  "type": "object",
  "required": ["action_type", "hypothesis_id", "rationale", "predicted_effect"],
  "additionalProperties": false,
  "properties": {
    "action_type": {
      "enum": ["set_params", "new_component", "patch_component",
               "select_component", "revert_to_best", "stop"]
    },
    "rationale": {"type": "string", "description": "为什么这么改(引用历史 trial 证据)"},
    "predicted_effect": {
      "type": "object", "additionalProperties": false,
      "required": ["metric", "direction"],
      "properties": {
        "metric": {"enum": ["energy_J", "path_ratio", "success_rate", "compute_time_ms", "smoothness", "vs_astar"]},
        "direction": {"enum": ["decrease", "increase"]},
        "magnitude_pct": {"type": "number"}
      }
    },
    "hypothesis_id": {"type": "string"},

    "params": {
      "type": "object",
      "description": "action_type=set_params 时生效; 键必须是 PlanningConfig 字段名",
      "additionalProperties": {"type": "number"},
      "propertyNames": {"enum": [
        "step_size","max_iterations","goal_sample_rate","search_radius",
        "planning_timeout","safety_margin","unknown_safe_threshold","flight_velocity",
        "weight_energy","weight_distance","weight_time",
        "energy_ref","distance_ref","time_ref",
        "dubins_turning_radius","dubins_max_climb_angle"]}
    },

    "component": {
      "type": "object", "additionalProperties": false,
      "description": "action_type ∈ {new_component, patch_component} 时生效",
      "required": ["kind", "name", "code"],
      "properties": {
        "kind": {"enum": ["sampler", "cost_term", "steer"]},
        "name": {"type": "string", "pattern": "^[a-z][a-z0-9_]{2,40}$"},
        "base_on": {"type": "string", "description": "patch 时基于哪个已有组件"},
        "code": {"type": "string", "description": "单个 Python 类源码, 实现规定的抽象接口"},
        "allowed_imports": {"type": "array", "items": {"enum": ["numpy", "math", "scipy", "typing"]}}
      }
    },

    "select": {
      "type": "object", "additionalProperties": false,
      "description": "action_type=select_component 时生效, 切换激活策略插件",
      "required": ["kind", "name"],
      "properties": {
        "kind": {"enum": ["sampler", "cost_term", "steer"]},
        "name": {"type": "string"}
      }
    }
  }
}
```

**两档能力**:参数级改动走 `set_params`(零风险、最高频);算法级改动走 `new_component`/`patch_component`/`select_component`(策略插件)。`safety_margin` 等安全字段列入 schema 只读禁改集下限保护(见 §5 参数表)。

### 4.3 策略插件接口(rrt_star.py 一次性插件化)

用 **`typing.Protocol`(零继承侵入)+ 冻结上下文对象 + 名字注册表**。三个天然算法组件抽成可注入策略,LLM 生成代码替换这里而非每次编辑类:

```python
# agentopt/strategies/base.py
from typing import Protocol, List
import numpy as np

class SampleContext:                 # 冻结的只读上下文, 避免生成代码访问 RRTStar 内部字段
    start: np.ndarray; goal: np.ndarray
    nodes: List[np.ndarray]; iteration: int
    bounds_min: np.ndarray; bounds_max: np.ndarray
    config: "PlanningConfig"; rng: np.random.Generator

class SamplingStrategy(Protocol):
    def sample(self, ctx: SampleContext) -> np.ndarray: ...        # 返回 3D 世界坐标
class CostTerm(Protocol):
    def compute_cost(self, a: np.ndarray, b: np.ndarray) -> float: ...   # >=0
    def compute_path_cost(self, path) -> float: ...
class SteerStrategy(Protocol):
    def steer(self, from_p, to_p, config) -> np.ndarray: ...
```

```python
# agentopt/strategies/registry.py
STRATEGY_REGISTRY = {"sampling": {}, "cost": {}, "steer": {}}
def register_strategy(kind, name):
    def deco(cls): STRATEGY_REGISTRY[kind][name] = cls; return cls
    return deco
def build_strategy(kind, name, **kw): return STRATEGY_REGISTRY[kind][name](**kw)
```

**默认策略 = 现有代码原样搬迁**(`strategies/builtin.py`,零回归):
- `@register_strategy("sampling","default_smart")` = 现 `_smart_sample` 主体(`np.random`→`ctx.rng`,`self.config`→`ctx.config`);顺手修掉硬编码 goal-bias `0.2` → 读 `ctx.config.goal_sample_rate`(现在该参数其实未被 `_smart_sample` 使用,是个隐藏 bug)。
- `@register_strategy("steer","straight")` = 现 `_steer`。
- `@register_strategy("cost","energy_aware")` = 复用 `EnergyAwareCostFunction`;另加 `"distance"` 纯距离版供消融。

**对 `rrt_star.py` 的最小改动(约 25 行)**:
1. `__init__` 增可选参数 `sampling_strategy/steer_strategy/cost_function/rng`,缺省用现有实现;`self.rng = rng or np.random.default_rng(getattr(config,'rng_seed',None))`。
2. `plan()` 主循环仅改两行:`sample = self.sampling_strategy.sample(self._make_ctx(...))`、`new_point = self.steer_strategy.steer(...)`。
3. **rewiring 与最近邻/碰撞检测暂不外抽**:rewiring 是 RRT\* 正确性核心、碰撞检测是硬安全约束,开放风险高;goal-bias 已通过 `goal_sample_rate` 参数化。平滑 `_smooth_path`、最近邻启发式列为**可选二级注入点**,中期不开放。
4. 保留 `_smart_sample`/`_steer` 原方法,默认策略内部复用,零回归。

生成代码只能通过冻结的 `SampleContext` 访问上下文,接口稳定;LLM 写"informed-RRT\* 椭球采样""桥测试采样""目标方向高斯采样"等新策略时不会因触碰内部字段而漂移。

### 4.4 evaluate 实验引擎 API

**唯一评测入口,`config` 是唯一自变量,便于 accept/reject 归因**:

```python
def evaluate(
    config: PlanningConfig,
    scenarios: list[str] | list[ScenarioSpec],   # 场景 id 或对象(固定)
    seeds: list[int],                            # 每场景重复的随机种子(固定)
    *,
    components: dict | None = None,              # 激活策略插件 {'sampler':..,'cost':..,'steer':..}
    baseline_cache: BaselineCache | None = None, # A* 基线缓存
    map_cache: MapCache | None = None,           # VoxelGrid+ESDF 缓存
) -> EvalResult:                                 # 普通 dict, 便于 JSON 落盘给 LLM 读
    ...
```

**返回结构(`EvalResult`)**:

```python
{
  "config_hash": "sha1(...)",
  "scenarios": {
    "blocks_straight": {
      "runs": [{"seed":0,"success":True,"path_length":74.2,"energy_j":5310.0,
                "vs_astar":1.06,"min_clearance":0.42,"smoothness":1.3, ...}, ...],
      "aggregate": {"success_rate":1.0,"path_length_mean":75.1,"path_length_std":1.8,
                    "energy_mean":5350.0,"vs_astar_mean":1.07,"min_clearance_min":0.38}},
    ...},
  "overall": {"success_rate":0.97,"mean_vs_astar":1.09,"mean_energy":5400.0,
              "mean_compute_ms":61.0,"score":0.812},
  "meta": {"seeds":[...], "n_scenarios":3, "wall_time_s":22.4,
           "baseline_from_cache":True, "hard_constraint_ok":True}
}
```

**单次 run 指标**:`success`、`path_length`、`energy_j`、`flight_time_s`、`compute_time_ms`、`path_ratio`、`vs_astar`(=length/A\* 最优)、`num_waypoints`、`smoothness`(Σθ²)、`curvature`、`sharp_turns`、`min_clearance`(沿路径最小 ESDF 距离,**验证安全硬约束未被绕过**)、`collision`(min_clearance<0)。逐场景聚合 `_mean/_std + success_rate`。

**统一目标函数(canonical fitness,越低越好,`metrics.fitness`)** —— 采用最完整的显式失败处理形式:

```
F = mean_over_scenarios[
      I_fail · P_penalty
    + I_success · ( w_r·PathRatio + w_e·(E/E*_astar) + w_t·(plan_time/t_ref) + w_s·(S/S_ref) )
    ]
    P_penalty=5.0,  w_r=1.0, w_e=0.5, w_t=0.2, w_s=0.3,  t_ref=100ms, S_ref=U形基线平滑度
```

硬约束优先于 F:任一场景 `min_clearance<0`(真碰撞)→ `F` 直接置惩罚值(等价 score=0)。**报告时同时给 F 标量与原始指标向量与多目标 HV**,不把结论藏在标量后。

**支撑"跑大量迭代"的三大加速**:
1. **A\* 基线缓存**:A\* 只依赖 `(scenario_id, safety_margin, grid_params)`,每组合只算一次进 `baseline_cache`,跨所有 config 迭代复用。
2. **地图缓存**:`VoxelGrid+ESDF` 只依赖场景与冻结的 `grid_size/voxel_size/origin`,按 `scenario_id` 缓存一次(省 ~40ms×N)。
3. **结果落盘缓存**:`config_hash` 命中直接读 `agentopt/runs/<hash>.json`,重复评估零成本。

**确定性可复现四动作**:①`RRTStar` 用 `np.random.default_rng(seed)` 实例、替换 6 处全局 `np.random`;②`deterministic=True` 时跳过基于墙钟的 `planning_timeout`,仅靠 `max_iterations` 收敛(消除跨机器/负载差异);③scenarios/seeds 按传入列表顺序执行、固定 key 排序聚合;④`config_hash` 由 `param_space.canonical(config)` 生成。

**评测两档**:`fast`(2–3 场景 × K=5–10 × 小迭代,<5s/次,给闭环刷几十上百轮)与 `full`(全场景 × K=20,里程碑收尾/出图时跑)—— 直接落实"快速台跑量 + AirSim 少量验证"决策。

### 4.5 知识库 / 实验数据库

**主存储:SQLite `agentopt/experiments.db`**(可复现、可查询、跨 run 累积)。MVP 阶段可先用 JSONL 账本(`ledger.append`)起步,再迁移到 SQLite。表结构:

- `trials(id, ts, parent_best_id, action_json, config_hash, status, score, accepted, reject_reason)`
- `metrics(trial_id, scenario, seed, energy_J, path_length, flight_time, path_ratio, success, compute_ms, safety_viol, smoothness)` —— **逐场景逐种子保存原始值,支撑配对统计**
- `hypotheses(id, trial_id, text, predicted_metric, predicted_dir, verified)` —— 记录假设是否被结果证实
- `components(name, kind, code_path, gen_trial_id, best_score)` —— 生成策略插件登记
- `insights(id, ts, text, evidence_trial_ids)` —— 蒸馏的可复用经验

**避免重复尝试**:Action 归一化后算 `config_hash`(参数排序 + 组件代码 AST 规范化后哈希);PROPOSE 落地前 `already_tried(hash)` 命中直接判 INVALID 并把"该改动已试过,结果 X"反馈给 LLM;System/User 常驻"最近已试 hash 摘要"。

**从失败中学习**:REJECT/崩溃 trial 全量入库并进 OPRO 轨迹(带 `reject_reason`),负样本与正样本一起塑造下一步。每 N 轮触发 **insight 蒸馏**:让 LLM 读近期轨迹产出一条压缩经验(如"goal_sample_rate>0.5 会在窄缝 Scenario C 掉 success"),写 `insights` 表并注入后续 System prompt,形成长期记忆而不必每轮重放全部历史。

**LLM 调用协议(OPRO / LLM-as-optimizer)**:
- **模型**:代码生成轮默认 `claude-opus-4-8`,纯调参低风险轮可降级 `claude-sonnet-4-6` 省成本;`thinking={"type":"adaptive"}` + `output_config={"effort":"high"}`;结构化输出用 **strict 工具**(`tool_choice={"type":"tool","name":"propose_action"}`,`input_schema` 带 `"strict": true`)。
- **Prompt 分层(为 prompt caching)**:渲染序 tools→system→messages,稳定内容放前并在末尾打 `cache_control`。**System(缓存,ephemeral)** = 研究目标 + 指标口径 + 硬约束/安全规则 + Action schema + 只读代码上下文(`config.py` 全文、三个策略插件点签名、`EnergyAwareCostFunction.compute_cost` 现有实现、evaluate 返回的指标键)。**User(易变,断点后)** = `current_best`(config JSON + 逐场景指标 + 聚合 score)+ OPRO 轨迹(top-K 按 score 排序)+ 最近失败清单 + 已试摘要 + 本轮指令。
- **OPRO 轨迹格式**(每行一条,按 score 从差到好,让模型看到"改动→结果"梯度):
  ```
  trial=017 action=set_params{goal_sample_rate:0.55} energy_J=612 path_ratio=1.28 succ=0.90 score=0.742 [rejected: succ regressed on Scenario C]
  trial=023 action=new_component{sampler:ellipsoid_informed} energy_J=548 path_ratio=1.19 succ=1.00 score=0.681 [ACCEPTED best]
  ...
  # 任务: 提出使 F 更低的下一个改动。先在 <hypothesis> 说明瓶颈, 再调用 propose_action。
  ```
- **缓存命中校验**:读 `response.usage.cache_read_input_tokens`,若持续为 0 说明前缀被 `datetime.now()`/随机 id 污染,需排查。长输出用 `client.messages.stream()` + `get_final_message()` 防超时。

### 4.6 代码安全沙箱

四道防线,LLM 生成代码永远写到 `agentopt/strategies/generated/gen_<trial_id>.py`,**绝不覆盖 `planning/`、`energy/` 核心文件**:

1. **AST 白名单静态检查(`sandbox.ast_guard`)**:`ast.parse` 保证语法;遍历 AST **拒绝**:非白名单 `Import/ImportFrom`(只许 `numpy/math/scipy/typing`)、`os/sys/subprocess/socket/shutil` 属性访问、`open`、`eval/exec/compile/__import__`、dunder 属性(`__globals__` 等)、`while True` 无 break 裸循环。失败→APPLY 失败回 PROPOSE。
2. **契约测试**:注册前用几组固定输入调用生成组件,断言返回类型/形状/非负/有限(非 NaN/Inf)且不修改入参。
3. **子进程隔离执行(`sandbox.run_in_subprocess`)**:evaluate 作为独立子进程运行(`subprocess.run([...], timeout=T)`),segfault/死循环/内存爆不会杀死 orchestrator;`preexec_fn` 里 `resource.setrlimit`(RLIMIT_CPU/RLIMIT_AS)限 CPU 与内存;子进程 JSON 序列化回传 `EvalResult`。超时/非零退出/NaN → 捕获为崩溃自动 REJECT。**新插件先小规模冒烟测试(1 场景 1 seed)**,通过才做完整 evaluate,把爆炸半径限制在单文件单子进程内。
4. **git 快照/回滚**:整个 run 在专用分支 `agentopt/run-<id>`。APPLY 前 `git_snapshot`;ACCEPT → 该 trial 的 config + 组件文件作为**一个 commit**(message 带 trial_id/score,便于溯源);REJECT/崩溃 → `git_rollback` 丢弃本轮改动。REPORT 阶段从最优 commit 导出 config 回 AirSim 验证。

**补丁注入机制 = 插件注册为主 + git 分支作审计层 + 参数 diff 作快速档**:参数改动走 `param_space.apply()`(零风险);算法组件写 `strategies/generated/` 并 `@register_strategy` 注册,**"回滚"= 下一版 config 不再引用该名字**(文件留存溯源,核心文件永不被改)。碰撞/边界检查始终由 `RRTStar` 把关,LLM 改不动 → 安全边界不被绕过。

---

## 5. 实验与评测协议

### 5.1 场景集与 train/val/test 划分

**固定场景(可解释锚点)**:
- **Blocks 族**(`benchmark_planning.py`):A 直穿 (0,0,-3)→(70,0,-3)、B 对角 →(70,20,-3)、C 反对角 →(70,-25,-3)。难度递增(A 的 PathRatio vs A\* 达 1.31,C 方差 ±17m),天然难度梯度。
- **墙/陷阱族**(`test_planning_simulation.py`):`build_wall_map`(窄墙/宽墙)、`build_u_shape_map`(U 形局部极小值陷阱)。

**参数化随机生成器 `generate_random_scenario(seed, difficulty)`**(`np.random.default_rng(seed)` 保复现):
```
map_extent    = (X=[0,70], Y=[-30,30], Z=[-12,0])         # 与 Blocks 一致
n_obstacles   ~ U{easy:3-5, medium:6-9, hard:10-14}
obstacle_types ∈ {box_wall(带缝隙), pillar(细柱), slab(悬空板, 占部分Z)}
gap_width     ~ U{easy:[6,10], medium:[3,6], hard:[2,3]} m # 缝隙宽度决定采样命中难度
start/goal    = 对角两端±抖动, 强制间距 ≥ 50m
```
三类障碍覆盖不同失败模式:box_wall 测缝隙命中、pillar 测密集绕障、**slab 测能量感知的爬升/下降决策**(补齐现有 Blocks 全同高度 Z=-3、能量权重无差异的盲区)。

**双级隔离划分(seed 隔离 + 结构族隔离)**:

| 集合 | 组成 | 数量 | 用途 | 地图 seed 池 |
|------|------|------|------|---------|
| **Train** | Blocks-A/B/C + 窄墙 + 宽墙 + U 形 + 6 随机(easy/med/hard 各 2) | 12 | 闭环 fitness 来源 | 100–105 |
| **Validation** | 6 随机(med/hard) | 6 | 过拟合监控 / ACCEPT 前复核 / 早停 | 200–205 |
| **Test** | 6 随机 + **3 迷宫族(held-out 结构)** + 3 森林/密柱族 | 12 | 最终泛化报告,**全程只评一次** | 900–905 / 950–952 / 970–972 |

Test 的迷宫/密柱族是训练时**从未出现的障碍拓扑**,测结构泛化 —— 答辩时区分"真优化"与"过拟合调参"的关键证据。

**可行性守卫**:每个生成场景先用 `AStarPlanner` 求解,无解则重采样(保证可解);A\* 最优长度缓存为该场景 PathRatio 分母(PathRatio≥1 恒成立且有意义)。

### 5.2 指标体系

| 指标 | 定义/来源 | 方向 |
|------|-----------|------|
| success | 找到路径 ∧ 全段 ESDF≥safety_margin ∧ 终点入 goal_tol | 大 |
| path_length (m) | `compute_path_length` | 小 |
| energy (J) | `physics_model.compute_energy_for_path(path, v=2.0)` | 小 |
| flight_time (s) | 同上 | 小 |
| **PathRatio** | path_length / A\*_length | →1 |
| plan_time (ms) | `RRTStar.plan` 墙钟 | 小 |
| **smoothness S** | Σθ_i²,θ=相邻段夹角 | 小 |
| curvature κ | mean(θ_i/seg_len) 与 max κ | 小 |
| sharp_turns | count(θ>60°),复用 `analyze_trajectory` | 小 |
| **min_clearance** | 沿路径最小 ESDF,验证安全未被绕过 | — |

平滑度/曲率补齐现有 benchmark 缺口(现只有长度/能耗/时间)。聚合为可最小化 fitness `F`(见 §4.4,含失败惩罚)。**多目标口径**:目标向量 O=(PathRatio, E/E\*, plan_time, S),固定参考点 r=(1.6,1.6,200ms,2·S_ref)下的 **dominated hypervolume(HV)** 作方法级前沿质量单一比较量 —— 比 fitness 更抗"权重可疑"质疑,建议作主结论之一。

### 5.3 统计协议

- **双层 seed**:`scenario_seed`(定地图,同场景所有方法共用同一张图,保证公平)+ `planner_seed`(定 RRT\* 采样)。每 (config × scenario) 跑 K 个 `planner_seed`:闭环 fast 档 K=10,ANALYZE 统计接受 K=20,最终 test 报告 K=20。每条 run 落盘 `{scenario_id, scenario_seed, planner_seed, 全指标}` 到 JSONL,完全可复现。
- **不确定性**:均值±std + **bootstrap 95% CI**(10k 重采样)。RRT\* 长度分布非正态(C 场景 ±17m 长尾),bootstrap CI 比 ±1.96σ 更诚实。
- **显著性 = 闭环 accept/reject 判据**:候选 c 与 incumbent c\* 在**相同 (scenario, planner_seed) 对**上比较(消除场景/seed 难度差异)→ **Wilcoxon 符号秩检验**(非参配对)对 fitness 差值 + 报告 **Cliff's delta / 中位改进量**作效应量。**accept 规则**:`p<0.05` 且中位 fitness 改进 > ε(相对 2%)且任一守卫指标无显著回退。配对(同种子)消除 RRT\* 随机性方差,远比独立均值±std 可靠。
- **硬约束守卫(任一不满足直接 reject,不看 score)**:每场景 `success_rate≥0.90`;`safety_viol==0`(路径穿障=0);`compute_ms≤3×基线`。专防"牺牲成功率/安全换低能耗"的 reward hacking。
- **防过拟合**:①train 上统计接受,ACCEPT 前必须在 **validation 集**复核(任一验证场景回退超容差如 energy +3% 则否决);②holdout 每 K 轮轮换划分;③跨种子方差检查(候选 score 方差显著大于 best 则视为不稳健不接受);④**最终配置强制在 test 集用全新 seed 重评一次**(消除 winner's curse,答辩防守命门);⑤每 N 次 accept 在 val 集评一次,train-fitness 降但 val-fitness 不降 → 判过拟合停。

### 5.4 对照基线与公平协议

| 方法 | 搜索空间 | 说明 |
|------|----------|------|
| **A\*** | — | oracle 上界(PathRatio/能耗分母) |
| **Manual(现状)** | 单点 | `config.py` 默认 + 4 组权重,专家先验,零搜索 |
| **Random Search** | PlanningConfig 域 | 均匀采样 |
| **Bayesian Opt** | 同上 | `skopt.gp_minimize`,GP 代理 |
| **CMA-ES** | 同上连续参数 | `cma` 包 |
| **LLM-agent(本方案)** | 同上 + **代码级改动** | 读结果→提假设→改参/改码→评测→accept/reject |

**公平协议(答辩必答)**:等评测预算 **B=50 候选**(1 候选 = N_train 场景 × K seeds 的完整规划;fast 档单候选 ≈6s,50 候选 ≈5 分钟/方法);等场景/等 seed/等 fitness/等 test 集。

**两条对比轨道**:
- **轨道 1(参数-only,公平对照)**:LLM 限制为**只改 PlanningConfig** vs Random/BO/CMA-ES。回答"LLM 是不是比朴素搜索更会调参?"—— LLM 能读懂"Scenario C 缝隙没命中"这类语义反馈定向调 `goal_sample_rate`,BO/CMA-ES 只看标量。
- **轨道 2(完整,头条结果)**:LLM 开启**代码级改动** —— 生成新采样策略(informed-RRT\*/桥测试)、新代价项、新扩展策略,这些**根本不在任何参数优化器的搜索空间内**,是 LLM 方案独有增益来源,即"为什么要用 LLM"的核心论据。

**主图口径**:sample-efficiency 曲线(x=已评测候选数,y=test-fitness running best),五条线叠加。一句话结论:"相同 50 次预算下,LLM-agent 收敛更快且终值更低;开启代码改动后突破所有参数优化器的下界(因触达参数空间之外的算法结构)。"

### 5.5 参数空间表(`param_space.PARAM_SPACE`)

LLM 产 `{参数名:值}` JSON patch,`apply()` 类型转换+越界裁剪,`validate()` 拦非法组合。

| 参数 | 类型 | 默认 | 建议范围 | 约束 | 可搜索 |
|---|---|---|---|---|---|
| `step_size` | float | 1.5 | 0.5–3.0 | ≤ `search_radius` | ✅ |
| `max_iterations` | int | 5000 | 500–8000 | — | ✅ |
| `goal_sample_rate` | float | 0.4 | 0.05–0.6 | — | ✅ |
| `search_radius` | float | 4.0 | 1.5–8.0 | ≥ `step_size` | ✅ |
| `planning_timeout` | float | 3.0 | 1.0–10.0 | deterministic 时忽略 | ⚠️ |
| `safety_margin` | float | 1.0 | **0.6–2.0** | **下限硬约束,不可再低** | ⚠️ |
| `unknown_safe_threshold` | float | 2.0 | 0.5–3.0 | ≥ `safety_margin` | ✅ |
| `flight_velocity` | float | 2.0 | 1.0–5.0 | >0 | ✅ |
| `weight_energy` | float | 0.6 | 0.0–1.0 | 三权重和>0,`apply` 后归一化 | ✅ |
| `weight_distance` | float | 0.3 | 0.0–1.0 | 同上 | ✅ |
| `weight_time` | float | 0.1 | 0.0–1.0 | 同上 | ✅ |
| `energy_aware` | bool | True | {T,F} | — | ✅ |
| `energy_ref` | float | 500 | 200–2000 | >0,**建议冻结**(改它扭曲 score 可比性) | ⚠️ |
| `distance_ref` | float | 10 | 5–30 | >0,建议冻结 | ⚠️ |
| `time_ref` | float | 5 | 2–15 | >0,建议冻结 | ⚠️ |
| `dubins_turning_radius` | float | 1.5 | 0.5–3.0 | — | ✅ |
| `dubins_max_climb_angle` | float | 30 | 10–45 | — | ✅ |
| `sampling_strategy` | str | "default_smart" | 注册表键 | 必须已注册 | ✅ |
| `cost_strategy` | str | "energy_aware" | 注册表键 | 必须已注册 | ✅ |
| `steer_strategy` | str | "straight" | 注册表键 | 必须已注册 | ✅ |
| `voxel_size/grid_size/origin/fov_deg/max_depth` | — | — | — | **冻结**(否则地图缓存与 A\* 基线失效) | ❌ |

`validate()` 关键规则:三权重归一化;`safety_margin≥0.6`;`unknown_safe_threshold≥safety_margin`;`search_radius≥step_size`;`⚠️` 标记默认进"锁定集",LLM 需显式解锁(降低误改归一化参考致 score 失真风险)。

### 5.6 消融实验

以完整系统 A0 为基准,每次只关一个组件,在 test 集(全新 seed)与 sample-efficiency 曲线上量化贡献:

| 编号 | 配置 | 验证的假设 |
|------|------|-----------|
| **A0** | 记忆 + 代码改动 + 强 LLM(Opus) + 统计 accept | 完整系统上界 |
| A1 | **去记忆**:每轮冷启动 | 知识库对采样效率的价值 |
| A2 | **只调参**:关闭代码生成(=轨道 1) | 隔离代码级改动净贡献(预期显著弱于 A0) |
| A3 | **去反思**:跳过分析失败原因步 | 反思环节价值 |
| A4 | **弱 LLM**:Opus→Haiku | 对推理能力的依赖度 |
| A5 | **去 A\* 信号**:prompt 不给 PathRatio/最优差距 | grounding 反馈价值 |
| A6 | **贪婪 accept**:任何 train 改进即接受,去 Wilcoxon | 统计门控防过拟合的作用(预期 test 上崩) |

每个消融报三诊断量:**final test-fitness**、**收敛所需评测数**(达 A0 90% 水平的候选数)、**有效提案率**(可编译且可行的代码提案/总提案)。A2 vs A0 量化"代码改动"、A6 vs A0 量化"统计门控"—— 消融里最能打的两条。

### 5.7 完整实验矩阵(方法 × 场景 × 指标)

每格 = 该场景组内 K seeds 的 mean±95%CI;`SR`=成功率、`PR`=PathRatio、`E/E*`=能耗比 A\*、`t`=plan_time ms、`S`=平滑度。

| 方法 \ 场景组 | Train:Blocks(A/B/C) | Train:墙/U | Train:随机 | **Test:随机** | **Test:迷宫(held-out)** | HV |
|---|---|---|---|---|---|---|
| A\*(oracle 上界) | 参考 | — | 参考 | 参考 | 参考 | — |
| Manual(现状) | ✓ | ✓ | ✓ | ✓ | ✓ | h₀ |
| Random Search | ✓ | ✓ | ✓ | ✓ | ✓ | h₁ |
| Bayesian Opt | ✓ | ✓ | ✓ | ✓ | ✓ | h₂ |
| CMA-ES | ✓ | ✓ | ✓ | ✓ | ✓ | h₃ |
| **LLM-param(轨道 1)** | ✓ | ✓ | ✓ | ✓ | ✓ | h₄ |
| **LLM-full(轨道 2,主张)** | ✓ | ✓ | ✓ | **★** | **★** | **h₅(最大)** |
| A1 去记忆 | — | — | ✓ | ✓ | ✓ | — |
| A2 只调参 | — | — | ✓ | ✓ | ✓ | — |
| A3 去反思 | — | — | ✓ | ✓ | ✓ | — |
| A4 弱 LLM | — | — | ✓ | ✓ | ✓ | — |
| A5 去 A\* 信号 | — | — | ✓ | ✓ | ✓ | — |
| A6 贪婪 accept | — | — | ✓ | ✓ | ✓ | — |

答辩三条主结论对应三块证据:**优于手动**(表第 2 vs 6/7 行)、**优于朴素搜索**(sample-efficiency 曲线)、**能泛化且触达参数空间之外**(Test 两列 + HV 的 h₅)。

---

## 6. Novelty 与文献定位

### 6.1 问题形式化(双层优化 / 自动科研)

规划器由(配置参数 θ ∈ `PlanningConfig`,算法组件代码 c ∈ {采样 `_smart_sample`、代价 `compute_cost`、扩展 `_steer`})共同决定:
- **内层(evaluate 实验引擎,客观)**:J(θ,c) = 场景集上聚合的多指标(PathRatio、能耗 J、成功率、计算耗时、轨迹质量);安全(ESDF 碰撞)为硬约束。
- **外层(研究智能体,自动)**:读 J 与实验记录 → 提假设 → 生成 θ 改动或 c 代码 diff → 触发内层评测 → 分析 → accept/reject → 写知识库 → 迭代。

**研究价值**:把只能白天靠专家跑的慢循环(`PlanningConfig` 约 20 个强耦合参数的 trade-off 知识目前只存在于代码注释和人脑;引入 DC-RRT 那类算法改进以周计)变成 24/7 自动运行、可探索远大于人工规模设计空间的闭环;LLM 天然携带文献知识(informed sampling、曲率引导、Dubins 连接),能把"读论文的直觉"直接落成可执行改动;客观可量化 reward(尤其 A\* 绝对基线)保证 accept/reject 严谨。

### 6.2 差异化定位

| 参照工作 | 它做什么 | 借鉴 | 本课题差异/更进一步 |
|---------|---------|------|------------------|
| **AI Scientist (Sakana)** | LLM 全自动科研,主在 ML 基准 | "LLM 作为自主研究者"闭环范式 | 落到**有物理仿真+硬安全约束的机器人域**;发现对象是**规划算法**;评测有客观物理量纲(焦耳、A\* 最优比) |
| **OPRO (Google)** | 历史(解,分数)轨迹 meta-prompt 迭代提优解 | 用历史尝试轨迹作 prompt 记忆 | 我们的"解"是**可执行规划器代码+配置**,由物理评测台打分 |
| **Eureka (NVIDIA)** | LLM 写 reward 函数进化搜索训 RL | LLM 直接写/改代价函数、采样策略;"多数样本失败靠选择保优" | Eureka 优化 RL 控制器奖励(要训练);我们优化**规划算法本身**,用 A\* 基线+BEMT 评测,**无需 RL 训练**,单次迭代秒级 |
| **AutoML/BO (Optuna/BOHB)** | 数值超参搜索,算法当黑盒 | "重复采样取统计"的严谨性 | BO **只能调数值参数、无法生成新代码组件、不会用自然语言读结果/给理由**;我们同时做**调参+代码结构改动+可读假设** |
| **Neural-RRT/MPNet** | 训练规划策略网络 | —— | 需大数据、产黑盒网络;我们保留**经典可解释规划器**,靠 LLM 写符号化代码改动,数据高效、可解释、可直接部署 AirSim |

**四个独特点(反复强调)**:
1. **域的选择使"自动科研"可信可验而非空谈**:UAV 3D 规划同时具备物理仿真、硬安全约束(LLM 无法作弊绕过)、`AStarPlanner` 给的绝对最优基线(刷不了分)、以及罕见的**两级评测**(秒级快速台跑海量迭代 + AirSim 高保真验证)。
2. **双优化范围**:既调参(AutoML 能做)又生成/替换算法组件(Eureka 那类),一个框架缝合"只调参"与"只改代码"。
3. **闭环含真实高保真验证**,直面纯 ML 自动科研普遍跳过的"基准过拟合/sim-to-real"问题。
4. **可验证 ground truth 让 accept/reject 有据可依**,区别于以往 LLM 自优化"自评自夸"的软评价。

---

## 7. 风险与缓解(去重汇总)

| # | 风险 | 缓解 |
|---|------|------|
| R1 | **Reward hacking**:LLM 靠降 success_rate 分母、放宽 safety_margin、钻评测台空子"虚假降能耗" | evaluate 引擎与 LLM 改动物理隔离,LLM 无法编辑评测代码;**硬约束守卫**(每场景 success_rate≥0.90、safety_viol=0、compute≤3×基线,任一不满足直接 reject 不看 score);evaluate 独立算 min_clearance,真碰撞则 F 置惩罚;`safety_margin≥0.6` 与 ref 类参数锁定 |
| R2 | **过拟合到静态场景**(Blocks/墙/U),回 AirSim 或换环境失效;winner's curse + 多重比较 | train/val/test 三分且 test 含 held-out 结构族(迷宫/密柱);ACCEPT 前 val 复核 + 每 K 轮 holdout 轮换 + val 早停;**最终配置全新 seed 在 test 上一次性无偏重评**;少量优胜配置回 AirSim 验证 |
| R3 | **RRT\* 随机性噪声**(C 场景 ±17m 长尾)把噪声当改善误接受 | 候选与 best 用**同组固定种子 K=20 配对 Wilcoxon**,要求 p<0.05 且中位改善>2% 且跨种子方差不显著增大;bootstrap CI;A6 贪婪 accept 消融量化噪声危害 |
| R4 | **LLM 生成代码含死循环/崩溃/恶意 IO**,拖垮或危害主控 | 四道防线:AST 白名单(禁危险 import 与 eval/exec/open)+ 契约测试 + 子进程 `subprocess.run(timeout)` + `setrlimit` + 小规模冒烟先行;崩溃/超时自动 reject + git 回滚;生成码只落 `strategies/generated/` 不覆盖核心 |
| R5 | **重复/近似提案**浪费预算与 token | `config_hash` 归一化去重(参数排序+代码 AST 规范化);`already_tried` 命中直接判 INVALID 并反馈"已试结果";System/User 常驻已试摘要与蒸馏 insight |
| R6 | **token 成本与延迟高**(每轮调 opus 生代码) | 仅 IDEATION/PROPOSE 调 LLM,EVALUATE/ANALYZE 纯本地;纯调参轮降级 sonnet;System+代码上下文用 prompt caching(ephemeral),按 `cache_read_input_tokens` 校验命中率;API 用 tool-use 强制 JSON + 重试超时;保留离线 MockProposer 与 `--replay` 兜底 |
| R7 | **生成组件访问 RRTStar 内部字段致接口漂移** | 冻结的 `SampleContext` 只读上下文 + `Protocol` 抽象基类固定三接口;注册前跑契约测试断言返回类型/形状/非负/有限且不改入参 |
| R8 | **RNG/超时重构致现有 benchmark 数值回归** | 默认策略为现有代码原样搬迁、默认走非 deterministic 分支;分阶段落地,先仅改 config/rrt_star 跑 benchmark 对比数值确认无回归再叠加 |
| R9 | **fitness 权重主观可调,结论不稳** | 同时报原始指标向量 + 多目标 hypervolume(固定参考点,不依赖标量权重)+ 权重敏感性附录 |
| R10 | **AirSim 接入成本高**(Windows/仿真器、单次慢),拖累进度 | 两级解耦:内层循环全在无 AirSim 快速台,AirSim 仅对最终 1-2 个配置确认、不进循环;中期设为可选/冲刺项,不具备条件时仅凭快速台完整演示,AirSim 全量下放终期(可降为预录视频) |
| R11 | **课题范围过大时间不足** | 分阶段交付:中期=调参闭环全通 + 一条 CostTerm 代码级改动 demo + 三张核心图;算法组件生成、泛化、AirSim 全量、BO/CMA-ES 消融下放终期;严格执行 §8 砍保清单 |

**诚实可行性判断**:纯调参闭环=低风险高把握(本质是 OPRO/AutoML 的 LLM 版,跑在已能工作的评测台);代码级改动=中风险已被三点去风险(只在定义好的注入点动刀、安全是评测台外部硬约束、接受"多数提案失败靠选择保优"的 Eureka 式心态,只要少数被接受的改动 + 框架本身贡献即成立);AirSim 验证=摩擦最大应降级可选。**总判断:中期"调参闭环全通 + 至少一个被接受的代码级改动 demo + 三张核心图"稳妥可达,即使部分提案退化课题结论不受影响。**

---

## 8. 从现在到中期的路线图与 MVP

### 8.1 里程碑 M0–M4

**M0 — evaluate 引擎与参数空间打通**
- 产出:`evaluator.py`、`param_space.py`(≥8 字段,每个带 type+min+max)、`metrics.py`(含 `fitness` + 硬约束)。
- 验收:①同 `(config,seed)` 调 `evaluate()` 两次数值**完全一致**;②两个不同 config 产**不同 score**;③A\* 结果缓存复用;④`fast` 档单次 <5s;⑤`validate` 拦越界/负值。

**M1 — 参数级自优化闭环跑通(LLM 只调参)**
- 产出:`proposer`(先 `MockProposer` 后 `LLMProposer`,tool-use 强制结构化 JSON)、`ledger`、`orchestrator/loop`;`python -m agentopt.loop --mode param --iters 20`。
- 验收:①≥15 轮不崩溃跑完;②至少一次 ACCEPT 使 objective 相对默认 config 改善且**重跑可复现**(判据 `Δmean>0.5×std`);③账本每行含 hypothesis/patch/metrics/decision/reason;④`best_config.json` 持久化;⑤断网用 MockProposer 也整轮跑通。

**M2 — 代码级组件替换接入**
- 产出:策略插件接口(`SamplingStrategy/CostTerm/SteerStrategy`)+ 改造 `RRTStar.__init__` 支持注入;`sandbox.py`(AST 白名单+冒烟);registry。
- 验收:①至少 1 个 LLM 生成组件(优先"新代价项",如加转弯/爬升平滑惩罚)通过沙箱进入完整闭环,被 accept 或**干净 reject**;②故意投喂危险代码(`import os`/`open`)时沙箱**必须拦截**;③生成组件不污染主命名空间;④组件替换与参数补丁走同一账本/回滚。

**M3 — 对照基线 + 结果图**
- 产出:`report.py`;三组对照——默认 config / 随机搜索(同预算)/ LLM 引导;输出 objective-vs-iteration 收敛曲线 + 前后指标表 + 路径对比图(复用 `generate_benchmark_figures`)。
- 验收:①三条收敛曲线**同预算**可比;②前后对比表覆盖各场景 mean±std 且可由账本复现;③LLM 引导在相同预算下**收敛更快或终值更优**(有统计说明);④一张 default-vs-best 俯视对比图。

**M4 — AirSim 少量验证 + 报告**
- 产出:top-1/2 config 注入 `fly_planned_path.py` 跑 1-2 次;中期报告。
- 验收:①至少一个优化后 config 在 AirSim 到达目标;②快速台趋势与 AirSim 定性一致(不要求数值吻合);③报告可复述闭环机制并附前后对比。

### 8.2 任务分解表

| # | 任务 | 依赖 | 优先级 | 工作量 |
|---|---|---|---|---|
| T1 | 从 `run_benchmark()` 抽纯函数 `evaluate()`,拆 print/savefig,加 seed 与 A\* 缓存 | — | P0 | 1.0 天 |
| T2 | `param_space.py`:PARAM_SPACE + `apply/validate` | T1 | P0 | 0.5 天 |
| T3 | `metrics.py`:`fitness()` + 硬约束(collision/success_rate) | T1 | P0 | 0.5 天 |
| T4 | `ledger`:JSONL 追加 + `load_best` + markdown 渲染 | — | P0 | 0.5 天 |
| T5 | `MockProposer` + `orchestrator/loop` 主控 + rollback | T2,T3,T4 | P0 | 1.0 天 |
| T6 | `LLMProposer`:Claude Messages API,strict tool-use(hypothesis+patch)+ 历史注入 | T5 | P0 | 1.5 天 |
| T7 | `fast/full` 双档 + `test_planning_simulation` 场景并入 evaluate | T1 | P1 | 0.5 天 |
| T8 | 策略插件接口(SamplingStrategy/CostTerm/SteerStrategy)+ 改造 RRTStar 支持注入 + RNG | T1 | P1 | 1.0 天 |
| T9 | `sandbox.py`:AST 白名单 + 冒烟测试 + registry | T8 | P1 | 1.0 天 |
| T10 | LLMProposer 扩展代码模式(生成组件源码字符串) | T6,T9 | P1 | 1.0 天 |
| T11 | 随机搜索基线脚本(同预算) | T5 | P1 | 0.5 天 |
| T12 | `report.py`:收敛曲线 + 前后对比表 + 路径对比图 | T5,T11 | P0 | 1.0 天 |
| T13 | `demo_short.py`:3-5 轮现场短迭代(挑可改善起点保证现场出 accept) | T6,T12 | P0 | 0.5 天 |
| T14 | AirSim 注入:best_config → `fly_planned_path.py` 跑 1-2 次 | T12 | P2 | 1.0 天 |
| T15 | 中期报告与 demo 排练 | T12,T13 | P0 | 1.0 天 |

P0 合计约 7.5 天,P0+P1 约 12.5 天,全量约 14.5 天(纯开发,不含调参/等实验)。按学生兼职每周~3 有效工作日:理想全量 4-5 周;**P0+P1(M0→M3 含一条代码级改动)约 3 周 = 稳妥中期目标**;仅 P0(M0→M1+报告)约 1.5-2 周 = 保底不翻车线。

### 8.3 砍量优先级

- **必保**:T1-T6(evaluate + 参数闭环 + Claude proposer)、T12-T13(报告+demo)、T11(随机搜索基线)。理由:没 evaluate 就没一切;随机搜索基线极便宜却是"LLM 是否真有用"的可信度关键;报告与 demo 是中期唯一被评产物。
- **保一条即可**:T8-T10 代码级改动**只做 CostTerm 一类**(不做采样泛化)。中期只需一条能被 accept 的代码级改动即证明能力。
- **可砍/降级**:T14 AirSim 实飞 → 降为**预录视频**;T9 沙箱 → 极限缺时间时降为"AST 白名单 + 人工确认"(但"拦危险代码"演示必留)。
- **绝不砍**:seed 可复现(T1)与硬约束/rollback(T3/T5)。没有可复现与约束,accept 可能是随机噪声,"自优化有效"论证会被一击致命。

---

## 9. 中期报告章节大纲 + 关键图表清单

### 9.1 章节大纲

1. **绪论** —— 背景(UAV 能量感知路径规划)、痛点(人工调参/设计慢且依赖专家)、提出"LLM 研究智能体驱动自优化闭环"、中期目标与贡献。
2. **相关工作** —— 采样式规划(RRT\*/DC-RRT)、能量感知规划、LLM 自动科研(AI Scientist/OPRO/Eureka)、AutoML/BO、learning-to-plan;本课题坐标定位(§6.2 表)。
3. **问题定义与总体框架** —— 双层优化形式化;系统架构图(研究智能体 ↔ evaluate 实验引擎 ↔ 知识库);优化对象 = `PlanningConfig` 参数 + 三个策略插件点。
4. **自优化闭环设计** —— 智能体七态状态机;结构化动作空间;OPRO 轨迹 prompt 与知识库记忆;代码安全沙箱(AST 白名单/子进程/git);多指标聚合成 fitness 的定义与硬约束守卫。
5. **evaluate 实验引擎与评测协议** —— 基于 `benchmark_planning.py`(A\* 基线、Blocks、多指标)与 `test_planning_simulation.py`(墙/U 场景、轨迹质量);确定性可复现;train/val/test 划分与配对统计。
6. **中期实现与初步实验** —— 跑通的闭环原型;一个纯调参案例 + 一个代码级改动案例(如重写 `_smart_sample` 或给 `compute_cost` 加曲率项);优化前后对比图/表;收敛曲线;LLM 改动样例卡片;随机搜索对照。
7. **风险分析与可行性判断** —— §7 逐条 + 诚实结论。
8. **后续工作(终期计划)** —— 算法组件生成扩展、更多场景与泛化、AirSim 全量验证、与 Random/BO/CMA-ES 的完整消融。

### 9.2 关键图表清单(按优先级)

1. **【必做·最重要】闭环架构流程图** —— 研究智能体 ↔ evaluate 实验引擎 ↔ 知识库三方,标注七步(读结果/提假设/改参或改码/评测/分析/accept-reject/记录)。回答"这个毕设到底在干嘛"。
2. **【必做】优化前后指标对比图/表** —— 复用 `generate_benchmark_figures` 风格:三场景(A/B/C)上 baseline vs LLM 优化后的路径长度、能耗 J、PathRatio、成功率柱状对比 + 误差棒,叠 A\* 最优水平线。
3. **【必做】优化收敛曲线** —— x=迭代次数、y=至今最优 fitness,accept/reject 点用不同标记(类比 OPRO/Eureka),直观证明闭环在变好;叠加随机搜索基线。
4. **【必做】LLM 改动样例卡片** —— 一个 assumption+rationale+真实 diff(如给 `compute_cost` 增曲率惩罚,或把 `_smart_sample` 改为目标方向 informed 高斯采样)+ 指标变化(Δ + accept/reject),展现"科研"质感。
5. **【建议】调参 vs 代码级改动贡献分解** —— 佐证 novelty 双范围。
6. **【选做/加分】AirSim 验证轨迹** —— 最优配置进 AirSim,复用 `fly_with_energy_visualization.py` 的 `energy_flight_visualization.png` 风格,证明两级回路闭合。

**汇总表**:行=指标(路径长度/能耗 J/PathRatio/成功率/计算耗时),列=Baseline / LLM-调参 / LLM-代码改动 / A\* 最优。

---

## 10. 中期演示脚本(15 分钟)

**核心策略**:预跑好的长迭代当"结果",现场短迭代当"活证据"(RRT\* 随机 + LLM 网络延迟不可控,忌现场等长迭代实时跑)。全程可 `--replay from ledger` 兜底防翻车。

1. **(1 min)一张图讲机制** —— 闭环框图(读结果→提假设→改参/改码→评测→accept/rollback→记账)。
2. **(3 min)亮结果先行** —— 打开预跑 30-40 轮账本渲染的 markdown + `report.py` 三条收敛曲线(默认/随机搜索/LLM)+ 前后指标表。一句话:"LLM 引导在同预算下把 energy/path_ratio 从 X 降到 Y。"
3. **(4 min)现场跑短迭代** —— `python -m agentopt.loop --mode param --iters 3`,**起点故意选已知可改善的 config**。逐轮念控制台:LLM 的 hypothesis → patch JSON → evaluate 指标 → **ACCEPT/REJECT + 理由** → 账本新增一行。让评委亲眼看到"提假设→验证→决策"。
4. **(3 min)代码级改动杀手锏** —— 回放一次 CostTerm 生成:LLM 生成源码片段 → 沙箱通过(**再演示一次投喂 `import os` 被拒**)→ 评测 → 决策。区别于"纯调参"的核心卖点,呼应双优化范围。
5. **(2 min)高保真兜底** —— 播放 AirSim 里优化后 config 的**预录视频**,对齐快速台结论。
6. **(2 min)收束** —— default vs best 指标表 + 下一步(泛化到更多组件/多场景/自动写实验结论)。

**防翻车**:现场 LLM/AirSim 出问题立即切预跑账本 `--replay` 回放,叙事不断。