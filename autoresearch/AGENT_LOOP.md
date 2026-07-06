# 研究智能体自优化闭环 (agent_loop)

> 在原有 `search.py`(经典参数搜索)+ 人工代码改写(Stage 2-4)的基础上,
> **把"提议"这一环交给 LLM**,实现"读结果→提假设→改参数或改代码→评测→keep/revert→迭代"的自动化闭环。
> 这是本课题从"autoresearch 方法论(人在环)"到"LLM 智能体在环"的关键增量。

## 与已有工作的关系

| | 提议者 | 优化对象 | 谁做代码改写 |
|---|---|---|---|
| `search.py`(已有) | 随机 + 高斯扰动 | 仅参数 | — |
| Stage 2-4(已有) | 人(+LLM 辅助) | 参数 + 代码 | **人工手动** |
| **`agent_loop.py`(新)** | **Proposer(LLM / Mock)** | **参数 + 代码** | **智能体自动** |

`agent_loop` 复用**同一个固定评测器 `evaluator.py`**(A\* 基线 + 三场景 + BEMT 能耗 + 100% 成功硬约束),
保证与已有 Stage 1-4 结果**同口径可比**——手动跑出的 4 阶段天然成为智能体要复现/超越的 ground-truth 轨迹。

## 架构

```
        ┌──────────── 研究智能体闭环 ────────────┐
program.md 策略 ─┐                                │
当前最优 + 历史 ─┼─► Proposer ── 结构化 Action ──┐ │
                 │   (LLM/Mock)  set_params /     │ │
                 │               rewrite_smoother │ │
                 │                                ▼ │
                 │      ┌── 参数动作 ──► evaluator.evaluate(overrides)
                 │      │
                 │      └── 代码动作 ─► sandbox.check_code(AST白名单)
                 │                     → candidate.contract_test
                 │                     → monkeypatch RRTStar._smooth_path
                 │                     → 墙钟超时保护 → evaluate
                 │                                │
        keep/revert ◄── score 更低? ◄────────────┘
        (JSONL 记录 + 持久化 best)
```

**安全**:LLM 生成的代码只落 `autoresearch/generated/`,经 AST 白名单(禁 os/sys/open/eval)、
契约测试、类级 monkeypatch(**不改 `drone_sim/planning/rrt_star.py`**)、SIGALRM 墙钟超时四道防护;
碰撞检测 `is_collision_free` 由 RRTStar 把关,平滑器改不动安全边界。

## 文件

| 文件 | 作用 |
|---|---|
| `agent_loop.py` | 闭环主体:propose→apply→evaluate→keep/revert→log |
| `proposer.py` | `MockProposer`(离线确定性)/ `LLMProposer`(Anthropic API, tool-use) |
| `actions.py` | 结构化动作空间 + 参数裁剪/权重归一化 |
| `candidate.py` | 代码进化目标 `smooth_path` 契约 + 加载 + monkeypatch |
| `sandbox.py` | AST 白名单 + SIGALRM 超时 |
| `report.py` | 出图:`fig_convergence.png` / `fig_before_after.png` |

## 运行

```bash
# 1) 离线闭环(无需 API key,证明机器跑通、出日志)
.venv/bin/python autoresearch/agent_loop.py --proposer mock --iters 12 --runs 3 --code

# 2) LLM 在环闭环(需先设 key)
export ANTHROPIC_API_KEY=sk-...
.venv/bin/python autoresearch/agent_loop.py --proposer llm --model claude-opus-4-8 --iters 20 --runs 3 --code

# 3) 出中期答辩图(收敛曲线 + 优化前后对比)
.venv/bin/python autoresearch/report.py 5
```

产物:`experiments/agent_log.jsonl`(每轮 假设/动作/指标/决策)、`experiments/agent_best.json`、
`generated/best_smoother.py`(若代码改动被接受)、`experiments/fig_*.png`。

## 诚实边界(答辩需说明)

- **LLMProposer 需要 `ANTHROPIC_API_KEY`**;当前开发机未配置,故用 `MockProposer` 验证闭环机制。
  接入真实 LLM 只需设置 key,架构与评测完全不变。
- MockProposer 的"提议"是启发式扰动 + 一个预置代码改写,用于**证明闭环管线正确**(sandbox/契约/
  monkeypatch/keep-revert 全通),**不代表 LLM 的研究能力**——后者由 LLMProposer 承载。
- 能耗是 BEMT 物理模型值,非真机实测(与原 REPORT.md 口径一致)。
