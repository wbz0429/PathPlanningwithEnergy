"""
candidate_energy.py — 代码级可进化组件:能耗模型 featurize(s)。经沙箱注入 m100_eval,绝不改评测器。
这是 UAV 能耗版 autoresearch 里让 LLM 进化的对象:进化"从运动学状态预测功率"的特征/函数形式,
对着真实 M100 功率的 held-out 误差评。与 radar 的 cluster/associate 完全平行。

契约:featurize(s) -> (N,K) 矩阵。s = dict of arrays:v_h,v_z,a_h,a_z,omega,payload,wind,speed。
安全:load 先过 sandbox.check_code(禁 os/sys/open/eval/dunder,只许 numpy/math/scipy)。
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # 找 autoresearch/sandbox.py
import numpy as np
from sandbox import check_code

DEFAULT_FEATURIZE_SRC = '''
def featurize(s):
    """默认 = 稳态 BEMT U 形:P ~ 1 + v + v²(仅水平速度)。"""
    return np.column_stack([np.ones_like(s["v_h"]), s["v_h"], s["v_h"] ** 2])
'''


def load_featurize(src):
    ok, reason = check_code(src)
    if not ok:
        raise ValueError(f"sandbox reject: {reason}")
    ns = {"np": np, "numpy": np}
    exec(compile(src, "<featurize>", "exec"), ns)
    fn = ns.get("featurize")
    if not callable(fn):
        raise ValueError("source must define callable `featurize(s)`")
    return fn


def contract_test_featurize(fn):
    s = {k: np.linspace(0.1, 5, 20) for k in
         ("v_h", "v_z", "a_h", "a_z", "omega", "payload", "wind", "speed")}
    X = np.asarray(fn(s), float)
    assert X.ndim == 2 and X.shape[0] == 20, "featurize 须返回 (N,K)"
    assert np.all(np.isfinite(X)), "特征须有限"


if __name__ == "__main__":
    from m100_eval import evaluate
    fn = load_featurize(DEFAULT_FEATURIZE_SRC)
    contract_test_featurize(fn)
    m = evaluate(fn, seed=0)
    print(f"默认 featurize 注入 ✓  held-out R²={m['r2']:.3f} 能量ARE={m['energy_ARE']*100:.2f}%")
    # 沙箱拦截
    for bad in ["import os\ndef featurize(s): return os.getcwd()",
                "def featurize(s): return s.__class__"]:
        try:
            load_featurize(bad); print("❌ 未拦截")
        except ValueError as e:
            print(f"沙箱拦截 ✓ ({str(e)[:36]})")
