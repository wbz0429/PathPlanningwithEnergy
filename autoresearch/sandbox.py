"""
sandbox.py — LLM 生成代码的安全校验 + 执行超时保护

Code safety for the autoresearch closed loop.
校验 LLM 生成的候选函数源码:AST 白名单(禁危险 import / eval / open / dunder),
并提供基于 SIGALRM 的墙钟超时上下文,防止候选代码死循环拖垮主控。
"""
import ast
import signal
from contextlib import contextmanager

# 只允许这些顶层模块被 import
ALLOWED_IMPORT_ROOTS = {"numpy", "math", "scipy", "typing"}
# 直接禁用的名字(调用/引用即拒)
FORBIDDEN_NAMES = {
    "os", "sys", "subprocess", "socket", "shutil", "pathlib", "importlib",
    "open", "eval", "exec", "compile", "__import__", "input",
    "exit", "quit", "globals", "locals", "vars", "getattr", "setattr", "delattr",
}


def check_code(src: str):
    """
    静态校验候选源码。

    Returns:
        (ok: bool, reason: str)
    """
    try:
        tree = ast.parse(src)
    except SyntaxError as e:
        return False, f"SyntaxError: {e}"

    for node in ast.walk(tree):
        # 1) import 白名单
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root not in ALLOWED_IMPORT_ROOTS:
                    return False, f"forbidden import: {alias.name}"
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".")[0]
            if root not in ALLOWED_IMPORT_ROOTS:
                return False, f"forbidden import-from: {node.module}"
        # 2) 危险名字调用/引用
        elif isinstance(node, ast.Name):
            if node.id in FORBIDDEN_NAMES:
                return False, f"forbidden name: {node.id}"
        # 3) dunder 属性访问(__globals__ / __subclasses__ ...)
        elif isinstance(node, ast.Attribute):
            if node.attr.startswith("__") and node.attr.endswith("__"):
                return False, f"forbidden dunder access: {node.attr}"
    return True, "ok"


class SandboxTimeout(Exception):
    pass


@contextmanager
def time_limit(seconds: float):
    """SIGALRM 墙钟超时(仅主线程可用)。超时抛 SandboxTimeout。"""
    def _handler(signum, frame):
        raise SandboxTimeout(f"execution exceeded {seconds}s")

    old = signal.signal(signal.SIGALRM, _handler)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)
