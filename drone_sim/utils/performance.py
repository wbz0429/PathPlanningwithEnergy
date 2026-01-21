"""
Performance monitoring utilities
"""

import time
import numpy as np
from collections import defaultdict
from contextlib import contextmanager


class PerformanceMonitor:
    """
    监控各模块耗时
    """

    def __init__(self):
        self.timings = defaultdict(list)
        self.current_timers = {}

    @contextmanager
    def measure(self, name: str):
        """
        测量代码块执行时间

        Usage:
            with perf_monitor.measure('mapping'):
                # code to measure
        """
        start = time.time()
        try:
            yield
        finally:
            elapsed = (time.time() - start) * 1000  # ms
            self.timings[name].append(elapsed)

    def get_average(self, name: str, last_n: int = 10) -> float:
        """获取最近 N 次的平均耗时"""
        if name not in self.timings or len(self.timings[name]) == 0:
            return 0.0
        return np.mean(self.timings[name][-last_n:])

    def get_total_average(self, last_n: int = 10) -> float:
        """获取所有模块的总耗时平均值"""
        total = 0.0
        for name in self.timings:
            total += self.get_average(name, last_n)
        return total

    def print_summary(self, last_n: int = 10):
        """打印性能摘要"""
        if not self.timings:
            return

        print("\n  Performance:")
        total = 0.0
        for name, times in self.timings.items():
            avg = self.get_average(name, last_n)
            total += avg
            print(f"    {name}: {avg:.1f}ms")

        print(f"    TOTAL: {total:.1f}ms")

        if total > 500:
            print("    [WARNING] Cycle time > 500ms!")

    def reset(self):
        """重置所有计时器"""
        self.timings.clear()
        self.current_timers.clear()
