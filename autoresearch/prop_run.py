# -*- coding: utf-8 -*-
"""prop_run.py — 逐个提交候选打分记账。我(proposer)每轮编辑 CANDIDATES 再跑。"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from llm_proposer_experiment import log
from candidate import DEFAULT_SMOOTHER_SRC

# === Claude proposer 候选 1:视线捷径 + 顶点回拉(能耗对转角敏感,拉直急转弯)===
LLM1 = '''
def smooth_path(path, is_collision_free, config):
    import numpy as np
    if len(path) <= 2:
        return path
    pts = [p.copy() for p in path]
    # 第1趟:多趟视线捷径(同基线,删冗余点)
    for _ in range(10):
        out = [pts[0]]; i = 0; shortened = False
        while i < len(pts) - 1:
            best_j = i + 1
            for j in range(len(pts) - 1, i + 1, -1):
                if is_collision_free(pts[i], pts[j]):
                    best_j = j; break
            if best_j > i + 1: shortened = True
            out.append(pts[best_j]); i = best_j
        pts = out
        if not shortened or len(pts) <= 2: break
    # 第2趟:顶点回拉——把每个中间点朝"前后点连线"方向移,拉直急转角(降剖面能耗)
    for _ in range(6):
        moved = False
        for i in range(1, len(pts) - 1):
            mid = 0.5 * (pts[i-1] + pts[i+1])          # 前后点中点
            for a in (0.7, 0.4, 0.2):                   # 试不同回拉强度
                cand = (1-a) * pts[i] + a * mid
                if is_collision_free(pts[i-1], cand) and is_collision_free(cand, pts[i+1]):
                    if np.linalg.norm(cand - pts[i]) > 1e-3:
                        pts[i] = cand; moved = True
                    break
        if not moved: break
    return pts
'''
log("llm_iter1", LLM1, "claude", "视线捷径+顶点回拉(拉直急转角降剖面能耗)")

# === 候选 2:视线捷径 + 转角圆弧化(能耗瓶颈=急转弯减速,给转角插入过渡点增大转弯半径)===
LLM2 = '''
def smooth_path(path, is_collision_free, config):
    import numpy as np
    if len(path) <= 2:
        return path
    pts = [p.copy() for p in path]
    # 第1趟:多趟视线捷径
    for _ in range(10):
        out = [pts[0]]; i = 0; sh = False
        while i < len(pts) - 1:
            bj = i + 1
            for j in range(len(pts) - 1, i + 1, -1):
                if is_collision_free(pts[i], pts[j]):
                    bj = j; break
            if bj > i + 1: sh = True
            out.append(pts[bj]); i = bj
        pts = out
        if not sh or len(pts) <= 2: break
    if len(pts) <= 2:
        return pts
    # 第2趟:转角圆弧化——每个转角用前后段各切一小段插两个过渡点,把尖角变钝(增大转弯半径→少减速)
    for _ in range(3):
        out = [pts[0]]; changed = False
        for i in range(1, len(pts) - 1):
            a, b, c = pts[i-1], pts[i], pts[i+1]
            d1 = np.linalg.norm(b - a); d2 = np.linalg.norm(c - b)
            if d1 < 1e-6 or d2 < 1e-6:
                out.append(b); continue
            # 转角越尖(夹角越小)切得越多;切段长度取相邻段的 25%
            cut = 0.25 * min(d1, d2)
            p1 = b - (b - a) / d1 * cut     # 进入点
            p2 = b + (c - b) / d2 * cut     # 离开点
            if is_collision_free(a, p1) and is_collision_free(p1, p2) and is_collision_free(p2, c):
                out.append(p1); out.append(p2); changed = True   # 用两个钝角过渡点替代尖角
            else:
                out.append(b)
        out.append(pts[-1]); pts = out
        if not changed: break
    return pts
'''
log("llm_iter2", LLM2, "claude", "视线捷径+转角圆弧化(增大转弯半径,直击急转减速能耗瓶颈)")

# === 候选 3:视线捷径 + 只对急转角做小步长回拉(避免破坏、只削尖角)===
LLM3 = '''
def smooth_path(path, is_collision_free, config):
    import numpy as np
    if len(path) <= 2:
        return path
    pts = [p.copy() for p in path]
    for _ in range(10):
        out = [pts[0]]; i = 0; sh = False
        while i < len(pts) - 1:
            bj = i + 1
            for j in range(len(pts) - 1, i + 1, -1):
                if is_collision_free(pts[i], pts[j]):
                    bj = j; break
            if bj > i + 1: sh = True
            out.append(pts[bj]); i = bj
        pts = out
        if not sh or len(pts) <= 2: break
    # 只对"急转角"(夹角<130度)做小步长回拉,缓解转弯减速;其余不动
    for _ in range(8):
        moved = False
        for i in range(1, len(pts) - 1):
            u = pts[i] - pts[i-1]; v = pts[i+1] - pts[i]
            cu, cv = np.linalg.norm(u), np.linalg.norm(v)
            if cu < 1e-6 or cv < 1e-6: continue
            ang = np.degrees(np.arccos(np.clip(u @ v / (cu*cv), -1, 1)))
            if ang <= 50:   # 转角平缓(方向变化<50度),不用管
                continue
            mid = 0.5 * (pts[i-1] + pts[i+1])
            cand = 0.75 * pts[i] + 0.25 * mid   # 小步长回拉
            if is_collision_free(pts[i-1], cand) and is_collision_free(cand, pts[i+1]):
                pts[i] = cand; moved = True
        if not moved: break
    return pts
'''
log("llm_iter3", LLM3, "claude", "视线捷径+急转角小步长回拉(只削尖角,不破坏平缓段)")

# === 候选 4:视线捷径 + CHOMP式全局平滑(最小化总曲率的联合梯度下降,MS2证过有效)===
LLM4 = '''
def smooth_path(path, is_collision_free, config):
    import numpy as np
    if len(path) <= 2:
        return path
    pts = [p.copy() for p in path]
    # 视线捷径
    for _ in range(10):
        out = [pts[0]]; i = 0; sh = False
        while i < len(pts) - 1:
            bj = i + 1
            for j in range(len(pts) - 1, i + 1, -1):
                if is_collision_free(pts[i], pts[j]):
                    bj = j; break
            if bj > i + 1: sh = True
            out.append(pts[bj]); i = bj
        pts = out
        if not sh or len(pts) <= 2: break
    if len(pts) <= 2:
        return pts
    # 在捷径后的骨架上按弧长重新密采样(给梯度平滑提供可动的中间点)
    seg = [np.linalg.norm(pts[i+1]-pts[i]) for i in range(len(pts)-1)]
    total = sum(seg)
    if total < 1e-6:
        return pts
    N = min(40, max(8, int(total / 3.0)))    # 每~3m 一个点
    cum = np.concatenate([[0], np.cumsum(seg)])
    ss = np.linspace(0, total, N)
    dense = []
    for s in ss:
        k = np.searchsorted(cum, s) - 1; k = max(0, min(k, len(pts)-2))
        t = (s - cum[k]) / max(seg[k], 1e-6)
        dense.append(pts[k] + t * (pts[k+1] - pts[k]))
    # CHOMP式平滑:最小化二阶差分(曲率),端点固定,梯度下降,每步查碰撞
    D = [p.copy() for p in dense]
    for _ in range(40):
        newD = [D[0]]
        for i in range(1, len(D)-1):
            lap = D[i-1] + D[i+1] - 2*D[i]        # 曲率梯度
            cand = D[i] + 0.3 * lap               # 朝平滑方向走
            if is_collision_free(D[i-1], cand) and is_collision_free(cand, D[i+1]):
                newD.append(cand)
            else:
                newD.append(D[i])
        newD.append(D[-1]); D = newD
    return D
'''
log("llm_iter4", LLM4, "claude", "视线捷径+弧长重采样+CHOMP式曲率梯度平滑(MS2的关键机制)")
