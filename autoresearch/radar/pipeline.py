"""
pipeline.py — 可调优的经典跟踪管线:DBSCAN 聚类 → Kalman(CV)跟踪 → 门控匈牙利关联。
纯 numpy/scipy(无 sklearn 依赖)。内存安全:逐帧 step(),只保留活跃航迹(小对象列表)。

PipelineParams = autoresearch 的动作空间(冻结层评测器不在这里):
    eps, min_samples   —— DBSCAN
    q, r               —— Kalman 过程/量测噪声
    gate               —— 关联门控距离(m)
    n_confirm, max_miss—— 航迹确认/删除生命周期
autoresearch 要打的就是这几个旋钮的【联合】最优 + 可进化的聚类/关联组件。
"""
import numpy as np
from dataclasses import dataclass
from scipy.spatial import cKDTree
from scipy.optimize import linear_sum_assignment


@dataclass
class PipelineParams:
    eps: float = 1.5
    min_samples: int = 2
    q: float = 1.0          # 过程噪声强度
    r: float = 0.5          # 量测噪声方差(m^2)
    gate: float = 3.0       # 关联门控(m)
    n_confirm: int = 2      # 命中多少次确认航迹
    max_miss: int = 3       # 连续丢失多少帧删除


def dbscan(X, eps, min_samples):
    """轻量 DBSCAN(cKDTree)。X:(N,d)。返回 labels(-1=噪声)。"""
    n = len(X)
    labels = np.full(n, -1, int)
    if n == 0:
        return labels
    tree = cKDTree(X)
    neigh = tree.query_ball_point(X, eps)
    core = np.array([len(neigh[i]) >= min_samples for i in range(n)])
    visited = np.zeros(n, bool)
    cid = 0
    for i in range(n):
        if visited[i] or not core[i]:
            continue
        labels[i] = cid; visited[i] = True
        stack = [i]
        while stack:
            pt = stack.pop()
            for q in neigh[pt]:
                if labels[q] == -1:
                    labels[q] = cid           # 边界点并入
                if not visited[q]:
                    visited[q] = True
                    if core[q]:
                        stack.append(q)       # 核心点才扩展
        cid += 1
    return labels


def cluster_centroids(points_xy, eps, min_samples):
    """点云 → 簇质心(检测)。返回 (K,2)。"""
    if len(points_xy) == 0:
        return np.empty((0, 2))
    lab = dbscan(points_xy, eps, min_samples)
    cents = [points_xy[lab == k].mean(axis=0) for k in range(lab.max() + 1)] if lab.max() >= 0 else []
    return np.array(cents) if cents else np.empty((0, 2))


class _Track:
    __slots__ = ("id", "x", "P", "hits", "miss", "confirmed")

    def __init__(self, tid, pos, r):
        self.id = tid
        self.x = np.array([pos[0], pos[1], 0.0, 0.0])   # [px,py,vx,vy]
        self.P = np.diag([r, r, 25.0, 25.0])
        self.hits = 1; self.miss = 0; self.confirmed = False

    def predict(self, dt, q):
        F = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1.]])
        dt2, dt3, dt4 = dt*dt, dt**3, dt**4
        Q = q * np.array([[dt4/4, 0, dt3/2, 0], [0, dt4/4, 0, dt3/2],
                          [dt3/2, 0, dt2, 0], [0, dt3/2, 0, dt2]])
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q

    def update(self, z, r):
        H = np.array([[1, 0, 0, 0], [0, 1, 0, 0.]])
        R = r * np.eye(2)
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        y = z - H @ self.x
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ H) @ self.P

    @property
    def pos(self):
        return self.x[:2]


class MultiTargetTracker:
    """逐帧流式跟踪器。内存 = 活跃航迹数(小)。"""
    def __init__(self, params: PipelineParams, cluster_fn=None):
        self.p = params
        self.tracks = []
        self._next_id = 1
        # 可插拔聚类组件(默认=固定eps DBSCAN);baseline 用它换自适应DBSCAN等
        self.cluster_fn = cluster_fn or (lambda pts, pr: cluster_centroids(pts, pr.eps, pr.min_samples))

    def step(self, points_xy, dt):
        p = self.p
        # 1) 预测
        for t in self.tracks:
            t.predict(dt, p.q)
        # 2) 聚类 → 检测(经可插拔组件)
        dets = self.cluster_fn(np.asarray(points_xy, float).reshape(-1, 2), p)
        # 3) 门控匈牙利关联(预测航迹 vs 检测)
        matched_tr, matched_de = set(), set()
        if self.tracks and len(dets):
            TP = np.array([t.pos for t in self.tracks])
            D = np.linalg.norm(TP[:, None, :] - dets[None, :, :], axis=2)
            BIG = 1e6
            Dc = np.where(D <= p.gate, D, BIG)
            ri, ci = linear_sum_assignment(Dc)
            for r_, c_ in zip(ri, ci):
                if Dc[r_, c_] >= BIG:
                    continue
                self.tracks[r_].update(dets[c_], p.r)
                self.tracks[r_].hits += 1; self.tracks[r_].miss = 0
                if self.tracks[r_].hits >= p.n_confirm:
                    self.tracks[r_].confirmed = True
                matched_tr.add(r_); matched_de.add(c_)
        # 4) 未匹配的【旧】航迹 → 丢失计数(必须在追加新航迹之前,否则索引错位)
        for i, t in enumerate(self.tracks):
            if i not in matched_tr:
                t.miss += 1
        # 5) 未匹配检测 → 新航迹(miss=0)
        for c_ in range(len(dets)):
            if c_ not in matched_de:
                self.tracks.append(_Track(self._next_id, dets[c_], p.r))
                self._next_id += 1
        # 6) 删除长期丢失
        self.tracks = [t for t in self.tracks if t.miss <= p.max_miss]
        # 6) 输出已确认航迹
        return [(t.id, t.pos.copy()) for t in self.tracks if t.confirmed]


if __name__ == "__main__":
    # DBSCAN 小自测
    X = np.array([[0, 0], [0.1, 0], [0, 0.1], [10, 10], [10.1, 10]], float)
    lab = dbscan(X, eps=0.5, min_samples=2)
    assert lab[0] == lab[1] == lab[2] and lab[3] == lab[4] and lab[0] != lab[3], lab
    cents = cluster_centroids(X, 0.5, 2)
    assert len(cents) == 2, cents
    print("pipeline DBSCAN 自测过 ✓  两簇质心:", np.round(cents, 2).tolist())
