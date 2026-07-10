"""
mota.py — 身份感知的第二把尺子:CLEAR-MOT(MOTA / MOTP + ID switch)。
Bernardin & Stiefelhagen 2008。逐帧 update(),只累加标量 + 一个 {gt_id->上次假设id} 的小字典。
内存安全:字典大小 = 见过的不同 GT 目标数(有界、小);不囤轨迹。冻结层尺子。

MOTA = 1 - (FN + FP + IDSW) / ΣGT   (越高越好,<=1)
MOTP = Σ匹配距离 / Σ匹配数           (越低越好,匹配的平均定位误差)
"""
import numpy as np
from scipy.optimize import linear_sum_assignment


class MOTAccumulator:
    def __init__(self, gate=2.0):
        self.gate = gate
        self.fp = 0; self.fn = 0; self.idsw = 0
        self.n_gt = 0; self.n_match = 0; self.dist_sum = 0.0
        self.prev = {}   # gt_id -> 上一次匹配到的 track_id(粘滞,用于跨帧检测 ID 切换)

    def update(self, gt_ids, gt_pos, tr_ids, tr_pos):
        gt_ids = list(gt_ids); tr_ids = list(tr_ids)
        gt_pos = np.asarray(gt_pos, float).reshape(-1, 2)
        tr_pos = np.asarray(tr_pos, float).reshape(-1, 2)
        self.n_gt += len(gt_ids)
        matches = {}                 # gt_id -> track_id(本帧)
        used_tr = set()
        # 1) 优先保持上一帧的匹配(仍在且在门内)→ 不算 ID 切换
        for i, g in enumerate(gt_ids):
            h = self.prev.get(g)
            if h is not None and h in tr_ids and h not in used_tr:
                j = tr_ids.index(h)
                d = float(np.linalg.norm(gt_pos[i] - tr_pos[j]))
                if d <= self.gate:
                    matches[g] = h; used_tr.add(h)
                    self.n_match += 1; self.dist_sum += d
        # 2) 剩余 GT / track 做门控最优分配(匈牙利)
        rem_gt = [i for i, g in enumerate(gt_ids) if g not in matches]
        rem_tr = [j for j, h in enumerate(tr_ids) if h not in used_tr]
        if rem_gt and rem_tr:
            D = np.linalg.norm(gt_pos[rem_gt][:, None, :] - tr_pos[rem_tr][None, :, :], axis=2)
            BIG = 1e6
            Dc = np.where(D <= self.gate, D, BIG)
            ri, ci = linear_sum_assignment(Dc)
            for r, c in zip(ri, ci):
                if Dc[r, c] >= BIG:
                    continue
                g = gt_ids[rem_gt[r]]; h = tr_ids[rem_tr[c]]
                matches[g] = h; used_tr.add(h)
                self.n_match += 1; self.dist_sum += float(D[r, c])
                prevh = self.prev.get(g)
                if prevh is not None and prevh != h:   # 该 GT 换了个假设 id → ID 切换
                    self.idsw += 1
        # 3) FP(没匹配上任何 GT 的 track) / FN(没被跟上的 GT)
        self.fp += len(tr_ids) - len(used_tr)
        self.fn += len(gt_ids) - len(matches)
        # 4) 粘滞更新:被匹配的 GT 记住其假设;未出现的保留旧记录(occlusion 后仍能查 ID 切换)
        self.prev.update(matches)

    def mota(self):
        return 1.0 - (self.fn + self.fp + self.idsw) / max(1, self.n_gt)

    def motp(self):
        return self.dist_sum / max(1, self.n_match)

    def summary(self):
        return {"MOTA": self.mota(), "MOTP": self.motp(), "FP": self.fp,
                "FN": self.fn, "IDSW": self.idsw, "GT": self.n_gt, "matches": self.n_match}


if __name__ == "__main__":
    # 合成自测(无需数据)
    # 1) 完美跟踪:est≡gt、id 一致 → MOTA=1, MOTP=0, IDSW=0
    acc = MOTAccumulator(gate=2.0)
    for t in range(10):
        gp = np.array([[t, 0.], [t, 5.]])
        acc.update([1, 2], gp, [1, 2], gp)
    s = acc.summary()
    assert abs(s["MOTA"] - 1.0) < 1e-9 and s["IDSW"] == 0 and s["FP"] == 0 and s["FN"] == 0, s
    assert s["MOTP"] < 1e-9, s

    # 2) 每帧漏一个 GT(只报 1 号)→ FN 累积,MOTA<1
    acc = MOTAccumulator(gate=2.0)
    for t in range(10):
        acc.update([1, 2], np.array([[t, 0.], [t, 5.]]), [1], np.array([[t, 0.]]))
    s = acc.summary()
    assert s["FN"] == 10 and abs(s["MOTA"] - (1 - 10/20)) < 1e-9, s

    # 3) ID 切换:前5帧 track1↔gt1,后5帧 track 变 99 → 应记 1 次 IDSW
    acc = MOTAccumulator(gate=2.0)
    for t in range(5):
        acc.update([1], np.array([[t, 0.]]), [1], np.array([[t, 0.]]))
    for t in range(5, 10):
        acc.update([1], np.array([[t, 0.]]), [99], np.array([[t, 0.]]))
    s = acc.summary()
    assert s["IDSW"] == 1, s

    # 4) 虚警:多报一个不存在的 track → FP 累积
    acc = MOTAccumulator(gate=2.0)
    for t in range(10):
        acc.update([1], np.array([[t, 0.]]), [1, 7], np.array([[t, 0.], [t, 50.]]))
    s = acc.summary()
    assert s["FP"] == 10, s

    # 5) 带定位误差的匹配 → MOTP 反映平均误差
    acc = MOTAccumulator(gate=2.0)
    for t in range(10):
        acc.update([1], np.array([[t, 0.]]), [1], np.array([[t, 0.3]]))   # 偏 0.3m
    s = acc.summary()
    assert abs(s["MOTP"] - 0.3) < 1e-9 and abs(s["MOTA"] - 1.0) < 1e-9, s

    print("MOTA/MOTP 自测全过 ✓ (完美/漏检/ID切换/虚警/定位误差 5 场景)")
