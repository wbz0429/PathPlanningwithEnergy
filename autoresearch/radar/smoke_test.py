"""
smoke_test.py — 合成流式 smoke test:造假雷达点云 → 跑整条管线 → OSPA/MOTA。
目的:在 RadarScenes 4GB 落地【之前】,验证 点云→DBSCAN→Kalman→关联→评测 整条闭环能跑通、
且行为合理。同时 evaluate_sequence() 就是后续 autoresearch 评测器的核心(冻结尺子 = OSPA/MOTA)。

内存安全:scene 是【生成器】,逐帧 yield;evaluate 逐帧消费,只留活跃航迹 + 标量累加器。
不同 seed = 不同场景 → 天然支持 held-out(训练 seed 调参,留出 seed 复核)。
"""
import numpy as np
from pipeline import PipelineParams, MultiTargetTracker
from ospa import ospa_distance
from mota import MOTAccumulator


def gen_scene(seed, n_frames=40, dt=0.1, clutter=(4, 9), pts_per_target=(6, 11), noise=0.3):
    """流式生成合成场景:3 个匀速目标,各喷一簇带噪点 + 均匀杂波。逐帧 yield。"""
    rng = np.random.default_rng(seed)
    targets = [(np.array([0., 0.]),  np.array([5., 0.]),   1),
               (np.array([0., 10.]), np.array([4., 0.6]),  2),
               (np.array([25., -4.]), np.array([-4., 1.2]), 3)]
    bbox_min, bbox_max = np.array([-5., -12.]), np.array([32., 16.])
    for f in range(n_frames):
        t = f * dt
        gt_ids, gt_pos, pts = [], [], []
        for pos0, vel, gid in targets:
            c = pos0 + vel * t
            gt_ids.append(gid); gt_pos.append(c)
            k = rng.integers(*pts_per_target)
            pts.append(c + rng.normal(0, noise, size=(k, 2)))
        nc = rng.integers(*clutter)
        pts.append(rng.uniform(bbox_min, bbox_max, size=(nc, 2)))
        yield np.vstack(pts), gt_ids, np.array(gt_pos)


def evaluate_sequence(scene_iter, params: PipelineParams, dt=0.1, ospa_c=2.0, mota_gate=2.0,
                      cluster_fn=None, associate_fn=None):
    """跑一条序列 → {OSPA_mean, MOTA, MOTP, ...}。流式、内存安全。这是评测器核心。"""
    trk = MultiTargetTracker(params, cluster_fn=cluster_fn, associate_fn=associate_fn)
    acc = MOTAccumulator(gate=mota_gate)
    ospas = []
    for pts, gt_ids, gt_pos in scene_iter:
        est = trk.step(pts, dt)                       # [(id, pos), ...] 已确认航迹
        est_ids = [e[0] for e in est]
        est_pos = np.array([e[1] for e in est]) if est else np.empty((0, 2))
        ospas.append(ospa_distance(est_pos, gt_pos, c=ospa_c))
        acc.update(gt_ids, gt_pos, est_ids, est_pos)
    m = acc.summary()
    m["OSPA_mean"] = float(np.mean(ospas)) if ospas else float(ospa_c)
    return m


if __name__ == "__main__":
    params = PipelineParams()          # 默认旋钮(未调优的基线)
    # 训练 seed 上评估
    m = evaluate_sequence(gen_scene(seed=0), params)
    print("默认参数 · seed0:", {k: round(v, 3) if isinstance(v, float) else v for k, v in m.items()})

    # 断言:整条闭环产出合理(默认参数就应能大致跟上 3 个目标)
    assert m["OSPA_mean"] < 0.9, f"OSPA 过大,管线没跟上: {m['OSPA_mean']}"
    assert m["MOTA"] > 0.6, f"MOTA 过低: {m['MOTA']}"
    assert m["GT"] == 120, m           # 3 目标 × 40 帧

    # 多 seed 复核(held-out 演示:换场景仍稳)
    vals = [evaluate_sequence(gen_scene(seed=s), params) for s in (1, 2, 3, 100, 200)]
    ospa_mu = np.mean([v["OSPA_mean"] for v in vals]); mota_mu = np.mean([v["MOTA"] for v in vals])
    print(f"留出 seeds(1,2,3,100,200): OSPA_mean={ospa_mu:.3f}  MOTA={mota_mu:.3f}")
    assert ospa_mu < 0.9 and mota_mu > 0.6, "留出场景不稳"

    print("\n✅ 合成 smoke test 全过 —— 点云→DBSCAN→Kalman→关联→OSPA/MOTA 整条闭环跑通,内存安全(流式)。")
    print("   evaluate_sequence() 即 autoresearch 评测器核心;换 seed = 换场景 = 天然 held-out。")
