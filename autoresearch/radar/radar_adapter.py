"""
radar_adapter.py — RadarScenes 真实数据 → evaluate_sequence 的流式适配器。
【API 已对 radar_scenes 包校验;逻辑用迷你假序列自测通过。真数据到后仅需校准 window/ROI。】

输出契约与合成 gen_scene 完全一致 —— 逐帧 yield (points_xy, gt_ids, gt_pos),
于是 smoke_test.evaluate_sequence / autoresearch_radar 一行不改就能换到真数据。

内存安全:一次只开一条 sequence 的 radar_data.h5,按 scene 惰性切片读(不整读),
一次只留 window 个 scene 的点。跨 158 序列 = 逐序列处理、处理完即释放,绝不整读 4GB。

RadarScenes 格式(已核验 radar_scenes 包源码):
  sequence_XXX/scenes.json  = {"sequence_name","first_timestamp","last_timestamp",
                               "scenes": {ts: {"radar_indices":[a,b], "sensor_id":k, ...}}}
  sequence_XXX/radar_data.h5 = dataset "radar_data"(结构化),字段含
      x_cc,y_cc(车体系 m)、track_id(bytes,目标id;杂波/静态为空)、label_id(u1;11=STATIC)、vr_compensated
"""
import os, json
import numpy as np
from collections import deque

STATIC_LABEL = 11                          # 已核验 radar_scenes.labels.Label.STATIC == 11
DEFAULT_ROI = (0.0, 60.0, -40.0, 40.0)     # x_min,x_max,y_min,y_max (m);待真数据校准


def _decode(t):
    return t.decode() if isinstance(t, (bytes, bytearray)) else str(t)


def stream_sequence(seq_dir, window=4, roi=DEFAULT_ROI, dynamic_only_gt=True, vr_thresh=0.5):
    """
    逐帧 yield (points_xy, gt_ids, gt_pos)。seq_dir 含 scenes.json + radar_data.h5。
    window:累积最近 window 个 scene(4 传感器轮流,单 scene 太稀)当一帧。
    vr_thresh:MTI 动目标指示——管线输入只保留 |vr_compensated|>vr_thresh 的【运动】点
              (滤掉静止杂波,真雷达 MOT 标准第一步)。None=不滤。
    GT = 动态目标(label!=STATIC 且 track_id 非空)按 track_id 聚合质心——【不】受 vr 滤影响
         (真值反映真实目标,与我们的预处理无关)。
    """
    import h5py
    xmin, xmax, ymin, ymax = roi
    with open(os.path.join(seq_dir, "scenes.json")) as f:
        meta = json.load(f)
    scenes = meta["scenes"]
    order = sorted(scenes, key=lambda t: int(t))
    with h5py.File(os.path.join(seq_dir, "radar_data.h5"), "r") as h5:
        data = h5["radar_data"]
        buf = deque(maxlen=window)
        for ts in order:
            a, b = scenes[ts]["radar_indices"]
            buf.append(data[a:b])                          # 惰性切片,只读这一小段
            recs = np.concatenate(list(buf)) if len(buf) > 1 else buf[0]
            x = np.asarray(recs["x_cc"], float); y = np.asarray(recs["y_cc"], float)
            keep = (x >= xmin) & (x <= xmax) & (y >= ymin) & (y <= ymax)
            x, y, recs = x[keep], y[keep], recs[keep]
            # GT:真动态目标(不受 vr 滤影响)
            tid = np.array([_decode(t) for t in recs["track_id"]])
            lab = np.asarray(recs["label_id"], int)
            dyn = (tid != "") & ((lab != STATIC_LABEL) if dynamic_only_gt else True)
            gt_ids, gt_pos = [], []
            for u in np.unique(tid[dyn]):
                mu = dyn & (tid == u)
                gt_ids.append(u); gt_pos.append([x[mu].mean(), y[mu].mean()])
            # 管线输入:MTI 动目标滤波(滤静止杂波)
            if vr_thresh is not None and "vr_compensated" in recs.dtype.names:
                mv = np.abs(np.asarray(recs["vr_compensated"], float)) > vr_thresh
                px, py = x[mv], y[mv]
            else:
                px, py = x, y
            points_xy = np.column_stack([px, py]) if len(px) else np.empty((0, 2))
            yield points_xy, gt_ids, (np.array(gt_pos) if gt_pos else np.empty((0, 2)))


def list_sequences(data_root):
    """列出所有 sequence_XXX 目录(train/test 按序列划分用)。"""
    d = os.path.join(data_root, "data")
    d = d if os.path.isdir(d) else data_root
    return sorted(p for p in (os.path.join(d, x) for x in os.listdir(d))
                  if os.path.isdir(p) and os.path.basename(p).startswith("sequence_"))


# ---------------------------------------------------------------------------
def _make_fake_sequence(dirpath, n_scenes=12):
    """造一个格式忠实的迷你假序列(scenes.json + radar_data.h5),用于自测适配器逻辑。"""
    import h5py
    os.makedirs(dirpath, exist_ok=True)
    dt = np.dtype([("x_cc", "<f4"), ("y_cc", "<f4"), ("vr_compensated", "<f4"),
                   ("track_id", "S32"), ("label_id", "u1")])
    rng = np.random.default_rng(0)
    all_rows, scenes = [], {}
    # 两个动态目标匀速 + 每 scene 若干杂波(静态)
    tgt = [(np.array([10., 0.]), np.array([1.0, 0.2]), b"obj_a"),
           (np.array([15., -5.]), np.array([0.5, 0.5]), b"obj_b")]
    for s in range(n_scenes):
        start = len(all_rows)
        for pos0, vel, tid in tgt:
            c = pos0 + vel * s
            for _ in range(rng.integers(3, 6)):
                p = c + rng.normal(0, 0.25, 2)
                all_rows.append((p[0], p[1], 0.0, tid, 0))          # label 0 = CAR(动态)
        for _ in range(rng.integers(2, 5)):                          # 杂波/静态
            all_rows.append((rng.uniform(0, 40), rng.uniform(-20, 20), 0.0, b"", STATIC_LABEL))
        scenes[str(100 + s)] = {"radar_indices": [start, len(all_rows)], "sensor_id": (s % 4) + 1}
    arr = np.array(all_rows, dtype=dt)
    with h5py.File(os.path.join(dirpath, "radar_data.h5"), "w") as h5:
        h5.create_dataset("radar_data", data=arr)
    with open(os.path.join(dirpath, "scenes.json"), "w") as f:
        json.dump({"sequence_name": "fake", "first_timestamp": 100,
                   "last_timestamp": 100 + n_scenes - 1, "scenes": scenes}, f)
    return dirpath


if __name__ == "__main__":
    import sys, tempfile
    if len(sys.argv) > 1:   # 真数据:python radar_adapter.py ~/datasets/radar_scenes/data/sequence_1
        seq = sys.argv[1]
        npf = []; ngt = []
        for i, (pts, gid, gpos) in enumerate(stream_sequence(seq)):
            npf.append(len(pts)); ngt.append(len(gid))
            if i >= 300:
                break
        print(f"真序列 {os.path.basename(seq)}: {len(npf)}帧 点/帧={np.mean(npf):.0f} 动态目标/帧={np.mean(ngt):.1f}")
    else:   # 自测:造假序列跑适配器逻辑
        d = _make_fake_sequence(os.path.join(tempfile.gettempdir(), "fake_seq"))
        frames = list(stream_sequence(d, window=1, roi=(-5, 50, -30, 30), vr_thresh=None))
        assert len(frames) == 12, len(frames)
        # 每帧应恢复 2 个动态目标;点云含杂波(>动态点)
        gt_counts = [len(g) for _, g, _ in frames]
        assert all(c == 2 for c in gt_counts), f"应每帧2动态目标: {gt_counts}"
        ids = set(frames[0][1]); assert ids == {"obj_a", "obj_b"}, ids
        # GT 质心应随时间移动(obj_a 从 x≈10 前移)
        xa0 = [p for i, p in zip(frames[0][1], frames[0][2]) if i == "obj_a"][0][0]
        xa5 = [p for i, p in zip(frames[5][1], frames[5][2]) if i == "obj_a"][0][0]
        assert xa5 > xa0 + 3, f"obj_a 应前移: {xa0:.1f}->{xa5:.1f}"
        # 契约兼容:能直接喂 evaluate_sequence
        from smoke_test import evaluate_sequence
        from pipeline import PipelineParams
        m = evaluate_sequence(stream_sequence(d, window=1, roi=(-5, 50, -30, 30), vr_thresh=None), PipelineParams())
        print("适配器自测全过 ✓  假序列 evaluate:",
              {k: round(v, 3) if isinstance(v, float) else v for k, v in m.items() if k in ("OSPA_mean", "MOTA", "GT")})
        print("契约与 gen_scene 一致 → 真数据到后 stream_sequence 直接替换 gen_scene 即可。")
