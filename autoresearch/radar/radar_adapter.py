"""
radar_adapter.py — RadarScenes 真实数据 → evaluate_sequence 的流式适配器。【v0,待数据+h5py 落地后验证】

设计:输出与合成 gen_scene 完全同契约 —— 逐帧 yield (points_xy, gt_ids, gt_pos),
于是 smoke_test.evaluate_sequence / autoresearch_radar 一行不改就能换到真数据上。
内存安全:用 h5py 惰性读、逐 scene 流式;绝不把整条序列/4GB 读进内存。

RadarScenes 格式(radar-scenes.com 文档):
  data/sequence_XXX/
    scenes.json    —— {"scenes": {ts: {"radar_indices":[start,end], "sensor_id":k, ...}}, ...}
    radar_data.h5  —— dataset "radar_data" = 结构化数组,字段含:
        timestamp, sensor_id, x_cc, y_cc(车体系位置 m), vr_compensated(补偿径向速度),
        rcs, track_id(目标实例 id,静态/杂波为空), label_id(类别; 11=STATIC)
类别 label_id: 0 CAR,1 LARGE_VEHICLE,2 TRUCK,3 BUS,4 TRAIN,5 BICYCLE,
              6 MOTORIZED_TWO_WHEELER,7 PEDESTRIAN,8 PEDESTRIAN_GROUP,9 ANIMAL,10 OTHER,11 STATIC

关键设计决定(v0,数据到后需实测校准):
  - 管线输入 points_xy = 该帧【全部】检测的 (x_cc, y_cc)(含杂波/静态,让 DBSCAN 去滤)。
  - GT = 该帧【动态】目标(label_id != 11 且 track_id 非空)按 track_id 聚合的质心。
  - "帧" = 累积最近 `window` 个 sensor 测量(RadarScenes 4 传感器轮流,单测量点太稀)。
  - 只保留 ego 前方 ROI(如 x∈[0,60], |y|<=40)减少远处噪声——阈值待实测调。
"""
import os, json
import numpy as np

STATIC_LABEL = 11
DEFAULT_ROI = (0.0, 60.0, -40.0, 40.0)   # x_min,x_max,y_min,y_max (m)


def _load_h5(seq_dir):
    import h5py                                  # 延迟导入:仅用真数据时才需要
    h5 = h5py.File(os.path.join(seq_dir, "radar_data.h5"), "r")
    return h5["radar_data"]                       # 结构化数组 dataset(惰性)


def stream_sequence(seq_dir, window=4, roi=DEFAULT_ROI, dynamic_only_gt=True):
    """
    流式 yield (points_xy, gt_ids, gt_pos)。seq_dir = .../data/sequence_XXX/。
    window: 累积最近 window 个 sensor 测量当一"帧"(点云去稀疏)。内存 = window 个测量的点。
    """
    with open(os.path.join(seq_dir, "scenes.json")) as f:
        meta = json.load(f)
    scenes = meta["scenes"]
    data = _load_h5(seq_dir)
    xmin, xmax, ymin, ymax = roi
    from collections import deque
    buf = deque(maxlen=window)
    for ts in sorted(scenes, key=lambda t: int(t)):
        s = scenes[ts]
        a, b = s["radar_indices"]
        rec = data[a:b]                            # 只读这一小段(惰性)
        buf.append(rec)
        # 拼当前窗口
        recs = np.concatenate(list(buf)) if len(buf) > 1 else buf[0]
        x = np.asarray(recs["x_cc"], float); y = np.asarray(recs["y_cc"], float)
        roi_m = (x >= xmin) & (x <= xmax) & (y >= ymin) & (y <= ymax)
        x, y, recs = x[roi_m], y[roi_m], recs[roi_m]
        points_xy = np.column_stack([x, y]) if len(x) else np.empty((0, 2))
        # GT:动态目标按 track_id 聚合质心
        tid = recs["track_id"]; lab = np.asarray(recs["label_id"], int)
        # track_id 是 bytes/str;空 = 杂波
        tid_str = np.array([t.decode() if isinstance(t, (bytes, bytearray)) else str(t) for t in tid])
        dyn = (lab != STATIC_LABEL) & (tid_str != "") if dynamic_only_gt else (tid_str != "")
        gt_ids, gt_pos = [], []
        for u in np.unique(tid_str[dyn]):
            m = dyn & (tid_str == u)
            gt_ids.append(u); gt_pos.append([x[m].mean(), y[m].mean()])
        yield points_xy, gt_ids, (np.array(gt_pos) if gt_pos else np.empty((0, 2)))
    data.file.close()


def list_sequences(data_root):
    """列出所有 sequence_XXX 目录(train/test 划分用)。"""
    d = os.path.join(data_root, "data") if os.path.isdir(os.path.join(data_root, "data")) else data_root
    return sorted(p for p in (os.path.join(d, x) for x in os.listdir(d))
                  if os.path.isdir(p) and os.path.basename(p).startswith("sequence_"))


if __name__ == "__main__":
    import sys
    # 用法: python radar_adapter.py <sequence_XXX 目录>  —— 数据到后冒烟验证
    if len(sys.argv) < 2:
        print("待数据落地后:python radar_adapter.py ~/datasets/radar_scenes/data/sequence_1")
        print("(需先 pip install h5py;当前 h5py 未装)")
        sys.exit(0)
    seq = sys.argv[1]
    n_frames = 0; n_pts = []; n_gt = []
    for pts, gid, gpos in stream_sequence(seq):
        n_frames += 1; n_pts.append(len(pts)); n_gt.append(len(gid))
        if n_frames >= 200:
            break
    print(f"序列 {os.path.basename(seq)}: 前{n_frames}帧  点/帧均值={np.mean(n_pts):.0f}  动态目标/帧均值={np.mean(n_gt):.1f}")
    print("契约与 gen_scene 一致 → 可直接喂 evaluate_sequence()。")
