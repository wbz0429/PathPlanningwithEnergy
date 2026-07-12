"""
planning_experiment.py — 工程缝合实验:三种代价(距离/教科书BEMT/真机M100)在同一 RRT*/A* 上对比。
把真机 M100 能耗模型包成 physics_eval.energy_astar 能吃的 `em` 接口(compute_energy_for_segment),
从而无缝插进已有的能量感知规划器。同一地图/起终点,只换代价 = 干净消融。

输出:①省能表(能量感知 vs 最短距离,省X%);②代价血统表(M100代价 vs BEMT代价选出不同路);
     ③双模型交叉验证(M100代价的路在两个尺子下都省 = 破循环)。
"""
import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # autoresearch/
import numpy as np
from planning_connect import PowerModel

HERE = os.path.dirname(os.path.abspath(__file__))
G = 9.81


class M100Em:
    """把真机 M100 能耗模型包成 energy_astar 的 em 接口。
    compute_energy_for_segment(start, end, velocity, dt) -> (energy_J, time_s)。
    可选 physics_climb=True:补上物理 mgh 载荷×爬升项(创新点③,修数据模型盲区)。"""
    def __init__(self, featurize_src, payload=250.0, physics_climb=False, eff=0.5, name="M100"):
        self.model = PowerModel(featurize_src, name)
        self.payload = payload
        self.physics_climb = physics_climb
        self.eff = eff
        self.name = name

    def compute_energy_for_segment(self, start_pos, end_pos, velocity, dt=0.1):
        disp = np.asarray(end_pos, float).reshape(-1)[:3] - np.asarray(start_pos, float).reshape(-1)[:3]
        length = float(np.linalg.norm(disp))
        if length < 1e-6:
            return 0.0, 0.0
        t = length / max(velocity, 0.5)
        lh = float(np.linalg.norm(disp[:2]))
        dz = float(disp[2])                          # NED: dz<0 = 上升
        v_h = lh / t
        v_z = -dz / t                                # 爬升率(正=上升):模型约定 正v_z=爬升(639W),
        #                                              NED里dz<0才是爬升→取负号。曾漏此符号→爬升被当下降低估152W
        st = {k: np.array([v]) for k, v in
              dict(v_h=v_h, v_z=v_z, a_h=0.0, a_z=0.0, omega=0.0,
                   payload=self.payload, wind=0.0, speed=velocity).items()}
        power = float(max(0.0, self.model.predict(st)[0]))
        energy = power * t
        if self.physics_climb:
            climb = max(0.0, -dz)                     # 上升高度(m)
            energy += (self.payload / 1000.0) * G * climb / self.eff   # 载荷爬升多做功
        return energy, t


class DistanceEm:
    """距离最短 baseline:代价=路径长度(能量感知规划器退化成最短路)。"""
    name = "距离最短"
    def compute_energy_for_segment(self, start_pos, end_pos, velocity, dt=0.1):
        length = float(np.linalg.norm(np.asarray(end_pos, float) - np.asarray(start_pos, float)))
        return length, length / max(velocity, 0.5)


def score_path(path, em, velocity=8.0):
    """用某个 em 当尺子,算一条路的总能耗(J)。"""
    if not path or len(path) < 2:
        return float("inf")
    tot = 0.0
    P = [np.asarray(p, float).reshape(-1)[:3] for p in path]
    for a, b in zip(P[:-1], P[1:]):
        e, _ = em.compute_energy_for_segment(np.zeros(3), b - a, velocity)
        tot += e
    return tot


if __name__ == "__main__":
    import physics_eval as pe
    print("载入地图 + 构造三种 em...")
    vg, esdf, bemt_em = pe.get_grounded_map()                 # 教科书 BEMT(仓库默认)
    best_src = open(os.path.join(HERE, "state", "best_featurize.py")).read()
    m100_em = M100Em(best_src, payload=250.0, name="M100真机")
    dist_em = DistanceEm()
    print("三种 em 就绪:距离 / BEMT / M100真机\n")

    # 冒烟:单场景,三代价各规划一次,验证 M100Em 能插进 energy_astar
    sc = pe.gen_scenarios([0])[0]
    s, g = sc["start"], sc["goal"]
    print(f"场景: start={np.round(s,1)} goal={np.round(g,1)}")
    for name, em in [("距离", dist_em), ("BEMT", bemt_em), ("M100", m100_em)]:
        t0 = time.time()
        path, _cost = pe.energy_astar(vg, esdf, em, s, g)    # 返回 (路径, 代价)
        dt = time.time() - t0
        e_m100 = score_path(path, m100_em)                    # M100 尺子评这条路
        e_len = score_path(path, dist_em)                     # 路长
        print(f"  代价={name:5s}: {len(path) if path else 0}航点, {dt:.0f}s, "
              f"路长={e_len:.0f}m, M100真机能耗={e_m100:.0f}J")
    print("\n✅ M100Em 成功插进 energy_astar —— 三种代价可跑。下一步:设计有障碍场景 + 批量对比。")
