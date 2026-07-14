"""px4_fly_demo.py — 让 PX4 SITL(真飞控固件:EKF+位置环+混控)飞能量感知走廊路线,录真实轨迹。
飞 M100 代价选出的"翻矮楼A + 绕高楼B"路线,记录 odometry(位置+速度)→ 存 npz + 画轨迹图。
前置:另一终端 PX4 SITL(gz_x500)已 Ready for takeoff。"""
import asyncio, os, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

# 走廊 M100 路线(NED,z 负=高度):起飞→翻矮楼A(升到5m)→绕高楼B(横向-12,不升)→终点
WPS = [
    (0, 0, -3), (18, 0, -3), (23, 0, -6), (30, -3, -4),
    (40, -11, -4), (48, -12, -4), (56, -7, -4), (66, -2, -3), (72, 0, -3),
]


async def main():
    from mavsdk import System
    from mavsdk.offboard import PositionNedYaw, OffboardError
    drone = System()
    await drone.connect(system_address="udpin://0.0.0.0:14540")
    print("等待飞控连接...")
    async for st in drone.core.connection_state():
        if st.is_connected:
            print("  已连接"); break
    print("等待位置估计就绪(EKF+GPS)...")
    async for h in drone.telemetry.health():
        if h.is_global_position_ok and h.is_home_position_ok:
            print("  位置就绪"); break

    log = {"t": [], "pos": [], "vel": []}
    stop = asyncio.Event()

    async def rec():
        t0 = time.time()
        async for od in drone.telemetry.position_velocity_ned():
            log["t"].append(time.time() - t0)
            log["pos"].append([od.position.north_m, od.position.east_m, od.position.down_m])
            log["vel"].append([od.velocity.north_m_s, od.velocity.east_m_s, od.velocity.down_m_s])
            if stop.is_set():
                break
    rectask = asyncio.create_task(rec())

    print("解锁 + 进入 offboard...")
    await drone.action.arm()
    await drone.offboard.set_position_ned(PositionNedYaw(0., 0., -3., 0.))
    try:
        await drone.offboard.start()
    except OffboardError as e:
        print("offboard 失败:", e); await drone.action.disarm(); stop.set(); return

    for i, (n, e, d) in enumerate(WPS):
        await drone.offboard.set_position_ned(PositionNedYaw(float(n), float(e), float(d), 0.))
        t0 = time.time()
        while time.time() - t0 < 20:
            p = log["pos"][-1] if log["pos"] else None
            if p and np.linalg.norm(np.array(p) - np.array([n, e, d])) < 1.2:
                break
            await asyncio.sleep(0.1)
        print(f"  航点 {i+1}/{len(WPS)} 到达 -> ({n},{e},{-d}m高)")

    print("降落...")
    await drone.offboard.stop()
    await drone.action.land()
    await asyncio.sleep(3)
    stop.set(); await asyncio.sleep(0.3); rectask.cancel()

    T = {k: np.array(v) for k, v in log.items()}
    np.savez(os.path.join(HERE, "px4_traj.npz"), **T)
    print(f"\n轨迹记录 {len(T['t'])} 点,时长 {T['t'][-1]:.1f}s,最高 {-T['pos'][:,2].min():.1f}m,"
          f"横向范围 [{T['pos'][:,1].min():.1f},{T['pos'][:,1].max():.1f}]m")
    _plot(T)


def _plot(T):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "Heiti TC"]; plt.rcParams["axes.unicode_minus"] = False
    x = T["pos"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.5))
    a1.add_patch(Rectangle((22, 0), 3, 5, color="gray", alpha=.5)); a1.text(23, 5.3, "楼A 5m", fontsize=8)
    a1.add_patch(Rectangle((45, 0), 3, 12, color="gray", alpha=.5)); a1.text(45, 12.3, "楼B 12m", fontsize=8)
    a1.plot(x[:, 0], -x[:, 2], "g-", lw=2); a1.set_xlabel("x 北 (m)"); a1.set_ylabel("高度 (m)")
    a1.set_title("PX4 真飞控栈·侧视(飞出的真实轨迹)")
    a2.add_patch(Rectangle((22, -26), 3, 52, color="gray", alpha=.4))
    a2.add_patch(Rectangle((45, -10), 3, 20, color="gray", alpha=.5))
    a2.plot(x[:, 0], x[:, 1], "g-", lw=2); a2.set_xlabel("x 北 (m)"); a2.set_ylabel("y 东 (m)")
    a2.set_title("PX4 真飞控栈·俯视(翻矮楼A + 绕高楼B)")
    plt.suptitle("PX4 SITL + Gazebo Harmonic:能量感知路线的真飞控飞行轨迹")
    plt.tight_layout()
    out = os.path.join(HERE, "px4_traj.png")
    plt.savefig(out, dpi=125); print(f"轨迹图存 {out}")


if __name__ == "__main__":
    asyncio.run(main())
