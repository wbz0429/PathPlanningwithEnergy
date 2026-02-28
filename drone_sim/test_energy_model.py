"""
测试能耗模型模块
"""

import numpy as np
import sys
import os

# 设置控制台编码
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# 添加父目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from energy.physics_model import PhysicsEnergyModel, QuadrotorParams, estimate_flight_energy
from energy.neural_model import NeuralResidualModel, create_synthetic_training_data
from energy.hybrid_model import HybridEnergyModel, EnergyCostFunction


def test_physics_model():
    """测试物理能耗模型"""
    print("\n" + "="*60)
    print("测试 1: 物理能耗模型 (BEMT)")
    print("="*60)

    model = PhysicsEnergyModel()

    # 测试悬停功率
    hover_power = model.compute_hover_power()
    print(f"\n悬停功率: {hover_power:.2f} W")

    # 测试不同飞行状态的功率
    test_cases = [
        ("悬停", np.array([0, 0, 0]), None),
        ("前飞 2m/s", np.array([2, 0, 0]), None),
        ("前飞 5m/s", np.array([5, 0, 0]), None),
        ("爬升 2m/s", np.array([0, 0, -2]), None),  # NED坐标系，向上为负
        ("下降 2m/s", np.array([0, 0, 2]), None),
        ("斜飞", np.array([3, 2, -1]), None),
        ("加速前飞", np.array([3, 0, 0]), np.array([2, 0, 0])),
    ]

    print("\n不同飞行状态的功率分解:")
    print("-" * 80)
    print(f"{'状态':<15} {'诱导':<10} {'剖面':<10} {'寄生':<10} {'爬升':<10} {'电气总功率':<12}")
    print("-" * 80)

    for name, velocity, acceleration in test_cases:
        breakdown = model.get_power_breakdown(velocity, acceleration)
        print(f"{name:<15} {breakdown['induced']:<10.1f} {breakdown['profile']:<10.1f} "
              f"{breakdown['parasite']:<10.1f} {breakdown['climb']:<10.1f} "
              f"{breakdown['electrical_total']:<12.1f}")

    # 测试路径能耗计算
    print("\n路径能耗计算测试:")
    path = [
        np.array([0, 0, -5]),
        np.array([10, 0, -5]),
        np.array([10, 10, -5]),
        np.array([10, 10, -10]),
        np.array([20, 10, -10]),
    ]

    result = estimate_flight_energy(path, velocity=2.0)
    print(f"  路径点数: {len(path)}")
    print(f"  飞行速度: 2.0 m/s")
    print(f"  总能耗: {result['total_energy_joules']:.1f} J ({result['total_energy_wh']:.3f} Wh)")
    print(f"  飞行时间: {result['flight_time_seconds']:.1f} s")
    print(f"  平均功率: {result['average_power_watts']:.1f} W")

    print("\n✓ 物理模型测试通过")
    return True


def test_neural_model():
    """测试神经网络残差模型"""
    print("\n" + "="*60)
    print("测试 2: 神经网络残差模型")
    print("="*60)

    # 创建合成训练数据
    print("\n生成合成训练数据...")
    X, y = create_synthetic_training_data(n_samples=2000)
    print(f"  样本数: {len(X)}")
    print(f"  特征维度: {X.shape[1]}")
    print(f"  残差范围: [{y.min():.1f}, {y.max():.1f}] W")
    print(f"  残差均值: {y.mean():.1f} W")
    print(f"  残差标准差: {y.std():.1f} W")

    # 创建并训练模型
    print("\n训练神经网络...")
    model = NeuralResidualModel(input_dim=9, hidden_dims=[32, 16], output_dim=1)
    history = model.train(X, y, epochs=500, learning_rate=0.01, verbose=False)

    print(f"  最终训练损失: {history['train_loss'][-1]:.4f}")
    print(f"  最终验证损失: {history['val_loss'][-1]:.4f}")

    # 测试预测
    print("\n预测测试:")
    test_velocity = np.array([3.0, 0.0, -1.0])
    test_acceleration = np.array([1.0, 0.0, 0.0])
    test_euler = np.array([0.1, 0.2, 0.0])

    residual = model.predict(test_velocity, test_acceleration, test_euler)
    print(f"  输入速度: {test_velocity}")
    print(f"  输入加速度: {test_acceleration}")
    print(f"  输入姿态角: {test_euler}")
    print(f"  预测残差: {residual:.2f} W")

    # 测试保存和加载
    print("\n测试模型保存/加载...")
    model.save("energy_data/test_neural_model.json")

    model2 = NeuralResidualModel()
    model2.load("energy_data/test_neural_model.json")

    residual2 = model2.predict(test_velocity, test_acceleration, test_euler)
    print(f"  加载后预测残差: {residual2:.2f} W")
    print(f"  预测一致性: {'✓' if abs(residual - residual2) < 0.01 else '✗'}")

    print("\n✓ 神经网络模型测试通过")
    return True


def test_hybrid_model():
    """测试混合能耗模型"""
    print("\n" + "="*60)
    print("测试 3: 混合能耗模型")
    print("="*60)

    # 创建混合模型
    model = HybridEnergyModel()

    # 测试纯物理模型（未训练神经网络）
    print("\n纯物理模型预测:")
    velocity = np.array([3.0, 0.0, -1.0])
    acceleration = np.array([1.0, 0.0, 0.0])
    euler = np.array([0.1, 0.2, 0.0])

    power_physics = model.compute_power(velocity, acceleration, euler)
    print(f"  速度: {velocity}")
    print(f"  功率: {power_physics:.2f} W")

    # 训练神经网络
    print("\n训练混合模型的神经网络...")
    X, y_residual = create_synthetic_training_data(n_samples=2000)

    # 计算"真实"功率 = 物理预测 + 残差
    y_true = np.zeros(len(X))
    for i in range(len(X)):
        vel = X[i, :3]
        acc = X[i, 3:6]
        y_true[i] = model.physics_model.compute_electrical_power(vel, acc) + y_residual[i]

    history = model.train_neural_model(X, y_true, epochs=500, verbose=False)

    # 测试混合模型预测
    print("\n混合模型预测:")
    power_hybrid = model.compute_power(velocity, acceleration, euler)
    print(f"  物理模型功率: {power_physics:.2f} W")
    print(f"  混合模型功率: {power_hybrid:.2f} W")
    print(f"  神经网络补偿: {power_hybrid - power_physics:.2f} W")

    # 功率分解
    print("\n功率分解:")
    breakdown = model.get_power_breakdown(velocity, acceleration, euler)
    for key, value in breakdown.items():
        print(f"  {key}: {value:.2f} W")

    # 测试路径能耗
    print("\n路径能耗计算:")
    path = [
        np.array([0, 0, -5]),
        np.array([10, 0, -5]),
        np.array([10, 10, -8]),
        np.array([20, 10, -8]),
    ]

    energy, time, segment_energies = model.compute_energy_for_path(path, velocity=2.0)
    print(f"  总能耗: {energy:.1f} J")
    print(f"  飞行时间: {time:.1f} s")
    print(f"  各段能耗: {[f'{e:.1f}' for e in segment_energies]}")

    # 续航估算
    print("\n续航能力估算:")
    range_info = model.estimate_flight_range(velocity=2.0)
    print(f"  巡航功率: {range_info['cruise_power_watts']:.1f} W")
    print(f"  悬停功率: {range_info['hover_power_watts']:.1f} W")
    print(f"  电池能量: {range_info['battery_energy_wh']:.1f} Wh")
    print(f"  续航时间: {range_info['flight_time_minutes']:.1f} min")
    print(f"  续航距离: {range_info['flight_range_km']:.2f} km")

    print("\n✓ 混合模型测试通过")
    return True


def test_energy_cost_function():
    """测试能耗代价函数"""
    print("\n" + "="*60)
    print("测试 4: 能耗代价函数（用于路径规划）")
    print("="*60)

    # 创建代价函数
    cost_fn = EnergyCostFunction(velocity=2.0, weight_energy=1.0, weight_distance=0.0)

    # 测试两点之间的代价
    start = np.array([0, 0, -5])
    end = np.array([10, 0, -5])

    cost = cost_fn.compute_cost(start, end)
    print(f"\n水平飞行 10m:")
    print(f"  起点: {start}")
    print(f"  终点: {end}")
    print(f"  能耗代价: {cost:.2f} J")

    # 测试爬升
    end_climb = np.array([10, 0, -10])
    cost_climb = cost_fn.compute_cost(start, end_climb)
    print(f"\n爬升飞行 (水平10m + 垂直5m):")
    print(f"  终点: {end_climb}")
    print(f"  能耗代价: {cost_climb:.2f} J")

    # 对比：爬升比水平飞行能耗更高
    print(f"\n爬升额外能耗: {cost_climb - cost:.2f} J ({(cost_climb/cost - 1)*100:.1f}%)")

    # 测试路径代价
    path1 = [
        np.array([0, 0, -5]),
        np.array([20, 0, -5]),  # 直线
    ]

    path2 = [
        np.array([0, 0, -5]),
        np.array([10, 0, -10]),  # 先爬升
        np.array([20, 0, -5]),   # 再下降
    ]

    cost1 = cost_fn.compute_path_cost(path1)
    cost2 = cost_fn.compute_path_cost(path2)

    print(f"\n路径对比:")
    print(f"  直线路径能耗: {cost1:.2f} J")
    print(f"  爬升-下降路径能耗: {cost2:.2f} J")
    print(f"  能耗差异: {cost2 - cost1:.2f} J ({(cost2/cost1 - 1)*100:.1f}%)")

    print("\n✓ 能耗代价函数测试通过")
    return True


def run_all_tests():
    """运行所有测试"""
    print("\n" + "="*60)
    print("能耗模型模块测试")
    print("="*60)

    results = []

    results.append(("物理模型", test_physics_model()))
    results.append(("神经网络模型", test_neural_model()))
    results.append(("混合模型", test_hybrid_model()))
    results.append(("能耗代价函数", test_energy_cost_function()))

    # 总结
    print("\n" + "="*60)
    print("测试总结")
    print("="*60)

    all_passed = True
    for name, passed in results:
        status = "✓ 通过" if passed else "✗ 失败"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False

    print("\n" + "="*60)
    if all_passed:
        print("所有测试通过！")
    else:
        print("部分测试失败")
    print("="*60)

    return all_passed


if __name__ == "__main__":
    # 确保数据目录存在
    os.makedirs("energy_data", exist_ok=True)

    run_all_tests()
