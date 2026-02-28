"""
Neural Network Residual Model for Energy Prediction

神经网络残差补偿模块
用于补偿物理模型在动态机动下的预测误差
"""

import numpy as np
from typing import Optional, Tuple, List
import os
import json


class NeuralResidualModel:
    """
    轻量级神经网络残差模型 (MLP)

    输入特征：
    - 速度 (vx, vy, vz)
    - 加速度 (ax, ay, az)
    - 姿态角 (roll, pitch, yaw)

    输出：
    - 功率残差 ΔP (W)
    """

    def __init__(self, input_dim: int = 9, hidden_dims: List[int] = [32, 16],
                 output_dim: int = 1):
        """
        初始化神经网络

        Args:
            input_dim: 输入维度 (默认9: 3速度 + 3加速度 + 3姿态角)
            hidden_dims: 隐藏层维度列表
            output_dim: 输出维度
        """
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim

        # 初始化权重
        self._init_weights()

        # 输入归一化参数
        self.input_mean = np.zeros(input_dim)
        self.input_std = np.ones(input_dim)

        # 输出归一化参数
        self.output_mean = 0.0
        self.output_std = 1.0

        # 训练状态
        self.is_trained = False

    def _init_weights(self):
        """Xavier 初始化权重"""
        self.weights = []
        self.biases = []

        dims = [self.input_dim] + self.hidden_dims + [self.output_dim]

        for i in range(len(dims) - 1):
            # Xavier 初始化
            std = np.sqrt(2.0 / (dims[i] + dims[i+1]))
            W = np.random.randn(dims[i], dims[i+1]) * std
            b = np.zeros(dims[i+1])
            self.weights.append(W)
            self.biases.append(b)

    def _relu(self, x: np.ndarray) -> np.ndarray:
        """ReLU 激活函数"""
        return np.maximum(0, x)

    def _relu_derivative(self, x: np.ndarray) -> np.ndarray:
        """ReLU 导数"""
        return (x > 0).astype(float)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        前向传播

        Args:
            x: 输入特征 (batch_size, input_dim) 或 (input_dim,)

        Returns:
            输出预测 (batch_size, output_dim) 或 (output_dim,)
        """
        # 确保输入是2D
        single_input = x.ndim == 1
        if single_input:
            x = x.reshape(1, -1)

        # 输入归一化
        x = (x - self.input_mean) / (self.input_std + 1e-8)

        # 前向传播
        self.activations = [x]

        for i in range(len(self.weights) - 1):
            z = x @ self.weights[i] + self.biases[i]
            x = self._relu(z)
            self.activations.append(x)

        # 输出层（无激活函数）
        output = x @ self.weights[-1] + self.biases[-1]

        # 输出反归一化
        output = output * self.output_std + self.output_mean

        if single_input:
            return output.flatten()
        return output

    def predict(self, velocity: np.ndarray, acceleration: np.ndarray,
                euler_angles: np.ndarray) -> float:
        """
        预测功率残差

        Args:
            velocity: 速度向量 [vx, vy, vz] (m/s)
            acceleration: 加速度向量 [ax, ay, az] (m/s^2)
            euler_angles: 姿态角 [roll, pitch, yaw] (rad)

        Returns:
            功率残差 ΔP (W)
        """
        # 构建输入特征
        features = np.concatenate([velocity, acceleration, euler_angles])

        # 前向传播
        residual = self.forward(features)

        return float(residual[0]) if isinstance(residual, np.ndarray) else float(residual)

    def train(self, X: np.ndarray, y: np.ndarray,
              epochs: int = 1000, learning_rate: float = 0.001,
              batch_size: int = 32, validation_split: float = 0.2,
              verbose: bool = True) -> dict:
        """
        训练神经网络

        Args:
            X: 输入特征 (n_samples, input_dim)
            y: 目标值 (n_samples,) 或 (n_samples, 1)
            epochs: 训练轮数
            learning_rate: 学习率
            batch_size: 批大小
            validation_split: 验证集比例
            verbose: 是否打印训练信息

        Returns:
            训练历史
        """
        # 确保 y 是 2D
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        # 计算归一化参数
        self.input_mean = np.mean(X, axis=0)
        self.input_std = np.std(X, axis=0)
        self.output_mean = np.mean(y)
        self.output_std = np.std(y)

        # 归一化
        X_norm = (X - self.input_mean) / (self.input_std + 1e-8)
        y_norm = (y - self.output_mean) / (self.output_std + 1e-8)

        # 划分训练集和验证集
        n_samples = X.shape[0]
        n_val = int(n_samples * validation_split)
        indices = np.random.permutation(n_samples)

        val_idx = indices[:n_val]
        train_idx = indices[n_val:]

        X_train, y_train = X_norm[train_idx], y_norm[train_idx]
        X_val, y_val = X_norm[val_idx], y_norm[val_idx]

        # 训练历史
        history = {'train_loss': [], 'val_loss': []}

        # 训练循环
        for epoch in range(epochs):
            # 打乱训练数据
            perm = np.random.permutation(len(X_train))
            X_train = X_train[perm]
            y_train = y_train[perm]

            # Mini-batch 训练
            epoch_loss = 0.0
            n_batches = 0

            for i in range(0, len(X_train), batch_size):
                X_batch = X_train[i:i+batch_size]
                y_batch = y_train[i:i+batch_size]

                # 前向传播
                activations = [X_batch]
                x = X_batch

                for j in range(len(self.weights) - 1):
                    z = x @ self.weights[j] + self.biases[j]
                    x = self._relu(z)
                    activations.append(x)

                output = x @ self.weights[-1] + self.biases[-1]

                # 计算损失 (MSE)
                loss = np.mean((output - y_batch) ** 2)
                epoch_loss += loss
                n_batches += 1

                # 反向传播
                delta = 2 * (output - y_batch) / len(y_batch)

                # 输出层梯度
                dW = activations[-1].T @ delta
                db = np.sum(delta, axis=0)

                self.weights[-1] -= learning_rate * dW
                self.biases[-1] -= learning_rate * db

                # 隐藏层梯度
                for j in range(len(self.weights) - 2, -1, -1):
                    delta = (delta @ self.weights[j+1].T) * self._relu_derivative(activations[j+1])
                    dW = activations[j].T @ delta
                    db = np.sum(delta, axis=0)

                    self.weights[j] -= learning_rate * dW
                    self.biases[j] -= learning_rate * db

            # 记录训练损失
            train_loss = epoch_loss / n_batches
            history['train_loss'].append(train_loss)

            # 计算验证损失
            val_pred = self._forward_normalized(X_val)
            val_loss = np.mean((val_pred - y_val) ** 2)
            history['val_loss'].append(val_loss)

            if verbose and (epoch + 1) % 100 == 0:
                print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

        self.is_trained = True

        return history

    def _forward_normalized(self, x: np.ndarray) -> np.ndarray:
        """对已归一化的输入进行前向传播"""
        for i in range(len(self.weights) - 1):
            z = x @ self.weights[i] + self.biases[i]
            x = self._relu(z)
        return x @ self.weights[-1] + self.biases[-1]

    def save(self, filepath: str):
        """
        保存模型

        Args:
            filepath: 保存路径
        """
        model_data = {
            'input_dim': self.input_dim,
            'hidden_dims': self.hidden_dims,
            'output_dim': self.output_dim,
            'weights': [w.tolist() for w in self.weights],
            'biases': [b.tolist() for b in self.biases],
            'input_mean': self.input_mean.tolist(),
            'input_std': self.input_std.tolist(),
            'output_mean': float(self.output_mean),
            'output_std': float(self.output_std),
            'is_trained': self.is_trained
        }

        with open(filepath, 'w') as f:
            json.dump(model_data, f, indent=2)

        print(f"Model saved to {filepath}")

    def load(self, filepath: str):
        """
        加载模型

        Args:
            filepath: 模型文件路径
        """
        with open(filepath, 'r') as f:
            model_data = json.load(f)

        self.input_dim = model_data['input_dim']
        self.hidden_dims = model_data['hidden_dims']
        self.output_dim = model_data['output_dim']
        self.weights = [np.array(w) for w in model_data['weights']]
        self.biases = [np.array(b) for b in model_data['biases']]
        self.input_mean = np.array(model_data['input_mean'])
        self.input_std = np.array(model_data['input_std'])
        self.output_mean = model_data['output_mean']
        self.output_std = model_data['output_std']
        self.is_trained = model_data['is_trained']

        print(f"Model loaded from {filepath}")


def create_synthetic_training_data(n_samples: int = 5000) -> Tuple[np.ndarray, np.ndarray]:
    """
    创建合成训练数据（用于测试，实际应从 AirSim 采集）

    模拟真实功率与物理模型预测之间的残差
    残差主要来源于：
    - 气动干扰
    - 电机非线性
    - 机身振动

    Args:
        n_samples: 样本数量

    Returns:
        (X, y): 特征和目标值
    """
    # 生成随机飞行状态
    velocities = np.random.uniform(-5, 5, (n_samples, 3))
    accelerations = np.random.uniform(-3, 3, (n_samples, 3))
    euler_angles = np.random.uniform(-0.5, 0.5, (n_samples, 3))  # rad

    # 构建特征
    X = np.hstack([velocities, accelerations, euler_angles])

    # 模拟残差（非线性函数）
    # 残差与速度平方、加速度、姿态角相关
    v_mag = np.linalg.norm(velocities, axis=1)
    a_mag = np.linalg.norm(accelerations, axis=1)
    pitch = euler_angles[:, 1]

    # 残差模型：考虑气动效应和非线性
    residual = (
        5.0 * v_mag ** 1.5 * np.sin(pitch) +  # 俯仰角影响
        2.0 * a_mag * v_mag +  # 加速度与速度耦合
        3.0 * np.abs(euler_angles[:, 0]) * v_mag +  # 横滚影响
        np.random.normal(0, 2, n_samples)  # 噪声
    )

    return X, residual
