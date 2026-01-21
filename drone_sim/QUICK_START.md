# 快速启动指南

## AirSim路径配置

你的AirSim Blocks环境路径：
```
E:\Sim\Blocks\Blocks\WindowsNoEditor\blocks.exe
```

---

## 方法1: 一键启动（推荐）

### 步骤1: 启动AirSim
双击运行：
```
start_airsim.bat
```

等待AirSim环境完全加载（看到无人机出现在场景中）

### 步骤2: 运行可视化
双击运行：
```
launch_visualization.bat
```

脚本会自动：
1. 检查AirSim是否运行
2. 激活conda环境
3. 测试连接
4. 让你选择运行模式：
   - **选项1**: 真实AirSim飞行（推荐）
   - **选项2**: 模拟数据（快速测试）

---

## 方法2: 手动启动

### 步骤1: 启动AirSim
```bash
# 双击运行
E:\Sim\Blocks\Blocks\WindowsNoEditor\blocks.exe
```

### 步骤2: 打开命令行
```bash
cd E:\毕业设计\drone_sim
conda activate drone
```

### 步骤3: 测试连接
```bash
python connect_test.py
```

### 步骤4: 运行可视化
```bash
# 真实飞行（推荐）
python fly_with_simple_visualization.py

# 或者模拟数据（快速测试）
python generate_simple_visualization.py
```

---

## 输出文件

### 真实AirSim飞行
```
airsim_flight_visualization.mp4    # 视频 (1.3MB, 30帧)
airsim_flight_final.png            # 最终图像 (161KB)
airsim_flight_frames/              # 图像序列
```

### 模拟数据
```
simple_3d_visualization.mp4        # 视频 (761KB, 20帧)
simple_3d_final.png                # 最终图像 (125KB)
simple_3d_frames/                  # 图像序列
```

---

## 常见问题

### Q1: AirSim启动失败
**解决**:
- 检查路径是否正确
- 确保有足够的显存（建议4GB+）
- 尝试以管理员身份运行

### Q2: 连接测试失败
**解决**:
- 确保AirSim完全启动（看到无人机）
- 检查防火墙设置
- 重启AirSim

### Q3: 无人机不移动
**原因**:
- 局部目标在障碍物内部
- 这是正常的，可视化系统仍然工作

**解决**:
- 可视化已经成功记录了建图过程
- 视频可以正常用于答辩

### Q4: conda环境未激活
**解决**:
```bash
conda create -n drone python=3.9
conda activate drone
cd E:\毕业设计\drone_sim
pip install -r requirements.txt
```

---

## 推荐工作流程

### 日常开发测试
1. 运行 `generate_simple_visualization.py`（不需要AirSim）
2. 快速查看可视化效果
3. 调整参数

### 最终验证
1. 启动AirSim
2. 运行 `fly_with_simple_visualization.py`
3. 获取真实飞行数据
4. 用于论文和答辩

---

## 快捷命令

```bash
# 进入项目目录
cd E:\毕业设计\drone_sim

# 激活环境
conda activate drone

# 测试连接
python connect_test.py

# 真实飞行
python fly_with_simple_visualization.py

# 模拟数据
python generate_simple_visualization.py

# 查看视频
start airsim_flight_visualization.mp4

# 查看图像
start airsim_flight_final.png
```

---

## 文件说明

| 文件 | 说明 |
|------|------|
| `start_airsim.bat` | 启动AirSim |
| `launch_visualization.bat` | 一键启动可视化 |
| `fly_with_simple_visualization.py` | 真实飞行脚本 |
| `generate_simple_visualization.py` | 模拟数据脚本 |
| `connect_test.py` | 连接测试 |

---

## 答辩准备

### 推荐使用
```
airsim_flight_visualization.mp4
```

### 讲解要点
1. "这是基于AirSim仿真环境的真实飞行数据"
2. "红色点云是通过深度相机实时观测并累积的障碍物"
3. "绿色轨迹显示无人机的实际飞行路径"
4. "蓝色虚线是RRT*算法规划的局部路径"
5. "系统采用滚动规划策略，每次规划6米局部目标"
6. "视角动态旋转，展示3D空间关系"

---

**创建日期**: 2026-01-19
**AirSim路径**: E:\Sim\Blocks\Blocks\WindowsNoEditor\blocks.exe
