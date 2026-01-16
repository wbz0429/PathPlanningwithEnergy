# 无人机路径规划系统 - 开发总结

## 项目概述

基于 AirSim 仿真平台的无人机自主飞行与避障系统，实现了 **OctoMap + ESDF + RRT*** 技术路线的可行性验证。

## 环境配置

### 1. 创建 Conda 环境

```cmd
conda create -n drone python=3.9
conda activate drone
```

### 2. 安装依赖

**方法一：使用安装脚本**
```cmd
cd E:\毕业设计\drone_sim
install_deps.bat
```

**方法二：手动安装（按顺序）**
```cmd
pip install numpy
pip install airsim
pip install opencv-python matplotlib pandas scipy
```

### 3. 依赖列表

| 包名 | 用途 |
|------|------|
| airsim | AirSim Python API |
| numpy | 数值计算 |
| opencv-python | 图像处理 |
| matplotlib | 数据可视化 |
| pandas | 数据分析 |
| scipy | ESDF 距离场计算 |

## 新增文件

### 1. `path_planning.py` - 路径规划核心模块

实现了三个核心组件：

#### VoxelGrid（3D 体素栅格）
- 简化版 OctoMap 实现
- 使用 numpy 3D 数组代替八叉树
- 支持从深度图更新地图
- 世界坐标与栅格索引转换

#### ESDF（欧几里得符号距离场）
- 使用 `scipy.ndimage.distance_transform_edt` 计算
- 提供任意点到最近障碍物的距离
- 支持梯度计算（用于势场法）

#### RRT*（路径规划算法）
- 基于采样的路径规划
- 支持重连接优化
- 碰撞检测基于 ESDF
- 路径平滑后处理

### 2. `test_path_planning.py` - 可行性验证测试

完整流程测试：
1. 连接 AirSim 获取深度图
2. 构建 3D 栅格地图
3. 计算 ESDF 距离场
4. RRT* 路径规划
5. 生成可视化结果

### 3. `fly_planned_path.py` - 路径跟踪飞行测试

控制无人机沿规划路径飞行：
1. 起飞并飞到指定高度
2. 获取深度图构建地图
3. 规划路径
4. 依次飞向各路径点
5. 降落

### 4. `install_deps.bat` - 依赖安装脚本

一键安装所有依赖包。

## 可行性验证结果

### 性能指标

| 模块 | 耗时 | 说明 |
|------|------|------|
| 3D 栅格地图构建 | ~40ms | 1898 个占据体素 |
| ESDF 距离场计算 | ~38ms | 80x80x40 栅格 |
| RRT* 路径规划 | ~24ms | ~90 次迭代 |

### 测试结果

- **地图构建**: 成功从深度图生成 3D 占据栅格
- **ESDF 计算**: 成功计算距离场
- **路径规划**: 成功规划绑过障碍物的路径
- **飞行测试**: 无人机成功沿规划路径飞行

## 运行方式

### 1. 启动 AirSim 仿真器

确保 AirSim 仿真环境已启动。

### 2. 运行可行性测试

```cmd
conda activate drone
cd E:\毕业设计\drone_sim
python test_path_planning.py
```

输出文件：`path_planning_result.png`

### 3. 运行飞行测试

```cmd
python fly_planned_path.py
```

## 技术路线

```
深度图 → 点云 → 3D体素栅格(OctoMap) → ESDF(距离场) → RRT*(路径规划) → 轨迹跟踪
```

### 当前实现

```
[深度图] ──→ [VoxelGrid] ──→ [ESDF] ──→ [RRT*] ──→ [路径点] ──→ [飞行控制]
   │            │              │           │
   │         40ms           38ms        24ms
   │
   └── AirSim DepthPlanar 图像
```

## 已知限制

1. **单摄像头视角有限**：只能看到前方 90° 范围内的障碍物
2. **单帧建图**：当前只使用单帧深度图，未实现增量建图
3. **简化坐标转换**：假设相机朝向 X 正方向

## 后续改进方向

1. **增量建图**：累积多帧深度图，构建完整环境地图
2. **多摄像头支持**：配置多个深度相机实现 360° 感知
3. **实时避障**：集成到主控制循环，实现动态避障
4. **轨迹优化**：添加轨迹平滑和速度规划
5. **动态障碍物**：支持移动障碍物检测和预测

## 文件结构

```
E:\毕业设计\drone_sim\
├── path_planning.py        # [新增] 路径规划核心模块
├── test_path_planning.py   # [新增] 可行性验证测试
├── fly_planned_path.py     # [新增] 路径跟踪飞行测试
├── install_deps.bat        # [新增] 依赖安装脚本
├── path_planning_result.png # [生成] 可视化结果
├── main_control.py         # 原有主控制程序
├── perception.py           # 原有障碍物检测模块
├── logger.py               # 原有飞行数据记录模块
├── show_3d_depth.py        # 原有 3D 深度可视化
├── visualize_depth.py      # 原有深度图可视化
├── connect_test.py         # 原有连接测试
├── requirements.txt        # 依赖列表
└── logs/                   # 飞行日志目录
```

## 参考

- AirSim 文档: https://microsoft.github.io/AirSim/
- OctoMap: https://octomap.github.io/
- RRT* 算法: Karaman & Frazzoli, 2011
