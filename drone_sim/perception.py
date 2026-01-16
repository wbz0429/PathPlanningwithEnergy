"""
perception.py - 障碍物检测模块
功能：处理深度图，识别前方障碍物距离，提供可视化
"""

import numpy as np
import cv2
from dataclasses import dataclass
from typing import Tuple, Optional


@dataclass
class DetectionResult:
    """障碍物检测结果"""
    has_obstacle: bool          # 是否检测到障碍物
    min_distance: float         # 最小距离（米）
    obstacle_position: str      # 障碍物大致位置描述
    roi_center: Tuple[int, int] # ROI 中心坐标
    roi_size: Tuple[int, int]   # ROI 尺寸


class ObstacleDetector:
    """
    障碍物检测器
    基于深度图分析前方障碍物距离
    """

    def __init__(self,
                 safety_threshold: float = 5.0,
                 roi_width_ratio: float = 0.4,
                 roi_height_ratio: float = 0.4,
                 max_valid_depth: float = 100.0):
        """
        初始化检测器

        Args:
            safety_threshold: 安全距离阈值（米），小于此值触发警告
            roi_width_ratio: ROI 宽度占图像宽度的比例
            roi_height_ratio: ROI 高度占图像高度的比例
            max_valid_depth: 最大有效深度值（米），超过此值视为无效
        """
        self.safety_threshold = safety_threshold
        self.roi_width_ratio = roi_width_ratio
        self.roi_height_ratio = roi_height_ratio
        self.max_valid_depth = max_valid_depth

    def _get_roi(self, depth_image: np.ndarray) -> Tuple[np.ndarray, int, int, int, int]:
        """
        获取图像中心区域 (ROI)

        Returns:
            roi: ROI 区域的深度数据
            x1, y1, x2, y2: ROI 的边界坐标
        """
        h, w = depth_image.shape[:2]

        roi_w = int(w * self.roi_width_ratio)
        roi_h = int(h * self.roi_height_ratio)

        x1 = (w - roi_w) // 2
        y1 = (h - roi_h) // 2
        x2 = x1 + roi_w
        y2 = y1 + roi_h

        roi = depth_image[y1:y2, x1:x2]
        return roi, x1, y1, x2, y2

    def _analyze_obstacle_position(self, roi: np.ndarray, threshold: float) -> str:
        """
        分析障碍物在 ROI 中的大致位置

        Args:
            roi: ROI 区域深度数据
            threshold: 距离阈值

        Returns:
            位置描述字符串
        """
        h, w = roi.shape[:2]

        # 将 ROI 分成 3x3 网格
        grid_h, grid_w = h // 3, w // 3

        positions = []
        position_names = [
            ["左上", "上方", "右上"],
            ["左侧", "正前方", "右侧"],
            ["左下", "下方", "右下"]
        ]

        for i in range(3):
            for j in range(3):
                cell = roi[i*grid_h:(i+1)*grid_h, j*grid_w:(j+1)*grid_w]
                valid_mask = cell < self.max_valid_depth
                if valid_mask.any():
                    cell_min = np.min(cell[valid_mask])
                    if cell_min < threshold:
                        positions.append(position_names[i][j])

        if not positions:
            return "未知"
        elif "正前方" in positions:
            return "正前方"
        else:
            return "、".join(positions[:2])  # 最多返回两个位置

    def detect(self, depth_image: np.ndarray) -> DetectionResult:
        """
        检测障碍物

        Args:
            depth_image: AirSim DepthPlanner 格式的深度图（浮点数矩阵，单位米）

        Returns:
            DetectionResult: 检测结果
        """
        # 获取 ROI
        roi, x1, y1, x2, y2 = self._get_roi(depth_image)

        # 过滤无效深度值
        valid_mask = (roi > 0) & (roi < self.max_valid_depth)

        if not valid_mask.any():
            # 没有有效深度数据
            return DetectionResult(
                has_obstacle=False,
                min_distance=float('inf'),
                obstacle_position="无有效数据",
                roi_center=((x1 + x2) // 2, (y1 + y2) // 2),
                roi_size=(x2 - x1, y2 - y1)
            )

        # 计算最小距离
        min_dist = np.min(roi[valid_mask])

        # 判断是否有障碍物
        has_obstacle = min_dist < self.safety_threshold

        # 分析障碍物位置
        obstacle_pos = "无" if not has_obstacle else self._analyze_obstacle_position(roi, self.safety_threshold)

        return DetectionResult(
            has_obstacle=has_obstacle,
            min_distance=min_dist,
            obstacle_position=obstacle_pos,
            roi_center=((x1 + x2) // 2, (y1 + y2) // 2),
            roi_size=(x2 - x1, y2 - y1)
        )

    def visualize(self,
                  rgb_image: np.ndarray,
                  detection_result: DetectionResult) -> np.ndarray:
        """
        在 RGB 图像上绘制检测结果

        Args:
            rgb_image: RGB 图像
            detection_result: 检测结果

        Returns:
            带有可视化标注的图像
        """
        vis_image = rgb_image.copy()

        # 计算 ROI 边界
        cx, cy = detection_result.roi_center
        rw, rh = detection_result.roi_size
        x1, y1 = cx - rw // 2, cy - rh // 2
        x2, y2 = cx + rw // 2, cy + rh // 2

        # 根据是否有障碍物选择颜色
        if detection_result.has_obstacle:
            color = (0, 0, 255)  # 红色警告
            thickness = 3
        else:
            color = (0, 255, 0)  # 绿色安全
            thickness = 2

        # 绘制 ROI 框
        cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, thickness)

        # 绘制距离信息
        dist_text = f"Dist: {detection_result.min_distance:.2f}m"
        cv2.putText(vis_image, dist_text, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # 如果有障碍物，显示警告
        if detection_result.has_obstacle:
            warning_text = f"WARNING: {detection_result.obstacle_position}"
            h, w = vis_image.shape[:2]
            cv2.putText(vis_image, warning_text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        return vis_image

    def depth_to_colormap(self, depth_image: np.ndarray) -> np.ndarray:
        """
        将深度图转换为彩色可视化图像

        Args:
            depth_image: 深度图（浮点数矩阵）

        Returns:
            彩色深度图
        """
        # 归一化到 0-255
        depth_normalized = np.clip(depth_image / self.max_valid_depth * 255, 0, 255)
        depth_uint8 = depth_normalized.astype(np.uint8)

        # 应用颜色映射
        depth_colormap = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_JET)

        return depth_colormap
