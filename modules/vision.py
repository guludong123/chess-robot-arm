"""
计算机视觉模块 - 棋子标定 + YOLO检测 + 棋盘线条检测
"""
import cv2
import numpy as np
import os
import json
import logging
import warnings
from typing import List, Dict, Optional, Tuple
import config
from modules.board_state import BoardStateManager

# 抑制 YOLO/Ultralytics 的详细日志
logging.getLogger('ultralytics').setLevel(logging.ERROR)
logging.getLogger('ultralytics.utils').setLevel(logging.ERROR)

# 抑制 Python 警告
warnings.filterwarnings('ignore', category=UserWarning)


class VisionSystem:
    """视觉系统 - 处理棋子标定、棋盘线条检测和棋子检测"""

    def __init__(self):
        # YOLO模型
        self.model = None
        self.load_model()

        # 标定状态
        self.calibration_complete = False
        self.calibration_type = None  # 'piece' 或 None
        self.transform_matrix = None
        self.calibration_error = 0.0
        self.calibration_points_data = []

        # 棋盘线条检测状态
        self.board_lines_horizontal = []  # 横线列表 [(x1, y1, x2, y2), ...]
        self.board_lines_vertical = []    # 竖线列表 [(x1, y1, x2, y2), ...]
        self.board_intersections = {}     # 交叉点 {(col, row): (x, y)}

        # 检测平滑器（解决类别抖动问题）
        self.smoother = None
        if config.DETECTION_SMOOTH_ENABLED:
            from modules.detection_smoother import DetectionSmoother
            self.smoother = DetectionSmoother(
                smooth_frames=config.DETECTION_SMOOTH_FRAMES,
                strategy=config.DETECTION_SMOOTH_STRATEGY,
                min_history_size=config.DETECTION_MIN_HISTORY_SIZE
            )
            print(f"[平滑器] 已启用，帧数={config.DETECTION_SMOOTH_FRAMES}, 策略={config.DETECTION_SMOOTH_STRATEGY}")

        # 加载已保存的标定
        self.load_calibration()

    def load_model(self):
        """加载YOLO模型"""
        try:
            from ultralytics import YOLO
            if os.path.exists(config.MODEL_PATH):
                self.model = YOLO(config.MODEL_PATH)
                print(f"YOLO模型加载成功: {config.MODEL_PATH}")
            else:
                print(f"模型文件不存在: {config.MODEL_PATH}")
        except Exception as e:
            print(f"模型加载失败: {e}")
            self.model = None

    def detect_board_corners_from_cars(self, detections: List[Dict]) -> Dict:
        """
        从四个车的检测结果获取棋盘四角

        旋转90度的棋盘：
        - 左边（X小）：红方 a1(上), i1(下)
        - 右边（X大）：黑方 a10(上), i10(下)

        所以：
        - X方向 → row (1→10)
        - Y方向 → col (a→i, 即0→8)

        Args:
            detections: YOLO 检测结果列表

        Returns:
            {'top_left': (x, y), 'top_right': (x, y), 'bottom_left': (x, y), 'bottom_right': (x, y)}
        """
        # 找出所有车
        cars = []
        for d in detections:
            class_name = d.get('class_name', '')
            if 'che' in class_name or 'ju' in class_name:  # 车/車
                cars.append(d)

        if len(cars) < 4:
            print(f"[棋盘定位] 只找到 {len(cars)} 个车，需要 4 个")
            return None

        # 取前4个（如果有更多，取置信度最高的）
        cars = sorted(cars, key=lambda c: c.get('confidence', 0), reverse=True)[:4]

        # 按 Y 坐标排序，分为上下两组
        cars_by_y = sorted(cars, key=lambda c: c['center'][1])
        top_cars = cars_by_y[:2]      # Y小 → 上方两个
        bottom_cars = cars_by_y[2:]   # Y大 → 下方两个

        # 上方两个按 X 排序：左(a1)、右(a10)
        top_cars_by_x = sorted(top_cars, key=lambda c: c['center'][0])
        top_left = top_cars_by_x[0]    # 左上 → a1
        top_right = top_cars_by_x[1]   # 右上 → a10

        # 下方两个按 X 排序：左(i1)、右(i10)
        bottom_cars_by_x = sorted(bottom_cars, key=lambda c: c['center'][0])
        bottom_left = bottom_cars_by_x[0]   # 左下 → i1
        bottom_right = bottom_cars_by_x[1]  # 右下 → i10

        corners = {
            'top_left': (int(top_left['center'][0]), int(top_left['center'][1])),
            'top_right': (int(top_right['center'][0]), int(top_right['center'][1])),
            'bottom_left': (int(bottom_left['center'][0]), int(bottom_left['center'][1])),
            'bottom_right': (int(bottom_right['center'][0]), int(bottom_right['center'][1])),
        }

        # 打印详细信息帮助调试
        for name, car in [('左上(a1)', top_left), ('右上(a10)', top_right), ('左下(i1)', bottom_left), ('右下(i10)', bottom_right)]:
            print(f"  {name}: {car.get('class_name')} @ 像素{car['center']}")

        print(f"[棋盘定位] 四角: 左上{corners['top_left']}, 右上{corners['top_right']}, " +
              f"左下{corners['bottom_left']}, 右下{corners['bottom_right']}")
        return corners

    def calculate_intersections_from_corners(self, corners: Dict, rotation: int = 90) -> Dict:
        """
        从棋盘四角计算所有交叉点（单应性变换）

        通过四组对应点计算单应性矩阵，将标准棋盘网格映射到图像中的四边形，
        比双线性插值更准确地处理透视畸变。

        支持棋盘旋转：
        - rotation=0: 正常方向（左上=a10, 右上=i10, 左下=a1, 右下=i1）
        - rotation=90: 顺时针旋转90度（左上=a1, 右上=a10, 左下=i1, 右下=i10）
        - rotation=180: 旋转180度
        - rotation=270: 逆时针旋转90度

        Args:
            corners: {'top_left': (x, y), 'top_right': (x, y), 'bottom_left': (x, y), 'bottom_right': (x, y)}
            rotation: 棋盘旋转角度

        Returns:
            {(col, row): (x, y)} 交叉点字典
        """
        if not corners or len(corners) < 4:
            return {}

        tl = corners['top_left']
        tr = corners['top_right']
        bl = corners['bottom_left']
        br = corners['bottom_right']

        # 标准棋盘网格的四个角点坐标 (col, row)
        # col: 0-8, row: 0-9
        board_corners = {
            0:   {'tl': (0, 9), 'tr': (8, 9), 'bl': (0, 0), 'br': (8, 0)},
            90:  {'tl': (0, 0), 'tr': (0, 9), 'bl': (8, 0), 'br': (8, 9)},
            180: {'tl': (8, 0), 'tr': (0, 0), 'bl': (8, 9), 'br': (0, 9)},
            270: {'tl': (8, 9), 'tr': (8, 0), 'bl': (0, 9), 'br': (0, 0)},
        }
        bc = board_corners.get(rotation, board_corners[90])

        # 源点：标准棋盘网格的四个角
        src = np.array([
            [bc['tl'][0], bc['tl'][1]],
            [bc['tr'][0], bc['tr'][1]],
            [bc['br'][0], bc['br'][1]],
            [bc['bl'][0], bc['bl'][1]],
        ], dtype=np.float32)

        # 目标点：图像中检测到的四个像素角点
        dst = np.array([
            [tl[0], tl[1]],
            [tr[0], tr[1]],
            [br[0], br[1]],
            [bl[0], bl[1]],
        ], dtype=np.float32)

        # 计算单应性矩阵
        H = cv2.getPerspectiveTransform(src, dst)

        # 生成所有交叉点的标准坐标
        grid = np.array([
            [[c, r]] for c in range(9) for r in range(10)
        ], dtype=np.float32)

        # 一次性变换所有交叉点
        transformed = cv2.perspectiveTransform(grid, H)

        intersections = {}
        for i, (c, r) in enumerate(((c, r) for c in range(9) for r in range(10))):
            x, y = transformed[i][0]
            intersections[(c, r)] = (int(x), int(y))

        print(f"[棋盘定位] 计算了 {len(intersections)} 个交叉点 (旋转{rotation}度, Homography)")
        return intersections

    def draw_board_lines(self, image: np.ndarray, h_lines: List = None, v_lines: List = None):
        """绘制检测到的棋盘线条"""
        if h_lines is None:
            h_lines = self.board_lines_horizontal
        if v_lines is None:
            v_lines = self.board_lines_vertical

        for line in h_lines:
            x1, y1, x2, y2 = line
            cv2.line(image, (int(x1), int(y1)), (int(x2), int(y2)), config.COLORS['green'], 2)

        for line in v_lines:
            x1, y1, x2, y2 = line
            cv2.line(image, (int(x1), int(y1)), (int(x2), int(y2)), config.COLORS['blue'], 2)

    def draw_intersections(self, image: np.ndarray, intersections: Dict = None):
        """绘制交叉点"""
        if intersections is None:
            intersections = self.board_intersections

        for (col, row), (x, y) in intersections.items():
            cv2.circle(image, (x, y), 4, config.COLORS['yellow'], -1)
            # 显示坐标标签（转换为用户习惯的 1-10 行号）
            display_row = row + 1
            label = f"{config.COL_NAMES[col]}{display_row}"
            cv2.putText(image, label, (x + 5, y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, config.COLORS['white'], 1)

    # ==================== 棋子标定 ====================

    def calculate_calibration(self, calibration_points: List[Dict]) -> bool:
        """
        计算棋子标定参数
        使用透视变换（8自由度），能处理透视畸变

        getPerspectiveTransform 要求点按顺序排列：
        左上 → 右上 → 右下 → 左下（顺时针）

        Args:
            calibration_points: 标定点数据列表
                [{'id': 'P0', 'pixel': (px, py), 'robot': (rx, ry)}, ...]
                需要 4 个点

        Returns:
            是否成功
        """
        if len(calibration_points) < 4:
            print(f"[标定] 需要 4 个点，当前只有 {len(calibration_points)} 个")
            return False

        # 按点ID排序，确保顺序一致
        # P2=左上, P3=右上, P0=右下, P1=左下
        point_order = ['P2', 'P3', 'P0', 'P1']
        sorted_points = []
        for pid in point_order:
            for p in calibration_points:
                if p['id'] == pid:
                    sorted_points.append(p)
                    break

        if len(sorted_points) != 4:
            print(f"[标定] 标定点不完整，缺少: {[pid for pid in point_order if pid not in [p['id'] for p in sorted_points]]}")
            return False

        # 收集像素坐标和机械臂坐标
        pixel_points = []
        robot_points = []

        for point in sorted_points:
            pixel_points.append(point['pixel'])
            robot_points.append(point['robot'])

        pixel_points = np.array(pixel_points, dtype=np.float32)
        robot_points = np.array(robot_points, dtype=np.float32)

        print(f"[标定] 点顺序: {point_order}")
        print(f"[标定] 像素坐标: {pixel_points.tolist()}")
        print(f"[标定] 机械臂坐标: {robot_points.tolist()}")

        try:
            # 使用透视变换（8自由度）
            # getPerspectiveTransform 需要 4 个点，返回 3x3 矩阵
            self.transform_matrix = cv2.getPerspectiveTransform(pixel_points, robot_points)

            if self.transform_matrix is None:
                print("[标定] 变换矩阵计算失败")
                return False

            self.calibration_type = 'piece_perspective'

            # 计算标定误差
            errors = []
            for pixel, robot in zip(pixel_points, robot_points):
                # 应用透视变换
                pt = np.array([[[pixel[0], pixel[1]]]], dtype=np.float32)
                transformed = cv2.perspectiveTransform(pt, self.transform_matrix)
                error = np.linalg.norm(robot - transformed[0][0])
                errors.append(error)
                print(f"[标定] 点误差: {error:.2f}mm")

            self.calibration_error = float(np.mean(errors))
            self.calibration_complete = True
            # 确保数据可 JSON 序列化
            self.calibration_points_data = [
                {
                    'id': p['id'],
                    'pixel': [float(p['pixel'][0]), float(p['pixel'][1])],
                    'robot': [float(p['robot'][0]), float(p['robot'][1])]
                }
                for p in sorted_points
            ]

            self.save_calibration()
            print(f"[标定] 完成（透视变换），平均误差: {self.calibration_error:.2f}mm")
            return True

        except Exception as e:
            print(f"[标定] 计算错误: {e}")
            import traceback
            traceback.print_exc()
            return False

    def pixel_to_robot(self, pixel_x: float, pixel_y: float) -> Tuple[float, float]:
        """
        像素坐标转机械臂坐标
        使用透视变换

        Args:
            pixel_x, pixel_y: 像素坐标

        Returns:
            (robot_x, robot_y): 机械臂坐标
        """
        if not self.calibration_complete or self.transform_matrix is None:
            # 使用简单缩放
            return pixel_x * 0.5, pixel_y * 0.5

        try:
            # 透视变换：需要 (1, 1, 2) 形状的输入
            pt = np.array([[[pixel_x, pixel_y]]], dtype=np.float32)
            transformed = cv2.perspectiveTransform(pt, self.transform_matrix)
            return float(transformed[0][0][0]), float(transformed[0][0][1])
        except Exception as e:
            print(f"坐标转换错误: {e}")
            return pixel_x * 0.5, pixel_y * 0.5

    def pixel_to_board(self, pixel_x: float, pixel_y: float) -> Tuple[Optional[int], Optional[int]]:
        """
        像素坐标转棋盘坐标 (col: 0-8, row: 0-9)
        row 0 = 红方底线，row 9 = 黑方底线
        使用交叉点查表方式
        """
        if not self.board_intersections:
            return None, None

        try:
            # 查找最近的交叉点
            min_dist = float('inf')
            nearest_col, nearest_row = None, None

            for (col, row), (ix, iy) in self.board_intersections.items():
                dist = (pixel_x - ix)**2 + (pixel_y - iy)**2
                if dist < min_dist:
                    min_dist = dist
                    nearest_col, nearest_row = col, row

            return nearest_col, nearest_row

        except Exception as e:
            print(f"棋盘坐标转换错误: {e}")
            return None, None

    def board_to_pixel(self, col: int, row: int) -> Tuple[Optional[int], Optional[int]]:
        """
        棋盘坐标转像素坐标
        使用交叉点查表方式
        """
        if not self.board_intersections:
            return None, None

        try:
            key = (col, row)
            if key in self.board_intersections:
                return self.board_intersections[key]
            return None, None
        except Exception as e:
            print(f"像素坐标转换错误: {e}")
            return None, None

    def board_to_robot(self, col: int, row: int) -> Tuple[Optional[float], Optional[float]]:
        """棋盘坐标转机械臂坐标"""
        pixel = self.board_to_pixel(col, row)
        if pixel[0] is None:
            return None, None
        return self.pixel_to_robot(pixel[0], pixel[1])

    def detect_chess_pieces(self, image: np.ndarray) -> List[Dict]:
        """使用YOLO检测棋子，只返回ROI区域内的结果"""
        if self.model is None:
            return []

        try:
            # 获取ROI参数
            roi = config.BOARD_ROI
            x_min, x_max = roi['x_min'], roi['x_max']
            y_min, y_max = roi['y_min'], roi['y_max']

            # 全图检测（保持原有识别效果）
            results = self.model.predict(
                image,
                verbose=False,
                conf=0.5,
            )
            detections = []

            if len(results) > 0 and len(results[0].boxes) > 0:
                boxes = results[0].boxes

                for i in range(len(boxes)):
                    box = boxes.xyxy[i].cpu().numpy()
                    confidence = float(boxes.conf[i].cpu().numpy())
                    class_id = int(boxes.cls[i].cpu().numpy())

                    class_name = self.model.names[class_id] if hasattr(self.model, 'names') else f"class_{class_id}"

                    center_x = (box[0] + box[2]) / 2
                    center_y = (box[1] + box[3]) / 2

                    # 只保留ROI区域内的检测结果
                    if center_x < x_min or center_x > x_max or center_y < y_min or center_y > y_max:
                        continue

                    # 计算棋盘坐标
                    board_col, board_row = self.pixel_to_board(center_x, center_y)

                    # 确定颜色
                    color = 'red' if 'red' in class_name else 'black'

                    detections.append({
                        'class_id': class_id,
                        'class_name': class_name,
                        'confidence': confidence,
                        'bbox': box.tolist(),
                        'center': (center_x, center_y),
                        'board_col': board_col,
                        'board_row': board_row,
                        'board_pos': self._format_position(board_col, board_row) if board_col is not None else None,
                        'color': color
                    })

            # 应用检测平滑
            if self.smoother and config.DETECTION_SMOOTH_ENABLED:
                detections = self.smoother.update(detections)

            return detections

        except Exception as e:
            print(f"棋子检测错误: {e}")
            return []

    def save_calibration(self):
        """保存标定参数"""
        data = {
            'calibration_complete': self.calibration_complete,
            'calibration_type': self.calibration_type,
            'transform_matrix': self.transform_matrix.tolist() if self.transform_matrix is not None else None,
            'calibration_error': self.calibration_error,
            'calibration_points': self.calibration_points_data,
            'board_lines': {
                'horizontal': self.board_lines_horizontal,
                'vertical': self.board_lines_vertical
            },
            'board_intersections': {
                f"{col},{row}": list(coord)
                for (col, row), coord in self.board_intersections.items()
            }
        }

        try:
            with open(config.CALIBRATION_FILE, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"保存标定参数失败: {e}")

    def load_calibration(self):
        """加载标定参数"""
        if not os.path.exists(config.CALIBRATION_FILE):
            return False

        try:
            with open(config.CALIBRATION_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)

            self.calibration_complete = data.get('calibration_complete', False)
            self.calibration_type = data.get('calibration_type', None)

            if data.get('transform_matrix'):
                self.transform_matrix = np.array(data['transform_matrix'])

            self.calibration_error = data.get('calibration_error', 0.0)
            self.calibration_points_data = data.get('calibration_points', [])

            # 加载棋盘线条数据
            board_lines = data.get('board_lines', {})
            self.board_lines_horizontal = [tuple(l) for l in board_lines.get('horizontal', [])]
            self.board_lines_vertical = [tuple(l) for l in board_lines.get('vertical', [])]

            # 加载交叉点数据（兼容旧坐标系统）
            intersections = data.get('board_intersections', {})
            self.board_intersections = {}

            # 检测是否是旧坐标系统：检查是否有row=10或row=1-10范围
            max_row = 0
            for key in intersections.keys():
                parts = key.split(',')
                if len(parts) == 2:
                    row = int(parts[1])
                    if row > max_row:
                        max_row = row

            need_conversion = (max_row == 10)  # 旧格式的特征是row最大值为10

            for key, coord in intersections.items():
                col, row = map(int, key.split(','))
                if need_conversion and row >= 1 and row <= 10:
                    # 旧坐标系统，需要转换
                    row = row - 1  # 转换为新坐标：新row = 旧row - 1
                self.board_intersections[(col, row)] = tuple(coord)

            # 保存转换后的数据（更新文件）
            if need_conversion and self.board_intersections:
                self.save_calibration()
                print("[标定] 已自动转换坐标格式（row 1-10 → row 0-9）")

            if self.calibration_complete:
                print(f"标定参数已加载，类型: {self.calibration_type}，误差: {self.calibration_error:.2f}mm")
                print(f"棋盘交叉点: {len(self.board_intersections)} 个")
            return True

        except Exception as e:
            print(f"加载标定参数失败: {e}")
            return False

    def reset_calibration(self):
        """重置标定"""
        self.calibration_complete = False
        self.calibration_type = None
        self.transform_matrix = None
        self.calibration_error = 0.0
        self.calibration_points_data = []
        self.board_lines_horizontal = []
        self.board_lines_vertical = []
        self.board_intersections = {}
        self.save_calibration()
        print("[标定] 已重置")

    def _format_position(self, col: Optional[int], row: Optional[int]) -> str:
        """
        格式化棋盘位置

        坐标系统（与引擎一致）：
        - col: 0-8 对应 a-i
        - row: 0-9（row 0 = 红方底线，row 9 = 黑方底线）
        """
        if col is None or row is None:
            return "??"
        col_names = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']
        if 0 <= col < 9 and 0 <= row <= 9:
            return f"{col_names[col]}{row}"
        return "??"

    def draw_detections(self, image: np.ndarray, detections: List[Dict]):
        """在图像上绘制检测结果"""
        for det in detections:
            bbox = det['bbox']
            center = det['center']
            class_name = det['class_name']
            confidence = det['confidence']
            color = config.COLORS[det['color']]

            # 绘制边界框
            cv2.rectangle(image,
                         (int(bbox[0]), int(bbox[1])),
                         (int(bbox[2]), int(bbox[3])), color, 2)

            # 绘制中心点
            cv2.circle(image, (int(center[0]), int(center[1])), 6, color, -1)

            # 绘制标签
            label = f"{class_name}:{confidence:.2f}"
            if det.get('board_pos'):
                label += f"@{det['board_pos']}"

            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(image,
                         (int(bbox[0]), int(bbox[1]) - label_size[1] - 10),
                         (int(bbox[0]) + label_size[0], int(bbox[1])),
                         color, -1)
            cv2.putText(image, label,
                       (int(bbox[0]), int(bbox[1]) - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    def draw_board_grid(self, image: np.ndarray):
        """绘制棋盘网格（使用检测到的线条和交叉点）"""
        # 绘制线条
        self.draw_board_lines(image)

        # 绘制交叉点
        self.draw_intersections(image)

    def reset_smoother(self):
        """重置检测平滑器（开局时调用）"""
        if self.smoother:
            self.smoother.reset()
            print("[平滑器] 已重置")
