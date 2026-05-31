"""
YOLO 检测平滑器 - 解决棋子类别抖动问题
"""
from typing import Dict, List, Optional, Tuple
from collections import defaultdict, deque
import numpy as np


class PositionHistory:
    """单个棋盘位置的历史记录"""

    def __init__(self, max_frames: int = 5):
        self.max_frames = max_frames
        self.class_history: deque = deque(maxlen=max_frames)  # [(class_name, confidence), ...]
        self.center_history: deque = deque(maxlen=max_frames)  # [(x, y), ...]

    def add(self, class_name: str, confidence: float, center: Tuple[float, float]):
        """添加一帧检测结果"""
        self.class_history.append((class_name, confidence))
        self.center_history.append(center)

    def get_smoothed_result(self, strategy: str = 'weighted_vote') -> Optional[Dict]:
        """
        获取平滑后的结果

        Args:
            strategy: 'weighted_vote' | 'majority' | 'highest_conf'

        Returns:
            {'class_name': str, 'confidence': float, 'center': (x, y)} 或 None
        """
        if len(self.class_history) == 0:
            return None

        if strategy == 'weighted_vote':
            return self._weighted_vote()
        elif strategy == 'majority':
            return self._majority_vote()
        elif strategy == 'highest_conf':
            return self._highest_confidence()

        return self._weighted_vote()

    def _weighted_vote(self) -> Dict:
        """置信度加权投票"""
        # 统计每个类别的加权得分
        class_scores = defaultdict(float)
        class_centers = defaultdict(list)

        for class_name, conf in self.class_history:
            if class_name:  # 忽略空检测
                class_scores[class_name] += conf
                class_centers[class_name].append(conf)

        if not class_scores:
            return None

        # 找得分最高的类别
        best_class = max(class_scores.keys(), key=lambda k: class_scores[k])

        # 计算平均置信度
        avg_confidence = class_scores[best_class] / len([c for c, _ in self.class_history if c])

        # 计算加权中心点
        centers_for_class = [
            self.center_history[i]
            for i, (cn, _) in enumerate(self.class_history)
            if cn == best_class
        ]
        if centers_for_class:
            avg_center = (
                float(np.mean([c[0] for c in centers_for_class])),
                float(np.mean([c[1] for c in centers_for_class]))
            )
        else:
            avg_center = self.center_history[-1] if self.center_history else (0.0, 0.0)

        return {
            'class_name': best_class,
            'confidence': float(avg_confidence),
            'center': avg_center
        }

    def _majority_vote(self) -> Dict:
        """简单众数投票"""
        class_counts = defaultdict(int)

        for class_name, _ in self.class_history:
            if class_name:
                class_counts[class_name] += 1

        if not class_counts:
            return None

        best_class = max(class_counts.keys(), key=lambda k: class_counts[k])

        # 取该类别的平均置信度和中心点
        confs = [conf for cn, conf in self.class_history if cn == best_class]
        centers = [
            self.center_history[i]
            for i, (cn, _) in enumerate(self.class_history)
            if cn == best_class
        ]

        return {
            'class_name': best_class,
            'confidence': float(np.mean(confs)) if confs else 0.5,
            'center': (
                float(np.mean([c[0] for c in centers])) if centers else (0.0, 0.0),
                float(np.mean([c[1] for c in centers])) if centers else (0.0, 0.0)
            )
        }

    def _highest_confidence(self) -> Dict:
        """取置信度最高的帧"""
        valid_entries = [(i, cn, conf) for i, (cn, conf) in enumerate(self.class_history) if cn]
        if not valid_entries:
            return None

        best_idx, best_class, best_conf = max(valid_entries, key=lambda x: x[2])

        return {
            'class_name': best_class,
            'confidence': float(best_conf),
            'center': self.center_history[best_idx]
        }

    def clear(self):
        """清空历史"""
        self.class_history.clear()
        self.center_history.clear()


class DetectionSmoother:
    """检测平滑器 - 管理所有棋盘位置的历史"""

    def __init__(self,
                 smooth_frames: int = 5,
                 strategy: str = 'weighted_vote',
                 min_history_size: int = 2):
        """
        Args:
            smooth_frames: 平滑帧数 (建议 3-5)
            strategy: 平滑策略 ('weighted_vote', 'majority', 'highest_conf')
            min_history_size: 最小历史帧数才输出平滑结果
        """
        self.smooth_frames = smooth_frames
        self.strategy = strategy
        self.min_history_size = min_history_size

        # 每个位置的历史: Dict[pos_name, PositionHistory]
        self.position_histories: Dict[str, PositionHistory] = {}

        # 初始化所有棋盘位置
        self._init_all_positions()

    def _init_all_positions(self):
        """
        初始化所有棋盘位置的历史缓冲

        坐标系统（与引擎一致）：
        - col: 0-8 对应 a-i
        - row: 0-9（row 0 = 红方底线，row 9 = 黑方底线）
        """
        col_names = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']
        for col in range(9):
            for row in range(10):  # row 0-9
                pos = f"{col_names[col]}{row}"
                self.position_histories[pos] = PositionHistory(self.smooth_frames)

    def update(self, detections: List[Dict]) -> List[Dict]:
        """
        更新历史并返回平滑后的检测结果

        Args:
            detections: 当前帧原始检测结果

        Returns:
            平滑后的检测结果列表
        """
        # 记录本次检测到的位置
        detected_positions = set()

        # 更新有检测的位置
        for det in detections:
            pos = det.get('board_pos')
            if pos and pos in self.position_histories:
                self.position_histories[pos].add(
                    det['class_name'],
                    det['confidence'],
                    det['center']
                )
                detected_positions.add(pos)

        # 为无检测的位置添加空记录（帮助处理棋子移除）
        for pos in self.position_histories:
            if pos not in detected_positions:
                self.position_histories[pos].add(None, 0.0, (0.0, 0.0))

        # 生成平滑结果
        smoothed_detections = []
        for det in detections:
            pos = det.get('board_pos')
            # 如果没有棋盘位置（标定未完成），直接返回原始检测结果
            if not pos:
                smoothed_detections.append(det)
                continue

            if pos in self.position_histories:
                history = self.position_histories[pos]

                # 历史帧数不足时，返回原始结果
                valid_history_count = len([cn for cn, _ in history.class_history if cn])
                if valid_history_count < self.min_history_size:
                    smoothed_detections.append(det)
                else:
                    smooth_result = history.get_smoothed_result(self.strategy)
                    if smooth_result and smooth_result['class_name']:
                        # 合并平滑结果与原始检测信息
                        smoothed_det = det.copy()
                        smoothed_det['class_name'] = smooth_result['class_name']
                        smoothed_det['confidence'] = smooth_result['confidence']
                        smoothed_det['center'] = smooth_result['center']
                        # 更新颜色
                        smoothed_det['color'] = 'red' if 'red' in smooth_result['class_name'] else 'black'
                        smoothed_detections.append(smoothed_det)

        return smoothed_detections

    def reset(self):
        """重置所有历史"""
        for history in self.position_histories.values():
            history.clear()

    def get_position_confidence(self, pos: str) -> float:
        """获取某位置的平滑置信度（用于调试）"""
        if pos in self.position_histories:
            history = self.position_histories[pos]
            if len(history.class_history) > 0:
                result = history.get_smoothed_result(self.strategy)
                return result.get('confidence', 0.0) if result else 0.0
        return 0.0

    def get_debug_info(self) -> Dict:
        """获取调试信息"""
        info = {}
        for pos, history in self.position_histories.items():
            valid_history = [(cn, conf) for cn, conf in history.class_history if cn]
            if len(valid_history) > 0:
                recent = valid_history[-3:]  # 最近3帧有效检测
                info[pos] = {
                    'recent_classes': [c[0] for c in recent],
                    'recent_confs': [round(c[1], 2) for c in recent],
                    'smoothed': history.get_smoothed_result(self.strategy)
                }
        return info