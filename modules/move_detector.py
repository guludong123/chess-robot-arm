"""
走棋检测器 - 检测人类走棋
"""
import time
from typing import Dict, Optional, List
from modules.board_state import BoardStateManager


class MoveDetector:
    """走棋检测器"""

    def __init__(self, stable_frames: int = 15):
        self.previous_state: Optional[BoardStateManager] = None
        self.current_state: Optional[BoardStateManager] = None
        self.stable_frame_count = 0
        self.required_stable_frames = stable_frames
        self.last_change_time = 0
        self.pending_move: Optional[Dict] = None

    def update(self, detections: List[Dict]) -> Optional[Dict]:
        """
        更新检测并检测走棋
        返回检测到的走棋，如果没有则返回None
        """
        self.current_state = BoardStateManager()
        self.current_state.from_detections(detections)

        if self.previous_state is None:
            self.previous_state = self.current_state.copy()
            return None

        # 比较状态
        diff = self.current_state.compare(self.previous_state)

        # 调试日志
        if diff['added'] or diff['removed'] or diff.get('changed'):
            print(f"[MoveDetector] 变化：added={len(diff['added'])}, removed={len(diff['removed'])}, changed={len(diff.get('changed', []))}")
            print(f"[MoveDetector] is_valid={diff['is_valid_move']}, from={diff['from_pos']}, to={diff['to_pos']}")
            if diff.get('captured'):
                print(f"[MoveDetector] 吃子：{diff['captured']}")

        if diff['is_valid_move']:
            # 检测到可能的走棋
            if self.pending_move is None:
                self.pending_move = diff
                self.stable_frame_count = 1
            else:
                # 检查是否是相同的走棋
                if (diff['from_pos'] == self.pending_move['from_pos'] and
                    diff['to_pos'] == self.pending_move['to_pos']):
                    self.stable_frame_count += 1
                else:
                    # 走棋变化，重新开始计数
                    self.pending_move = diff
                    self.stable_frame_count = 1

            # 检查是否稳定
            if self.stable_frame_count >= self.required_stable_frames:
                # 确认走棋
                move = self.pending_move
                self.previous_state = self.current_state.copy()
                self.pending_move = None
                self.stable_frame_count = 0
                return move

        else:
            # 没有检测到走棋或状态不稳定
            if diff['added'] or diff['removed']:
                # 有变化但不是合法走棋，可能是正在摆棋
                self.stable_frame_count = 0
                self.pending_move = None
            else:
                # 状态稳定，更新previous_state
                if self.stable_frame_count == 0:
                    self.previous_state = self.current_state.copy()

        return None

    def reset(self):
        """重置检测器"""
        self.previous_state = None
        self.current_state = None
        self.stable_frame_count = 0
        self.pending_move = None

    def force_update(self, board_state: BoardStateManager):
        """强制更新状态"""
        self.previous_state = board_state.copy()
        self.current_state = board_state.copy()
        self.pending_move = None
        self.stable_frame_count = 0
