# modules/__init__.py
"""
中国象棋对战机器人系统 - 模块包
"""

__version__ = "1.0.0"
__author__ = "Chess Robot Team"

from .camera import CameraManager
from .vision import VisionSystem
from .robot_arm import RobotArm
from .board_state import BoardStateManager, Piece
from .move_detector import MoveDetector
from .chess_ai import ChineseChessAI

__all__ = [
    'CameraManager',
    'VisionSystem',
    'RobotArm',
    'BoardStateManager',
    'Piece',
    'MoveDetector',
    'ChineseChessAI',
]
