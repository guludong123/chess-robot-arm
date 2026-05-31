"""
对话模块入口
"""
from .history import DialogueHistory, DialogueMessage
from .session import DialogueSession

__all__ = [
    'DialogueHistory',
    'DialogueMessage',
    'DialogueSession',
]