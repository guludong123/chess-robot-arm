"""
LLM 提供者抽象基类
"""
from abc import ABC, abstractmethod
from typing import Dict, Optional


class LLMProviderBase(ABC):
    """LLM 提供者抽象基类"""

    @abstractmethod
    async def generate_commentary(
        self,
        board_state: Dict,
        human_move: Dict,
        ai_move: Dict,
        move_history: list,
        context: Optional[Dict] = None
    ) -> str:
        """
        生成棋局解说

        Args:
            board_state: 当前棋盘状态
            human_move: 人类走棋信息 {'from_pos', 'to_pos', 'moving_piece', 'captured'}
            ai_move: AI 走棋信息 {'from', 'to', 'piece'}
            move_history: 走棋历史列表
            context: 额外上下文（如局势评估）

        Returns:
            解说文本
        """
        pass

    @abstractmethod
    async def process_voice_command(
        self,
        transcript: str,
        game_state: Dict
    ) -> Dict:
        """
        处理语音命令

        Args:
            transcript: 语音转文字结果
            game_state: 当前游戏状态

        Returns:
            {'action': str, 'params': Dict, 'response': str}
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """提供者名称"""
        pass