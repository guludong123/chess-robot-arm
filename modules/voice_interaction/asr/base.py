"""
ASR 提供者抽象基类 (Phase 2)
"""
from abc import ABC, abstractmethod
from typing import AsyncGenerator, Optional


class ASRProviderBase(ABC):
    """ASR 提供者抽象基类"""

    @abstractmethod
    async def listen(
        self,
        timeout: float = 5.0,
        silence_threshold: float = 1.0
    ) -> Optional[str]:
        """
        监听并转录语音

        Args:
            timeout: 最大监听时间
            silence_threshold: 静音检测阈值

        Returns:
            转录文本，如果无语音则返回 None
        """
        pass

    @abstractmethod
    async def start_continuous_listening(self) -> AsyncGenerator[str, None]:
        """
        开始连续监听模式（流式返回结果）

        Yields:
            转录文本片段
        """
        pass

    @abstractmethod
    async def stop_listening(self):
        """停止监听"""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """提供者名称"""
        pass