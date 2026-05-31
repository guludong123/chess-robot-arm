"""
TTS 提供者抽象基类
"""
from abc import ABC, abstractmethod
from typing import Optional
import asyncio


class TTSProviderBase(ABC):
    """TTS 提供者抽象基类"""

    @abstractmethod
    async def speak(
        self,
        text: str,
        interrupt_event: Optional[asyncio.Event] = None
    ) -> bool:
        """
        播报文本

        Args:
            text: 要播报的文本
            interrupt_event: 打断信号事件

        Returns:
            是否成功完成播报
        """
        pass

    @abstractmethod
    async def stop(self):
        """停止当前播报"""
        pass

    @abstractmethod
    def is_speaking(self) -> bool:
        """是否正在播报"""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """提供者名称"""
        pass