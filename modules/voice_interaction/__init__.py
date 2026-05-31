"""
语音交互模块 - LLM 解说 + TTS 播报 + ASR 语音命令
"""

from .manager import VoiceInteractionManager
from .state import VoiceState, VoiceStatus

# LLM 提供者
from .llm.base import LLMProviderBase
from .llm.dashscope_provider import DashScopeLLMProvider

# TTS 提供者
from .tts.base import TTSProviderBase
from .tts.edge_tts import EdgeTTSProvider

# ASR 提供者 (Phase 2)
from .asr.base import ASRProviderBase

__all__ = [
    'VoiceInteractionManager',
    'VoiceState',
    'VoiceStatus',
    'LLMProviderBase',
    'DashScopeLLMProvider',
    'TTSProviderBase',
    'EdgeTTSProvider',
    'ASRProviderBase',
]