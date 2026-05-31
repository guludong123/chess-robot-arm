"""
TTS 提供者模块
"""
from .base import TTSProviderBase
from .edge_tts import EdgeTTSProvider

__all__ = [
    'TTSProviderBase',
    'EdgeTTSProvider',
]