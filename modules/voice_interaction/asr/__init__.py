"""
ASR 提供者模块

支持多种 ASR 提供者：
- DashScope Paraformer (云端)
- FunASR (本地部署)

使用工厂模式创建：
    from modules.voice_interaction.asr import create_asr_provider
    asr = create_asr_provider()  # 根据 config.ASR_PROVIDER 自动选择
"""
from .base import ASRProviderBase
from .factory import create_asr_provider, get_available_providers, get_provider_info

# 条件导出 DashScope ASR
try:
    from .dashscope_asr import DashScopeASRProvider
    __all__ = ['ASRProviderBase', 'DashScopeASRProvider', 'create_asr_provider', 'get_available_providers', 'get_provider_info']
except ImportError:
    __all__ = ['ASRProviderBase', 'create_asr_provider', 'get_available_providers', 'get_provider_info']

# 条件导出 FunASR
try:
    from .funasr_asr import FunASRProvider
    __all__.append('FunASRProvider')
except ImportError:
    pass