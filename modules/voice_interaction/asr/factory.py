"""
ASR Provider 工厂

根据配置动态创建对应的 ASR 实例
支持：
- 'dashscope_paraformer': 阿里云 DashScope (云端)
- 'funasr': FunASR 本地部署
"""
import os
from typing import Optional
from .base import ASRProviderBase

# 条件导入 DashScope
try:
    from .dashscope_asr import DashScopeASRProvider
    DASHSCOPE_AVAILABLE = True
except ImportError:
    DASHSCOPE_AVAILABLE = False
    print("[ASR Factory] DashScope ASR 不可用")

# 条件导入 FunASR
try:
    from .funasr_asr import FunASRProvider
    FUNASR_AVAILABLE = True
except ImportError:
    FUNASR_AVAILABLE = False
    print("[ASR Factory] FunASR 不可用，请安装: pip install funasr")


def create_asr_provider(
    provider_type: str = None,
    **kwargs
) -> Optional[ASRProviderBase]:
    """
    创建 ASR Provider

    Args:
        provider_type: 提供者类型
            - 'dashscope_paraformer': 阿里云 DashScope (云端)
            - 'funasr': FunASR 本地部署
            - None: 使用 config.ASR_PROVIDER 默认值
        **kwargs: 提供者特定参数

    Returns:
        ASR Provider 实例，如果创建失败返回 None
    """
    import config

    # 从配置获取默认提供者类型
    provider_type = provider_type or getattr(config, 'ASR_PROVIDER', 'dashscope_paraformer')

    print(f"[ASR Factory] 创建 ASR Provider: {provider_type}")

    if provider_type == 'dashscope_paraformer':
        if not DASHSCOPE_AVAILABLE:
            print("[ASR Factory] DashScope ASR 不可用，请检查依赖")
            # 尝试回退到 FunASR
            if FUNASR_AVAILABLE:
                print("[ASR Factory] 回退到 FunASR")
                return _create_funasr_provider(config, kwargs)
            return None

        return DashScopeASRProvider(
            api_key=kwargs.get('api_key', config.DASHSCOPE_API_KEY),
            sample_rate=kwargs.get('sample_rate', getattr(config, 'ASR_SAMPLE_RATE', 16000)),
            silence_threshold=kwargs.get('silence_threshold', getattr(config, 'ASR_SILENCE_THRESHOLD', 1.5)),
            silence_energy_threshold=kwargs.get('silence_energy_threshold', getattr(config, 'ASR_SILENCE_ENERGY_THRESHOLD', 300))
        )

    elif provider_type == 'funasr':
        if not FUNASR_AVAILABLE:
            print("[ASR Factory] FunASR 不可用，请安装: pip install funasr")
            # 尝试回退到 DashScope
            if DASHSCOPE_AVAILABLE:
                print("[ASR Factory] 回退到 DashScope")
                return _create_dashscope_provider(config, kwargs)
            return None

        return _create_funasr_provider(config, kwargs)

    else:
        print(f"[ASR Factory] 未知的 ASR 提供者: {provider_type}")
        # 尝试使用可用提供者
        if FUNASR_AVAILABLE:
            print("[ASR Factory] 使用 FunASR")
            return _create_funasr_provider(config, kwargs)
        elif DASHSCOPE_AVAILABLE:
            print("[ASR Factory] 使用 DashScope")
            return _create_dashscope_provider(config, kwargs)
        return None


def _create_funasr_provider(config, kwargs) -> Optional[ASRProviderBase]:
    """创建 FunASR Provider"""
    return FunASRProvider(
        model_id=kwargs.get('model_id', getattr(config, 'FUNASR_MODEL', None)),
        vad_model_id=kwargs.get('vad_model_id', getattr(config, 'FUNASR_VAD_MODEL', None)),
        sample_rate=kwargs.get('sample_rate', getattr(config, 'ASR_SAMPLE_RATE', 16000)),
        silence_threshold=kwargs.get('silence_threshold', getattr(config, 'ASR_SILENCE_THRESHOLD', 1.5)),
        device=kwargs.get('device', getattr(config, 'FUNASR_DEVICE', 'cuda')),
        cache_dir=kwargs.get('cache_dir', getattr(config, 'FUNASR_CACHE_DIR', None))
    )


def _create_dashscope_provider(config, kwargs) -> Optional[ASRProviderBase]:
    """创建 DashScope Provider"""
    return DashScopeASRProvider(
        api_key=kwargs.get('api_key', config.DASHSCOPE_API_KEY),
        sample_rate=kwargs.get('sample_rate', getattr(config, 'ASR_SAMPLE_RATE', 16000)),
        silence_threshold=kwargs.get('silence_threshold', getattr(config, 'ASR_SILENCE_THRESHOLD', 1.5)),
        silence_energy_threshold=kwargs.get('silence_energy_threshold', getattr(config, 'ASR_SILENCE_ENERGY_THRESHOLD', 300))
    )


def get_available_providers() -> list:
    """
    获取可用的 ASR 提供者列表

    Returns:
        可用提供者名称列表
    """
    providers = []
    if DASHSCOPE_AVAILABLE:
        providers.append('dashscope_paraformer')
    if FUNASR_AVAILABLE:
        providers.append('funasr')
    return providers


def get_provider_info() -> dict:
    """
    获取提供者信息

    Returns:
        提供者详细信息字典
    """
    info = {
        'available': get_available_providers(),
        'providers': {}
    }

    if DASHSCOPE_AVAILABLE:
        info['providers']['dashscope_paraformer'] = {
            'name': '阿里云 DashScope Paraformer',
            'type': '云端',
            'latency': '~2秒',
            'requires': ['DASHSCOPE_API_KEY']
        }

    if FUNASR_AVAILABLE:
        info['providers']['funasr'] = {
            'name': 'FunASR 本地部署',
            'type': '本地',
            'latency': '200-500ms (GPU)',
            'requires': ['CUDA (可选)', '模型下载 (~1.2GB)']
        }

    return info