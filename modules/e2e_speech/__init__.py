"""端到端语音对话模块"""

from .base import S2SProviderBase
from .stepfun_provider import StepFunProvider

__all__ = ['S2SProviderBase', 'StepFunProvider']
