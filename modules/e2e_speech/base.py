"""端到端语音对话 Provider 基类"""

from abc import ABC, abstractmethod
from typing import Callable, Optional


class S2SProviderBase(ABC):
    """端到端语音对话 Provider 基类

    StepFun Realtime 等 API 是单一 WebSocket 会话，
    音频输入/输出/文字交织在一起，无法拆分为独立的 ASR + TTS。
    此基类定义统一的 S2S 接口。

    所有回调均为可选，签名：
        on_user_speech_start()
        on_user_speech_end()
        on_user_transcript(text: str)
        on_ai_audio_chunk(audio_bytes: bytes)  # pcm16 原始音频
        on_ai_transcript_chunk(text: str)
        on_ai_response_done()
        on_error(error_type: str, message: str)
    """

    def __init__(self):
        # 回调函数（外部注册）
        self.on_user_speech_start: Optional[Callable[[], None]] = None
        self.on_user_speech_end: Optional[Callable[[], None]] = None
        self.on_user_transcript: Optional[Callable[[str], None]] = None
        self.on_ai_audio_chunk: Optional[Callable[[bytes], None]] = None
        self.on_ai_transcript_chunk: Optional[Callable[[str], None]] = None
        self.on_ai_response_done: Optional[Callable[[], None]] = None
        self.on_error: Optional[Callable[[str, str], None]] = None

    # --- 生命周期 ---

    @abstractmethod
    async def connect(self) -> bool:
        """建立连接并初始化会话，成功返回 True"""

    @abstractmethod
    async def disconnect(self):
        """断开连接并清理资源"""

    # --- 会话配置 ---

    @abstractmethod
    async def update_session(self, instructions: str = None, voice: str = None):
        """运行时更新会话配置（instructions / voice）"""

    # --- 音频流控制 ---

    @abstractmethod
    async def send_audio(self, audio_bytes: bytes):
        """发送一帧音频数据（pcm16 原始字节）"""

    @abstractmethod
    async def send_text(self, text: str):
        """发送文字消息（用于文字+语音混合交互）"""

    @abstractmethod
    async def cancel_response(self):
        """打断 AI 当前回复"""

    # --- 状态 ---

    @property
    @abstractmethod
    def is_connected(self) -> bool:
        """是否已连接"""

    @property
    @abstractmethod
    def is_ai_speaking(self) -> bool:
        """AI 是否正在说话"""

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider 名称"""

    # --- 工具方法 ---

    def _fire(self, callback: Optional[Callable], *args):
        """安全调用回调，忽略异常"""
        if callback:
            try:
                callback(*args)
            except Exception as e:
                print(f"[S2S] 回调异常: {e}")
