"""StepFun Realtime API 端到端语音对话 Provider"""

import asyncio
import base64
import json
import os
from typing import Optional

try:
    import websockets
    WEBSOCKETS_AVAILABLE = True
except ImportError:
    WEBSOCKETS_AVAILABLE = False

from .base import S2SProviderBase

# 尝试从 config.py 读取默认配置
try:
    from config import (
        STEPFUN_API_KEY as _CFG_API_KEY,
        STEPFUN_MODEL as _CFG_MODEL,
        STEPFUN_VOICE as _CFG_VOICE,
        STEPFUN_SAMPLE_RATE as _CFG_SAMPLE_RATE,
        STEPFUN_INSTRUCTIONS as _CFG_INSTRUCTIONS,
        STEPFUN_VAD_THRESHOLD as _CFG_VAD_THRESHOLD,
        STEPFUN_VAD_PREFIX_PADDING_MS as _CFG_VAD_PREFIX_MS,
        STEPFUN_VAD_SILENCE_DURATION_MS as _CFG_VAD_SILENCE_MS,
    )
except ImportError:
    _CFG_API_KEY = ''
    _CFG_MODEL = 'step-1o-audio'
    _CFG_VOICE = 'qingchunshaonv'
    _CFG_SAMPLE_RATE = 24000
    _CFG_INSTRUCTIONS = '你是象棋对弈助手，用简洁中文回复用户。'
    _CFG_VAD_THRESHOLD = 0.4
    _CFG_VAD_PREFIX_MS = 200
    _CFG_VAD_SILENCE_MS = 400


# StepFun Realtime API 配置
STEPFUN_WS_URL = "wss://api.stepfun.com/v1/realtime"
STEPFUN_DEFAULT_MODEL = _CFG_MODEL
STEPFUN_DEFAULT_VOICE = _CFG_VOICE
STEPFUN_SAMPLE_RATE = _CFG_SAMPLE_RATE

# 可用音色
STEPFUN_VOICES = [
    "qingchunshaonv",       # 青春少女
    "wenrounansheng",       # 温柔男声
    "elegantgentle-female", # 优雅女声
    "livelybreezy-female",  # 活泼女声
]


class StepFunProvider(S2SProviderBase):
    """StepFun Realtime API 端到端语音对话 Provider

    使用 WebSocket 双向流式通信，server_vad 自动端点检测。
    音频格式：pcm16, 24kHz, mono。
    """

    def __init__(
        self,
        api_key: str = None,
        model: str = None,
        voice: str = None,
        instructions: str = None,
        sample_rate: int = None,
    ):
        super().__init__()

        self._api_key = api_key or os.environ.get('STEPFUN_API_KEY', '') or _CFG_API_KEY
        self._model = model or STEPFUN_DEFAULT_MODEL
        self._voice = voice or STEPFUN_DEFAULT_VOICE
        self._instructions = instructions or _CFG_INSTRUCTIONS
        self._sample_rate = sample_rate or STEPFUN_SAMPLE_RATE
        self._vad_threshold = _CFG_VAD_THRESHOLD
        self._vad_prefix_ms = _CFG_VAD_PREFIX_MS
        self._vad_silence_ms = _CFG_VAD_SILENCE_MS

        # WebSocket
        self._ws = None
        self._connected = False
        self._session_ready = False
        self._send_lock = asyncio.Lock()

        # 状态
        self._ai_speaking = False
        self._event_id = 0

        # 后台任务
        self._receive_task: Optional[asyncio.Task] = None

    @property
    def is_connected(self) -> bool:
        return self._connected and self._ws is not None

    @property
    def is_ai_speaking(self) -> bool:
        return self._ai_speaking

    @property
    def name(self) -> str:
        return f"stepfun-{self._model}"

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    # --- 生命周期 ---

    async def connect(self) -> bool:
        """建立 WebSocket 连接并初始化会话"""
        if not WEBSOCKETS_AVAILABLE:
            self._fire(self.on_error, "import_error", "websockets 未安装")
            return False

        if not self._api_key:
            self._fire(self.on_error, "config_error", "未设置 API Key")
            return False

        try:
            url = f"{STEPFUN_WS_URL}?model={self._model}"
            headers = {"Authorization": f"Bearer {self._api_key}"}

            # 禁用内置 ping，StepFun 服务端自己管理心跳
            # 避免高频 send_audio 阻塞事件循环导致 ping timeout
            self._ws = await websockets.connect(
                url,
                additional_headers=headers,
                ping_interval=None,
                ping_timeout=None,
                max_size=2**20,  # 1MB，防止大消息被拒
            )
            self._connected = True

            # 等待 session.created
            msg = await self._ws.recv()
            data = json.loads(msg)

            if data.get('type') != 'session.created':
                self._fire(self.on_error, "protocol_error",
                           f"期望 session.created，收到 {data.get('type')}")
                await self._ws.close()
                self._connected = False
                return False

            # 发送 session.update
            await self._send_session_update()

            # 启动接收循环
            self._receive_task = asyncio.create_task(self._receive_loop())

            return True

        except Exception as e:
            self._fire(self.on_error, "connect_error", str(e))
            self._connected = False
            return False

    async def disconnect(self):
        """断开连接并清理"""
        self._connected = False
        self._session_ready = False
        self._ai_speaking = False

        if self._receive_task and not self._receive_task.done():
            self._receive_task.cancel()
            try:
                await self._receive_task
            except asyncio.CancelledError:
                pass
            self._receive_task = None

        if self._ws:
            try:
                await self._ws.close()
            except Exception:
                pass
            self._ws = None

    # --- 会话配置 ---

    async def update_session(self, instructions: str = None, voice: str = None):
        """运行时更新会话配置"""
        if instructions is not None:
            self._instructions = instructions
        if voice is not None:
            self._voice = voice

        if not self.is_connected:
            return

        event = {
            "type": "session.update",
            "event_id": self._next_event_id(),
            "session": {
                "modalities": ["text", "audio"],
                "instructions": self._instructions,
                "voice": self._voice,
                "input_audio_format": "pcm16",
                "output_audio_format": "pcm16",
                "turn_detection": {
                    "type": "server_vad",
                    "threshold": self._vad_threshold,
                    "prefix_padding_ms": self._vad_prefix_ms,
                    "silence_duration_ms": self._vad_silence_ms,
                },
            }
        }
        await self._ws.send(json.dumps(event))

    # --- 音频流控制 ---

    async def send_audio(self, audio_bytes: bytes):
        """发送一帧音频数据（pcm16 原始字节，20ms @ 24kHz = 480 samples）"""
        if not self.is_connected or not self._session_ready:
            return

        # 预编码 base64，减少锁内耗时
        audio_b64 = base64.b64encode(audio_bytes).decode('ascii')
        msg = '{"type":"input_audio_buffer.append","event_id":"%s","audio":"%s"}' % (
            self._next_event_id(), audio_b64
        )

        async with self._send_lock:
            await self._ws.send(msg)

    async def send_text(self, text: str):
        """发送文字消息"""
        if not self.is_connected or not self._session_ready:
            return

        event = {
            "type": "conversation.item.create",
            "event_id": self._next_event_id(),
            "item": {
                "type": "message",
                "role": "user",
                "content": [
                    {"type": "input_text", "text": text}
                ]
            }
        }
        async with self._send_lock:
            await self._ws.send(json.dumps(event))

        # 触发响应
        response_event = {
            "type": "response.create",
            "event_id": self._next_event_id(),
        }
        async with self._send_lock:
            await self._ws.send(json.dumps(response_event))

    async def cancel_response(self):
        """打断 AI 当前回复"""
        if not self.is_connected:
            return

        event = {
            "type": "response.cancel",
            "event_id": self._next_event_id(),
        }
        async with self._send_lock:
            await self._ws.send(json.dumps(event))
        self._ai_speaking = False

    # --- 内部方法 ---

    def _next_event_id(self) -> str:
        self._event_id += 1
        return f"evt_{self._event_id}"

    async def _send_session_update(self):
        """发送初始会话配置"""
        event = {
            "type": "session.update",
            "event_id": self._next_event_id(),
            "session": {
                "modalities": ["text", "audio"],
                "instructions": self._instructions,
                "voice": self._voice,
                "input_audio_format": "pcm16",
                "output_audio_format": "pcm16",
                "turn_detection": {
                    "type": "server_vad",
                    "threshold": self._vad_threshold,
                    "prefix_padding_ms": self._vad_prefix_ms,
                    "silence_duration_ms": self._vad_silence_ms,
                },
            }
        }
        await self._ws.send(json.dumps(event))

        # 等待 session.updated
        msg = await self._ws.recv()
        data = json.loads(msg)

        if data.get('type') == 'session.updated':
            self._session_ready = True
        else:
            self._fire(self.on_error, "protocol_error",
                       f"期望 session.updated，收到 {data.get('type')}")

    async def _receive_loop(self):
        """接收并处理服务器事件"""
        try:
            while self._connected and self._ws:
                try:
                    msg = await self._ws.recv()
                    data = json.loads(msg)
                    self._handle_event(data)
                except websockets.exceptions.ConnectionClosed:
                    self._connected = False
                    self._fire(self.on_error, "connection_closed", "WebSocket 连接关闭")
                    break
        except asyncio.CancelledError:
            pass
        except Exception as e:
            if self._connected:
                self._fire(self.on_error, "receive_error", str(e))

    def _handle_event(self, data: dict):
        """处理单个服务器事件，分发到回调"""
        event_type = data.get('type', '')

        # VAD 事件
        if event_type == 'input_audio_buffer.speech_started':
            if self._ai_speaking:
                # 用户打断 AI
                asyncio.create_task(self.cancel_response())
            self._fire(self.on_user_speech_start)

        elif event_type == 'input_audio_buffer.speech_stopped':
            self._fire(self.on_user_speech_end)

        # 用户语音转录
        elif event_type == 'conversation.item.input_audio_transcription.completed':
            transcript = data.get('transcript', '')
            if transcript:
                self._fire(self.on_user_transcript, transcript)

        # AI 响应事件
        elif event_type == 'response.created':
            self._ai_speaking = True

        elif event_type == 'response.audio.delta':
            delta = data.get('delta', '')
            if delta:
                audio_bytes = base64.b64decode(delta)
                self._fire(self.on_ai_audio_chunk, audio_bytes)

        elif event_type == 'response.audio_transcript.delta':
            delta = data.get('delta', '')
            if delta:
                self._fire(self.on_ai_transcript_chunk, delta)

        elif event_type == 'response.done':
            self._ai_speaking = False
            self._fire(self.on_ai_response_done)

        # 错误
        elif event_type == 'error':
            error = data.get('error', {})
            self._fire(self.on_error,
                       error.get('type', 'unknown'),
                       error.get('message', '未知错误'))
