"""
FunASR 本地语音识别提供者

使用 FunASR + FSMN-VAD：
- FSMN-VAD 检测语音活动（智能区分语音和噪音）
- Paraformer 进行语音识别
- GPU 加速

安装：pip install funasr modelscope pyaudio
"""
import asyncio
import queue
import threading
import time
import os
import numpy as np
from typing import Optional, AsyncGenerator, Dict, List
import logging

# 抑制 verbose 日志
logging.getLogger('modelscope').setLevel(logging.ERROR)
logging.getLogger('funasr').setLevel(logging.ERROR)

try:
    import pyaudio
    PYAUDIO_AVAILABLE = True
except ImportError:
    PYAUDIO_AVAILABLE = False

try:
    from funasr import AutoModel
    FUNASR_AVAILABLE = True
except ImportError:
    FUNASR_AVAILABLE = False

from .base import ASRProviderBase


class FunASRProvider(ASRProviderBase):
    """
    FunASR 本地语音识别提供者

    使用 FSMN-VAD 进行语音活动检测
    """

    def __init__(
        self,
        model_id: str = None,
        vad_model_id: str = None,
        sample_rate: int = 16000,
        silence_threshold: float = 1.5,
        device: str = None,
        cache_dir: str = None
    ):
        import config
        self.model_id = model_id or getattr(config, 'FUNASR_MODEL', 'paraformer-zh-streaming')
        self.vad_model_id = vad_model_id or getattr(config, 'FUNASR_VAD_MODEL', 'fsmn-vad')
        self.sample_rate = sample_rate
        self.silence_threshold = silence_threshold
        self.device = device or getattr(config, 'FUNASR_DEVICE', 'cuda')
        self.cache_dir = cache_dir or getattr(config, 'FUNASR_CACHE_DIR',
            os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), 'models', 'funasr'))

        # VAD 混合检测参数
        self.vad_accumulate_frames = getattr(config, 'ASR_VAD_ACCUMULATE_FRAMES', 5)
        self.energy_threshold = getattr(config, 'ASR_ENERGY_THRESHOLD', 0.015)
        self.speech_confirm_count = getattr(config, 'ASR_SPEECH_CONFIRM_COUNT', 2)
        self.vad_end_count = getattr(config, 'ASR_VAD_END_COUNT', 2)  # VAD 无语音次数就结束

        # 模型
        self._asr_model = None
        self._vad_model = None
        self._initialized = False

        # PyAudio
        self._pyaudio = None
        self._audio_stream = None

        # 控制
        self._is_listening = False
        self._stop_event = threading.Event()
        self._audio_thread = None
        self._result_queue = queue.Queue()

        # 转录结果
        self._transcript = ""

    @property
    def name(self) -> str:
        return "FunASR Local"

    def _init_models(self):
        """初始化模型"""
        if self._initialized:
            return True

        if not FUNASR_AVAILABLE:
            print("[FunASR] funasr 未安装")
            return False

        try:
            print(f"[FunASR] 加载 ASR: {self.model_id}")
            self._asr_model = AutoModel(
                model=self.model_id,
                device=self.device,
                cache_dir=self.cache_dir,
                disable_update=True,
                disable_log=True
            )

            print(f"[FunASR] 加载 VAD: {self.vad_model_id}")
            self._vad_model = AutoModel(
                model=self.vad_model_id,
                device=self.device,
                cache_dir=self.cache_dir,
                disable_update=True,
                disable_log=True
            )

            # 验证 CUDA
            if self.device == 'cuda':
                try:
                    import torch
                    if torch.cuda.is_available():
                        print(f"[FunASR] GPU: {torch.cuda.get_device_name(0)}")
                    else:
                        self.device = 'cpu'
                        print("[FunASR] CUDA 不可用")
                except ImportError:
                    pass

            self._initialized = True
            print("[FunASR] 模型就绪")
            return True

        except Exception as e:
            print(f"[FunASR] 模型加载失败: {e}")
            return False

    def _check_dependencies(self) -> bool:
        if not PYAUDIO_AVAILABLE:
            print("[FunASR] PyAudio 未安装")
            return False
        if not self._init_models():
            return False
        return True

    def _init_audio_capture(self):
        """初始化音频采集"""
        if self._pyaudio is None:
            self._pyaudio = pyaudio.PyAudio()

        input_device = None
        for i in range(self._pyaudio.get_device_count()):
            info = self._pyaudio.get_device_info_by_index(i)
            if info['maxInputChannels'] > 0:
                input_device = i
                break

        if input_device is None:
            raise RuntimeError("未找到麦克风")

        self._audio_stream = self._pyaudio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.sample_rate,
            input=True,
            input_device_index=input_device,
            frames_per_buffer=1024
        )

    def _audio_loop(self):
        """音频处理循环 - 能量前端 + VAD 累积确认"""
        audio_buffer = []          # ASR 识别用完整缓冲
        vad_buffer = []            # VAD 检测用滑动缓冲
        speech_confirmed = False   # VAD 是否确认有语音
        vad_confirm_count = 0      # VAD 连续确认计数
        vad_no_speech_count = 0    # VAD 连续无语音计数
        silence_start = None       # 静音开始时间
        chunk_size = 1024          # ~64ms at 16kHz

        try:
            while self._is_listening and not self._stop_event.is_set():
                audio_data = self._audio_stream.read(chunk_size, exception_on_overflow=False)

                # 计算 RMS
                audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
                rms = np.sqrt(np.mean(audio_array ** 2))

                # 能量前端检测（快速响应）
                potential_speech = rms > self.energy_threshold

                # 状态1: 等待语音开始
                if not speech_confirmed:
                    # VAD 缓冲区始终接收数据（让 VAD 自己判断是否是语音）
                    vad_buffer.append(audio_data)
                    if potential_speech:
                        audio_buffer.append(audio_data)

                    # VAD 累积确认
                    if len(vad_buffer) >= self.vad_accumulate_frames:
                        combined = b''.join(vad_buffer[-self.vad_accumulate_frames:])
                        is_speech = self._detect_speech_vad(combined)

                        if is_speech:
                            vad_confirm_count += 1
                            vad_no_speech_count = 0
                            if vad_confirm_count >= self.speech_confirm_count:
                                speech_confirmed = True  # VAD 确认有语音
                                # 把 VAD 缓冲中的音频补入录音缓冲（防止丢失开头轻声部分）
                                audio_buffer.extend(vad_buffer)
                                vad_buffer = []  # 清空，准备监测结束
                        else:
                            vad_confirm_count = 0
                            vad_buffer = vad_buffer[-(self.vad_accumulate_frames - 1):]  # 滑动窗口

                # 状态2: 语音已确认，等待结束
                else:
                    # 始终添加到 VAD 缓冲（用于检测结束）
                    vad_buffer.append(audio_data)

                    # 能量高 或 VAD 有语音时都添加到录音缓冲
                    if potential_speech:
                        audio_buffer.append(audio_data)

                    # VAD 检测是否结束（主要判断条件）
                    if len(vad_buffer) >= self.vad_accumulate_frames:
                        combined = b''.join(vad_buffer[-self.vad_accumulate_frames:])
                        is_speech = self._detect_speech_vad(combined)

                        if is_speech:
                            vad_no_speech_count = 0
                            vad_buffer = []  # 清空，重新累积
                            # VAD 有语音时，把当前帧也加入录音缓冲
                            if not potential_speech:
                                audio_buffer.append(audio_data)
                        else:
                            vad_no_speech_count += 1
                            vad_buffer = vad_buffer[-(self.vad_accumulate_frames - 1):]

                    # 静音计时（备用条件）
                    if not potential_speech:
                        if silence_start is None:
                            silence_start = time.time()
                    else:
                        silence_start = None

                    # 结束条件：VAD 连续无语音（主要条件）
                    # 或 静音超时（备用条件）
                    silence_duration = 0 if silence_start is None else time.time() - silence_start

                    if vad_no_speech_count >= self.vad_end_count or silence_duration > self.silence_threshold:
                        # 语音结束，ASR 识别
                        if len(audio_buffer) > 0:
                            text = self._recognize(audio_buffer)
                            if text:
                                self._transcript = text
                        # 发送结果
                        self._result_queue.put({'text': self._transcript, 'is_final': True})
                        break

        except Exception as e:
            self._result_queue.put({'error': str(e)})

    def _detect_speech_vad(self, audio_data: bytes) -> bool:
        """VAD 累积检测 - 需要足够长的音频"""
        try:
            audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0

            result = self._vad_model.generate(audio_array)

            if result and len(result) > 0:
                vad_info = result[0]
                # 检查 value 字段（语音段列表）
                value = vad_info.get('value', [])
                if isinstance(value, list) and len(value) > 0:
                    return True  # VAD 检测到语音段

            return False

        except Exception:
            return False

    def _recognize(self, audio_buffer: List[bytes]) -> str:
        """语音识别"""
        try:
            audio_data = b''.join(audio_buffer)
            audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0

            result = self._asr_model.generate(audio_array)

            if result and len(result) > 0:
                return result[0].get('text', '')

            return ""

        except Exception:
            return ""

    async def listen(self, timeout: float = 10.0, silence_threshold: float = None) -> Optional[str]:
        """单次语音识别"""
        if not self._check_dependencies():
            return None

        silence_threshold = silence_threshold or self.silence_threshold
        self._stop_event.clear()
        self._result_queue = queue.Queue()
        self._transcript = ""

        try:
            self._init_audio_capture()
            self._is_listening = True

            self._audio_thread = threading.Thread(target=self._audio_loop, daemon=True)
            self._audio_thread.start()

            # 等待结果
            start_time = time.time()
            while True:
                if time.time() - start_time > timeout:
                    return self._transcript if self._transcript else None

                if self._stop_event.is_set():
                    return None

                try:
                    item = self._result_queue.get(timeout=0.1)
                    if 'error' in item:
                        return None
                    if 'text' in item and item['text']:
                        return item['text']
                except queue.Empty:
                    continue

        except Exception as e:
            print(f"[FunASR] 错误: {e}")
            return None

        finally:
            self._cleanup()

    async def start_continuous_listening(self) -> AsyncGenerator[str, None]:
        """连续监听"""
        if not self._check_dependencies():
            return

        self._stop_event.clear()

        try:
            self._init_audio_capture()
            self._is_listening = True

            while self._is_listening and not self._stop_event.is_set():
                text = await self.listen(timeout=5.0)
                if text:
                    yield text

        finally:
            self._cleanup()

    async def stop_listening(self):
        """停止监听"""
        self._stop_event.set()
        self._is_listening = False
        self._cleanup()

    def _cleanup(self):
        """清理"""
        self._is_listening = False
        self._stop_event.set()

        if self._audio_thread:
            self._audio_thread.join(timeout=1.0)
            self._audio_thread = None

        if self._audio_stream:
            try:
                self._audio_stream.stop_stream()
                self._audio_stream.close()
            except Exception:
                pass
            self._audio_stream = None

    def close(self):
        """清理资源"""
        self._cleanup()
        if self._pyaudio:
            try:
                self._pyaudio.terminate()
            except Exception:
                pass
            self._pyaudio = None