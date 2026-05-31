"""
阿里云 DashScope Paraformer 实时语音识别 (ASR)

使用 DashScope SDK 的 Recognition 类进行实时流式语音识别：
- PyAudio 采集麦克风音频 (16kHz PCM)
- SDK 内部处理 WebSocket 连接
- 实时发送音频数据，接收转录结果
- 静音检测自动结束识别
"""
import asyncio
import queue
import threading
import time
import os
from typing import Optional

try:
    import pyaudio
    PYAUDIO_AVAILABLE = True
except ImportError:
    PYAUDIO_AVAILABLE = False
    print("[ASR] PyAudio 未安装，语音识别功能不可用")

from dashscope.audio.asr import Recognition, RecognitionCallback, RecognitionResult

from .base import ASRProviderBase


class DashScopeASRProvider(ASRProviderBase):
    """
    阿里云 DashScope Paraformer 实时语音识别

    使用 dashscope SDK 的 Recognition 类：
    https://help.aliyun.com/zh/model-studio/developer-reference/paraformer-real-time-speech-recognition

    使用流程:
    1. 初始化 PyAudio 音频采集
    2. 创建 Recognition 实例，设置回调
    3. 调用 start() 开始识别
    4. 持续调用 send_audio_frame() 发送音频
    5. 通过回调接收实时转录结果
    6. 静音检测后调用 stop() 结束
    """

    def __init__(
        self,
        api_key: str = None,
        sample_rate: int = 16000,
        silence_threshold: float = 1.5,
        silence_energy_threshold: int = 300,
        chunk_size: int = 1024
    ):
        """
        初始化 ASR 提供者

        Args:
            api_key: DashScope API Key（如不提供则从环境变量获取）
            sample_rate: 音频采样率（默认 16kHz）
            silence_threshold: 静音检测时间阈值（秒）
            silence_energy_threshold: 静音能量阈值
            chunk_size: 音频块大小
        """
        # 导入 config 获取 API Key
        import sys
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
        import config

        self.api_key = api_key or config.DASHSCOPE_API_KEY
        self.sample_rate = sample_rate
        self.silence_threshold = silence_threshold
        self.silence_energy_threshold = silence_energy_threshold
        self.chunk_size = chunk_size

        # 设置 API Key
        if self.api_key:
            import dashscope
            dashscope.api_key = self.api_key

        # PyAudio 实例
        self._pyaudio = None
        self._audio_stream = None

        # Recognition 实例
        self._recognition = None

        # 结果队列（线程安全）
        self._result_queue = queue.Queue()

        # 控制标志
        self._is_listening = False
        self._stop_event = threading.Event()

        # 转录结果缓存
        self._transcript_buffer = ""
        self._final_transcript = ""

        # 音频采集线程
        self._audio_thread = None

    @property
    def name(self) -> str:
        return "DashScope Paraformer ASR"

    def _check_dependencies(self) -> bool:
        """检查依赖是否可用"""
        if not PYAUDIO_AVAILABLE:
            print("[ASR] 错误: PyAudio 未安装，请运行 pip install pyaudio")
            return False
        if not self.api_key:
            print("[ASR] 错误: DASHSCOPE_API_KEY 未设置")
            return False
        return True

    def _create_callback(self) -> RecognitionCallback:
        """创建识别回调"""
        class ASRCallback(RecognitionCallback):
            def __init__(self, result_queue):
                self.result_queue = result_queue

            def on_open(self):
                print("[ASR] WebSocket 连接已建立")

            def on_event(self, result: RecognitionResult):
                """收到识别结果"""
                sentence = result.get_sentence()
                if sentence:
                    text = sentence.get('text', '')
                    # end_time 存在且非 None 表示句子结束
                    end_time = sentence.get('end_time')
                    is_final = end_time is not None and end_time > 0
                    self.result_queue.put({
                        'text': text,
                        'is_final': is_final
                    })

            def on_error(self, result: RecognitionResult):
                """错误回调"""
                # 安全地获取错误信息（不同错误类型下 result 结构可能不同）
                try:
                    # 尝试多种方式获取错误信息
                    if hasattr(result, 'get_error_msg'):
                        error_msg = result.get_error_msg()
                    elif hasattr(result, 'error_msg'):
                        error_msg = result.error_msg
                    elif isinstance(result, dict) and 'error_msg' in result:
                        error_msg = result['error_msg']
                    else:
                        # 对于连接错误等特殊情况
                        error_msg = str(result) if result else "未知错误"
                except Exception as e:
                    error_msg = f"错误获取失败: {e}"

                print(f"[ASR] 服务错误: {error_msg}")
                self.result_queue.put({
                    'error': error_msg
                })

            def on_complete(self):
                """识别完成"""
                print("[ASR] 识别完成")
                self.result_queue.put({
                    'complete': True
                })

            def on_close(self):
                """连接关闭"""
                print("[ASR] WebSocket 连接已关闭")
                self.result_queue.put({
                    'closed': True
                })

        return ASRCallback(self._result_queue)

    def _init_audio_capture(self):
        """初始化 PyAudio 音频采集"""
        if self._pyaudio is None:
            self._pyaudio = pyaudio.PyAudio()

        # 查找可用的输入设备
        device_count = self._pyaudio.get_device_count()
        input_device = None
        for i in range(device_count):
            device_info = self._pyaudio.get_device_info_by_index(i)
            if device_info['maxInputChannels'] > 0:
                input_device = i
                break

        if input_device is None:
            raise RuntimeError("未找到可用的麦克风设备")

        # 打开音频流
        self._audio_stream = self._pyaudio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.sample_rate,
            input=True,
            input_device_index=input_device,
            frames_per_buffer=self.chunk_size
        )

        print(f"[ASR] 音频采集已启动: {self.sample_rate}Hz, 设备 {input_device}")

    def _audio_capture_thread(self):
        """音频采集线程"""
        try:
            while self._is_listening and not self._stop_event.is_set():
                # 从音频流读取数据
                audio_data = self._audio_stream.read(
                    self.chunk_size,
                    exception_on_overflow=False
                )

                # 发送到 Recognition
                if self._recognition:
                    try:
                        self._recognition.send_audio_frame(audio_data)
                    except Exception as e:
                        print(f"[ASR] 发送音频失败: {e}")

        except Exception as e:
            print(f"[ASR] 音频采集错误: {e}")

    def _detect_silence(self, audio_data: bytes) -> bool:
        """
        检测静音

        计算音频能量（RMS），低于阈值则判定为静音
        """
        import struct

        # 将 bytes 转为 16-bit 整数数组
        samples = struct.unpack(f'{len(audio_data)//2}h', audio_data)

        # 计算 RMS 能量
        if len(samples) == 0:
            return True

        rms = sum(abs(s) for s in samples) / len(samples)

        return rms < self.silence_energy_threshold

    async def listen(
        self,
        timeout: float = 10.0,
        silence_threshold: float = None
    ) -> Optional[str]:
        """
        单次语音识别

        Args:
            timeout: 最大监听时间（秒）
            silence_threshold: 静音检测阈值（秒），覆盖默认值

        Returns:
            转录文本，如果无有效语音则返回 None
        """
        if not self._check_dependencies():
            return None

        silence_threshold = silence_threshold or self.silence_threshold
        self._final_transcript = ""
        self._transcript_buffer = ""
        self._stop_event.clear()
        self._result_queue = queue.Queue()

        try:
            # 初始化音频采集
            self._init_audio_capture()

            # 创建 Recognition 实例
            callback = self._create_callback()
            self._recognition = Recognition(
                model='paraformer-realtime-v2',
                callback=callback,
                format='pcm',
                sample_rate=self.sample_rate
            )

            # 开始识别
            self._recognition.start()

            self._is_listening = True

            # 启动音频采集线程
            self._audio_thread = threading.Thread(
                target=self._audio_capture_thread,
                daemon=True
            )
            self._audio_thread.start()

            # 处理结果队列
            result = await self._process_results(
                timeout=timeout,
                silence_threshold=silence_threshold
            )

            return result

        except Exception as e:
            print(f"[ASR] 识别失败: {e}")
            return None

        finally:
            self._cleanup()

    async def _process_results(
        self,
        timeout: float,
        silence_threshold: float
    ) -> Optional[str]:
        """
        处理识别结果

        Args:
            timeout: 最大监听时间
            silence_threshold: 静音检测阈值

        Returns:
            最终转录文本
        """
        start_time = time.time()
        silence_start = None
        has_speech = False

        while True:
            # 检查超时
            elapsed = time.time() - start_time
            if elapsed > timeout:
                print(f"[ASR] 超时结束 ({timeout}s)")
                break

            # 检查停止信号
            if self._stop_event.is_set():
                break

            try:
                # 从队列获取结果（非阻塞检查）
                try:
                    item = self._result_queue.get(timeout=0.1)
                except queue.Empty:
                    # 检查静音（基于音频能量）
                    # 注意：简化处理，实际应结合音频能量检测
                    if has_speech and silence_start is None:
                        silence_start = time.time()
                    elif has_speech and silence_start:
                        silence_duration = time.time() - silence_start
                        if silence_duration > silence_threshold:
                            print(f"[ASR] 静音结束 ({silence_threshold}s)")
                            break
                    continue

                # 处理结果
                if 'error' in item:
                    print(f"[ASR] 错误: {item['error']}")
                    break

                if 'complete' in item or 'closed' in item:
                    break

                if 'text' in item:
                    text = item['text']
                    is_final = item.get('is_final', False)

                    if text:
                        self._transcript_buffer = text
                        has_speech = True
                        silence_start = None  # 有语音，重置静音计时
                        print(f"[ASR] 转录: {text}")

                    # 最终结果
                    if is_final:
                        self._final_transcript = text
                        print(f"[ASR] 最终结果: {text}")
                        break

            except Exception as e:
                print(f"[ASR] 处理结果错误: {e}")
                break

        # 返回最终结果
        result = self._final_transcript or self._transcript_buffer
        return result if result.strip() else None

    def _cleanup(self):
        """清理资源"""
        # 停止识别
        if self._recognition:
            try:
                self._recognition.stop()
            except Exception:
                pass
            self._recognition = None

        # 停止音频采集
        self._is_listening = False
        self._stop_event.set()

        if self._audio_thread:
            self._audio_thread.join(timeout=1.0)
            self._audio_thread = None

        # 关闭音频流
        if self._audio_stream:
            try:
                self._audio_stream.stop_stream()
                self._audio_stream.close()
            except Exception:
                pass
            self._audio_stream = None

    async def start_continuous_listening(self):
        """
        连续监听模式（流式返回结果）

        使用异步生成器模式
        """
        if not self._check_dependencies():
            return

        self._stop_event.clear()
        self._result_queue = queue.Queue()

        try:
            self._init_audio_capture()

            callback = self._create_callback()
            self._recognition = Recognition(
                model='paraformer-realtime-v2',
                callback=callback,
                format='pcm',
                sample_rate=self.sample_rate
            )

            self._recognition.start()
            self._is_listening = True

            # 启动音频采集线程
            self._audio_thread = threading.Thread(
                target=self._audio_capture_thread,
                daemon=True
            )
            self._audio_thread.start()

            # 持续处理消息，直到收到停止信号
            while self._is_listening and not self._stop_event.is_set():
                try:
                    item = self._result_queue.get(timeout=0.1)

                    if 'text' in item:
                        yield item['text']

                    if 'complete' in item or 'closed' in item or 'error' in item:
                        break

                except queue.Empty:
                    continue

        finally:
            self._cleanup()

    async def stop_listening(self):
        """停止监听"""
        self._stop_event.set()
        self._is_listening = False
        self._cleanup()

    def close(self):
        """同步清理资源（用于非异步环境）"""
        self._cleanup()

        if self._pyaudio:
            try:
                self._pyaudio.terminate()
            except Exception:
                pass
            self._pyaudio = None