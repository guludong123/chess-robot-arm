"""
E2E Speech Module 独立测试脚本

使用 StepFun Realtime API 进行端到端语音对话测试。

用法：
    python test_e2e_speech.py
    python test_e2e_speech.py --voice wenrounansheng
    python test_e2e_speech.py --device 27
    python test_e2e_speech.py --list-devices
    python test_e2e_speech.py --no-preprocess  # 禁用音频预处理

交互命令（运行时输入）：
    t <文字>  - 发送文字消息
    i         - 打断 AI 回复
    v <音色>  - 切换音色
    q         - 退出
"""

import argparse
import asyncio
import signal
import struct
import sys
import threading
import time

try:
    import pyaudio
    PYAUDIO_AVAILABLE = True
except ImportError:
    PYAUDIO_AVAILABLE = False

from modules.e2e_speech import StepFunProvider
from modules.e2e_speech.stepfun_provider import STEPFUN_SAMPLE_RATE, STEPFUN_VOICES

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

# 音频参数
FORMAT = pyaudio.paInt16 if PYAUDIO_AVAILABLE else None
CHANNELS = 1


class AudioPreprocessor:
    """音频预处理器：噪声门 + 增益归一化"""

    def __init__(self, noise_gate_threshold=0.005, gain_target_rms=0.08,
                 gain_max=10.0, enabled=True):
        self.enabled = enabled and NUMPY_AVAILABLE
        self.noise_gate_threshold = noise_gate_threshold
        self.gain_target_rms = gain_target_rms
        self.gain_max = gain_max

        # 自动增益控制（平滑增益变化）
        self._current_gain = 1.0
        self._gain_smoothing = 0.3  # 增益变化平滑系数

        # 噪底（启动时校准）
        self.noise_floor = noise_gate_threshold

    def calibrate_noise_floor(self, stream, chunk_size, duration=2.0):
        """校准噪底：录几秒静音，计算环境噪声 RMS"""
        if not self.enabled:
            return

        print(f"[校准] 测量环境噪声 ({duration}s)...")
        rms_list = []

        start = time.time()
        while time.time() - start < duration:
            data = stream.read(chunk_size, exception_on_overflow=False)
            arr = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
            rms = float(np.sqrt(np.mean(arr ** 2)))
            rms_list.append(rms)

        self.noise_floor = float(np.mean(rms_list))
        # 噪声门阈值 = 噪底的 2 倍，留余量
        self.noise_gate_threshold = max(self.noise_gate_threshold, self.noise_floor * 2)
        print(f"[校准] 噪底 RMS: {self.noise_floor:.4f}, 噪声门阈值: {self.noise_gate_threshold:.4f}")

    def process(self, audio_data: bytes) -> bytes:
        """处理一帧音频：噪声门 + 增益归一化"""
        if not self.enabled:
            return audio_data

        arr = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
        rms = float(np.sqrt(np.mean(arr ** 2)))

        # 噪声门：低于阈值的帧静音
        if rms < self.noise_gate_threshold:
            return b'\x00' * len(audio_data)

        # 增益归一化：把 RMS 拉到目标值
        if rms > 0.001:  # 避免除以极小值
            target_gain = self.gain_target_rms / rms
            target_gain = min(target_gain, self.gain_max)
            # 平滑增益变化，避免突变
            self._current_gain = (self._gain_smoothing * target_gain +
                                  (1 - self._gain_smoothing) * self._current_gain)

        # 应用增益
        arr = arr * self._current_gain
        # 钳位到 [-1, 1]
        arr = np.clip(arr, -1.0, 1.0)
        return (arr * 32768.0).astype(np.int16).tobytes()


def list_audio_devices():
    """列出所有音频输入设备"""
    if not PYAUDIO_AVAILABLE:
        print("[错误] PyAudio 未安装")
        return

    pa = pyaudio.PyAudio()
    print("\n可用音频输入设备:")
    print("-" * 60)
    for i in range(pa.get_device_count()):
        info = pa.get_device_info_by_index(i)
        if info['maxInputChannels'] > 0:
            default_rate = int(info['defaultSampleRate'])
            print(f"  [{i}] {info['name']}  (采样率: {default_rate}Hz, 通道: {info['maxInputChannels']})")
    print("-" * 60)
    pa.terminate()


def resample_pcm16(data: bytes, src_rate: int, dst_rate: int) -> bytes:
    """PCM16 音频重采样"""
    if src_rate == dst_rate:
        return data

    # 整数比降采样（如 48000->24000 = 2:1），直接取每隔 N 个点
    if src_rate > dst_rate and src_rate % dst_rate == 0:
        step = src_rate // dst_rate
        n = len(data) // 2
        samples = struct.unpack(f'<{n}h', data)
        out = samples[::step]
        return struct.pack(f'<{len(out)}h', *out)

    # 整数比升采样（如 24000->48000 = 1:2），每个点重复 N 次
    if dst_rate > src_rate and dst_rate % src_rate == 0:
        repeat = dst_rate // src_rate
        n = len(data) // 2
        samples = struct.unpack(f'<{n}h', data)
        out = []
        for s in samples:
            out.extend([s] * repeat)
        return struct.pack(f'<{len(out)}h', *out)

    # 通用线性插值（非整数比）
    samples = struct.unpack(f'<{len(data) // 2}h', data)
    ratio = src_rate / dst_rate
    n_out = int(len(samples) / ratio)
    out = []
    for i in range(n_out):
        pos = i * ratio
        idx = int(pos)
        frac = pos - idx
        if idx + 1 < len(samples):
            val = samples[idx] * (1 - frac) + samples[idx + 1] * frac
        else:
            val = samples[idx]
        out.append(int(val))
    return struct.pack(f'<{len(out)}h', *out)


class E2ESpeechTester:
    """E2E 语音模块测试器"""

    def __init__(self, provider: StepFunProvider, device_index: int = None,
                 enable_preprocess: bool = True):
        self.provider = provider
        self.device_index = device_index
        self.running = False
        self.enable_preprocess = enable_preprocess

        # PyAudio
        self.pa = None
        self.input_stream = None
        self.output_stream = None

        # 设备参数
        self.device_sample_rate = None
        self.device_channels = None  # 设备原生通道数
        self.device_chunk_size = None

        # API 采样率
        self.api_sample_rate = provider.sample_rate

        # 统计
        self.audio_recv_count = 0

    def init_audio(self):
        """初始化 PyAudio 设备"""
        if not PYAUDIO_AVAILABLE:
            raise RuntimeError("PyAudio 未安装")

        self.pa = pyaudio.PyAudio()

        # 选择输入设备
        input_device = self.device_index
        if input_device is not None:
            info = self.pa.get_device_info_by_index(input_device)
            self.device_sample_rate = int(info['defaultSampleRate'])
            self.device_channels = int(info['maxInputChannels'])
            print(f"[音频] 使用指定设备 [{input_device}]: {info['name']}")
        else:
            for i in range(self.pa.get_device_count()):
                info = self.pa.get_device_info_by_index(i)
                if info['maxInputChannels'] > 0:
                    input_device = i
                    self.device_sample_rate = int(info['defaultSampleRate'])
                    self.device_channels = int(info['maxInputChannels'])
                    print(f"[音频] 自动选择设备 [{i}]: {info['name']}")
                    break

        if input_device is None:
            raise RuntimeError("未找到麦克风")

        print(f"[音频] 设备采样率: {self.device_sample_rate}Hz, 通道: {self.device_channels}")

        # 设备端每帧采样数（20ms）
        self.device_chunk_size = int(self.device_sample_rate * 0.02)

        # 输入流 - 使用设备原生参数（关键：用设备的通道数，不要强制单声道）
        self.input_stream = self.pa.open(
            format=FORMAT,
            channels=self.device_channels,
            rate=self.device_sample_rate,
            input=True,
            input_device_index=input_device,
            frames_per_buffer=self.device_chunk_size,
        )

        # 输出流 - 使用默认输出设备
        self.output_stream = self.pa.open(
            format=FORMAT,
            channels=1,
            rate=self.device_sample_rate,
            output=True,
            frames_per_buffer=self.device_chunk_size * 4,
        )

        print(f"[音频] 初始化完成")

    def cleanup_audio(self):
        """清理音频设备"""
        for stream in [self.input_stream, self.output_stream]:
            if stream:
                try:
                    stream.stop_stream()
                    stream.close()
                except Exception:
                    pass
        if self.pa:
            try:
                self.pa.terminate()
            except Exception:
                pass
        self.input_stream = None
        self.output_stream = None
        self.pa = None

    def setup_callbacks(self):
        """注册回调函数"""

        def on_user_speech_start():
            print("\n[VAD] 用户开始说话")

        def on_user_speech_end():
            print("[VAD] 用户停止说话")

        def on_user_transcript(text):
            print(f"[用户] {text}")

        def on_ai_audio_chunk(audio_bytes):
            """播放 AI 音频（需要上采样到设备原生采样率）"""
            self.audio_recv_count += 1
            if self.output_stream:
                try:
                    # AI 音频是 24kHz，需要上采样到设备原生采样率
                    if self.device_sample_rate != self.api_sample_rate:
                        audio_bytes = resample_pcm16(
                            audio_bytes, self.api_sample_rate, self.device_sample_rate
                        )
                    self.output_stream.write(audio_bytes)
                except Exception as e:
                    print(f"[错误] 音频播放: {e}")

        def on_ai_transcript_chunk(text):
            print(text, end='', flush=True)

        def on_ai_response_done():
            if self.audio_recv_count > 0:
                print()
            print("[AI] 回复完成")
            self.audio_recv_count = 0

        def on_error(error_type, message):
            print(f"\n[错误] {error_type}: {message}")

        self.provider.on_user_speech_start = on_user_speech_start
        self.provider.on_user_speech_end = on_user_speech_end
        self.provider.on_user_transcript = on_user_transcript
        self.provider.on_ai_audio_chunk = on_ai_audio_chunk
        self.provider.on_ai_transcript_chunk = on_ai_transcript_chunk
        self.provider.on_ai_response_done = on_ai_response_done
        self.provider.on_error = on_error

    async def send_audio_loop(self):
        """麦克风采集并发送音频"""
        count = 0
        need_resample = self.device_sample_rate != self.api_sample_rate
        need_mono = self.device_channels > 1

        while self.running and self.provider.is_connected:
            try:
                # 读取设备原生音频（可能是多通道）
                audio_data = self.input_stream.read(
                    self.device_chunk_size, exception_on_overflow=False
                )

                # 多通道转单声道（取所有通道的平均值）
                if need_mono:
                    n = len(audio_data) // 2
                    samples = struct.unpack(f'<{n}h', audio_data)
                    ch = self.device_channels
                    mono = []
                    for i in range(0, n, ch):
                        mono.append(sum(samples[i:i+ch]) // ch)
                    audio_data = struct.pack(f'<{len(mono)}h', *mono)

                # 重采样到 API 采样率
                if need_resample:
                    audio_data = resample_pcm16(
                        audio_data, self.device_sample_rate, self.api_sample_rate
                    )

                # 音频预处理（噪声门 + 增益归一化）
                if self.preprocessor:
                    audio_data = self.preprocessor.process(audio_data)

                # 计算能量（调试用）
                samples = struct.unpack(f'{len(audio_data) // 2}h', audio_data)
                energy = sum(abs(s) for s in samples) / len(samples) if samples else 0

                count += 1
                if count % 50 == 0:
                    if energy > 500:
                        label = "[说话]"
                    elif energy > 200:
                        label = "[噪音]"
                    else:
                        label = "[静音]"
                    gain_info = ""
                    if self.preprocessor:
                        gain_info = f" gain={self.preprocessor._current_gain:.1f}x"
                    print(f"[音频] #{count} energy={energy:.0f} {label}{gain_info}")

                await self.provider.send_audio(audio_data)

                # yield 控制权，让 receive_loop 有机会处理消息
                await asyncio.sleep(0)

            except Exception as e:
                if self.running:
                    print(f"[错误] 音频发送: {e}")
                break

    def input_loop(self):
        """键盘输入循环（在独立线程中运行）"""
        while self.running:
            try:
                line = input().strip()
                if not line:
                    continue

                cmd = line[0].lower()

                if cmd == 'q':
                    print("[命令] 退出")
                    self.running = False
                    break

                elif cmd == 't':
                    text = line[1:].strip()
                    if text:
                        print(f"[文字] 发送: {text}")
                        asyncio.run_coroutine_threadsafe(
                            self.provider.send_text(text),
                            self._loop,
                        )

                elif cmd == 'i':
                    print("[命令] 打断 AI")
                    asyncio.run_coroutine_threadsafe(
                        self.provider.cancel_response(),
                        self._loop,
                    )

                elif cmd == 'v':
                    voice = line[1:].strip()
                    if voice in STEPFUN_VOICES:
                        print(f"[命令] 切换音色: {voice}")
                        asyncio.run_coroutine_threadsafe(
                            self.provider.update_session(voice=voice),
                            self._loop,
                        )
                    else:
                        print(f"[命令] 可用音色: {', '.join(STEPFUN_VOICES)}")

                else:
                    print(f"[文字] 发送: {line}")
                    asyncio.run_coroutine_threadsafe(
                        self.provider.send_text(line),
                        self._loop,
                    )

            except EOFError:
                break
            except Exception as e:
                print(f"[错误] 输入: {e}")

    async def run(self):
        """运行测试"""
        self._loop = asyncio.get_event_loop()

        # 初始化音频
        self.init_audio()

        # 初始化音频预处理器并校准噪底
        self.preprocessor = None
        if self.enable_preprocess:
            try:
                import config
                self.preprocessor = AudioPreprocessor(
                    noise_gate_threshold=getattr(config, 'AUDIO_NOISE_GATE_THRESHOLD', 0.005),
                    gain_target_rms=getattr(config, 'AUDIO_GAIN_TARGET_RMS', 0.08),
                    gain_max=getattr(config, 'AUDIO_GAIN_MAX', 10.0),
                )
                self.preprocessor.calibrate_noise_floor(
                    self.input_stream, self.device_chunk_size, duration=2.0
                )
            except ImportError:
                pass

        # 注册回调
        self.setup_callbacks()

        # 连接
        print("[连接] 正在连接 StepFun Realtime API...")
        ok = await self.provider.connect()
        if not ok:
            print("[错误] 连接失败")
            return

        self.running = True

        print("\n" + "=" * 60)
        print("E2E 语音对话测试")
        print("=" * 60)
        print(f"模型: {self.provider.name}")
        print(f"API采样率: {self.api_sample_rate}Hz")
        print(f"设备采样率: {self.device_sample_rate}Hz")
        print("对着麦克风说话，AI 会自动回复")
        print()
        print("命令:")
        print("  t <文字>  发送文字消息")
        print("  i         打断 AI 回复")
        print("  v <音色>  切换音色")
        print("  q         退出")
        print("=" * 60 + "\n")

        # 启动键盘输入线程
        input_thread = threading.Thread(target=self.input_loop, daemon=True)
        input_thread.start()

        # 运行音频采集
        try:
            await self.send_audio_loop()
        except asyncio.CancelledError:
            pass
        finally:
            await self.stop()

    async def stop(self):
        """停止并清理"""
        self.running = False
        print("\n[系统] 正在停止...")
        await self.provider.disconnect()
        self.cleanup_audio()
        print("[系统] 已停止")


def main():
    parser = argparse.ArgumentParser(description="E2E 语音模块测试")
    parser.add_argument('--api-key', default=None, help="StepFun API Key")
    parser.add_argument('--model', default='step-1o-audio', help="模型名称")
    parser.add_argument('--voice', default='qingchunshaonv',
                        choices=STEPFUN_VOICES, help="音色")
    parser.add_argument('--instructions', default='你是象棋对弈助手，用简洁中文回复用户。',
                        help="系统指令")
    parser.add_argument('--device', type=int, default=None,
                        help="音频输入设备索引 (用 --list-devices 查看)")
    parser.add_argument('--list-devices', action='store_true',
                        help="列出所有音频输入设备")
    parser.add_argument('--no-preprocess', action='store_true',
                        help="禁用音频预处理（噪声门+增益归一化）")
    args = parser.parse_args()

    if not PYAUDIO_AVAILABLE:
        print("[错误] PyAudio 未安装: pip install pyaudio")
        sys.exit(1)

    if args.list_devices:
        list_audio_devices()
        sys.exit(0)

    # 创建 Provider
    provider = StepFunProvider(
        api_key=args.api_key,
        model=args.model,
        voice=args.voice,
        instructions=args.instructions,
    )

    tester = E2ESpeechTester(provider, device_index=args.device,
                             enable_preprocess=not args.no_preprocess)

    # Ctrl+C 处理
    def signal_handler():
        print("\n[用户] Ctrl+C")
        tester.running = False

    if sys.platform == 'win32':
        signal.signal(signal.SIGINT, lambda s, f: signal_handler())
    else:
        loop = asyncio.get_event_loop()
        loop.add_signal_handler(signal.SIGINT, signal_handler)

    try:
        asyncio.run(tester.run())
    except KeyboardInterrupt:
        print("\n[用户] 退出")


if __name__ == '__main__':
    main()
