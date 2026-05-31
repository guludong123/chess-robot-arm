"""
FunASR 模型对比测试

对比流式模型 vs 离线模型的识别准确率
"""
import pyaudio
import numpy as np
import time
import sys
import os
import logging

# 抑制日志
logging.getLogger('modelscope').setLevel(logging.ERROR)
logging.getLogger('funasr').setLevel(logging.ERROR)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from funasr import AutoModel
import config

# 测试参数
SAMPLE_RATE = 16000
CHUNK_SIZE = 1024
VAD_ACCUMULATE_FRAMES = 5
ENERGY_THRESHOLD = 0.015
SILENCE_THRESHOLD = 1.5

def load_models():
    """加载 VAD 和两种 ASR 模型"""
    print("=" * 60)
    print("加载模型...")
    print("=" * 60)

    # VAD 模型
    print("  VAD: fsmn-vad")
    vad_model = AutoModel(
        model='fsmn-vad',
        device=config.FUNASR_DEVICE,
        disable_update=True,
        disable_log=True
    )

    # 流式模型（当前使用）
    print("  ASR (流式): paraformer-zh-streaming")
    streaming_model = AutoModel(
        model='paraformer-zh-streaming',
        device=config.FUNASR_DEVICE,
        disable_update=True,
        disable_log=True
    )

    # 离线模型（准确率更高）
    print("  ASR (离线): paraformer-zh")
    offline_model = AutoModel(
        model='paraformer-zh',  # 非流式，准确率更高
        device=config.FUNASR_DEVICE,
        disable_update=True,
        disable_log=True
    )

    print("✓ 模型加载完成")
    return vad_model, streaming_model, offline_model


def detect_speech_vad(vad_model, audio_data):
    """VAD 检测"""
    audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
    result = vad_model.generate(audio_array)
    if result and len(result) > 0:
        value = result[0].get('value', [])
        if isinstance(value, list) and len(value) > 0:
            return True
    return False


def recognize(asr_model, audio_buffer):
    """ASR 识别"""
    audio_data = b''.join(audio_buffer)
    audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
    result = asr_model.generate(audio_array)
    if result and len(result) > 0:
        return result[0].get('text', '')
    return ''


def test_model_comparison():
    """对比测试两种模型"""
    vad_model, streaming_model, offline_model = load_models()

    # 初始化音频
    p = pyaudio.PyAudio()

    # 显示可用设备
    print("\n可用麦克风设备:")
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        if info['maxInputChannels'] > 0:
            print(f"  [{i}] {info['name']}")

    print("\n使用设备: [0] Microsoft 声音映射器")

    stream = p.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=SAMPLE_RATE,
        input=True,
        input_device_index=0,
        frames_per_buffer=CHUNK_SIZE
    )

    print("\n" + "=" * 60)
    print("语音识别对比测试")
    print("说一句话，对比两种模型的识别结果")
    print("按 Ctrl+C 退出")
    print("=" * 60)

    try:
        test_count = 0
        while True:
            print("\n[监听] 请说话...")
            test_count += 1

            # 状态变量
            audio_buffer = []
            vad_buffer = []
            speech_confirmed = False
            vad_confirm_count = 0
            vad_no_speech_count = 0
            start_time = time.time()

            while time.time() - start_time < 15:
                audio_data = stream.read(CHUNK_SIZE, exception_on_overflow=False)

                # 计算 RMS
                audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
                rms = np.sqrt(np.mean(audio_array ** 2))
                potential_speech = rms > ENERGY_THRESHOLD

                # 状态1: 等待语音开始
                if not speech_confirmed:
                    if potential_speech:
                        audio_buffer.append(audio_data)
                        vad_buffer.append(audio_data)

                    if len(vad_buffer) >= VAD_ACCUMULATE_FRAMES:
                        if detect_speech_vad(vad_model, b''.join(vad_buffer[-VAD_ACCUMULATE_FRAMES:])):
                            vad_confirm_count += 1
                            if vad_confirm_count >= 2:
                                speech_confirmed = True
                                vad_buffer = []
                                print("  [VAD] 确认语音开始")
                        else:
                            vad_confirm_count = 0
                            vad_buffer = vad_buffer[-(VAD_ACCUMULATE_FRAMES - 1):]

                # 状态2: 录音中，等待结束
                else:
                    vad_buffer.append(audio_data)
                    if potential_speech:
                        audio_buffer.append(audio_data)

                    if len(vad_buffer) >= VAD_ACCUMULATE_FRAMES:
                        if detect_speech_vad(vad_model, b''.join(vad_buffer[-VAD_ACCUMULATE_FRAMES:])):
                            vad_no_speech_count = 0
                            vad_buffer = []
                            audio_buffer.append(audio_data)
                        else:
                            vad_no_speech_count += 1
                            vad_buffer = vad_buffer[-(VAD_ACCUMULATE_FRAMES - 1):]

                    # 结束条件
                    if vad_no_speech_count >= 3:
                        print(f"  [结束] 录音 {len(audio_buffer)} 帧 ({len(audio_buffer) * CHUNK_SIZE / SAMPLE_RATE:.2f}s)")

                        # 对比识别
                        print("\n  识别结果对比:")

                        # 流式模型
                        streaming_result = recognize(streaming_model, audio_buffer)
                        print(f"    流式模型: \"{streaming_result}\"")

                        # 离线模型
                        offline_result = recognize(offline_model, audio_buffer)
                        print(f"    离线模型: \"{offline_result}\"")

                        break

    except KeyboardInterrupt:
        print("\n退出测试")

    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()


if __name__ == '__main__':
    test_model_comparison()