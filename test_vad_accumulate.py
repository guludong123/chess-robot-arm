"""
VAD-ASR 完整链路测试

测试流程：
1. 能量前端快速检测
2. FSMN-VAD 累积确认语音
3. Paraformer ASR 识别语音内容

使用方法：
f:/miniconda/envs/chessrobot/python.exe test_vad_accumulate.py
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

# 参数
SAMPLE_RATE = config.ASR_SAMPLE_RATE
CHUNK_SIZE = 1024
VAD_ACCUMULATE_FRAMES = config.ASR_VAD_ACCUMULATE_FRAMES
ENERGY_THRESHOLD = config.ASR_ENERGY_THRESHOLD
SILENCE_THRESHOLD = config.ASR_SILENCE_THRESHOLD
SPEECH_CONFIRM_COUNT = config.ASR_SPEECH_CONFIRM_COUNT


VAD_END_COUNT = getattr(config, 'ASR_VAD_END_COUNT', 2)  # VAD 无语音次数就结束


def test_vad_asr_pipeline():
    print("=" * 60)
    print("VAD-ASR 完整链路测试")
    print(f"能量阈值: {ENERGY_THRESHOLD}")
    print(f"VAD 累积帧数: {VAD_ACCUMULATE_FRAMES} (~{VAD_ACCUMULATE_FRAMES * CHUNK_SIZE / SAMPLE_RATE * 1000:.0f}ms)")
    print(f"VAD 确认次数: {SPEECH_CONFIRM_COUNT}")
    print(f"VAD 结束次数: {VAD_END_COUNT}")
    print(f"静音阈值: {SILENCE_THRESHOLD}s")
    print("=" * 60)

    # 加载模型
    print("\n加载模型...")
    print(f"  ASR: {config.FUNASR_MODEL}")
    print(f"  VAD: {config.FUNASR_VAD_MODEL}")

    asr_model = AutoModel(
        model=config.FUNASR_MODEL,
        device=config.FUNASR_DEVICE,
        disable_update=True,
        disable_log=True
    )
    vad_model = AutoModel(
        model=config.FUNASR_VAD_MODEL,
        device=config.FUNASR_DEVICE,
        disable_update=True,
        disable_log=True
    )
    print("✓ 模型就绪")

    # 初始化音频
    p = pyaudio.PyAudio()
    stream = p.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=SAMPLE_RATE,
        input=True,
        input_device_index=0,
        frames_per_buffer=CHUNK_SIZE
    )

    print("\n" + "=" * 60)
    print(f"[监听] 请说话，说完后停顿 {SILENCE_THRESHOLD} 秒自动结束...")
    print("=" * 60)

    # 状态变量
    audio_buffer = []          # ASR 识别用完整缓冲
    vad_buffer = []            # VAD 检测用滑动缓冲
    speech_confirmed = False   # VAD 是否确认有语音
    vad_confirm_count = 0      # VAD 连续确认计数
    vad_no_speech_count = 0    # VAD 连续无语音计数
    silence_start = None       # 静音开始时间
    start_time = time.time()
    max_duration = 30

    try:
        while time.time() - start_time < max_duration:
            audio_data = stream.read(CHUNK_SIZE, exception_on_overflow=False)

            # 计算 RMS
            audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
            rms = np.sqrt(np.mean(audio_array ** 2))

            # 能量前端检测
            potential_speech = rms > ENERGY_THRESHOLD

            # 显示状态
            elapsed = time.time() - start_time
            bar_len = int(rms * 50)
            bar = "█" * min(bar_len, 30) + "░" * (30 - min(bar_len, 30))

            # 状态1: 等待语音开始
            if not speech_confirmed:
                energy_status = "等待语音" if not potential_speech else "潜在语音"

                if potential_speech:
                    audio_buffer.append(audio_data)
                    vad_buffer.append(audio_data)

                # VAD 累积确认
                vad_status = ""
                if len(vad_buffer) >= VAD_ACCUMULATE_FRAMES:
                    combined = b''.join(vad_buffer[-VAD_ACCUMULATE_FRAMES:])
                    audio_combined = np.frombuffer(combined, dtype=np.int16).astype(np.float32) / 32768.0

                    result = vad_model.generate(audio_combined)
                    is_speech = False
                    if result and len(result) > 0:
                        value = result[0].get('value', [])
                        if isinstance(value, list) and len(value) > 0:
                            is_speech = True

                    if is_speech:
                        vad_confirm_count += 1
                        vad_status = f"VAD确认({vad_confirm_count}/{SPEECH_CONFIRM_COUNT})"
                        if vad_confirm_count >= SPEECH_CONFIRM_COUNT:
                            speech_confirmed = True
                            vad_buffer = []
                            print(f"\n>>> [VAD] 确认语音开始！开始录音...")
                    else:
                        vad_confirm_count = 0
                        vad_buffer = vad_buffer[-(VAD_ACCUMULATE_FRAMES - 1):]
                        vad_status = "VAD:无"

                print(f"[{elapsed:.1f}s] RMS={rms:.4f} |{bar}| {energy_status} | {vad_status} | 缓冲:{len(audio_buffer)}帧")

            # 状态2: 语音已确认，等待结束
            else:
                # 始终添加到缓冲区（用于 VAD 检测）
                vad_buffer.append(audio_data)

                if potential_speech:
                    # 能量高时添加到录音缓冲
                    audio_buffer.append(audio_data)
                    silence_start = None

                # VAD 检测是否结束
                vad_status = ""
                if len(vad_buffer) >= VAD_ACCUMULATE_FRAMES:
                    combined = b''.join(vad_buffer[-VAD_ACCUMULATE_FRAMES:])
                    audio_combined = np.frombuffer(combined, dtype=np.int16).astype(np.float32) / 32768.0

                    result = vad_model.generate(audio_combined)
                    is_speech = False
                    if result and len(result) > 0:
                        value = result[0].get('value', [])
                        if isinstance(value, list) and len(value) > 0:
                            is_speech = True

                    if is_speech:
                        vad_no_speech_count = 0
                        vad_buffer = []  # 清空，重新累积
                        vad_status = "VAD:有"
                        # VAD 有语音时也添加到录音缓冲
                        audio_buffer.append(audio_data)
                    else:
                        vad_no_speech_count += 1
                        # 滑动窗口，保留部分帧
                        vad_buffer = vad_buffer[-(VAD_ACCUMULATE_FRAMES - 1):]
                        vad_status = f"VAD:无({vad_no_speech_count})"

                # 静音计时（能量低时）
                if not potential_speech:
                    if silence_start is None:
                        silence_start = time.time()

                silence_duration = 0 if silence_start is None else time.time() - silence_start
                silence_status = f"静音:{silence_duration:.1f}s" if silence_start else ""

                status = "录音中" if potential_speech else "等待结束"
                print(f"[{elapsed:.1f}s] RMS={rms:.4f} |{bar}| {status} | {vad_status} | {silence_status} | 缓冲:{len(audio_buffer)}帧")

                # 结束条件：VAD 连续无语音（主要条件）
                # 或 静音超时（备用条件，当能量确实很低时）
                if vad_no_speech_count >= VAD_END_COUNT:
                    print(f"\n>>> [结束] VAD 连续 {vad_no_speech_count} 次无语音，开始 ASR 识别...")
                    print(f"    音频缓冲: {len(audio_buffer)} 帧 ({len(audio_buffer) * CHUNK_SIZE / SAMPLE_RATE:.2f}s)")
                elif silence_duration > SILENCE_THRESHOLD:
                    print(f"\n>>> [结束] 静音超时 {silence_duration:.1f}s，开始 ASR 识别...")
                    print(f"    音频缓冲: {len(audio_buffer)} 帧 ({len(audio_buffer) * CHUNK_SIZE / SAMPLE_RATE:.2f}s)")

                # 触发识别
                if vad_no_speech_count >= VAD_END_COUNT or silence_duration > SILENCE_THRESHOLD:
                    print(f"\n>>> [结束] 静音超时，开始 ASR 识别...")
                    print(f"    音频缓冲: {len(audio_buffer)} 帧 ({len(audio_buffer) * CHUNK_SIZE / SAMPLE_RATE:.2f}s)")

                    # ASR 识别
                    if len(audio_buffer) > 0:
                        audio_final = b''.join(audio_buffer)
                        audio_array_final = np.frombuffer(audio_final, dtype=np.int16).astype(np.float32) / 32768.0

                        print("    正在识别...")
                        asr_result = asr_model.generate(audio_array_final)
                        if asr_result and len(asr_result) > 0:
                            text = asr_result[0].get('text', '')
                            print(f"\n>>> [ASR结果] \"{text}\"")
                        else:
                            print("\n>>> [ASR结果] 识别失败，无结果")
                    else:
                        print("    缓冲区为空，跳过识别")

                    # 重置继续监听
                    audio_buffer = []
                    vad_buffer = []
                    speech_confirmed = False
                    vad_confirm_count = 0
                    vad_no_speech_count = 0
                    start_time = time.time()
                    print("\n[监听] 请继续说话...")

    except KeyboardInterrupt:
        print("\n\n退出测试")

    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()


if __name__ == '__main__':
    test_vad_asr_pipeline()