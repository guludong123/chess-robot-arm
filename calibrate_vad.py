"""
VAD 语音检测校准工具

自动测量环境噪底和语音特征，计算最优 VAD 参数并写入 config.py。

使用方法：
f:/miniconda/envs/chessrobot/python.exe calibrate_vad.py
"""
import pyaudio
import numpy as np
import time
import sys
import os
import re
import logging
import argparse
from dataclasses import dataclass

logging.getLogger('modelscope').setLevel(logging.ERROR)
logging.getLogger('funasr').setLevel(logging.ERROR)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from funasr import AutoModel
import config

SAMPLE_RATE = config.ASR_SAMPLE_RATE
CHUNK_SIZE = 1024
FRAME_DURATION = CHUNK_SIZE / SAMPLE_RATE  # ~0.064s


@dataclass
class CalibrationResult:
    noise_rms_mean: float = 0.0
    noise_rms_peak: float = 0.0
    speech_rms_mean: float = 0.0
    speech_rms_std: float = 0.0
    speech_rms_min: float = 0.0
    speech_rms_max: float = 0.0
    speech_vad_hit_rate: float = 0.0
    avg_speech_duration: float = 0.0
    energy_threshold: float = 0.0
    vad_accumulate_frames: int = 3
    speech_confirm_count: int = 1
    vad_end_count: int = 3
    silence_threshold: float = 0.8
    verification_passed: bool = False
    detection_count: int = 0
    false_trigger_count: int = 0


def load_vad_model():
    print(f"加载 VAD 模型: {config.FUNASR_VAD_MODEL} ...")
    model = AutoModel(
        model=config.FUNASR_VAD_MODEL,
        device=config.FUNASR_DEVICE,
        cache_dir=getattr(config, 'FUNASR_CACHE_DIR', None),
        disable_update=True,
        disable_log=True
    )
    print("VAD 模型就绪")
    return model


def open_audio_stream():
    p = pyaudio.PyAudio()
    try:
        stream = p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=SAMPLE_RATE,
            input=True,
            input_device_index=0,
            frames_per_buffer=CHUNK_SIZE
        )
    except OSError as e:
        print(f"未找到麦克风: {e}")
        print("请检查音频设备连接")
        p.terminate()
        sys.exit(1)
    return p, stream


def compute_rms(audio_data: bytes) -> float:
    arr = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
    return float(np.sqrt(np.mean(arr ** 2)))


def detect_vad(vad_model, audio_data: bytes) -> bool:
    arr = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
    try:
        result = vad_model.generate(arr)
        if result and len(result) > 0:
            value = result[0].get('value', [])
            if isinstance(value, list) and len(value) > 0:
                return True
    except Exception:
        pass
    return False


def rms_bar(rms: float, width: int = 30) -> str:
    bar_len = int(rms * 200)
    filled = min(bar_len, width)
    return "█" * filled + "░" * (width - filled)


# ─── Phase 1: 噪底测量 ────────────────────────────────────

def measure_noise_floor(stream, duration: float = 3.0):
    print(f"\n[Phase 1/5] 测量环境噪声")
    print(f"请保持安静 {duration} 秒...")

    rms_list = []
    num_frames = int(duration / FRAME_DURATION)
    start = time.time()

    for i in range(num_frames):
        data = stream.read(CHUNK_SIZE, exception_on_overflow=False)
        rms = compute_rms(data)
        rms_list.append(rms)

        elapsed = time.time() - start
        print(f"\r  [{elapsed:.1f}s] RMS={rms:.4f} |{rms_bar(rms)}|", end="", flush=True)

    noise_mean = float(np.mean(rms_list))
    noise_peak = float(np.max(rms_list))
    print(f"\n  噪底均值: {noise_mean:.4f}, 峰值: {noise_peak:.4f}")
    return noise_mean, noise_peak


# ─── Phase 2: 语音采样 ────────────────────────────────────

def sample_speech(stream, vad_model, noise_floor: float, num_phrases: int = 3):
    print(f"\n[Phase 2/5] 语音采样")
    print(f"请说 {num_phrases} 句话（例如：马二进三、炮八平五）")
    print(f"每句说完后停顿 1 秒自动进入下一句\n")

    energy_gate = noise_floor * 3
    all_rms = []
    all_durations = []
    vad_hits = 0
    vad_total = 0

    for phrase_idx in range(num_phrases):
        print(f"  第 {phrase_idx + 1}/{num_phrases} 句：等待说话...")

        # 等待语音开始（超时 15 秒）
        speech_start_wait = time.time()
        speech_started = False
        while time.time() - speech_start_wait < 15.0:
            data = stream.read(CHUNK_SIZE, exception_on_overflow=False)
            rms = compute_rms(data)
            if rms > energy_gate:
                speech_started = True
                break

        if not speech_started:
            print(f"  超时，跳过")
            continue

        # 录音：能量降到门限以下 1 秒算结束
        phrase_rms = []
        phrase_vad_buffer = []
        silence_start = None
        recording_start = time.time()

        while True:
            data = stream.read(CHUNK_SIZE, exception_on_overflow=False)
            rms = compute_rms(data)
            phrase_rms.append(rms)

            # VAD 检测
            phrase_vad_buffer.append(data)
            if len(phrase_vad_buffer) >= 3:
                combined = b''.join(phrase_vad_buffer[-3:])
                is_speech = detect_vad(vad_model, combined)
                vad_total += 1
                if is_speech:
                    vad_hits += 1
                    phrase_vad_buffer = []
                else:
                    phrase_vad_buffer = phrase_vad_buffer[-2:]

            # 结束判断
            if rms < energy_gate:
                if silence_start is None:
                    silence_start = time.time()
                elif time.time() - silence_start > 1.0:
                    break
            else:
                silence_start = None

            elapsed = time.time() - recording_start
            if elapsed > 10.0:
                break

        duration = time.time() - recording_start
        all_durations.append(duration)
        all_rms.extend(phrase_rms)
        print(f"  录到 {duration:.1f}s, 平均 RMS={np.mean(phrase_rms):.4f}")

    if not all_rms:
        print("  未录到任何语音，使用保守默认值")
        return 0.03, 0.01, 0.01, 0.06, 0.5, 2.0

    speech_mean = float(np.mean(all_rms))
    speech_std = float(np.std(all_rms))
    speech_min = float(np.min(all_rms))
    speech_max = float(np.max(all_rms))
    hit_rate = vad_hits / max(vad_total, 1)
    avg_dur = float(np.mean(all_durations))

    print(f"  语音均值: {speech_mean:.4f}, 标准差: {speech_std:.4f}")
    print(f"  范围: {speech_min:.4f} - {speech_max:.4f}")
    print(f"  VAD 命中率: {hit_rate:.1%}")
    print(f"  平均语句时长: {avg_dur:.1f}s")

    return speech_mean, speech_std, speech_min, speech_max, hit_rate, avg_dur


# ─── Phase 3a: 评估累积帧数 ───────────────────────────────

def evaluate_accumulate_frames(stream, vad_model, noise_floor: float):
    print(f"\n[Phase 3a/5] 测试不同累积帧数...")
    print('请说 1 句短话（如"马二进三"），说 3 次\n')

    energy_gate = noise_floor * 3
    candidates = [2, 3, 5]
    scores = {}

    for frames in candidates:
        print(f"  测试 {frames} 帧 (~{frames * FRAME_DURATION * 1000:.0f}ms)...")
        detections = 0
        total_latency = 0
        trials = 0

        for trial in range(3):
            print(f"    第 {trial + 1}/3 次: 等待说话...", end="", flush=True)

            # 等待语音
            wait_start = time.time()
            speech_started = False
            while time.time() - wait_start < 10.0:
                data = stream.read(CHUNK_SIZE, exception_on_overflow=False)
                rms = compute_rms(data)
                if rms > energy_gate:
                    speech_started = True
                    break

            if not speech_started:
                print(" 超时")
                continue

            trials += 1
            vad_buffer = []
            confirm_count = 0
            chunk_count = 0
            detected = False

            # 用目标帧数跑检测
            while chunk_count < 50:  # 最多 ~3.2s
                data = stream.read(CHUNK_SIZE, exception_on_overflow=False)
                rms = compute_rms(data)
                chunk_count += 1
                vad_buffer.append(data)

                if len(vad_buffer) >= frames:
                    combined = b''.join(vad_buffer[-frames:])
                    is_speech = detect_vad(vad_model, combined)
                    if is_speech:
                        confirm_count += 1
                        if confirm_count >= 1:
                            detected = True
                            total_latency += chunk_count
                            break
                        vad_buffer = []
                    else:
                        confirm_count = 0
                        vad_buffer = vad_buffer[-(frames - 1):]

                # 能量低且超时就停
                if rms < energy_gate and chunk_count > 10:
                    break

            if detected:
                detections += 1
                print(f" 检测到 (延迟 {chunk_count} 帧)")
            else:
                print(" 未检测到")

            # 等说完再测下一次
            time.sleep(0.5)
            while True:
                data = stream.read(CHUNK_SIZE, exception_on_overflow=False)
                if compute_rms(data) < energy_gate:
                    break

        score = detections * 100 - (total_latency / max(detections, 1))
        scores[frames] = score
        print(f"  结果: {detections}/{trials} 检测到, 平均延迟 {total_latency / max(detections, 1):.0f} 帧, 分数 {score:.0f}\n")

    best = max(scores, key=scores.get)
    print(f"  最优累积帧数: {best}")
    return best


# ─── Phase 3b: 计算参数 ───────────────────────────────────

def calculate_parameters(noise_floor, noise_peak, speech_mean, speech_std,
                         speech_min, speech_max, vad_hit_rate, best_frames, avg_duration):
    # 能量阈值：噪底到语音均值的 30% 位置
    energy_threshold = noise_floor + (speech_mean - noise_floor) * 0.3
    energy_threshold = max(noise_floor * 2, min(energy_threshold, speech_mean * 0.5))

    # 累积帧数
    vad_accumulate_frames = best_frames

    # 确认次数：基于信噪比
    snr = speech_mean / max(noise_floor, 1e-6)
    speech_confirm_count = 1 if snr > 10 else 2

    # 结束计数：基于 VAD 命中率
    vad_end_count = 3 if vad_hit_rate > 0.8 else 4

    # 静音阈值：基于语速
    if avg_duration > 3.0:
        silence_threshold = 1.2
    elif avg_duration > 1.5:
        silence_threshold = 0.8
    else:
        silence_threshold = 0.6

    return {
        'energy_threshold': round(energy_threshold, 4),
        'vad_accumulate_frames': vad_accumulate_frames,
        'speech_confirm_count': speech_confirm_count,
        'vad_end_count': vad_end_count,
        'silence_threshold': silence_threshold,
    }


# ─── Phase 4: 验证 ────────────────────────────────────────

def verify_parameters(stream, vad_model, params, noise_floor: float):
    print(f"\n[Phase 4/5] 验证参数")
    print(f"请说 3 句话来测试检测效果（20 秒内）\n")

    energy_threshold = params['energy_threshold']
    accum_frames = params['vad_accumulate_frames']
    confirm_count = params['speech_confirm_count']
    end_count = params['vad_end_count']
    silence_thresh = params['silence_threshold']

    detections = 0
    false_triggers = 0
    start = time.time()

    audio_buffer = []
    vad_buffer = []
    speech_confirmed = False
    vad_confirm = 0
    vad_no_speech = 0
    silence_start = None

    while time.time() - start < 20.0 and detections < 5:
        data = stream.read(CHUNK_SIZE, exception_on_overflow=False)
        rms = compute_rms(data)
        potential = rms > energy_threshold

        if not speech_confirmed:
            vad_buffer.append(data)
            if potential:
                audio_buffer.append(data)

            if len(vad_buffer) >= accum_frames:
                combined = b''.join(vad_buffer[-accum_frames:])
                is_speech = detect_vad(vad_model, combined)
                if is_speech:
                    vad_confirm += 1
                    if vad_confirm >= confirm_count:
                        speech_confirmed = True
                        audio_buffer.extend(vad_buffer)
                        vad_buffer = []
                else:
                    vad_confirm = 0
                    vad_buffer = vad_buffer[-(accum_frames - 1):]

        else:
            vad_buffer.append(data)
            if potential:
                audio_buffer.append(data)

            if len(vad_buffer) >= accum_frames:
                combined = b''.join(vad_buffer[-accum_frames:])
                is_speech = detect_vad(vad_model, combined)
                if is_speech:
                    vad_no_speech = 0
                    vad_buffer = []
                    if not potential:
                        audio_buffer.append(data)
                else:
                    vad_no_speech += 1
                    vad_buffer = vad_buffer[-(accum_frames - 1):]

            if not potential:
                if silence_start is None:
                    silence_start = time.time()
            else:
                silence_start = None

            sil_dur = 0 if silence_start is None else time.time() - silence_start

            if vad_no_speech >= end_count or sil_dur > silence_thresh:
                is_false = rms < noise_floor * 2 and len(audio_buffer) < 5
                if is_false:
                    false_triggers += 1
                    print(f"  检测到 #{detections + false_triggers}: 误触")
                else:
                    detections += 1
                    dur = len(audio_buffer) * FRAME_DURATION
                    print(f"  检测到 #{detections}: 语音 ({dur:.1f}s)")

                audio_buffer = []
                vad_buffer = []
                speech_confirmed = False
                vad_confirm = 0
                vad_no_speech = 0
                silence_start = None

    passed = detections >= 2 and false_triggers == 0
    return passed, detections, false_triggers


# ─── Phase 5: 写入 config ─────────────────────────────────

def write_config(params: dict):
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.py')

    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except IOError as e:
        print(f"读取 config.py 失败: {e}")
        print("请手动更新以下参数:")
        for k, v in params.items():
            print(f"  ASR_{k.upper()} = {v}")
        return

    replacements = {
        'ASR_VAD_ACCUMULATE_FRAMES': params['vad_accumulate_frames'],
        'ASR_ENERGY_THRESHOLD': params['energy_threshold'],
        'ASR_SPEECH_CONFIRM_COUNT': params['speech_confirm_count'],
        'ASR_VAD_END_COUNT': params['vad_end_count'],
        'ASR_SILENCE_THRESHOLD': params['silence_threshold'],
    }

    original = content
    for key, value in replacements.items():
        if isinstance(value, float):
            value_str = f"{value}"
        else:
            value_str = str(value)

        pattern = rf"^({key}\s*=\s*)(\S+)(.*?)$"
        match = re.search(pattern, content, re.MULTILINE)
        if match:
            old_val = match.group(2)
            comment = match.group(3)
            content = re.sub(
                pattern,
                rf"\g<1>{value_str}{comment}",
                content,
                count=1,
                flags=re.MULTILINE
            )
            print(f"  {key}: {old_val} -> {value_str}")

    if content == original:
        print("config.py 无变化")
        return

    try:
        with open(config_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print("\n配置已写入 config.py")
    except IOError as e:
        print(f"写入 config.py 失败: {e}")
        print("请手动更新以下参数:")
        for k, v in replacements.items():
            print(f"  {k} = {v}")


# ─── 结果输出 ─────────────────────────────────────────────

def print_summary(result: CalibrationResult):
    snr = result.speech_rms_mean / max(result.noise_rms_mean, 1e-6)
    print("\n" + "=" * 55)
    print("VAD 校准结果")
    print("=" * 55)
    print(f"环境噪声:")
    print(f"  噪底均值 RMS: {result.noise_rms_mean:.4f}")
    print(f"  噪底峰值 RMS: {result.noise_rms_peak:.4f}")
    print(f"语音采样:")
    print(f"  语音均值 RMS: {result.speech_rms_mean:.4f}")
    print(f"  语音标准差:   {result.speech_rms_std:.4f}")
    print(f"  语音范围:    {result.speech_rms_min:.4f} - {result.speech_rms_max:.4f}")
    print(f"  VAD 命中率:  {result.speech_vad_hit_rate:.1%}")
    print(f"  信噪比 (SNR): {snr:.1f}x")
    print(f"推荐参数:")
    print(f"  ASR_ENERGY_THRESHOLD      = {result.energy_threshold}")
    print(f"  ASR_VAD_ACCUMULATE_FRAMES = {result.vad_accumulate_frames}")
    print(f"  ASR_SPEECH_CONFIRM_COUNT  = {result.speech_confirm_count}")
    print(f"  ASR_VAD_END_COUNT         = {result.vad_end_count}")
    print(f"  ASR_SILENCE_THRESHOLD     = {result.silence_threshold}")
    status = "通过" if result.verification_passed else "未通过"
    print(f"验证结果: {status} (检测: {result.detection_count}次, 误触: {result.false_trigger_count}次)")
    print("=" * 55)


# ─── StepFun 模式校准 ─────────────────────────────────────

def measure_speech_levels(stream, num_phrases=3):
    """测量语音能量级别（不需要 VAD 模型）"""
    print(f"\n[语音采样] 请说 {num_phrases} 句话（例如：马二进三）")

    all_rms = []
    all_durations = []

    for i in range(num_phrases):
        print(f"  第 {i + 1}/{num_phrases} 句：等待说话...", end="", flush=True)

        # 等待语音开始
        wait_start = time.time()
        speech_started = False
        while time.time() - wait_start < 15.0:
            data = stream.read(CHUNK_SIZE, exception_on_overflow=False)
            rms = compute_rms(data)
            if rms > 0.01:
                speech_started = True
                break

        if not speech_started:
            print(" 超时")
            continue

        # 录音
        phrase_rms = []
        silence_start = None
        rec_start = time.time()

        while True:
            data = stream.read(CHUNK_SIZE, exception_on_overflow=False)
            rms = compute_rms(data)
            phrase_rms.append(rms)

            if rms < 0.01:
                if silence_start is None:
                    silence_start = time.time()
                elif time.time() - silence_start > 1.0:
                    break
            else:
                silence_start = None

            if time.time() - rec_start > 10.0:
                break

        duration = time.time() - rec_start
        all_durations.append(duration)
        all_rms.extend(phrase_rms)
        print(f" 录到 {duration:.1f}s, RMS={np.mean(phrase_rms):.4f}")

    if not all_rms:
        print("  未录到语音，使用默认值")
        return 0.03, 0.01, 0.06, 2.0

    return (float(np.mean(all_rms)), float(np.std(all_rms)),
            float(np.min(all_rms)), float(np.mean(all_durations)))


def write_stepfun_config(params: dict):
    """写入 StepFun 相关配置"""
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.py')

    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except IOError as e:
        print(f"读取 config.py 失败: {e}")
        for k, v in params.items():
            print(f"  {k} = {v}")
        return

    original = content
    for key, value in params.items():
        value_str = f"{value}" if isinstance(value, float) else str(value)
        pattern = rf"^({key}\s*=\s*)(\S+)(.*?)$"
        match = re.search(pattern, content, re.MULTILINE)
        if match:
            old_val = match.group(2)
            comment = match.group(3)
            content = re.sub(pattern, rf"\g<1>{value_str}{comment}",
                             content, count=1, flags=re.MULTILINE)
            print(f"  {key}: {old_val} -> {value_str}")

    if content == original:
        print("config.py 无变化")
        return

    try:
        with open(config_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print("\n配置已写入 config.py")
    except IOError as e:
        print(f"写入失败: {e}")
        for k, v in params.items():
            print(f"  {k} = {v}")


def calibrate_stepfun():
    """StepFun 对讲模式校准：优化音频预处理和服务端 VAD 参数"""
    print("=" * 55)
    print("StepFun 对讲模式校准")
    print("（优化音频预处理 + 服务端 VAD 参数）")
    print("=" * 55)

    p, stream = open_audio_stream()

    try:
        # Phase 1: 噪底
        print("\n[1/3] 测量环境噪声...")
        noise_mean, noise_peak = measure_noise_floor(stream, duration=3.0)

        # Phase 2: 语音
        print("\n[2/3] 语音采样...")
        speech_mean, speech_std, speech_min, avg_dur = measure_speech_levels(stream, num_phrases=3)

        # Phase 3: 计算参数
        print("\n[3/3] 计算最优参数...")
        snr = speech_mean / max(noise_mean, 1e-6)

        # 噪声门阈值：噪底的 2 倍
        noise_gate = round(noise_mean * 2, 4)
        noise_gate = max(noise_gate, 0.002)  # 最低 0.002

        # 增益目标：基于语音均值，拉到 0.08 附近
        gain_target = 0.08
        if speech_mean < 0.03:
            gain_target = 0.10  # 说话轻，拉高一点
        elif speech_mean > 0.15:
            gain_target = 0.06  # 说话响，拉低一点

        # VAD 阈值：基于 SNR
        if snr > 15:
            vad_threshold = 0.3   # 安静环境，更灵敏
        elif snr > 8:
            vad_threshold = 0.4   # 正常环境
        elif snr > 4:
            vad_threshold = 0.5   # 稍吵
        else:
            vad_threshold = 0.6   # 嘈杂环境，减少误触

        # 静音判定时长
        if avg_dur > 3.0:
            silence_ms = 600
        elif avg_dur > 1.5:
            silence_ms = 400
        else:
            silence_ms = 300

        params = {
            'AUDIO_NOISE_GATE_THRESHOLD': noise_gate,
            'AUDIO_GAIN_TARGET_RMS': round(gain_target, 2),
            'STEPFUN_VAD_THRESHOLD': vad_threshold,
            'STEPFUN_VAD_SILENCE_DURATION_MS': silence_ms,
        }

        # 输出结果
        print(f"\n{'=' * 55}")
        print(f"StepFun 校准结果")
        print(f"{'=' * 55}")
        print(f"环境噪声:")
        print(f"  噪底均值 RMS: {noise_mean:.4f}")
        print(f"  噪底峰值 RMS: {noise_peak:.4f}")
        print(f"语音采样:")
        print(f"  语音均值 RMS: {speech_mean:.4f}")
        print(f"  语音标准差:   {speech_std:.4f}")
        print(f"  信噪比 (SNR): {snr:.1f}x")
        print(f"  平均语句时长: {avg_dur:.1f}s")
        print(f"推荐参数:")
        for k, v in params.items():
            print(f"  {k} = {v}")
        print(f"{'=' * 55}")

        # 写入
        print(f"\n写入配置...")
        write_stepfun_config(params)

    except KeyboardInterrupt:
        print("\n\n校准已取消")

    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()


# ─── FunASR 模式校准 ──────────────────────────────────────

def calibrate_funasr():
    """FunASR 本地 VAD 校准"""
    print("=" * 55)
    print("FunASR VAD 校准")
    print("=" * 55)

    try:
        vad_model = load_vad_model()
    except Exception as e:
        print(f"FSMN-VAD 模型加载失败: {e}")
        print("请先运行一次 python app.py 下载模型")
        sys.exit(1)

    p, stream = open_audio_stream()

    try:
        # Phase 1
        noise_mean, noise_peak = measure_noise_floor(stream, duration=3.0)

        # Phase 2
        speech_mean, speech_std, speech_min, speech_max, hit_rate, avg_dur = \
            sample_speech(stream, vad_model, noise_floor=noise_mean, num_phrases=3)

        # Phase 3a
        best_frames = evaluate_accumulate_frames(stream, vad_model, noise_mean)

        # Phase 3b
        params = calculate_parameters(
            noise_mean, noise_peak, speech_mean, speech_std,
            speech_min, speech_max, hit_rate, best_frames, avg_dur
        )

        print(f"\n[Phase 3b/5] 计算完成:")
        for k, v in params.items():
            print(f"  {k} = {v}")

        # Phase 4
        passed, detections, false_triggers = verify_parameters(
            stream, vad_model, params, noise_mean
        )

        # 结果
        result = CalibrationResult(
            noise_rms_mean=noise_mean,
            noise_rms_peak=noise_peak,
            speech_rms_mean=speech_mean,
            speech_rms_std=speech_std,
            speech_rms_min=speech_min,
            speech_rms_max=speech_max,
            speech_vad_hit_rate=hit_rate,
            avg_speech_duration=avg_dur,
            **params,
            verification_passed=passed,
            detection_count=detections,
            false_trigger_count=false_triggers,
        )

        print_summary(result)

        # Phase 5
        if passed or detections >= 1:
            print(f"\n[Phase 5/5] 写入配置...")
            write_config(params)
        else:
            print("\n验证未通过，配置未写入。请重新运行校准。")
            print("你也可以手动参考上面的推荐参数修改 config.py")

    except KeyboardInterrupt:
        print("\n\n校准已取消")

    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()


# ─── 主流程 ───────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="VAD 语音检测校准工具")
    parser.add_argument('--stepfun', action='store_true',
                        help="StepFun 对讲模式校准（音频预处理 + 服务端 VAD）")
    args = parser.parse_args()

    if args.stepfun:
        calibrate_stepfun()
    else:
        calibrate_funasr()


if __name__ == '__main__':
    main()
