"""
FunASR 解说模式测试脚本

测试流程：
1. FunASR 离线模型语音识别（能量前端 + VAD 累积确认）
2. 检测"下好了"、"该你了"等关键词（模糊匹配）
3. 触发预设回复（旁观者口吻）

使用方法：
f:/miniconda/envs/chessrobot/python.exe test_funasr_commentary.py
"""
import asyncio
import time
import sys
import os
import logging

# 抑制 verbose 日志
logging.getLogger('modelscope').setLevel(logging.WARNING)
logging.getLogger('funasr').setLevel(logging.WARNING)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from modules.voice_interaction.asr import create_asr_provider
from modules.voice_interaction.intent import detect_intent_with_confidence, HIGH_CONFIDENCE_KEYWORDS, CORE_CHARS
from modules.voice_interaction.llm.prompts import get_commentary_move_response, get_commentary_hurry_response


async def main():
    print("=" * 60)
    print("FunASR 解说模式测试（离线模型 + 极简预设回复）")
    print("=" * 60)
    print(f"ASR 模型: {config.FUNASR_MODEL}")
    print(f"静音阈值: {config.ASR_SILENCE_THRESHOLD}s (快速响应)")
    print(f"VAD 结束次数: {config.ASR_VAD_END_COUNT}")
    print(f"关键词: {HIGH_CONFIDENCE_KEYWORDS}")
    print(f"核心字: {CORE_CHARS}")
    print("=" * 60)
    print("说'下好了'或'该你下了'触发走棋检测")
    print("按 Ctrl+C 退出")
    print("=" * 60)

    # 创建 ASR
    asr = create_asr_provider('funasr')
    if not asr:
        print("ASR 创建失败")
        return

    try:
        while True:
            print("\n[监听] 请说话...")
            start = time.time()

            transcript = await asr.listen(
                timeout=10.0,
                silence_threshold=1.5
            )

            latency = time.time() - start

            if transcript:
                print(f"\n[{latency:.2f}s] 识别结果: \"{transcript}\"")

                intent, confidence = detect_intent_with_confidence(transcript)
                print(f"  → 意图: {intent} (置信度: {confidence})")

                if intent == "move" and confidence == "high":
                    response = get_commentary_move_response()
                    print(f"\n>>> [预设回复] \"{response}\"")
                elif intent == "cancel":
                    print(f"\n>>> [取消] 已取消当前操作")
                else:
                    # 检查是否是催促类
                    if "快" in transcript or "快点" in transcript:
                        response = get_commentary_hurry_response()
                        print(f"\n>>> [催促回复] \"{response}\"")
                    else:
                        print(f"\n>>> [对话] 请说'下好了'告知完成走棋")
            else:
                print(f"[{latency:.2f}s] 未检测到语音")

    except KeyboardInterrupt:
        print("\n退出测试")
    finally:
        asr.close()


if __name__ == '__main__':
    asyncio.run(main())