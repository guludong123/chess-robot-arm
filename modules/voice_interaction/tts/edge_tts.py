"""
Microsoft Edge TTS 提供者 (免费)
"""
import asyncio
import tempfile
import os
from typing import Optional
import edge_tts
from edge_tts import exceptions as edge_tts_exceptions
from .base import TTSProviderBase
import config


class EdgeTTSProvider(TTSProviderBase):
    """Microsoft Edge TTS 提供者（免费）"""

    def __init__(
        self,
        voice: str = None,
        rate: str = None,
        volume: str = None
    ):
        self.voice = voice or config.TTS_VOICE
        self.rate = rate or getattr(config, 'TTS_RATE', '+0%')
        self.volume = volume or getattr(config, 'TTS_VOLUME', '+0%')
        self._is_speaking = False
        self._stop_event = asyncio.Event()
        self._pygame_initialized = False

    @property
    def name(self) -> str:
        return f"Edge TTS ({self.voice})"

    async def speak(
        self,
        text: str,
        interrupt_event: Optional[asyncio.Event] = None,
        voice: Optional[str] = None,
        rate: Optional[str] = None,
        volume: Optional[str] = None
    ) -> bool:
        """
        播报文本

        Args:
            text: 要播报的文本
            interrupt_event: 打断事件（调用时会被清除）
            voice: 语音 ID（覆盖默认值）
            rate: 语速（覆盖默认值）
            volume: 音量（覆盖默认值）

        Returns:
            是否成功播报
        """
        # 【关键】清除打断信号，确保播报能执行
        if interrupt_event and interrupt_event.is_set():
            print("[EdgeTTS] 检测到打断信号，正在清除...")
            interrupt_event.clear()

        if self._is_speaking:
            print("[EdgeTTS] 正在播报中，先停止")
            await self.stop()

        self._is_speaking = True
        self._stop_event.clear()

        # 使用传入参数或默认值
        use_voice = voice or self.voice
        use_rate = rate or self.rate
        use_volume = volume or self.volume

        # 确保 rate 格式正确（edge_tts 需要 +X% 或 -X% 格式）
        if use_rate:
            if not use_rate.endswith('%'):
                use_rate = use_rate + '%'
            if not use_rate.startswith('+') and not use_rate.startswith('-'):
                use_rate = '+' + use_rate

        print(f"[EdgeTTS] 开始生成语音: voice={use_voice}, rate={use_rate}, volume={use_volume}, text=\"{text}\"")
        print(f"[EdgeTTS] 打断信号状态: interrupt_event={interrupt_event.is_set() if interrupt_event else 'None'}, stop_event={self._stop_event.is_set()}")

        temp_file = None
        max_retries = 3

        try:
            audio_data = []

            # 尝试最多 3 次
            for retry in range(max_retries):
                try:
                    if retry > 0:
                        print(f"[EdgeTTS] 第 {retry + 1} 次重试...")

                    # 创建 TTS 通信对象
                    communicate = edge_tts.Communicate(
                        text,
                        use_voice,
                        rate=use_rate,
                        volume=use_volume
                    )

                    # 收集音频数据
                    audio_data = []
                    chunk_count = 0
                    async for chunk in communicate.stream():
                        chunk_count += 1
                        # 检查打断信号
                        if interrupt_event and interrupt_event.is_set():
                            print(f"[EdgeTTS] 被打断 (外部信号) at chunk {chunk_count}")
                            break
                        if self._stop_event.is_set():
                            print(f"[EdgeTTS] 被打断 (内部信号) at chunk {chunk_count}")
                            break
                        if chunk["type"] == "audio":
                            audio_data.append(chunk["data"])

                    print(f"[EdgeTTS] 收集完成: chunks={chunk_count}, audio_chunks={len(audio_data)}")

                    # 成功获取音频，跳出重试循环
                    if audio_data and not self._stop_event.is_set():
                        break

                except edge_tts_exceptions.NoAudioReceived as e:
                    print(f"[EdgeTTS] NoAudioReceived 错误 (重试 {retry + 1}/{max_retries}): {e}")
                    if retry < max_retries - 1:
                        await asyncio.sleep(0.5)  # 等待后重试
                        continue
                    else:
                        print("[EdgeTTS] 所有重试都失败")
                        self._is_speaking = False
                        return False

                except Exception as inner_e:
                    print(f"[EdgeTTS] 内部错误 (重试 {retry + 1}/{max_retries}): {inner_e}")
                    if retry < max_retries - 1:
                        await asyncio.sleep(1.0)
                        continue
                    else:
                        print("[EdgeTTS] 所有重试都失败")
                        self._is_speaking = False
                        return False

            if not audio_data or self._stop_event.is_set():
                print(f"[EdgeTTS] 无音频数据或被打断: audio_len={len(audio_data)}, stopped={self._stop_event.is_set()}")
                return False

            print(f"[EdgeTTS] 音频数据大小: {len(b''.join(audio_data))} bytes，准备播放...")

            # 播放音频
            await self._play_audio(audio_data, interrupt_event)
            print("[EdgeTTS] 播放完成")
            return True

        except Exception as e:
            print(f"[EdgeTTS] 播报失败: {e}")
            import traceback
            traceback.print_exc()
            return False

        finally:
            self._is_speaking = False
            # 清理临时文件
            if temp_file and os.path.exists(temp_file):
                try:
                    os.unlink(temp_file)
                except:
                    pass

    async def stop(self):
        """停止播报"""
        self._stop_event.set()
        self._is_speaking = False

        # 停止 pygame 播放
        if self._pygame_initialized:
            try:
                import pygame
                if pygame.mixer.music.get_busy():
                    pygame.mixer.music.stop()
            except:
                pass

    def is_speaking(self) -> bool:
        """是否正在播报"""
        return self._is_speaking

    async def _play_audio(self, audio_data: list, interrupt_event: Optional[asyncio.Event] = None):
        """播放音频数据"""
        print("[EdgeTTS] _play_audio 开始...")

        try:
            import pygame
        except ImportError:
            print("[EdgeTTS] pygame 未安装，无法播放音频")
            return

        # 初始化 pygame
        if not self._pygame_initialized:
            print("[EdgeTTS] 初始化 pygame.mixer...")
            try:
                pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
                self._pygame_initialized = True
                print("[EdgeTTS] pygame.mixer 初始化成功")
            except Exception as e:
                print(f"[EdgeTTS] pygame 初始化失败: {e}")
                import traceback
                traceback.print_exc()
                return

        # 合并音频数据
        full_audio = b"".join(audio_data)
        print(f"[EdgeTTS] 音频总大小: {len(full_audio)} bytes")

        # 保存临时文件
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            f.write(full_audio)
            temp_file = f.name
        print(f"[EdgeTTS] 临时文件: {temp_file}")

        try:
            print("[EdgeTTS] 加载音频文件...")
            pygame.mixer.music.load(temp_file)
            print("[EdgeTTS] 开始播放...")
            pygame.mixer.music.play()

            # 等待播放完成
            play_count = 0
            while pygame.mixer.music.get_busy():
                play_count += 1
                if play_count % 10 == 0:  # 每1秒打印一次
                    print(f"[EdgeTTS] 播放中... ({play_count * 0.1:.1f}s)")

                if self._stop_event.is_set():
                    print("[EdgeTTS] 播放被停止 (内部信号)")
                    pygame.mixer.music.stop()
                    break
                if interrupt_event and interrupt_event.is_set():
                    print("[EdgeTTS] 播放被停止 (外部信号)")
                    pygame.mixer.music.stop()
                    break
                await asyncio.sleep(0.1)

            print(f"[EdgeTTS] 播放结束，总时长约 {play_count * 0.1:.1f}s")

        except Exception as e:
            print(f"[EdgeTTS] 播放异常: {e}")
            import traceback
            traceback.print_exc()

        finally:
            # 清理临时文件
            try:
                os.unlink(temp_file)
                print(f"[EdgeTTS] 清理临时文件: {temp_file}")
            except Exception as e:
                print(f"[EdgeTTS] 清理临时文件失败: {e}")