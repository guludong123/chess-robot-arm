"""
语音交互管理器 - 协调 LLM/TTS/ASR 组件

功能：
- 解说生成（走棋后触发）
- 语音对话（用户主动发起）
- 角色系统（切换不同 AI 人设）
- TTS 播报（支持角色化语音）
- 会话模式（持续监听 + 关键词触发）
"""
import asyncio
import random
import threading
import uuid
from typing import Dict, Optional, List, Callable

from .llm.base import LLMProviderBase
from .tts.base import TTSProviderBase
from .asr.base import ASRProviderBase
from .state import VoiceState, VoiceStatus, VoiceMode
from .character import CharacterManager
from .character.presets import COMMENTARY_CHARACTERS, get_commentary_character
from .dialogue import DialogueSession, DialogueHistory
from .intent import detect_intent, detect_intent_with_confidence, ErrorType, classify_error
from .llm.prompts import IMMEDIATE_REPLIES, get_commentary_move_response, get_commentary_hurry_response
import config


# 走棋检测回调类型
MoveDetectCallback = Callable[[], tuple]  # 返回 (move, error)


class VoiceInteractionManager:
    """
    语音交互管理器 - 协调 LLM/TTS/ASR 组件
    """

    def __init__(
        self,
        llm_provider: LLMProviderBase,
        tts_provider: TTSProviderBase,
        asr_provider: Optional[ASRProviderBase] = None,
        socketio=None
    ):
        self.llm = llm_provider
        self.tts = tts_provider
        self.asr = asr_provider
        self.socketio = socketio

        self.state = VoiceState()

        # 角色管理器
        self.character_manager = CharacterManager(
            default_character_id=config.DEFAULT_CHARACTER
        )

        # 解说角色（解说模式专用）
        self._commentary_character_id = "professional"  # 默认专业解说
        self._commentary_character = get_commentary_character("professional")

        # 对话 Session
        self.dialogue_session: Optional[DialogueSession] = None

        # 异步事件循环
        self._async_loop = None
        self._worker_thread = None

    def start(self):
        """启动语音交互服务"""
        if self._worker_thread and self._worker_thread.is_alive():
            return

        self._worker_thread = threading.Thread(
            target=self._run_async_loop,
            daemon=True
        )
        self._worker_thread.start()
        print(f"[VoiceManager] 服务已启动 - LLM: {self.llm.name}, TTS: {self.tts.name}")

    def stop(self):
        """停止服务"""
        if self._async_loop:
            self._async_loop.call_soon_threadsafe(
                self._async_loop.stop
            )
        print("[VoiceManager] 服务已停止")

    def on_move_complete(
        self,
        board_state: Dict,
        human_move: Dict,
        ai_move: Dict,
        move_history: list
    ):
        """
        走棋完成事件回调（从 app.py 调用）

        在 execute_and_notify() 成功后调用此方法
        """
        if not self._async_loop:
            print("[VoiceManager] 警告: 事件循环未启动")
            return

        # 如果正在对话模式，跳过自动解说
        if self.state.dialogue_active:
            print(f"[VoiceManager] 对话模式激活中，跳过自动解说 (dialogue_active={self.state.dialogue_active}, mode={self.state.mode.value})")
            return

        # 将任务放入队列，异步执行
        asyncio.run_coroutine_threadsafe(
            self._handle_move_complete(
                board_state, human_move, ai_move, move_history
            ),
            self._async_loop
        )

    async def _handle_move_complete(
        self,
        board_state: Dict,
        human_move: Dict,
        ai_move: Dict,
        move_history: list
    ):
        """异步处理走棋完成（使用解说角色风格）"""
        try:
            self.state.set_status(VoiceStatus.GENERATING)

            # 1. 获取解说角色的系统提示词
            character_prompt = None
            if self._commentary_character:
                character_prompt = self._commentary_character.get_system_prompt()

            # 2. LLM 生成解说（使用解说角色风格）
            commentary = await self.llm.generate_commentary(
                board_state, human_move, ai_move, move_history,
                character_system_prompt=character_prompt
            )

            # 格式化解说（角色风格）
            if self._commentary_character:
                commentary = self._commentary_character.format_commentary(commentary)

            # 记录解说历史
            self.state.add_commentary(commentary, {
                'human_move': human_move,
                'ai_move': ai_move
            })

            # 3. 发送解说文本到前端
            if self.socketio:
                self.socketio.emit('commentary_generated', {
                    'text': commentary,
                    'human_move': human_move,
                    'ai_move': ai_move,
                    'character': self._commentary_character.get_name() if self._commentary_character else '解说员'
                })

            # 4. TTS 播报（使用角色语音）
            self.state.set_status(VoiceStatus.SPEAKING)
            self.state.interrupt_event.clear()  # 清除打断信号，确保播报能执行

            voice_config = {}
            if self._commentary_character:
                voice_config = self._commentary_character.get_voice_config()

            print(f"[VoiceManager] 开始 TTS 播报，语音配置: voice={voice_config.get('voice')}, text长度={len(commentary)}")

            try:
                tts_success = await self.tts.speak(
                    commentary,
                    voice=voice_config.get('voice'),
                    rate=voice_config.get('rate'),
                    interrupt_event=self.state.interrupt_event
                )
                print(f"[VoiceManager] TTS 播报结果: {tts_success}")
                if not tts_success:
                    print("[VoiceManager] TTS 播报失败，1秒后重试...")
                    await asyncio.sleep(1.0)
                    tts_success = await self.tts.speak(
                        commentary,
                        voice=voice_config.get('voice'),
                        rate=voice_config.get('rate'),
                        interrupt_event=self.state.interrupt_event
                    )
                    print(f"[VoiceManager] TTS 重试结果: {tts_success}")
            except Exception as tts_error:
                print(f"[VoiceManager] TTS 播报异常: {tts_error}")
                import traceback
                traceback.print_exc()

            # 恢复状态：如果解说监听激活，恢复为监听状态
            if self.state.is_commentary_listening_active():
                self.state.set_status(VoiceStatus.COMMENTARY_LISTENING)
            else:
                self.state.set_status(VoiceStatus.IDLE)

        except Exception as e:
            print(f"[VoiceManager] 处理走棋事件失败: {e}")
            self.state.set_status(VoiceStatus.ERROR)
            if self.socketio:
                self.socketio.emit('voice_error', {
                    'error': str(e)
                })

            # 异常后恢复监听状态
            if self.state.is_commentary_listening_active():
                await asyncio.sleep(1.0)  # 等待一会儿再恢复
                self.state.set_status(VoiceStatus.COMMENTARY_LISTENING)

    async def handle_commentary_error(self, error: str):
        """
        处理解说模式下的走棋错误（语音提示）
        使用预设模板，不调用LLM

        Args:
            error: 错误信息
        """
        try:
            # 分类错误类型
            from .intent import classify_error
            from .llm.prompts import get_error_template

            error_type = classify_error(error) if error else "invalid_move"

            # 直接使用预设模板，不调用LLM
            error_response = get_error_template(error_type)

            # 发送到前端
            if self.socketio:
                self.socketio.emit('commentary_error', {
                    'text': error_response,
                    'error_type': error_type
                })

            # TTS 播报（使用解说角色语音）
            self.state.set_status(VoiceStatus.SPEAKING)
            self.state.interrupt_event.clear()  # 清除打断信号

            voice_config = {}
            if self._commentary_character:
                voice_config = self._commentary_character.get_voice_config()

            await self.tts.speak(
                error_response,
                voice=voice_config.get('voice'),
                rate=voice_config.get('rate'),
                interrupt_event=self.state.interrupt_event
            )

            self.state.set_status(VoiceStatus.IDLE)

        except Exception as e:
            print(f"[VoiceManager] 处理解说错误失败: {e}")
            self.state.set_status(VoiceStatus.IDLE)

    async def on_game_over(self, winner: str, reason: str):
        """
        处理游戏结束（语音播报）

        Args:
            winner: 获胜方 ('red' 或 'black')
            reason: 结束原因（如 '将死'）
        """
        try:
            self.state.set_status(VoiceStatus.GENERATING)

            winner_name = "红方" if winner == 'red' else "黑方"

            # 使用解说角色风格生成游戏结束语音
            character_name = self._commentary_character.get_name() if self._commentary_character else "解说员"

            # 直接生成游戏结束播报
            game_over_msg = f"{winner_name}获胜！{reason}！"

            # 发送到前端
            if self.socketio:
                self.socketio.emit('game_over_announcement', {
                    'text': game_over_msg,
                    'winner': winner_name,
                    'reason': reason,
                    'character': character_name
                })

            # TTS 播报
            self.state.set_status(VoiceStatus.SPEAKING)
            self.state.interrupt_event.clear()  # 清除打断信号

            voice_config = {}
            if self._commentary_character:
                voice_config = self._commentary_character.get_voice_config()

            await self.tts.speak(
                game_over_msg,
                voice=voice_config.get('voice'),
                rate=voice_config.get('rate'),
                interrupt_event=self.state.interrupt_event
            )

            self.state.set_status(VoiceStatus.IDLE)

        except Exception as e:
            print(f"[VoiceManager] 处理游戏结束失败: {e}")
            self.state.set_status(VoiceStatus.IDLE)

    async def process_voice_input(self) -> Optional[Dict]:
        """处理语音输入命令"""
        if not self.asr:
            print("[VoiceManager] ASR 未配置")
            return None

        try:
            self.state.set_status(VoiceStatus.LISTENING)

            # 停止当前播报
            await self.tts.stop()

            # 监听语音
            transcript = await self.asr.listen(timeout=5.0)

            if not transcript:
                self.state.set_status(VoiceStatus.IDLE)
                return None

            # LLM 解析命令
            self.state.set_status(VoiceStatus.PROCESSING)
            result = await self.llm.process_voice_command(
                transcript,
                self.state.game_context
            )

            # 播报响应
            if result.get('response'):
                self.state.set_status(VoiceStatus.SPEAKING)
                self.state.interrupt_event.clear()  # 清除打断信号
                await self.tts.speak(result['response'])

            self.state.set_status(VoiceStatus.IDLE)
            return result

        except Exception as e:
            print(f"[VoiceManager] 处理语音输入失败: {e}")
            self.state.set_status(VoiceStatus.ERROR)
            if self.socketio:
                self.socketio.emit('voice_error', {
                    'error': str(e)
                })
            return None

    def interrupt_speech(self):
        """打断当前播报"""
        self.state.interrupt_event.set()
        if self.tts.is_speaking() and self._async_loop:
            asyncio.run_coroutine_threadsafe(
                self.tts.stop(),
                self._async_loop
            )

    def update_game_context(
        self,
        board_state: Dict,
        current_player: str,
        move_history: list
    ):
        """更新游戏上下文"""
        self.state.update_game_context(
            board_state, current_player, move_history
        )

    def get_status(self) -> Dict:
        """获取当前状态"""
        return {
            'status': self.state.status.value,
            'is_speaking': self.tts.is_speaking(),
            'can_interrupt': self.state.can_interrupt(),
            'dialogue_active': self.state.dialogue_active,
            'current_character': self.character_manager.get_character_config()
        }

    # ========== 对话模式方法 ==========

    def start_dialogue(self) -> Dict:
        """
        启动对话模式

        Returns:
            结果字典 {'success': bool, 'greeting': str, 'character': dict}
        """
        if not self.asr:
            print("[VoiceManager] ASR 未配置，无法启动对话")
            return {'success': False, 'error': 'ASR 未配置'}

        if self.state.dialogue_active:
            return {'success': False, 'error': '对话模式已激活'}

        # 打断正在进行的解说/播报
        if self.tts.is_speaking():
            print("[VoiceManager] 打断当前播报，启动对话模式")
            self.interrupt_speech()

        if not self._async_loop:
            return {'success': False, 'error': '服务未启动'}

        # 异步启动对话
        future = asyncio.run_coroutine_threadsafe(
            self._start_dialogue_async(),
            self._async_loop
        )

        try:
            return future.result(timeout=10.0)
        except Exception as e:
            print(f"[VoiceManager] 启动对话失败: {e}")
            return {'success': False, 'error': str(e)}

    async def _start_dialogue_async(self) -> Dict:
        """异步启动对话模式"""
        try:
            # 创建对话 Session
            self.dialogue_session = DialogueSession(
                session_id=uuid.uuid4().hex[:8]
            )

            # 设置当前角色
            character = self.character_manager.get_current_character()
            if character:
                self.dialogue_session.set_character(character.get_id())
                self.state.set_character(character.get_id(), character.get_name())

            # 激活对话模式
            self.state.set_dialogue_mode(True)

            # 获取问候语
            greeting = ""
            if character:
                greeting = character.get_greeting()
                voice_config = character.get_voice_config()

                # 播报问候语
                self.state.set_status(VoiceStatus.SPEAKING)
                self.state.interrupt_event.clear()  # 清除打断信号
                await self.tts.speak(
                    greeting,
                    voice=voice_config.get('voice'),
                    rate=voice_config.get('rate'),
                    interrupt_event=self.state.interrupt_event
                )

                # 记录问候语
                self.dialogue_session.history.add_assistant_message(
                    greeting,
                    metadata={'character': character.get_name()}
                )
                self.state.add_dialogue_message('assistant', greeting)

            self.state.set_status(VoiceStatus.DIALOGUE_ACTIVE)

            # 通知前端
            if self.socketio:
                self.socketio.emit('dialogue_start', {
                    'success': True,
                    'greeting': greeting,
                    'character': self.character_manager.get_character_config()
                })

            print(f"[VoiceManager] 对话模式已启动 - 角色: {self.state.current_character_name}")

            return {
                'success': True,
                'greeting': greeting,
                'character': self.character_manager.get_character_config()
            }

        except Exception as e:
            print(f"[VoiceManager] 启动对话失败: {e}")
            self.state.set_status(VoiceStatus.ERROR)
            return {'success': False, 'error': str(e)}

    def stop_dialogue(self) -> Dict:
        """停止对话模式"""
        if not self.state.dialogue_active:
            return {'success': True}

        self.state.set_dialogue_mode(False)
        self.state.clear_dialogue_history()

        if self.dialogue_session:
            self.dialogue_session.clear()
            self.dialogue_session = None

        # 通知前端
        if self.socketio:
            self.socketio.emit('dialogue_stop', {'success': True})

        print("[VoiceManager] 对话模式已停止")
        return {'success': True}

    def process_dialogue_input(self, text_input: str = None) -> Dict:
        """
        处理对话输入

        Args:
            text_input: 文本输入（可选，如果提供则跳过 ASR）

        Returns:
            结果字典 {'success': bool, 'transcript': str, 'response': str}
        """
        if not self._async_loop:
            return {'success': False, 'error': '服务未启动'}

        future = asyncio.run_coroutine_threadsafe(
            self._process_dialogue_input_async(text_input),
            self._async_loop
        )

        try:
            return future.result(timeout=30.0)
        except Exception as e:
            print(f"[VoiceManager] 处理对话输入失败: {e}")
            return {'success': False, 'error': str(e)}

    async def _process_dialogue_input_async(self, text_input: str = None) -> Dict:
        """异步处理对话输入"""
        if not self.state.dialogue_active:
            return {'success': False, 'error': '对话模式未激活'}

        try:
            # 停止当前播报
            await self.tts.stop()

            # 1. 获取用户输入
            transcript = text_input
            if not transcript and self.asr:
                self.state.set_status(VoiceStatus.LISTENING)

                # 通知前端开始监听
                if self.socketio:
                    self.socketio.emit('listening_status', {'status': 'listening'})

                transcript = await self.asr.listen(
                    timeout=config.ASR_TIMEOUT,
                    silence_threshold=config.ASR_SILENCE_THRESHOLD
                )

                # 通知前端结束监听
                if self.socketio:
                    self.socketio.emit('listening_status', {'status': 'idle'})

            if not transcript:
                self.state.set_status(VoiceStatus.DIALOGUE_ACTIVE)
                return {'success': False, 'error': '无语音输入'}

            # 2. 记录用户输入
            self.dialogue_session.history.add_user_message(transcript)
            self.state.add_dialogue_message('user', transcript)

            # 3. LLM 生成回复
            self.state.set_status(VoiceStatus.PROCESSING)

            character = self.character_manager.get_current_character()
            character_prompt = character.get_system_prompt() if character else None

            response = await self.llm.generate_dialogue_response(
                transcript=transcript,
                history=self.dialogue_session.history.get_recent_messages(10),
                game_context=self.dialogue_session.game_context,
                character_system_prompt=character_prompt
            )

            # 格式化回复（角色风格）
            if character:
                response = character.format_dialogue_response(response)

            # 4. 记录回复
            self.dialogue_session.history.add_assistant_message(
                response,
                metadata={'character': character.get_name() if character else 'AI'}
            )
            self.state.add_dialogue_message('assistant', response, {
                'character': character.get_name() if character else 'AI'
            })

            # 5. TTS 播报（角色语音）
            self.state.set_status(VoiceStatus.SPEAKING)
            self.state.interrupt_event.clear()  # 清除打断信号

            voice_config = character.get_voice_config() if character else {}
            await self.tts.speak(
                response,
                voice=voice_config.get('voice'),
                rate=voice_config.get('rate'),
                interrupt_event=self.state.interrupt_event
            )

            # 6. 发送对话事件到前端
            if self.socketio:
                self.socketio.emit('dialogue_message', {
                    'user': transcript,
                    'assistant': response,
                    'character': character.get_name() if character else 'AI'
                })

            self.state.set_status(VoiceStatus.DIALOGUE_ACTIVE)

            return {
                'success': True,
                'transcript': transcript,
                'response': response
            }

        except Exception as e:
            print(f"[VoiceManager] 处理对话失败: {e}")
            self.state.set_status(VoiceStatus.ERROR)

            if self.socketio:
                self.socketio.emit('voice_error', {'error': str(e)})

            return {'success': False, 'error': str(e)}

    # ========== 角色管理方法 ==========

    def set_character(self, character_id: str) -> Dict:
        """
        设置当前角色

        Args:
            character_id: 角色 ID

        Returns:
            结果字典 {'success': bool, 'character': dict}
        """
        success = self.character_manager.set_character(character_id)

        if success:
            character = self.character_manager.get_current_character()
            self.state.set_character(character.get_id(), character.get_name())

            # 更新对话 Session 的角色
            if self.dialogue_session:
                self.dialogue_session.set_character(character_id)

            # 通知前端
            if self.socketio:
                self.socketio.emit('character_changed', {
                    'character': self.character_manager.get_character_config()
                })

            print(f"[VoiceManager] 角色已切换: {character.get_name()}")

            return {
                'success': True,
                'character': self.character_manager.get_character_config()
            }

        return {'success': False, 'error': f'角色 {character_id} 不存在'}

    def get_characters(self) -> List[Dict]:
        """获取可用角色列表"""
        return self.character_manager.list_available_characters()

    def get_current_character(self) -> Dict:
        """获取当前角色"""
        return self.character_manager.get_character_config()

    # ========== 解说角色管理方法 ==========

    def set_commentary_character(self, character_id: str) -> Dict:
        """
        设置解说角色

        Args:
            character_id: 角色 ID ("professional" 或 "humorous")

        Returns:
            结果字典 {'success': bool, 'character': dict}
        """
        character = get_commentary_character(character_id)

        if character:
            self._commentary_character_id = character_id
            self._commentary_character = character

            print(f"[VoiceManager] 解说角色已切换: {character.get_name()}")

            # 通知前端
            if self.socketio:
                self.socketio.emit('commentary_character_changed', {
                    'character_id': character_id,
                    'character_name': character.get_name()
                })

            return {
                'success': True,
                'character': {
                    'id': character_id,
                    'name': character.get_name(),
                    'description': character.get_description()
                }
            }

        return {'success': False, 'error': f'解说角色 {character_id} 不存在'}

    def get_commentary_character(self) -> Dict:
        """获取当前解说角色"""
        if self._commentary_character:
            return {
                'id': self._commentary_character_id,
                'name': self._commentary_character.get_name(),
                'description': self._commentary_character.get_description()
            }
        return {'id': 'professional', 'name': '专业解说', 'description': '专业术语，深入分析'}

    def get_commentary_characters(self) -> List[Dict]:
        """获取可用解说角色列表"""
        from .character.presets import list_commentary_characters
        return list_commentary_characters()

    def get_dialogue_history(self) -> Dict:
        """获取对话历史"""
        if self.dialogue_session:
            return self.dialogue_session.history.to_dict()
        return {'messages': [], 'message_count': 0}

    def clear_dialogue_history(self) -> Dict:
        """清空对话历史"""
        self.state.clear_dialogue_history()
        if self.dialogue_session:
            self.dialogue_session.history.clear()
        return {'success': True}

    def _run_async_loop(self):
        """运行异步事件循环（在独立线程中）"""
        self._async_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._async_loop)
        self._async_loop.run_forever()

    # ========== 解说模式持续监听方法 ==========

    def start_commentary_listening(self) -> Dict:
        """
        启动解说模式的持续监听

        Returns:
            结果字典 {'success': bool, 'error': str}
        """
        if not self.asr:
            return {'success': False, 'error': 'ASR 未配置'}

        if self.state.is_commentary_listening_active():
            return {'success': False, 'error': '监听已激活'}

        if not self._async_loop:
            return {'success': False, 'error': '服务未启动'}

        # 打断当前播报
        if self.tts.is_speaking():
            self.interrupt_speech()

        # 启动解说监听
        self.state.start_commentary_listening()

        # 启动后台监听循环
        asyncio.run_coroutine_threadsafe(
            self._commentary_listening_loop(),
            self._async_loop
        )

        # 通知前端
        if self.socketio:
            self.socketio.emit('commentary_listening_started', {
                'success': True
            })

        print("[VoiceManager] 解说模式持续监听已启动")
        return {'success': True}

    def stop_commentary_listening(self) -> Dict:
        """
        停止解说模式的持续监听

        Returns:
            结果字典
        """
        if not self.state.is_commentary_listening_active():
            return {'success': True}

        self.state.stop_commentary_listening()

        # 通知前端
        if self.socketio:
            self.socketio.emit('commentary_listening_stopped', {
                'success': True
            })

        print("[VoiceManager] 解说模式持续监听已停止")
        return {'success': True}

    async def _commentary_listening_loop(self):
        """解说模式的持续监听循环"""
        print("[VoiceManager] 解说监听循环启动")

        while self.state.is_commentary_listening_active():
            try:
                # 如果正在生成解说或播报，等待完成后再继续监听
                if self.state.status in [VoiceStatus.GENERATING, VoiceStatus.SPEAKING, VoiceStatus.WAITING_MOVE_RESULT]:
                    await asyncio.sleep(0.5)
                    continue

                # 检查打断信号
                if self.state.interrupt_event.is_set():
                    self.state.interrupt_event.clear()

                # 监听语音
                self.state.set_status(VoiceStatus.COMMENTARY_LISTENING)

                if self.socketio:
                    self.socketio.emit('listening_status', {'status': 'listening'})

                transcript = await self.asr.listen(
                    timeout=config.ASR_TIMEOUT,
                    silence_threshold=config.ASR_SILENCE_THRESHOLD
                )

                if self.socketio:
                    self.socketio.emit('listening_status', {'status': 'processing'})

                if not transcript:
                    # 无语音输入，继续监听
                    continue

                print(f"[VoiceManager] 解说监听识别到: {transcript}")

                # 意图判断（带置信度）
                intent, confidence = detect_intent_with_confidence(transcript)
                print(f"[VoiceManager] 意图: {intent}, 置信度: {confidence}")

                if intent == "move" and confidence == "high":
                    # 高置信度走棋意图
                    await self._handle_commentary_move_intent(transcript)

                elif intent == "cancel":
                    # 取消意图
                    await self.tts.stop()
                    self.state.set_status(VoiceStatus.COMMENTARY_LISTENING)

                # 其他情况（chat 或低置信度 move）：忽略，继续监听

            except Exception as e:
                print(f"[VoiceManager] 解说监听循环错误: {e}")
                await asyncio.sleep(0.5)
                # 恢复监听状态
                if self.state.is_commentary_listening_active():
                    self.state.set_status(VoiceStatus.COMMENTARY_LISTENING)

        print("[VoiceManager] 解说监听循环结束")

    async def _handle_commentary_move_intent(self, transcript: str):
        """
        处理解说模式下的走棋意图

        Args:
            transcript: 识别到的语音文本
        """
        # 游戏状态验证
        can_move, reason = await self._validate_game_state_for_move()

        if not can_move:
            # 不能走棋，播报原因
            self.state.set_status(VoiceStatus.SPEAKING)
            if self.state.interrupt_event.is_set():
                self.state.interrupt_event.clear()

            voice_config = {}
            if self._commentary_character:
                voice_config = self._commentary_character.get_voice_config()

            # 确保 rate 格式正确（edge_tts 需要 +X% 或 -X% 格式）
            rate = voice_config.get('rate', '+0%')
            if rate and not rate.startswith('+') and not rate.startswith('-'):
                rate = '+' + rate if not rate.startswith('%') else '+0%'

            await self.tts.speak(
                reason,
                voice=voice_config.get('voice'),
                rate=rate,
                interrupt_event=self.state.interrupt_event
            )

            # 恢复监听
            self.state.set_status(VoiceStatus.COMMENTARY_LISTENING)
            return

        # 直接触发走棋检测（不播报预设回复）
        self.state.set_waiting_move_result()

        # 发送事件到前端，触发走棋检测
        if self.socketio:
            self.socketio.emit('commentary_move_intent', {
                'transcript': transcript
            })

        print(f"[VoiceManager] 已触发走棋检测: {transcript}")

    async def _validate_game_state_for_move(self) -> tuple:
        """
        验证是否可以触发走棋检测

        Returns:
            (can_move, reason): 是否可以走棋，以及不能走棋的原因
        """
        try:
            import requests

            # 调用 API 获取游戏状态
            result = await asyncio.to_thread(
                requests.get,
                "http://localhost:5000/api/game/state_for_voice",
                timeout=2.0
            )

            if result.status_code != 200:
                return False, "无法获取游戏状态"

            data = result.json()
            can_move = data.get('can_move', False)
            reason = data.get('reason', '')

            return can_move, reason

        except Exception as e:
            print(f"[VoiceManager] 验证游戏状态失败: {e}")
            return False, "无法获取游戏状态"

    async def handle_commentary_move_result(self, data: Dict):
        """
        处理解说模式的走棋结果（从 WebSocket 事件接收）

        Args:
            data: {'success': bool, 'error': str, 'human_move': dict, 'ai_move': dict}
        """
        if not self.state.is_commentary_listening_active():
            return

        success = data.get('success', False)

        if success:
            # 成功：解说会在 on_move_complete() 中处理
            # 只需恢复监听状态
            print("[VoiceManager] 走棋成功，等待解说完成")
            self.state.set_move_result_received()
        else:
            # 失败：播报错误
            error = data.get('error', '未知错误')

            # 分类错误并生成提示
            error_type = classify_error(error) if error else ErrorType.INVALID_MOVE
            from .llm.prompts import get_error_template
            error_response = get_error_template(error_type)

            self.state.set_status(VoiceStatus.SPEAKING)
            self.state.interrupt_event.clear()  # 清除打断信号

            voice_config = {}
            if self._commentary_character:
                voice_config = self._commentary_character.get_voice_config()

            await self.tts.speak(
                error_response,
                voice=voice_config.get('voice'),
                rate=voice_config.get('rate'),
                interrupt_event=self.state.interrupt_event
            )

            # 恢复监听
            self.state.set_status(VoiceStatus.COMMENTARY_LISTENING)

    def is_commentary_listening_active(self) -> bool:
        """解说模式监听是否激活"""
        return self.state.is_commentary_listening_active()

    # ========== 会话模式方法 ==========

    def set_mode(self, mode: str) -> Dict:
        """
        设置交互模式

        Args:
            mode: "commentary" 或 "session"

        Returns:
            结果字典
        """
        if mode == "commentary":
            self.state.set_mode(VoiceMode.COMMENTARY)
            # 停止会话
            if self.state.session_active:
                self.stop_session()
            print("[VoiceManager] 切换到解说模式")
            return {'success': True, 'mode': 'commentary'}

        elif mode == "session":
            self.state.set_mode(VoiceMode.SESSION)
            print("[VoiceManager] 切换到会话模式")
            return {'success': True, 'mode': 'session'}

        return {'success': False, 'error': f'未知模式: {mode}'}

    def get_mode(self) -> str:
        """获取当前模式"""
        return self.state.mode.value

    def start_session(self) -> Dict:
        """
        启动会话模式（持续监听）

        Returns:
            结果字典
        """
        if not self.asr:
            return {'success': False, 'error': 'ASR 未配置'}

        if self.state.session_active:
            return {'success': False, 'error': '会话已激活'}

        if not self._async_loop:
            return {'success': False, 'error': '服务未启动'}

        # 打断当前播报
        if self.tts.is_speaking():
            self.interrupt_speech()

        # 启动会话
        self.state.start_session()

        # 创建对话 Session
        self.dialogue_session = DialogueSession(
            session_id=uuid.uuid4().hex[:8]
        )

        # 设置角色
        character = self.character_manager.get_current_character()
        if character:
            self.dialogue_session.set_character(character.get_id())

        # 启动持续监听循环
        asyncio.run_coroutine_threadsafe(
            self._continuous_listening_loop(),
            self._async_loop
        )

        # 播报问候语
        greeting = ""
        if character:
            greeting = character.get_greeting()

            async def speak_greeting():
                self.state.interrupt_event.clear()  # 清除打断信号
                voice_config = character.get_voice_config()
                await self.tts.speak(
                    greeting,
                    voice=voice_config.get('voice'),
                    rate=voice_config.get('rate'),
                    interrupt_event=self.state.interrupt_event
                )
                self.state.set_status(VoiceStatus.LISTENING)

            asyncio.run_coroutine_threadsafe(
                speak_greeting(),
                self._async_loop
            )

        # 通知前端
        if self.socketio:
            self.socketio.emit('session_start', {
                'success': True,
                'greeting': greeting,
                'character': self.character_manager.get_character_config()
            })

        print(f"[VoiceManager] 会话模式已启动 - 角色: {self.state.current_character_name}")

        return {
            'success': True,
            'greeting': greeting,
            'character': self.character_manager.get_character_config()
        }

    def stop_session(self) -> Dict:
        """停止会话模式"""
        self.state.stop_session()

        if self.dialogue_session:
            self.dialogue_session.clear()
            self.dialogue_session = None

        # 通知前端
        if self.socketio:
            self.socketio.emit('session_stop', {'success': True})

        print("[VoiceManager] 会话模式已停止")
        return {'success': True}

    async def _continuous_listening_loop(self):
        """持续监听循环"""
        print("[VoiceManager] 持续监听循环启动")

        while self.state.session_active:
            try:
                # 检查打断信号
                if self.state.interrupt_event.is_set():
                    self.state.interrupt_event.clear()

                # 监听语音
                self.state.set_status(VoiceStatus.LISTENING)

                if self.socketio:
                    self.socketio.emit('listening_status', {'status': 'listening'})

                transcript = await self.asr.listen(
                    timeout=config.ASR_TIMEOUT,
                    silence_threshold=config.ASR_SILENCE_THRESHOLD
                )

                if self.socketio:
                    self.socketio.emit('listening_status', {'status': 'processing'})

                if not transcript:
                    # 无语音输入，继续监听
                    continue

                print(f"[VoiceManager] 识别到: {transcript}")

                # 意图判断
                intent = detect_intent(transcript)
                print(f"[VoiceManager] 意图: {intent}")

                if intent == "move":
                    # 走棋意图 - 发送事件让前端处理走棋检测
                    if self.socketio:
                        self.socketio.emit('move_intent_detected', {
                            'transcript': transcript
                        })
                    # 继续监听（走棋处理由 app.py 回调）
                    continue

                elif intent == "cancel":
                    # 取消意图
                    await self.tts.stop()
                    self.state.set_status(VoiceStatus.LISTENING)
                    continue

                else:
                    # 对话意图
                    await self._handle_chat_intent(transcript)

            except Exception as e:
                print(f"[VoiceManager] 监听循环错误: {e}")
                await asyncio.sleep(0.5)

        print("[VoiceManager] 持续监听循环结束")

    async def _handle_chat_intent(self, transcript: str):
        """处理对话意图"""
        try:
            # 记录用户输入
            self.dialogue_session.history.add_user_message(transcript)
            self.state.add_dialogue_message('user', transcript)

            # 通知前端
            if self.socketio:
                self.socketio.emit('session_message', {
                    'role': 'user',
                    'content': transcript
                })

            # LLM 生成回复
            self.state.set_status(VoiceStatus.PROCESSING)

            character = self.character_manager.get_current_character()
            character_prompt = character.get_system_prompt() if character else None

            response = await self.llm.generate_dialogue_response(
                transcript=transcript,
                history=self.dialogue_session.history.get_recent_messages(10),
                game_context=self.dialogue_session.game_context,
                character_system_prompt=character_prompt
            )

            # 格式化回复
            if character:
                response = character.format_dialogue_response(response)

            # 记录回复
            self.dialogue_session.history.add_assistant_message(
                response,
                metadata={'character': character.get_name() if character else 'AI'}
            )
            self.state.add_dialogue_message('assistant', response)

            # TTS 播报
            self.state.set_status(VoiceStatus.SPEAKING)
            self.state.interrupt_event.clear()  # 清除打断信号

            voice_config = character.get_voice_config() if character else {}
            await self.tts.speak(
                response,
                voice=voice_config.get('voice'),
                rate=voice_config.get('rate'),
                interrupt_event=self.state.interrupt_event
            )

            # 通知前端
            if self.socketio:
                self.socketio.emit('session_message', {
                    'role': 'assistant',
                    'content': response,
                    'character': character.get_name() if character else 'AI'
                })

            # 恢复监听
            self.state.set_status(VoiceStatus.LISTENING)

        except Exception as e:
            print(f"[VoiceManager] 处理对话失败: {e}")
            self.state.set_status(VoiceStatus.LISTENING)

    async def handle_move_result(
        self,
        success: bool,
        error: str = None,
        user_move: Dict = None,
        ai_move: Dict = None,
        board_status: Dict = None
    ):
        """
        处理走棋结果（由 app.py 在走棋检测完成后调用）

        Args:
            success: 是否成功
            error: 错误信息（如果失败）
            user_move: 用户走棋信息
            ai_move: AI 走棋信息
            board_status: 棋盘状态
        """
        if not success:
            # 走棋失败，生成友好错误提示
            error_type = classify_error(error) if error else ErrorType.INVALID_MOVE
            character_name = self.state.current_character_name or "新手导师"

            error_response = await self.llm.generate_error_response(
                error_type=error_type,
                error_detail=error,
                character_name=character_name
            )

            # TTS 播报
            self.state.set_status(VoiceStatus.SPEAKING)
            self.state.interrupt_event.clear()  # 清除打断信号

            character = self.character_manager.get_current_character()
            voice_config = character.get_voice_config() if character else {}

            await self.tts.speak(
                error_response,
                voice=voice_config.get('voice'),
                rate=voice_config.get('rate'),
                interrupt_event=self.state.interrupt_event
            )

            # 通知前端
            if self.socketio:
                self.socketio.emit('session_message', {
                    'role': 'assistant',
                    'content': error_response,
                    'type': 'error'
                })

            # 恢复监听
            self.state.set_status(VoiceStatus.LISTENING)
            return

        # 走棋成功，生成解说
        if user_move and ai_move:
            character_name = self.state.current_character_name or "新手导师"

            commentary = await self.llm.generate_move_commentary(
                user_move=user_move,
                ai_move=ai_move,
                board_status=board_status or {},
                character_name=character_name
            )

            # TTS 播报
            self.state.set_status(VoiceStatus.SPEAKING)
            self.state.interrupt_event.clear()  # 清除打断信号

            character = self.character_manager.get_current_character()
            voice_config = character.get_voice_config() if character else {}

            await self.tts.speak(
                commentary,
                voice=voice_config.get('voice'),
                rate=voice_config.get('rate'),
                interrupt_event=self.state.interrupt_event
            )

            # 通知前端
            if self.socketio:
                self.socketio.emit('session_message', {
                    'role': 'assistant',
                    'content': commentary,
                    'type': 'commentary',
                    'user_move': user_move,
                    'ai_move': ai_move
                })

        # 恢复监听
        if self.state.session_active:
            self.state.set_status(VoiceStatus.LISTENING)

    def process_text_input(self, text: str) -> Dict:
        """
        处理文本输入（用于测试或前端手动输入）

        Args:
            text: 用户输入文本

        Returns:
            结果字典，包含 response 字段
        """
        if not self._async_loop:
            return {'success': False, 'error': '服务未启动'}

        # 如果会话未激活，自动启动
        if not self.state.session_active:
            print("[VoiceManager] 会话未激活，自动启动...")
            self.state.start_session()

            # 创建对话 Session
            if not self.dialogue_session:
                self.dialogue_session = DialogueSession(
                    session_id=uuid.uuid4().hex[:8]
                )

            # 设置角色
            character = self.character_manager.get_current_character()
            if character:
                self.dialogue_session.set_character(character.get_id())

        future = asyncio.run_coroutine_threadsafe(
            self._handle_chat_intent_sync(text),
            self._async_loop
        )

        try:
            result = future.result(timeout=30.0)
            return result
        except Exception as e:
            print(f"[VoiceManager] 处理文本输入失败: {e}")
            return {'success': False, 'error': str(e)}

    async def _handle_chat_intent_sync(self, transcript: str) -> Dict:
        """处理对话意图并返回结果"""
        try:
            # 记录用户输入
            self.dialogue_session.history.add_user_message(transcript)
            self.state.add_dialogue_message('user', transcript)

            # LLM 生成回复
            self.state.set_status(VoiceStatus.PROCESSING)

            character = self.character_manager.get_current_character()
            character_prompt = character.get_system_prompt() if character else None

            response = await self.llm.generate_dialogue_response(
                transcript=transcript,
                history=self.dialogue_session.history.get_recent_messages(10),
                game_context=self.dialogue_session.game_context if self.dialogue_session else {},
                character_system_prompt=character_prompt
            )

            # 格式化回复
            if character:
                response = character.format_dialogue_response(response)

            # 记录回复
            self.dialogue_session.history.add_assistant_message(
                response,
                metadata={'character': character.get_name() if character else 'AI'}
            )
            self.state.add_dialogue_message('assistant', response)

            # TTS 播报（可选）
            self.state.set_status(VoiceStatus.SPEAKING)
            self.state.interrupt_event.clear()  # 清除打断信号

            voice_config = character.get_voice_config() if character else {}
            await self.tts.speak(
                response,
                voice=voice_config.get('voice'),
                rate=voice_config.get('rate'),
                interrupt_event=self.state.interrupt_event
            )

            # 通知前端（通过 socket）
            if self.socketio:
                self.socketio.emit('session_message', {
                    'role': 'assistant',
                    'content': response,
                    'character': character.get_name() if character else 'AI'
                })

            self.state.set_status(VoiceStatus.LISTENING)

            return {
                'success': True,
                'response': response,
                'character': character.get_name() if character else 'AI'
            }

        except Exception as e:
            print(f"[VoiceManager] 处理对话失败: {e}")
            self.state.set_status(VoiceStatus.LISTENING)
            return {'success': False, 'error': str(e)}