"""
语音交互状态管理
"""
import asyncio
from enum import Enum
from typing import Dict, Optional, List
import time


class VoiceStatus(Enum):
    """语音交互状态"""
    IDLE = 'idle'               # 待命
    GENERATING = 'generating'   # 正在生成解说
    SPEAKING = 'speaking'       # 正在播报
    LISTENING = 'listening'     # 正在监听语音
    PROCESSING = 'processing'   # 正在处理命令
    DIALOGUE_ACTIVE = 'dialogue_active'  # 会话模式激活
    COMMENTARY_LISTENING = 'commentary_listening'  # 解说模式监听中
    WAITING_MOVE_RESULT = 'waiting_move_result'    # 等待走棋结果
    ERROR = 'error'             # 错误状态
    DISABLED = 'disabled'       # 已禁用


class VoiceMode(Enum):
    """语音交互模式"""
    COMMENTARY = 'commentary'   # 解说模式（被动，走棋后自动解说）
    SESSION = 'session'         # 会话模式（主动，持续监听+交互）


class VoiceState:
    """语音交互状态管理器"""

    def __init__(self):
        self.status = VoiceStatus.IDLE
        self.interrupt_event = asyncio.Event()
        self.game_context: Dict = {}
        self.last_commentary: Optional[str] = None
        self.commentary_history: List[Dict] = []

        # ========== 模式状态 ==========
        self.mode: VoiceMode = VoiceMode.COMMENTARY  # 当前模式（默认解说模式）
        self.session_active: bool = False            # 会话模式是否激活

        # ========== 解说模式监听状态 ==========
        self._commentary_listening_active: bool = False  # 解说模式持续监听激活
        self._move_result_received: bool = False          # 走棋结果是否已接收

        # ========== 对话状态扩展 ==========
        self.dialogue_active: bool = False          # 对话模式是否激活
        self.dialogue_history: List[Dict] = []      # 对话历史（简化版）
        self.current_character_id: Optional[str] = None  # 当前角色 ID
        self.current_character_name: Optional[str] = None  # 当前角色名称

    def set_status(self, status: VoiceStatus):
        """设置状态"""
        self.status = status
        # 状态改变时清除打断信号
        if status != VoiceStatus.SPEAKING:
            self.interrupt_event.clear()

    def update_game_context(
        self,
        board_state: Dict,
        current_player: str,
        move_history: list
    ):
        """更新游戏上下文"""
        self.game_context = {
            'board_state': board_state,
            'current_player': current_player,
            'move_history': move_history,
            'move_count': len(move_history)
        }

    def can_listen(self) -> bool:
        """是否可以开始监听"""
        return self.status in [VoiceStatus.IDLE, VoiceStatus.SPEAKING]

    def can_interrupt(self) -> bool:
        """是否可以打断"""
        return self.status == VoiceStatus.SPEAKING

    def add_commentary(self, text: str, metadata: Dict = None):
        """记录解说历史"""
        self.last_commentary = text
        self.commentary_history.append({
            'text': text,
            'metadata': metadata,
            'timestamp': time.time()
        })
        # 限制历史长度
        if len(self.commentary_history) > 20:
            self.commentary_history = self.commentary_history[-20:]

    # ========== 对话状态方法 ==========

    def set_dialogue_mode(self, active: bool):
        """设置对话模式"""
        self.dialogue_active = active
        if active:
            self.set_status(VoiceStatus.DIALOGUE_ACTIVE)
        else:
            self.set_status(VoiceStatus.IDLE)

    def add_dialogue_message(self, role: str, content: str, metadata: Dict = None):
        """
        添加对话消息

        Args:
            role: 角色 ('user' 或 'assistant')
            content: 消息内容
            metadata: 元数据（如角色名称、触发方式等）
        """
        self.dialogue_history.append({
            'role': role,
            'content': content,
            'metadata': metadata,
            'timestamp': time.time()
        })
        # 限制对话历史长度
        if len(self.dialogue_history) > 50:
            self.dialogue_history = self.dialogue_history[-50:]

    def clear_dialogue_history(self):
        """清空对话历史"""
        self.dialogue_history.clear()

    def set_character(self, character_id: str, character_name: str = None):
        """设置当前角色"""
        self.current_character_id = character_id
        self.current_character_name = character_name or character_id

    def get_last_dialogue_messages(self, count: int = 10) -> List[Dict]:
        """获取最近的对话消息"""
        return self.dialogue_history[-count:] if len(self.dialogue_history) > count else self.dialogue_history

    def is_dialogue_enabled(self) -> bool:
        """是否处于对话模式"""
        return self.dialogue_active and self.status != VoiceStatus.DISABLED

    # ========== 模式切换方法 ==========

    def set_mode(self, mode: VoiceMode):
        """设置交互模式"""
        self.mode = mode
        if mode == VoiceMode.COMMENTARY:
            # 切换到解说模式，停止会话
            self.session_active = False
            self.dialogue_active = False
            self.set_status(VoiceStatus.IDLE)
        elif mode == VoiceMode.SESSION:
            # 切换到会话模式，等待用户启动
            self.session_active = False
            self.set_status(VoiceStatus.IDLE)

    def get_mode(self) -> VoiceMode:
        """获取当前模式"""
        return self.mode

    def is_commentary_mode(self) -> bool:
        """是否为解说模式"""
        return self.mode == VoiceMode.COMMENTARY

    def is_session_mode(self) -> bool:
        """是否为会话模式"""
        return self.mode == VoiceMode.SESSION

    def start_session(self):
        """启动会话（会话模式下的监听）"""
        self.session_active = True
        self.dialogue_active = True
        self.set_status(VoiceStatus.LISTENING)

    def stop_session(self):
        """停止会话"""
        self.session_active = False
        self.dialogue_active = False
        self.set_status(VoiceStatus.IDLE)

    # ========== 解说模式监听方法 ==========

    def start_commentary_listening(self):
        """启动解说模式的持续监听"""
        self._commentary_listening_active = True
        self._move_result_received = False
        self.set_status(VoiceStatus.COMMENTARY_LISTENING)

    def stop_commentary_listening(self):
        """停止解说模式的持续监听"""
        self._commentary_listening_active = False
        self._move_result_received = False
        self.set_status(VoiceStatus.IDLE)

    def is_commentary_listening_active(self) -> bool:
        """解说模式监听是否激活"""
        return self._commentary_listening_active

    def set_waiting_move_result(self):
        """设置为等待走棋结果状态"""
        self._move_result_received = False
        self.set_status(VoiceStatus.WAITING_MOVE_RESULT)

    def set_move_result_received(self):
        """标记走棋结果已接收"""
        self._move_result_received = True

    def is_waiting_move_result(self) -> bool:
        """是否在等待走棋结果"""
        return self.status == VoiceStatus.WAITING_MOVE_RESULT

    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            'status': self.status.value,
            'last_commentary': self.last_commentary,
            'can_listen': self.can_listen(),
            'can_interrupt': self.can_interrupt(),
            # 模式状态
            'mode': self.mode.value,
            'session_active': self.session_active,
            # 对话状态
            'dialogue_active': self.dialogue_active,
            'dialogue_history_count': len(self.dialogue_history),
            'current_character_id': self.current_character_id,
            'current_character_name': self.current_character_name
        }