"""
Session 级别对话上下文

管理一个对话 Session 的完整上下文：
- 对话历史
- 游戏状态上下文
- 当前角色
"""
import time
import uuid
from typing import Dict, Optional
from dataclasses import dataclass, field

from .history import DialogueHistory


@dataclass
class DialogueSession:
    """
    对话 Session

    一个 Session 包含：
    - session_id: 会话唯一标识
    - history: 对话历史
    - game_context: 游戏状态上下文
    - character_id: 当前角色 ID
    - created_at: 创建时间
    """

    session_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    history: DialogueHistory = field(default_factory=DialogueHistory)
    game_context: Dict = field(default_factory=dict)
    character_id: Optional[str] = None
    created_at: float = field(default_factory=time.time)

    def update_game_context(
        self,
        board_state: Dict,
        current_player: str,
        move_history: list
    ):
        """
        更新游戏上下文

        Args:
            board_state: 当前棋盘状态
            current_player: 当前执棋方 ('red' 或 'black')
            move_history: 走棋历史列表
        """
        self.game_context = {
            "board_state": board_state,
            "current_player": current_player,
            "move_history": move_history,
            "move_count": len(move_history),
            "updated_at": time.time()
        }

    def set_character(self, character_id: str):
        """设置当前角色"""
        self.character_id = character_id

    def get_context_summary(self) -> str:
        """
        获取上下文摘要文本

        用于 LLM Prompt 中添加游戏上下文信息
        """
        if not self.game_context:
            return ""

        player = self.game_context.get('current_player', '未知')
        move_count = self.game_context.get('move_count', 0)

        player_text = '红方' if player == 'red' else '黑方'

        return f"当前棋局：已走 {move_count} 步，{player_text} 执棋。"

    def clear(self):
        """清空 Session"""
        self.history.clear()
        self.game_context.clear()
        self.character_id = None

    def reset(self):
        """重置 Session（保留 session_id）"""
        self.clear()
        self.created_at = time.time()

    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            "session_id": self.session_id,
            "history": self.history.to_dict(),
            "game_context": self.game_context,
            "character_id": self.character_id,
            "created_at": self.created_at,
            "duration": time.time() - self.created_at
        }

    def __repr__(self) -> str:
        return f"DialogueSession(id={self.session_id}, msgs={len(self.history)}, char={self.character_id})"


class SessionManager:
    """
    Session 管理器

    管理多个对话 Session：
    - 当前活跃 Session
    - Session 创建/销毁
    """

    def __init__(self, max_sessions: int = 5):
        """
        初始化 Session 管理器

        Args:
            max_sessions: 最大 Session 数量
        """
        self._sessions: Dict[str, DialogueSession] = {}
        self._current_session: Optional[DialogueSession] = None
        self._max_sessions = max_sessions

    def create_session(self) -> DialogueSession:
        """
        创建新 Session

        Returns:
            新创建的 Session
        """
        session = DialogueSession()
        self._sessions[session.session_id] = session
        self._current_session = session

        # 清理过多的 Session
        self._cleanup_old_sessions()

        return session

    def get_current_session(self) -> Optional[DialogueSession]:
        """获取当前 Session"""
        return self._current_session

    def set_current_session(self, session_id: str) -> bool:
        """
        设置当前 Session

        Args:
            session_id: Session ID

        Returns:
            是否成功设置
        """
        if session_id in self._sessions:
            self._current_session = self._sessions[session_id]
            return True
        return False

    def get_session(self, session_id: str) -> Optional[DialogueSession]:
        """获取指定 Session"""
        return self._sessions.get(session_id)

    def destroy_session(self, session_id: str) -> bool:
        """
        销毁 Session

        Args:
            session_id: Session ID

        Returns:
            是否成功销毁
        """
        if session_id in self._sessions:
            session = self._sessions.pop(session_id)
            session.clear()

            # 如果销毁的是当前 Session，清空引用
            if self._current_session and self._current_session.session_id == session_id:
                self._current_session = None

            return True
        return False

    def clear_all(self):
        """清空所有 Session"""
        for session in self._sessions.values():
            session.clear()
        self._sessions.clear()
        self._current_session = None

    def _cleanup_old_sessions(self):
        """清理过多的旧 Session"""
        if len(self._sessions) > self._max_sessions:
            # 按创建时间排序，删除最旧的
            sorted_sessions = sorted(
                self._sessions.items(),
                key=lambda x: x[1].created_at
            )

            # 删除最旧的 Session
            for session_id, _ in sorted_sessions[:-self._max_sessions]:
                self.destroy_session(session_id)

    def __len__(self) -> int:
        return len(self._sessions)

    def __repr__(self) -> str:
        return f"SessionManager(count={len(self._sessions)}, current={self._current_session})"