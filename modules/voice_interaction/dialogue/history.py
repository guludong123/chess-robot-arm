"""
对话历史管理

管理用户与 AI 的对话记录：
- 存储对话消息（用户消息 + AI 消息）
- 支持历史长度限制
- 提供 LLM 格式的消息列表
"""
import time
from typing import List, Dict, Optional
from dataclasses import dataclass, field


@dataclass
class DialogueMessage:
    """
    对话消息数据结构

    Attributes:
        role: 角色 ("user" 或 "assistant")
        content: 消息内容
        timestamp: 时间戳
        metadata: 元数据（如触发方式、角色信息等）
    """
    role: str
    content: str
    timestamp: float = field(default_factory=time.time)
    metadata: Optional[Dict] = None

    def to_dict(self) -> Dict:
        """转换为字典格式"""
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp,
            "metadata": self.metadata
        }

    def to_llm_format(self) -> Dict:
        """转换为 LLM 调用格式"""
        return {
            "role": self.role,
            "content": self.content
        }


class DialogueHistory:
    """
    对话历史管理器

    功能：
    - 存储对话消息
    - 限制历史长度（避免 LLM token 过多）
    - 提供多种格式输出
    """

    def __init__(self, max_history: int = 50):
        """
        初始化对话历史

        Args:
            max_history: 最大历史消息数（超过后自动删除旧消息）
        """
        self._messages: List[DialogueMessage] = []
        self._max_history = max_history
        self._session_start_time: float = time.time()

    def add_user_message(
        self,
        content: str,
        metadata: Optional[Dict] = None
    ) -> DialogueMessage:
        """
        添加用户消息

        Args:
            content: 消息内容
            metadata: 元数据（如来源：语音/文字）

        Returns:
            创建的消息对象
        """
        message = DialogueMessage(
            role="user",
            content=content,
            metadata=metadata
        )
        self._messages.append(message)
        self._trim_history()
        return message

    def add_assistant_message(
        self,
        content: str,
        metadata: Optional[Dict] = None
    ) -> DialogueMessage:
        """
        添加助手消息

        Args:
            content: 消息内容
            metadata: 元数据（如角色名称、生成时间等）

        Returns:
            创建的消息对象
        """
        message = DialogueMessage(
            role="assistant",
            content=content,
            metadata=metadata
        )
        self._messages.append(message)
        self._trim_history()
        return message

    def get_messages(self) -> List[DialogueMessage]:
        """获取所有消息"""
        return self._messages.copy()

    def get_llm_messages(self) -> List[Dict]:
        """
        获取 LLM 格式的消息列表

        用于 DashScope 等 LLM API 调用
        """
        return [msg.to_llm_format() for msg in self._messages]

    def get_recent_messages(self, count: int = 10) -> List[Dict]:
        """
        获取最近 N 条消息（LLM 格式）

        Args:
            count: 消息数量

        Returns:
            LLM 格式的消息列表
        """
        recent = self._messages[-count:] if len(self._messages) > count else self._messages
        return [msg.to_llm_format() for msg in recent]

    def get_last_user_message(self) -> Optional[DialogueMessage]:
        """获取最后一条用户消息"""
        for msg in reversed(self._messages):
            if msg.role == "user":
                return msg
        return None

    def get_last_assistant_message(self) -> Optional[DialogueMessage]:
        """获取最后一条助手消息"""
        for msg in reversed(self._messages):
            if msg.role == "assistant":
                return msg
        return None

    def clear(self):
        """清空历史"""
        self._messages.clear()
        self._session_start_time = time.time()

    def to_dict(self) -> Dict:
        """
        转换为字典（用于前端显示）

        Returns:
            包含所有消息和元数据的字典
        """
        return {
            "messages": [msg.to_dict() for msg in self._messages],
            "session_start": self._session_start_time,
            "message_count": len(self._messages),
            "max_history": self._max_history
        }

    def get_summary(self) -> Dict:
        """获取历史摘要"""
        user_count = sum(1 for msg in self._messages if msg.role == "user")
        assistant_count = sum(1 for msg in self._messages if msg.role == "assistant")
        return {
            "total_messages": len(self._messages),
            "user_messages": user_count,
            "assistant_messages": assistant_count,
            "session_duration": time.time() - self._session_start_time
        }

    def _trim_history(self):
        """修剪历史长度"""
        if len(self._messages) > self._max_history:
            # 保留最近的消息
            self._messages = self._messages[-self._max_history:]

    def __len__(self) -> int:
        return len(self._messages)

    def __repr__(self) -> str:
        return f"DialogueHistory(messages={len(self._messages)}, max={self._max_history})"