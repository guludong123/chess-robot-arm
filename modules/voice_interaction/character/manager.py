"""
角色管理器

管理角色的切换和使用：
- 当前角色设置
- 角色列表获取
- 使用历史记录
"""
import time
from typing import Optional, Dict, List

from .base import CharacterBase
from .presets import get_character, list_characters


class CharacterManager:
    """
    角色管理器

    功能：
    - 设置/获取当前角色
    - 列出可用角色
    - 记录角色使用历史
    """

    def __init__(self, default_character_id: str = "novice_teacher"):
        """
        初始化角色管理器

        Args:
            default_character_id: 默认角色 ID
        """
        self._current_character: Optional[CharacterBase] = None
        self._character_history: Dict[str, Dict] = {}  # 使用历史

        # 设置默认角色
        self.set_character(default_character_id)

    def set_character(self, character_id: str) -> bool:
        """
        设置当前角色

        Args:
            character_id: 角色 ID

        Returns:
            是否成功设置
        """
        character = get_character(character_id)
        if character:
            self._current_character = character
            self._record_character_usage(character_id)
            print(f"[CharacterManager] 角色已切换: {character.get_name()}")
            return True

        print(f"[CharacterManager] 角色 {character_id} 不存在")
        return False

    def get_current_character(self) -> Optional[CharacterBase]:
        """获取当前角色"""
        return self._current_character

    def get_current_character_id(self) -> Optional[str]:
        """获取当前角色 ID"""
        if self._current_character:
            return self._current_character.get_id()
        return None

    def get_current_character_name(self) -> Optional[str]:
        """获取当前角色名称"""
        if self._current_character:
            return self._current_character.get_name()
        return None

    def get_character_config(self) -> Dict:
        """
        获取当前角色配置（用于前端）

        Returns:
            角色配置字典，如果无角色则返回默认配置
        """
        if self._current_character:
            return self._current_character.to_dict()

        return {
            "id": "default",
            "name": "默认解说员",
            "description": "标准解说风格",
            "greeting": "",
            "voice_config": {
                "voice": "zh-CN-XiaoxiaoNeural",
                "rate": "+0%",
                "volume": "+0%"
            }
        }

    def list_available_characters(self) -> List[Dict]:
        """
        列出可用角色

        Returns:
            角色信息列表
        """
        characters = list_characters()

        # 标记当前角色
        current_id = self.get_current_character_id()
        for char in characters:
            char['is_current'] = (char['id'] == current_id)

        return characters

    def get_character_usage_history(self) -> Dict:
        """获取角色使用历史"""
        return self._character_history.copy()

    def _record_character_usage(self, character_id: str):
        """记录角色使用历史"""
        if character_id not in self._character_history:
            self._character_history[character_id] = {
                "first_used": time.time(),
                "use_count": 0
            }

        self._character_history[character_id]["use_count"] += 1
        self._character_history[character_id]["last_used"] = time.time()

    def __repr__(self) -> str:
        char_name = self.get_current_character_name() or "无"
        return f"CharacterManager(current={char_name})"