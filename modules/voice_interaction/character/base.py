"""
角色抽象基类和数据结构

定义角色的核心属性和行为：
- 角色配置（名称、描述、语音风格等）
- 系统提示词生成
- 解说和对话风格格式化
"""
from abc import ABC, abstractmethod
from typing import Dict, Optional, List
from dataclasses import dataclass


@dataclass
class CharacterConfig:
    """
    角色配置数据结构

    Attributes:
        id: 角色 ID（用于标识和切换）
        name: 角色显示名称
        description: 角色简短描述
        system_prompt: 系统提示词模板
        greeting: 问候语（对话开始时播报）
        voice_style: 语音风格配置（TTS 参数）
        personality_traits: 性格特点列表
        commentary_style: 解说风格描述
        dialogue_style: 对话风格描述
        avatar: 角色头像 URL（可选）
    """
    id: str
    name: str
    description: str
    system_prompt: str
    greeting: str
    voice_style: Dict
    personality_traits: List[str] = None
    commentary_style: str = ""
    dialogue_style: str = ""
    avatar: Optional[str] = None

    def __post_init__(self):
        if self.personality_traits is None:
            self.personality_traits = []


class CharacterBase(ABC):
    """
    角色抽象基类

    所有角色必须实现以下方法：
    - get_id(): 获取角色 ID
    - get_name(): 获取角色名称
    - get_system_prompt(): 获取系统提示词
    - get_greeting(): 获取问候语
    - get_voice_config(): 获取语音配置
    - format_commentary(): 格式化解说
    - format_dialogue_response(): 格式化对话响应
    """

    @abstractmethod
    def get_id(self) -> str:
        """获取角色 ID"""
        pass

    @abstractmethod
    def get_name(self) -> str:
        """获取角色名称"""
        pass

    @abstractmethod
    def get_description(self) -> str:
        """获取角色描述"""
        pass

    @abstractmethod
    def get_system_prompt(self, context: Optional[Dict] = None) -> str:
        """
        获取系统提示词

        Args:
            context: 上下文信息（如游戏状态、对话历史等）

        Returns:
            构建后的系统提示词
        """
        pass

    @abstractmethod
    def get_greeting(self) -> str:
        """获取问候语"""
        pass

    @abstractmethod
    def get_voice_config(self) -> Dict:
        """
        获取语音配置

        Returns:
            TTS 参数字典：voice, rate, volume 等
        """
        pass

    @abstractmethod
    def format_commentary(self, text: str) -> str:
        """
        格式化解说文本

        根据角色风格调整解说表达
        """
        pass

    @abstractmethod
    def format_dialogue_response(self, text: str) -> str:
        """
        格式化对话响应

        根据角色风格调整对话表达
        """
        pass

    def to_dict(self) -> Dict:
        """
        转换为字典（用于前端展示）

        Returns:
            角色信息字典
        """
        return {
            'id': self.get_id(),
            'name': self.get_name(),
            'description': self.get_description(),
            'greeting': self.get_greeting(),
            'voice_config': self.get_voice_config()
        }

    def __repr__(self) -> str:
        return f"Character(id={self.get_id()}, name={self.get_name()})"