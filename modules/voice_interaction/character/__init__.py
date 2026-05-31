"""
角色模块入口
"""
from .base import CharacterBase, CharacterConfig
from .presets import (
    PRESET_CHARACTERS,
    ChessMasterCharacter,
    NoviceTeacherCharacter,
    HumorousCommentatorCharacter,
    ClassicalChessPlayerCharacter,
    get_character,
    list_characters
)
from .manager import CharacterManager

__all__ = [
    'CharacterBase',
    'CharacterConfig',
    'PRESET_CHARACTERS',
    'ChessMasterCharacter',
    'NoviceTeacherCharacter',
    'HumorousCommentatorCharacter',
    'ClassicalChessPlayerCharacter',
    'get_character',
    'list_characters',
    'CharacterManager',
]