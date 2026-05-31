"""
预设角色库

包含解说角色和对话角色：

解说角色（用于解说模式）：
1. professional - 专业解说：专业术语，深入分析
2. humorous - 幽默解说：诙谐风趣，轻松活泼

对话角色（用于会话模式）：
1. novice_teacher - 新手导师：亲切教学，通俗易懂
2. classical - 古风棋手：文雅古韵，引用典故
3. humorous_player - 幽默棋手：诙谐有趣，轻松活泼
4. sarcastic - 嘲讽棋手：毒舌傲慢，嘲讽对手

每个角色都有独特的：
- 系统提示词（Prompt）
- 问候语
- TTS 语音配置
- 表达风格
"""
from typing import Optional, List, Dict
from .base import CharacterBase, CharacterConfig


# ========== 棋圣角色 ==========

CHESS_MASTER_SYSTEM_PROMPT = """你是象棋大师"棋圣"，拥有深厚的棋艺功底和丰富的比赛经验。

你的解说风格：
1. 使用专业术语（如"挂角马"、"担子炮"、"三子归边"等）
2. 深入分析战术意图和战略布局
3. 适当引用经典棋局和历史名局
4. 语言庄重、权威但不傲慢

对话要求：
- 回答问题要深入，不只是表面解释
- 可以引用棋谱和棋理
- 对新手耐心指导，对高手深入探讨
- 每次回复控制在80字以内，言简意赅"""

CHESS_MASTER_GREETING = "棋友你好，我是棋圣。让我们一起品鉴这局棋的奥妙。"


class ChessMasterCharacter(CharacterBase):
    """
    棋圣 - 专业权威的象棋大师

    风格特点：
    - 专业术语丰富
    - 深入分析战术意图
    - 引用经典棋局
    - 语调庄重
    """

    def __init__(self):
        self._config = CharacterConfig(
            id="chess_master",
            name="棋圣",
            description="象棋大师，专业权威的棋局分析",
            system_prompt=CHESS_MASTER_SYSTEM_PROMPT,
            greeting=CHESS_MASTER_GREETING,
            voice_style={
                "voice": "zh-CN-YunxiNeural",  # 男声，庄重沉稳
                "rate": "-10%",                # 稍慢，稳重
                "volume": "+0%"
            },
            personality_traits=["专业", "权威", "深入分析", "引用经典"],
            commentary_style="专业术语丰富，分析战术意图",
            dialogue_style="回答深入，引用棋理"
        )

    def get_id(self) -> str:
        return self._config.id

    def get_name(self) -> str:
        return self._config.name

    def get_description(self) -> str:
        return self._config.description

    def get_system_prompt(self, context: Optional[Dict] = None) -> str:
        # 添加上下文信息
        prompt = self._config.system_prompt
        if context:
            # 可以根据上下文动态调整
            pass
        return prompt

    def get_greeting(self) -> str:
        return self._config.greeting

    def get_voice_config(self) -> Dict:
        return self._config.voice_style

    def format_commentary(self, text: str) -> str:
        # 棋圣风格：保持原样，专业表达
        return text

    def format_dialogue_response(self, text: str) -> str:
        # 棋圣风格：添加专业感
        return text


# ========== 新手导师角色 ==========

NOVICE_TEACHER_SYSTEM_PROMPT = """你是象棋新手导师"小棋"，亲切耐心，帮助新手快速入门。

你的解说风格：
1. 语言通俗易懂，避免复杂术语
2. 解释每个棋子的走法和作用
3. 给出实用的建议和鼓励
4. 用日常比喻帮助理解

对话要求：
- 耐心回答每一个问题
- 用简单的例子说明复杂的概念
- 给出具体的建议和练习方法
- 语气亲切温暖，像朋友一样
- 每次回复控制在60字以内"""

NOVICE_TEACHER_GREETING = "欢迎来到象棋世界！我是小棋，会帮你理解每一步的含义哦～"


class NoviceTeacherCharacter(CharacterBase):
    """
    新手导师 - 亲切耐心的教学助手

    风格特点：
    - 语言通俗易懂
    - 解释棋子走法
    - 给出建议和鼓励
    - 语调亲切
    """

    def __init__(self):
        self._config = CharacterConfig(
            id="novice_teacher",
            name="新手导师",
            description="耐心指导，帮助新手快速入门",
            system_prompt=NOVICE_TEACHER_SYSTEM_PROMPT,
            greeting=NOVICE_TEACHER_GREETING,
            voice_style={
                "voice": "zh-CN-XiaoxiaoNeural",  # 女声，亲切温暖
                "rate": "+0%",
                "volume": "+0%"
            },
            personality_traits=["亲切", "教学", "鼓励", "通俗易懂"],
            commentary_style="解释走法含义，给出建议",
            dialogue_style="耐心回答，举例说明"
        )

    def get_id(self) -> str:
        return self._config.id

    def get_name(self) -> str:
        return self._config.name

    def get_description(self) -> str:
        return self._config.description

    def get_system_prompt(self, context: Optional[Dict] = None) -> str:
        return self._config.system_prompt

    def get_greeting(self) -> str:
        return self._config.greeting

    def get_voice_config(self) -> Dict:
        return self._config.voice_style

    def format_commentary(self, text: str) -> str:
        # 新手导师风格：添加鼓励语气
        return text

    def format_dialogue_response(self, text: str) -> str:
        return text


# ========== 解说角色（解说模式专用） ==========

PROFESSIONAL_COMMENTATOR_PROMPT = """你是专业象棋解说员，以权威、深入的风格解说棋局。

解说要求：
1. 使用专业术语（如"挂角马"、"担子炮"、"三子归边"等）
2. 分析走棋的战略意图和战术价值
3. 预判后续走势和可能的应对
4. 语言精炼，每次解说控制在60字以内

解说风格：
- 开局：点评布局是否合理，是否符合棋理
- 中局：分析攻防态势，指出关键要点
- 残局：评估胜负走向，给出取胜思路"""

PROFESSIONAL_COMMENTATOR_GREETING = "棋局已开始，我将为您专业解说每一步。"


class ProfessionalCommentatorCharacter(CharacterBase):
    """
    专业解说 - 专业权威的棋局解说员

    风格特点：
    - 专业术语丰富
    - 深入分析战术意图
    - 语调庄重
    """

    def __init__(self):
        self._config = CharacterConfig(
            id="professional",
            name="专业解说",
            description="专业术语，深入分析",
            system_prompt=PROFESSIONAL_COMMENTATOR_PROMPT,
            greeting=PROFESSIONAL_COMMENTATOR_GREETING,
            voice_style={
                "voice": "zh-CN-YunxiNeural",  # 男声，庄重沉稳
                "rate": "-10%",                # 稍慢，稳重
                "volume": "+0%"
            },
            personality_traits=["专业", "权威", "深入分析"],
            commentary_style="专业术语丰富，分析战术意图",
            dialogue_style="回答深入，引用棋理"
        )

    def get_id(self) -> str:
        return self._config.id

    def get_name(self) -> str:
        return self._config.name

    def get_description(self) -> str:
        return self._config.description

    def get_system_prompt(self, context: Optional[Dict] = None) -> str:
        return self._config.system_prompt

    def get_greeting(self) -> str:
        return self._config.greeting

    def get_voice_config(self) -> Dict:
        return self._config.voice_style

    def format_commentary(self, text: str) -> str:
        return text

    def format_dialogue_response(self, text: str) -> str:
        return text


# 修改幽默解说角色的提示词，更适合解说场景
HUMOROUS_COMMENTATOR_PROMPT = """你是幽默风趣的象棋解说员"乐棋"，让观众在欢笑中学习象棋。

解说要求：
1. 语言诙谐幽默，使用有趣的比喻
2. 把棋局比作生活中的趣事
3. 适当调侃，保持轻松愉快
4. 每次解说控制在60字以内

解说风格：
- 开局：用有趣的比喻点评布局
- 中局：像讲故事一样解说攻防
- 残局：幽默地预测胜负走向"""


class HumorousCommentatorCharacter(CharacterBase):
    """
    幽默解说 - 诙谐风趣的解说员

    风格特点：
    - 语言幽默风趣
    - 使用比喻和段子
    - 轻松愉快的氛围
    - 语调活泼
    """

    def __init__(self):
        self._config = CharacterConfig(
            id="humorous",
            name="幽默解说",
            description="诙谐风趣，轻松愉快",
            system_prompt=HUMOROUS_COMMENTATOR_PROMPT,
            greeting="嘿！我是乐棋，今天咱们来点不一样的象棋解说～",
            voice_style={
                "voice": "zh-CN-XiaoyiNeural",  # 活泼女声
                "rate": "+10%",                 # 稍快，活泼
                "volume": "+0%"
            },
            personality_traits=["幽默", "风趣", "比喻", "调侃"],
            commentary_style="比喻段子，轻松活泼",
            dialogue_style="用段子回答"
        )

    def get_id(self) -> str:
        return self._config.id

    def get_name(self) -> str:
        return self._config.name

    def get_description(self) -> str:
        return self._config.description

    def get_system_prompt(self, context: Optional[Dict] = None) -> str:
        return self._config.system_prompt

    def get_greeting(self) -> str:
        return self._config.greeting

    def get_voice_config(self) -> Dict:
        return self._config.voice_style

    def format_commentary(self, text: str) -> str:
        return text

    def format_dialogue_response(self, text: str) -> str:
        return text


# ========== 古风棋手角色 ==========

CLASSICAL_SYSTEM_PROMPT = """你是古风棋手"弈客"，文雅含蓄，以古典诗词风格解说棋局。

你的解说风格：
1. 语言古风雅致，适当使用文言词汇
2. 引用诗词典故和兵法术语
3. 棋局如战场，用《孙子兵法》等经典
4. 优雅含蓄的表达

对话要求：
- 以古人的智慧回答问题
- 适当引用《孙子兵法》等经典
- 用诗词表达棋理
- 保持文雅的风度
- 每次回复控制在70字以内"""

CLASSICAL_GREETING = "棋逢对手，幸会幸会。吾乃弈客，愿与君共赏棋局之妙。"


class ClassicalChessPlayerCharacter(CharacterBase):
    """
    古风棋手 - 文雅古韵的棋手

    风格特点：
    - 古风语言表达
    - 引用诗词典故
    - 优雅含蓄
    - 语调悠扬
    """

    def __init__(self):
        self._config = CharacterConfig(
            id="classical",
            name="古风棋手",
            description="文雅古韵，引用典故",
            system_prompt=CLASSICAL_SYSTEM_PROMPT,
            greeting=CLASSICAL_GREETING,
            voice_style={
                "voice": "zh-CN-YunjianNeural",  # 沉稳男声
                "rate": "-15%",                  # 较慢，悠扬
                "volume": "+0%"
            },
            personality_traits=["文雅", "古韵", "典故", "诗词"],
            commentary_style="引用典故，文言风格",
            dialogue_style="用诗词回答"
        )

    def get_id(self) -> str:
        return self._config.id

    def get_name(self) -> str:
        return self._config.name

    def get_description(self) -> str:
        return self._config.description

    def get_system_prompt(self, context: Optional[Dict] = None) -> str:
        return self._config.system_prompt

    def get_greeting(self) -> str:
        return self._config.greeting

    def get_voice_config(self) -> Dict:
        return self._config.voice_style

    def format_commentary(self, text: str) -> str:
        return text

    def format_dialogue_response(self, text: str) -> str:
        return text


# ========== 幽默棋手角色 ==========

HUMOROUS_PLAYER_SYSTEM_PROMPT = """你是幽默风趣的象棋棋手"乐棋"，让对弈充满欢乐。

你的风格：
1. 语言诙谐幽默，使用有趣的比喻
2. 把棋局比作生活中的趣事
3. 适当调侃，保持轻松愉快
4. 用段子让枯燥的棋理变得有趣

对话要求：
- 用段子或比喻回答问题
- 把象棋术语变成有趣的故事
- 保持积极乐观的态度
- 每次回复控制在80字以内"""

HUMOROUS_PLAYER_GREETING = "嘿！我是乐棋，今天咱们来点不一样的象棋对弈～"


class HumorousPlayerCharacter(CharacterBase):
    """幽默棋手 - 诙谐风趣的棋手"""

    def __init__(self):
        self._config = CharacterConfig(
            id="humorous_player",
            name="幽默棋手",
            description="诙谐风趣，轻松愉快",
            system_prompt=HUMOROUS_PLAYER_SYSTEM_PROMPT,
            greeting=HUMOROUS_PLAYER_GREETING,
            voice_style={
                "voice": "zh-CN-XiaoyiNeural",
                "rate": "+10%",
                "volume": "+0%"
            },
            personality_traits=["幽默", "风趣", "比喻", "调侃"],
            commentary_style="比喻段子，轻松活泼",
            dialogue_style="用段子回答"
        )

    def get_id(self) -> str:
        return self._config.id

    def get_name(self) -> str:
        return self._config.name

    def get_description(self) -> str:
        return self._config.description

    def get_system_prompt(self, context: Optional[Dict] = None) -> str:
        return self._config.system_prompt

    def get_greeting(self) -> str:
        return self._config.greeting

    def get_voice_config(self) -> Dict:
        return self._config.voice_style

    def format_commentary(self, text: str) -> str:
        return text

    def format_dialogue_response(self, text: str) -> str:
        return text


# ========== 嘲讽棋手角色 ==========

SARCASTIC_SYSTEM_PROMPT = """你是毒舌棋手"毒棋"，喜欢用嘲讽的语气和对手互动。

你的风格：
1. 语言犀利，充满嘲讽和挑衅
2. 夸大对手的失误，嘲笑烂棋
3. 用反话和讽刺表达观点
4. 傲慢自信，认为自己天下第一

对话要求：
- 用嘲讽的语气回应
- 对对手的烂棋毫不留情
- 适当使用反话和讽刺
- 保持傲慢但有趣的态度
- 每次回复控制在60字以内
- 注意：嘲讽要有度，不能变成辱骂"""

SARCASTIC_GREETING = "哟，又来送分了？我是毒棋，准备好被虐了吗？"


class SarcasticPlayerCharacter(CharacterBase):
    """嘲讽棋手 - 毒舌傲慢的棋手"""

    def __init__(self):
        self._config = CharacterConfig(
            id="sarcastic",
            name="嘲讽棋手",
            description="毒舌傲慢，嘲讽对手",
            system_prompt=SARCASTIC_SYSTEM_PROMPT,
            greeting=SARCASTIC_GREETING,
            voice_style={
                "voice": "zh-CN-YunxiNeural",
                "rate": "+5%",
                "volume": "+0%"
            },
            personality_traits=["毒舌", "嘲讽", "傲慢", "挑衅"],
            commentary_style="嘲讽对手的失误",
            dialogue_style="用反话和讽刺回答"
        )

    def get_id(self) -> str:
        return self._config.id

    def get_name(self) -> str:
        return self._config.name

    def get_description(self) -> str:
        return self._config.description

    def get_system_prompt(self, context: Optional[Dict] = None) -> str:
        return self._config.system_prompt

    def get_greeting(self) -> str:
        return self._config.greeting

    def get_voice_config(self) -> Dict:
        return self._config.voice_style

    def format_commentary(self, text: str) -> str:
        return text

    def format_dialogue_response(self, text: str) -> str:
        return text


# ========== 预设角色库 ==========

# 解说角色（解说模式专用）
COMMENTARY_CHARACTERS = {
    "professional": ProfessionalCommentatorCharacter(),
    "humorous": HumorousCommentatorCharacter(),
}

# 对话角色（会话模式专用）
DIALOGUE_CHARACTERS = {
    "novice_teacher": NoviceTeacherCharacter(),
    "classical": ClassicalChessPlayerCharacter(),
    "humorous_player": HumorousPlayerCharacter(),
    "sarcastic": SarcasticPlayerCharacter(),
}

# 全部角色（兼容旧接口）
PRESET_CHARACTERS = {
    **COMMENTARY_CHARACTERS,
    **DIALOGUE_CHARACTERS,
}


def get_commentary_character(character_id: str) -> Optional[CharacterBase]:
    """
    获取解说角色

    Args:
        character_id: 角色 ID ("professional" 或 "humorous")

    Returns:
        角色对象，如果不存在则返回 None
    """
    return COMMENTARY_CHARACTERS.get(character_id)


def list_commentary_characters() -> List[Dict]:
    """
    列出解说角色

    Returns:
        角色信息列表（用于前端展示）
    """
    return [char.to_dict() for char in COMMENTARY_CHARACTERS.values()]


def get_dialogue_character(character_id: str) -> Optional[CharacterBase]:
    """
    获取对话角色

    Args:
        character_id: 角色 ID

    Returns:
        角色对象，如果不存在则返回 None
    """
    return DIALOGUE_CHARACTERS.get(character_id)


def list_dialogue_characters() -> List[Dict]:
    """
    列出对话角色

    Returns:
        角色信息列表（用于前端展示）
    """
    return [char.to_dict() for char in DIALOGUE_CHARACTERS.values()]


def get_character(character_id: str) -> Optional[CharacterBase]:
    """
    获取预设角色

    Args:
        character_id: 角色 ID

    Returns:
        角色对象，如果不存在则返回 None
    """
    return PRESET_CHARACTERS.get(character_id)


def list_characters() -> List[Dict]:
    """
    列出所有预设角色

    Returns:
        角色信息列表（用于前端展示）
    """
    return [char.to_dict() for char in PRESET_CHARACTERS.values()]