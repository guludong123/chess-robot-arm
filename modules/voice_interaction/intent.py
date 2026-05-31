"""
意图判断模块

根据用户语音内容判断意图：
- move: 走棋意图（触发走棋检测）
- chat: 对话意图（直接 LLM 回复）
"""
from typing import Literal, Tuple
import config


# 高置信度走棋关键词（明确是走棋意图，应触发检测）
HIGH_CONFIDENCE_KEYWORDS = [
    "下好了", "我下好了", "该你了", "到你走了",
    "我走完了", "走棋", "下完了", "可以走了",
    "好了你下", "你走", "到你了", "下", "该你下"
]

# 核心字（用于模糊匹配）
CORE_CHARS = ["下", "走", "该"]

# 低置信度走棋关键词（可能是泛用词，不触发检测）
LOW_CONFIDENCE_KEYWORDS = ["好了", "好"]

# 走棋关键词列表（兼容旧接口）
MOVE_KEYWORDS = HIGH_CONFIDENCE_KEYWORDS + LOW_CONFIDENCE_KEYWORDS

# 取消/打断关键词
CANCEL_KEYWORDS = [
    "取消", "算了", "不要了", "停下", "等等"
]


def detect_intent(transcript: str) -> Literal["move", "chat", "cancel"]:
    """
    判断用户意图

    Args:
        transcript: 用户语音转文字结果

    Returns:
        "move" - 走棋意图，触发走棋检测
        "chat" - 对话意图，直接 LLM 回复
        "cancel" - 取消意图，打断当前操作
    """
    if not transcript:
        return "chat"

    transcript = transcript.strip().lower()

    # 检查取消关键词
    for keyword in CANCEL_KEYWORDS:
        if keyword in transcript:
            return "cancel"

    # 检查走棋关键词
    for keyword in MOVE_KEYWORDS:
        if keyword in transcript:
            return "move"

    # 默认为对话意图
    return "chat"


def detect_intent_with_confidence(transcript: str) -> Tuple[Literal["move", "chat", "cancel"], Literal["high", "low"]]:
    """
    判断用户意图并返回置信度

    Args:
        transcript: 用户语音转文字结果

    Returns:
        (intent, confidence)
        - intent: "move", "chat", "cancel"
        - confidence: "high" 或 "low"
        - 只有 high 置信度的 move 才应触发走棋检测
    """
    if not transcript:
        return "chat", "low"

    # 去除空格（ASR 可能识别出带空格的结果）
    transcript_clean = transcript.strip().lower().replace(" ", "")
    transcript_original = transcript.strip().lower()

    # 检查取消关键词（高置信度）
    for keyword in CANCEL_KEYWORDS:
        if keyword in transcript_original or keyword in transcript_clean:
            return "cancel", "high"

    # 检查高置信度走棋关键词（精确匹配，同时检查带空格和不带空格的版本）
    for keyword in HIGH_CONFIDENCE_KEYWORDS:
        if keyword in transcript_original or keyword in transcript_clean:
            return "move", "high"

    # 模糊匹配：包含核心字组合（如"该你下"、"下棋"等）
    # 需要同时包含"你"或"我" + 核心字，避免误触发
    has_person = "你" in transcript_clean or "我" in transcript_clean
    has_core = any(char in transcript_clean for char in CORE_CHARS)

    if has_person and has_core:
        return "move", "high"

    # 检查低置信度关键词（如单独"好了"，可能是泛用词）
    if transcript_clean in LOW_CONFIDENCE_KEYWORDS:
        return "move", "low"

    # 默认为对话意图
    return "chat", "low"


def is_move_intent(transcript: str) -> bool:
    """判断是否为走棋意图"""
    return detect_intent(transcript) == "move"


def is_cancel_intent(transcript: str) -> bool:
    """判断是否为取消意图"""
    return detect_intent(transcript) == "cancel"


def get_matched_keyword(transcript: str) -> str:
    """
    获取匹配到的关键词

    Returns:
        匹配到的关键词，如果没有匹配则返回空字符串
    """
    if not transcript:
        return ""

    transcript = transcript.strip().lower()

    for keyword in MOVE_KEYWORDS:
        if keyword in transcript:
            return keyword

    return ""


# 错误类型枚举
class ErrorType:
    """错误类型"""
    INVALID_MOVE = "invalid_move"           # 走法不合法（如马走直线）
    WRONG_PIECE = "wrong_piece"             # 移动了黑方棋子
    NO_MOVE_DETECTED = "no_move_detected"   # 未检测到走棋
    NOT_YOUR_TURN = "not_your_turn"         # 不是红方回合
    AI_THINKING = "ai_thinking"             # AI 正在思考
    GAME_NOT_STARTED = "game_not_started"   # 游戏未开始


def classify_error(error_message: str) -> str:
    """
    根据错误消息分类错误类型

    Args:
        error_message: 系统返回的错误消息

    Returns:
        错误类型标识
    """
    error_lower = error_message.lower()

    if "不合法" in error_message or "规则" in error_message:
        return ErrorType.INVALID_MOVE

    if "黑方棋子" in error_message:
        return ErrorType.WRONG_PIECE

    if "未检测到" in error_message or "没有走棋" in error_message:
        return ErrorType.NO_MOVE_DETECTED

    if "黑方回合" in error_message or "等待ai" in error_message:
        return ErrorType.NOT_YOUR_TURN

    if "游戏状态" in error_message or "不能走棋" in error_message:
        return ErrorType.GAME_NOT_STARTED

    return ErrorType.INVALID_MOVE