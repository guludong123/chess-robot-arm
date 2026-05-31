"""
Prompt 模板

包含：
- 解说系统提示词
- 命令解析系统提示词
- 对话系统提示词
- 各角色的 Prompt 模板
- 预设错误回复模板
"""
import random
from typing import Dict, Optional, List


# ========== 解说系统提示词 ==========

COMMENTARY_SYSTEM_PROMPT = """你是专业中国象棋解说员，解说风格生动有趣。

解说要求：
1. 语言活泼自然，避免机械化套路
2. 使用象棋术语（挂角马、担子炮、三子归边、卧槽马等）
3. 分析战术意图和后续变化，不只是描述走棋
4. 可以适当调侃、幽默，或用比喻手法
5. 根据局势紧张程度调整语气（将死时语气激动，均势时从容分析）
6. 每次解说要有不同的切入角度，避免重复套路
7. 解说长度灵活，关键局面可深入分析（最多80字）
8. 开局阶段侧重布局思路，中局侧重战术攻防，残局侧重杀法和子力效率"""

# 多样化引导词（随机选取，避免每次都一样）
COMMENTARY_INTRO_VARIANTS = [
    "这步棋很有意思！",
    "来看这轮交锋：",
    "精彩的一幕：",
    "棋盘上风云变幻：",
    "这一回合：",
    "局势有了新变化：",
    "双方你来我往：",
    "这一招：",
]


# ========== 预设错误回复模板 ==========

ERROR_TEMPLATES = {
    "invalid_move": [
        "这步棋不符合象棋规则，请重新走。",
        "这个走法不合法，再想想吧。",
        "棋子不是这样走的，请检查规则。"
    ],
    "wrong_piece": [
        "这是黑方的棋子，你只能走红方。",
        "嘿，那是对方的棋子！"
    ],
    "no_move_detected": [
        "我没看到你走棋呢，走好了告诉我。",
        "棋盘上好像没有变化哦。"
    ],
    "not_your_turn": [
        "等等，现在轮到AI走了。",
        "稍等，让AI先下完这步。"
    ],
    "game_not_started": [
        "游戏还没开始，先点击开始游戏。",
        "记得先开始游戏才能走棋。"
    ]
}

# ========== 解说模式预设回复（旁观者口吻） ==========

# 当用户说"下好了"触发走棋检测时的预设回复
COMMENTARY_MOVE_CONFIRMED_RESPONSES = [
    "好的",
    "收到",
    "明白",
]

# 当用户说"该你了"等催促时的预设回复
COMMENTARY_HURRY_RESPONSES = [
    "好",
    "收到",
]


def get_commentary_move_response() -> str:
    """获取走棋确认的预设回复（旁观者口吻）"""
    return random.choice(COMMENTARY_MOVE_CONFIRMED_RESPONSES)


def get_commentary_hurry_response() -> str:
    """获取催促时的预设回复"""
    return random.choice(COMMENTARY_HURRY_RESPONSES)


def get_error_template(error_type: str) -> str:
    """获取预设的错误回复模板（不调用LLM）"""
    templates = ERROR_TEMPLATES.get(error_type, ["遇到了一点问题，请重试。"])
    return random.choice(templates)


def get_commentary_prompt(
    move_description: str,
    situation: Dict = None,
    move_history: List[Dict] = None,
    context: Optional[Dict] = None,
    board_analysis: Optional[Dict] = None,
    board_global: Optional[Dict] = None
) -> str:
    """
    生成生动解说 Prompt

    Args:
        move_description: 走棋描述
        situation: 局势分析 {advantage, score_cp, is_checkmate, mate_in, score_change}
        move_history: 走棋历史
        context: 额外上下文
        board_analysis: 棋盘态势分析 {threats, active_pieces, key_areas}
        board_global: 全局分析 {phase, pieces_summary}
    """
    import random

    # 随机引导词（增加多样性）
    intro = random.choice(COMMENTARY_INTRO_VARIANTS)

    # 局势描述（更详细）
    situation_str = ""
    if situation:
        advantage = situation.get('advantage', '均势')
        score_cp = situation.get('score_cp', 0)
        score_change = situation.get('score_change', 0)

        # 分数变化描述
        if abs(score_change) > 50:
            change_desc = f"（分数变化{score_change:+d}）"
        else:
            change_desc = ""

        situation_str = f"【局势】{advantage}{change_desc}"

        # 将死信息
        if situation.get('is_checkmate'):
            situation_str += " ⚠️ 即将将死！"
        elif situation.get('mate_in'):
            situation_str += f"，{situation.get('mate_in')}步可胜"

    # 棋盘态势（威胁分析等）
    analysis_str = ""
    if board_analysis:
        threats = board_analysis.get('threats', [])
        key_areas = board_analysis.get('key_areas', [])
        engine_intent = board_analysis.get('engine_intent', [])  # 引擎真实意图

        if threats:
            threat_desc = "、".join(threats[:3])  # 最多显示3个威胁
            analysis_str += f"【威胁】{threat_desc}"

        if key_areas:
            area_desc = "、".join(key_areas[:2])
            analysis_str += f"【焦点区域】{area_desc}"

        # 引擎意图（基于 PV 分析）
        if engine_intent:
            intent_desc = "、".join(engine_intent)
            analysis_str += f"【引擎意图】{intent_desc}"

    # 棋局阶段与子力统计
    global_str = ""
    if board_global:
        phase = board_global.get('phase', '')
        pieces_summary = board_global.get('pieces_summary', '')
        if phase:
            global_str += f"【阶段】{phase}"
        if pieces_summary:
            global_str += f"【子力】{pieces_summary}"

    # 历史走棋（含吃子信息）
    CAP_NAME_MAP = {'che': '车', 'ma': '马', 'pao': '炮',
                    'bing': '兵', 'zu': '卒', 'shi': '仕',
                    'xiang': '相', 'shuai': '帅', 'jiang': '将'}
    history_str = ""
    if move_history:
        history_str = "【近期】"
        for move in move_history[-3:]:
            player = "红" if move.get('player') == 'red' else "黑"
            piece_short = move.get('piece_short', '')
            captured = move.get('captured')
            if captured and isinstance(captured, dict):
                cap_name = captured.get('class_name', '').split('_')[-1]
                cap_cn = CAP_NAME_MAP.get(cap_name, '')
                if cap_cn:
                    history_str += f"{player}{piece_short}吃{cap_cn} "
                else:
                    history_str += f"{player}{piece_short} "
            else:
                history_str += f"{player}{piece_short} "

    prompt = f"""{intro}

本轮走棋：{move_description}

{situation_str}
{analysis_str}
{global_str}
{history_str}

请用生动有趣的语言解说这步棋：
1. 分析战术意图和后续可能变化
2. 结合局势变化评价得失
3. 语言自然活泼，避免机械套路
4. 可以适当幽默或用比喻
5. 根据局势紧张程度调整语气
6. 解说长度灵活，最多80字"""

    if context:
        prompt += f"\n\n额外信息：{context}"

    return prompt

# ========== 命令解析系统提示词 ==========

COMMAND_SYSTEM_PROMPT = """你是象棋机器人语音命令解析器。
将用户的语音命令转换为结构化指令。
返回 JSON 格式：
{"action": "动作名", "params": {"参数": "值"}, "response": "回复用户的话"}

支持的动作：
- start_game: 开始游戏
- stop_game: 停止游戏
- reset_game: 重置游戏
- ai_move: AI走棋
- scan_board: 扫描棋盘
- connect_robot: 连接机械臂
- disconnect_robot: 断开机械臂
- set_difficulty: 设置难度 (params: {"level": 1-20})
- interrupt: 打断播报
- explain_position: 解说局势

如果无法理解，返回 {"action": "unknown", "response": "抱歉的回复"}"""


def get_command_prompt(transcript: str, game_state: Dict) -> str:
    """生成命令解析 Prompt"""

    status = game_state.get('status', 'unknown')
    player = game_state.get('current_player', 'unknown')
    move_count = len(game_state.get('move_history', []))

    return f"""用户说："{transcript}"

当前状态：
- 游戏状态：{status}
- 当前玩家：{'红方' if player == 'red' else '黑方'}
- 已走棋数：{move_count}

请解析用户的意图，返回 JSON 格式的命令。"""


# ========== 对话 Prompt 模板 ==========

DEFAULT_DIALOGUE_SYSTEM_PROMPT = """你是象棋对战机器人的AI助手。

你的职责：
1. 与用户进行自然对话
2. 回答关于象棋的问题
3. 解释棋局和走法
4. 提供游戏建议

对话要求：
- 语言自然流畅
- 保持友善态度
- 回答简洁明了
- 每次回复不超过100字"""


def get_dialogue_prompt(
    transcript: str,
    history: List[Dict],
    game_context: Dict,
    character_system_prompt: str = None
) -> str:
    """
    生成对话 Prompt

    Args:
        transcript: 用户输入文本
        history: 对话历史列表 [{"role": "user/assistant", "content": "..."}]
        game_context: 游戏上下文（棋盘状态等）
        character_system_prompt: 角色系统提示词（可选）

    Returns:
        构建 Prompt
    """
    # 构建历史对话
    history_str = ""
    if history:
        history_str = "对话历史：\n"
        for msg in history[-6:]:  # 最近6轮
            role = "用户" if msg.get("role") == "user" else "AI"
            content = msg.get("content", "")
            history_str += f"{role}: {content}\n"

    # 构建游戏上下文
    context_str = ""
    if game_context:
        player = game_context.get('current_player', '未知')
        move_count = game_context.get('move_count', 0)
        context_str = f"当前棋局：已走 {move_count} 步，{'红方' if player == 'red' else '黑方'}执棋。"

    prompt = f"""{history_str}

{context_str}

用户最新问题：{transcript}

请根据你的角色设定，给出合适的回复。"""

    return prompt


def build_dialogue_messages(
    system_prompt: str,
    history: List[Dict],
    current_input: str
) -> List[Dict]:
    """
    构建完整的对话消息列表（用于 LLM API 调用）

    Args:
        system_prompt: 系统提示词
        history: 对话历史
        current_input: 当前用户输入

    Returns:
        LLM API 格式的消息列表
    """
    messages = [
        {"role": "system", "content": system_prompt}
    ]

    # 添加历史对话（最多8轮）
    for msg in history[-8:]:
        role = msg.get("role")
        content = msg.get("content")
        if role in ["user", "assistant"] and content:
            messages.append({"role": role, "content": content})

    # 添加当前输入
    messages.append({"role": "user", "content": current_input})

    return messages


def get_character_commentary_prompt(
    move_description: str,
    character_prompt: str,
    context: Optional[Dict] = None
) -> str:
    """
    生成角色化解说 Prompt

    Args:
        move_description: 走棋描述
        character_prompt: 角色系统提示词
        context: 额外上下文

    Returns:
        构建后的 Prompt
    """
    prompt = f"""请解说本轮走棋：
{move_description}

根据你的角色风格，给出简短解说。"""

    if context:
        prompt += f"\n\n额外信息：{context}"

    return prompt


# ========== 会话模式 Prompt ==========

# 立即反馈（固定回复，无 LLM）
IMMEDIATE_REPLIES = [
    "好的，让我想想...",
    "收到，我来走棋~",
    "好，让我看看怎么走",
    "嗯，让我算算...",
]


def get_error_response_prompt(
    error_type: str,
    error_detail: str,
    character_name: str = "新手导师"
) -> str:
    """
    生成错误处理 Prompt

    Args:
        error_type: 错误类型（invalid_move, wrong_piece, no_move_detected 等）
        error_detail: 错误详情
        character_name: 角色名称

    Returns:
        Prompt 字符串
    """
    # 中国象棋坐标系统说明
    coord_note = """【中国象棋坐标系统 - 重要】
棋盘是9列(a-i) × 10行(1-10)。红方在下方(行1-5靠近红方)，黑方在上方(行6-10靠近黑方)。

判断红方走法方向：
- 行数增加 = 前进（如行4→行5是前进一步）
- 行数减少 = 后退（如行4→行3是后退一步）
- 列号变化 = 左右移动（a是最左，i是最右）

示例分析：
- e4→d5：行4到行5（前进），e到d（左移），所以是"左前方"前进
- e4→e5：纯前进
- e4→e3：后退
- e4→f4：右移"""

    error_prompts = {
        "invalid_move": f"""用户走了一步不合法的棋（{error_detail}）。
{coord_note}
请以{character_name}的身份，友好地告诉用户问题所在。
如果涉及走法方向判断，请根据坐标系统正确分析。回复不超过50字。""",

        "wrong_piece": f"""用户试图移动黑方棋子，但他只能走红方棋子。
请以{character_name}的身份，友好地提醒用户这一点。
回复不超过40字。""",

        "no_move_detected": f"""用户说"下好了"，但系统没有检测到棋盘上有走棋变化。
请以{character_name}的身份，提醒用户先走棋再通知你。
回复不超过40字。""",

        "not_your_turn": f"""现在轮到AI走棋了，用户不能在这个时候走棋。
请以{character_name}的身份，告诉用户稍等一下。
回复不超过30字。""",

        "game_not_started": f"""游戏还没有开始，用户不能走棋。
请以{character_name}的身份，提醒用户先开始游戏。
回复不超过30字。""",

        "ai_thinking": f"""AI正在思考走法。
请以{character_name}的身份，告诉用户稍等。
回复不超过20字。"""
    }

    return error_prompts.get(error_type, f"用户遇到了问题：{error_detail}。请友好地告诉用户。回复不超过40字。")


def get_move_commentary_prompt(
    user_move: Dict,
    ai_move: Dict,
    board_status: Dict,
    character_name: str = "新手导师"
) -> str:
    """
    生成走棋解说 Prompt（AI 计算完成后立即使用）

    Args:
        user_move: 用户走棋信息 {piece, from, to, captured}
        ai_move: AI 走棋信息 {piece, from, to, captured}
        board_status: 棋盘状态 {advantage, move_count}
        character_name: 角色名称

    Returns:
        Prompt 字符串
    """
    # 构建用户走棋描述
    user_desc = f"用户（红方）走了 {user_move.get('piece', '棋子')} {user_move.get('from', '')} → {user_move.get('to', '')}"
    if user_move.get('captured'):
        user_desc += f"，吃掉了黑方的{user_move.get('captured')}"

    # 构建 AI 走棋描述
    ai_desc = f"AI（黑方）走了 {ai_move.get('piece', '棋子')} {ai_move.get('from', '')} → {ai_move.get('to', '')}"
    if ai_move.get('captured'):
        ai_desc += f"，吃掉了红方的{ai_move.get('captured')}"

    # 局势描述
    advantage_map = {
        "red_big": "红方大优",
        "red_slight": "红方略优",
        "equal": "势均力敌",
        "black_slight": "黑方略优",
        "black_big": "黑方大优"
    }
    advantage = advantage_map.get(board_status.get('advantage', 'equal'), "局势均衡")

    prompt = f"""你是{character_name}，刚刚和用户完成一轮走棋。

走棋情况：
- {user_desc}
- {ai_desc}

当前局势：{advantage}，已走 {board_status.get('move_count', 0)} 步。

请用你的角色风格，边解说边告诉用户你的走棋。回复不超过60字。
可以同时提到用户的走棋和你的应对。"""

    return prompt


def get_listening_prompt(character_name: str = "新手导师") -> str:
    """生成持续监听状态提示"""
    return f"你是{character_name}，正在等待用户说话。保持准备状态。"