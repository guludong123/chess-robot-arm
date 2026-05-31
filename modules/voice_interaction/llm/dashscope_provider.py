"""
阿里云 DashScope LLM 提供者 (Qwen3.5-Flash)

支持：
- 解说生成
- 命令解析
- 对话生成（角色化）
"""
import asyncio
import json
from typing import Dict, Optional, List
import dashscope
from dashscope import Generation
from .base import LLMProviderBase
from .prompts import (
    get_commentary_prompt,
    get_command_prompt,
    get_dialogue_prompt,
    build_dialogue_messages,
    get_character_commentary_prompt,
    COMMENTARY_SYSTEM_PROMPT,
    COMMAND_SYSTEM_PROMPT,
    DEFAULT_DIALOGUE_SYSTEM_PROMPT
)
import config


class DashScopeLLMProvider(LLMProviderBase):
    """阿里云 DashScope API LLM 提供者"""

    def __init__(
        self,
        api_key: str = None,
        model: str = None
    ):
        self.api_key = api_key or config.DASHSCOPE_API_KEY
        self.model = model or config.DASHSCOPE_MODEL
        dashscope.api_key = self.api_key

    @property
    def name(self) -> str:
        return f"DashScope ({self.model})"

    async def generate_commentary(
        self,
        board_state: Dict,
        human_move: Dict,
        ai_move: Dict,
        move_history: list,
        context: Optional[Dict] = None,
        character_system_prompt: str = None
    ) -> str:
        """生成棋局解说（生动风格，包含局势分析和全局态势）"""

        # 分析局势（包含分数变化）
        situation = self._analyze_situation_enhanced(ai_move, human_move)

        # 分析棋盘态势（威胁、焦点区域）
        board_analysis = self._analyze_board_situation(board_state, human_move, ai_move)

        # 分析全局棋子状况（阶段、子力统计）
        board_global = self._analyze_board_global(board_state, move_history)

        # 构建生动的走棋描述
        move_desc = self._describe_moves_vivid(human_move, ai_move, situation)

        # 使用角色系统提示词或默认
        system_prompt = character_system_prompt or COMMENTARY_SYSTEM_PROMPT

        # 获取 Prompt（传入局势信息和态势分析）
        prompt = get_commentary_prompt(
            move_description=move_desc,
            situation=situation,
            move_history=move_history[-5:] if move_history else [],
            context=context,
            board_analysis=board_analysis,
            board_global=board_global
        )

        # 调用 DashScope API
        try:
            print(f"[DashScopeLLM] 调用API生成解说...")
            response = await asyncio.to_thread(
                self._call_api,
                system_prompt,
                prompt
            )

            if response is None:
                print(f"[DashScopeLLM] API 返回 None，使用备用解说")
                return self._fallback_commentary(human_move, ai_move)

            print(f"[DashScopeLLM] API status_code: {response.status_code}")

            if response.status_code == 200:
                result = response.output.text.strip()
                print(f"[DashScopeLLM] 解说生成成功，长度: {len(result)}")
                return result
            else:
                error_msg = getattr(response, 'message', '未知错误')
                error_code = getattr(response, 'code', 'N/A')
                print(f"[DashScopeLLM] API 调用失败: code={error_code}, message={error_msg}")
                return self._fallback_commentary(human_move, ai_move)

        except Exception as e:
            print(f"[DashScopeLLM] 生成解说异常: {e}")
            import traceback
            traceback.print_exc()
            return self._fallback_commentary(human_move, ai_move)

    def _analyze_situation_enhanced(self, ai_move: Dict, human_move: Dict = None) -> Dict:
        """
        增强版局势分析（包含分数变化幅度）

        Args:
            ai_move: AI走棋信息，包含 score_cp
            human_move: 人类走棋信息，包含 before_score_cp（可选）

        Returns:
            局势分析结果
        """
        score_cp = ai_move.get('score_cp') if ai_move else None
        before_score = human_move.get('before_score_cp') if human_move else None

        # 处理 None 值（如将死情况下 score_cp 为 None）
        if score_cp is None:
            score_cp = 0
        if before_score is None:
            before_score = 0

        score_change = score_cp - before_score

        # 判断优势方（score_cp 从黑方视角）
        if score_cp > 300:
            advantage = "黑方大优"
        elif score_cp > 100:
            advantage = "黑方占优"
        elif score_cp > 30:
            advantage = "黑方略优"
        elif score_cp < -300:
            advantage = "红方大优"
        elif score_cp < -100:
            advantage = "红方占优"
        elif score_cp < -30:
            advantage = "红方略优"
        else:
            advantage = "均势"

        # 分数变化解读
        change_desc = ""
        if abs(score_change) > 200:
            change_desc = "局势剧变"
        elif abs(score_change) > 50:
            change_desc = "明显变化"
        elif abs(score_change) > 20:
            change_desc = "小幅波动"

        return {
            'score_cp': score_cp,
            'advantage': advantage,
            'score_change': score_change,
            'change_desc': change_desc,
            'is_checkmate': ai_move.get('is_checkmate', False) if ai_move else False,
            'mate_in': ai_move.get('mate_in')
        }

    def _analyze_board_situation(self, board_state: Dict, human_move: Dict, ai_move: Dict) -> Dict:
        """
        分析棋盘全局态势（威胁、焦点区域、引擎意图）

        Args:
            board_state: 棋盘状态（包含所有棋子位置）
            human_move: 人类走棋
            ai_move: AI走棋（包含 PV 信息）

        Returns:
            棋盘态势分析
        """
        threats = []
        key_areas = []
        engine_intent = []  # 引擎意图分析（基于 PV）

        # 分析吃子威胁
        if ai_move and ai_move.get('captured'):
            captured_piece = self._piece_to_chinese_name(
                ai_move.get('captured', '').split('_')[-1]
            )
            threats.append(f"红方{captured_piece}被吃")

        if human_move and human_move.get('captured'):
            captured_piece = self._piece_to_chinese(human_move.get('captured'))
            threats.append(f"黑方{captured_piece}被吃")

        # 分析焦点区域（走棋位置附近）
        if human_move:
            to_pos = human_move.get('to_pos', '')
            if to_pos:
                col = to_pos[0]
                key_areas.append(f"{col}路")

        if ai_move:
            to_pos = ai_move.get('to', '')
            if to_pos:
                col = to_pos[0]
                key_areas.append(f"{col}路")

        # 【核心新增】基于 PV 分析引擎真实意图
        if ai_move and ai_move.get('pv'):
            engine_intent = self._analyze_pv_intent(ai_move.get('pv', []), board_state)

        # 分析特殊战术（简单的启发式分析）
        if ai_move:
            piece_name = ai_move.get('piece', '').split('_')[-1]

            # 马的威胁位置（卧槽马、挂角马）
            if piece_name == 'ma':
                to_pos = ai_move.get('to', '')
                if to_pos:
                    row = int(to_pos[1:])
                    # 马靠近帅/将位置
                    if row <= 2:
                        threats.append("马逼近帅位")
                    elif row >= 8:
                        threats.append("马逼近将位")

            # 车的威胁
            if piece_name == 'che':
                to_pos = ai_move.get('to', '')
                if to_pos:
                    # 车进入中路或底线
                    col = to_pos[0]
                    if col in ['d', 'e', 'f']:
                        threats.append("车占中路")
                    row = int(to_pos[1:])
                    if row <= 1:
                        threats.append("车压底线")

            # 炮的威胁
            if piece_name == 'pao':
                # 炮平移形成担子炮
                from_pos = ai_move.get('from', '')
                to_pos = ai_move.get('to', '')
                if from_pos and to_pos and from_pos[0] != to_pos[0]:
                    # 炮横向移动
                    threats.append("炮调整位置")

        return {
            'threats': threats,
            'key_areas': key_areas[:2],  # 最多显示2个焦点区域
            'has_capture': bool(ai_move.get('captured') or (human_move and human_move.get('captured'))),
            'engine_intent': engine_intent  # 引擎意图分析
        }

    def _analyze_board_global(self, board_state: Dict, move_history: list) -> Dict:
        """
        分析全局棋子状况：存活棋子统计、棋局阶段
        """
        pieces = board_state.get('pieces', {})

        # 统计双方存活棋子
        red_pieces = {}
        black_pieces = {}
        for pos, piece_info in pieces.items():
            color = piece_info.get('color', '')
            class_name = piece_info.get('class_name', '')
            piece_type = class_name.split('_')[-1] if class_name else ''
            if color == 'red':
                red_pieces[piece_type] = red_pieces.get(piece_type, 0) + 1
            elif color == 'black':
                black_pieces[piece_type] = black_pieces.get(piece_type, 0) + 1

        PIECE_VALUES = {
            'shuai': 10000, 'jiang': 10000,
            'che': 900, 'ma': 400, 'pao': 450,
            'xiang': 200, 'shi': 200,
            'bing': 100, 'zu': 100,
        }

        red_value = sum(PIECE_VALUES.get(pt, 0) * cnt for pt, cnt in red_pieces.items())
        black_value = sum(PIECE_VALUES.get(pt, 0) * cnt for pt, cnt in black_pieces.items())

        # 存活大子简述
        MAJOR_PIECES = {'che': '车', 'ma': '马', 'pao': '炮'}
        red_major = []
        black_major = []
        for pt, cn in MAJOR_PIECES.items():
            if red_pieces.get(pt, 0) > 0:
                red_major.append(f"{cn}{red_pieces[pt]}" if red_pieces[pt] > 1 else cn)
            if black_pieces.get(pt, 0) > 0:
                black_major.append(f"{cn}{black_pieces[pt]}" if black_pieces[pt] > 1 else cn)

        # 棋局阶段判断
        total_major = sum(red_pieces.get(p, 0) + black_pieces.get(p, 0) for p in ['che', 'ma', 'pao'])
        move_count = len(move_history) if move_history else 0

        if move_count <= 10 and total_major >= 8:
            phase = "开局"
        elif total_major <= 4 or (red_value + black_value < 3000):
            phase = "残局"
        else:
            phase = "中局"

        pieces_desc_parts = []
        if red_major:
            pieces_desc_parts.append(f"红: {''.join(red_major)}")
        if black_major:
            pieces_desc_parts.append(f"黑: {''.join(black_major)}")

        return {
            'phase': phase,
            'red_major': red_major,
            'black_major': black_major,
            'red_value': red_value,
            'black_value': black_value,
            'pieces_summary': "，".join(pieces_desc_parts),
        }

    def _analyze_pv_intent(self, pv: List[str], board_state: Dict) -> List[str]:
        """
        基于 PV 序列分析引擎真实意图

        PV (Principal Variation) 是引擎认为后续会发生的最佳走法序列，
        通过分析 PV 可以推断引擎的战术意图。

        Args:
            pv: PV 走法序列，如 ['e2e4', 'e7e5', 'g1f3']
            board_state: 棋盘状态

        Returns:
            引擎意图描述列表
        """
        intents = []

        if not pv or len(pv) < 2:
            return intents

        # 分析 PV 中的走法模式
        try:
            # 1. 检查是否有将军意图（PV 中目标位置靠近将/帅）
            for i, move in enumerate(pv[:3]):  # 只看前3步
                if len(move) >= 4:
                    to_pos = move[2:4]
                    to_row = int(to_pos[1])

                    # 黑方走法（偶数索引 0, 2, 4...），目标是红方区域
                    if i % 2 == 0 and to_row <= 2:
                        intents.append("逼近帅位")
                        break
                    # 红方走法（奇数索引），目标是黑方区域
                    elif i % 2 == 1 and to_row >= 8:
                        intents.append("威胁将位")
                        break

            # 2. 检查是否有连续进攻（多个走法都向前推进）
            forward_moves = 0
            for i, move in enumerate(pv[:4]):
                if len(move) >= 4:
                    from_row = int(move[1])
                    to_row = int(move[3])

                    # 黑方向红方推进（row 变小）
                    if i % 2 == 0 and to_row < from_row:
                        forward_moves += 1
                    # 红方向黑方推进（row 变大）
                    elif i % 2 == 1 and to_row > from_row:
                        forward_moves += 1

            if forward_moves >= 2:
                intents.append("连续进攻")

            # 3. 检查是否有防守意图（退回己方区域）
            backward_moves = 0
            for i, move in enumerate(pv[:4]):
                if len(move) >= 4:
                    from_row = int(move[1])
                    to_row = int(move[3])

                    # 黑方后退（row 变大）
                    if i % 2 == 0 and to_row > from_row:
                        backward_moves += 1
                    # 红方后退（row 变小）
                    elif i % 2 == 1 and to_row < from_row:
                        backward_moves += 1

            if backward_moves >= 2:
                intents.append("防守调整")

            # 4. 检查中路争夺（多个走法涉及 d/e/f 列）
            center_moves = 0
            for move in pv[:3]:
                if len(move) >= 4:
                    from_col = move[0]
                    to_col = move[2]
                    if from_col in ['d', 'e', 'f'] or to_col in ['d', 'e', 'f']:
                        center_moves += 1

            if center_moves >= 2:
                intents.append("中路争夺")

            # 5. 检查是否有兑子意图（走法是吃子且位置互换）
            # PV 中第1步是黑方吃子，第2步是红方吃回同一位置
            if len(pv) >= 2:
                move1 = pv[0]
                move2 = pv[1]
                if len(move1) >= 4 and len(move2) >= 4:
                    # 如果红方吃回黑方刚吃掉的位置
                    if move2[2:4] == move1[2:4]:
                        intents.append("预期兑子")

        except (ValueError, IndexError) as e:
            print(f"[DashScopeLLM] PV 分析失败: {e}")

        return intents[:3]  # 最多返回3个意图描述

    def _describe_moves_vivid(self, human_move: Dict, ai_move: Dict, situation: Dict) -> str:
        """
        生动版走棋描述（加入战术意图推测）

        Args:
            human_move: 人类走棋
            ai_move: AI走棋
            situation: 局势分析

        Returns:
            生动的走棋描述
        """
        desc = []

        if human_move:
            piece = self._piece_to_chinese(human_move.get('moving_piece', {}))
            from_pos = human_move.get('from_pos', '')
            to_pos = human_move.get('to_pos', '')
            captured = human_move.get('captured')

            # 红方棋子
            direction = self._describe_move_direction(from_pos, to_pos, is_red=True)
            move_str = f"红{piece}{direction}"

            if captured:
                captured_name = self._piece_to_chinese(captured)
                tier = self._piece_value_tier(captured)
                move_str += f"吃{captured_name}"
                if tier:
                    move_str += f"({tier})"

            # 加入意图推测（简单启发式）
            intent = self._guess_move_intent(human_move, 'red')
            if intent:
                move_str += f"，{intent}"

            desc.append(move_str)

        if ai_move:
            piece = self._piece_to_chinese_name(ai_move.get('piece', ''))
            from_pos = ai_move.get('from', '')
            to_pos = ai_move.get('to', '')
            captured = ai_move.get('captured')
            pv = ai_move.get('pv', [])  # 引擎 PV 序列

            # 黑方棋子
            direction = self._describe_move_direction(from_pos, to_pos, is_red=False)
            move_str = f"黑{piece}{direction}"

            if captured:
                captured_name = self._piece_to_chinese(captured)
                tier = self._piece_value_tier(captured)
                move_str += f"吃{captured_name}"
                if tier:
                    move_str += f"({tier})"

            # 加入意图：优先使用引擎 PV 分析，否则用启发式
            if pv and len(pv) >= 2:
                # 使用引擎 PV 分析的意图（更准确）
                pv_intent = self._get_intent_from_pv(pv)
                if pv_intent:
                    move_str += f"，{pv_intent}"
            else:
                # 回退到启发式意图
                intent = self._guess_move_intent(ai_move, 'black')
                if intent:
                    move_str += f"，{intent}"

            desc.append(move_str)

        return "；".join(desc)

    def _get_intent_from_pv(self, pv: List[str]) -> str:
        """
        从 PV 序列提取一句话描述意图

        Args:
            pv: PV 走法序列

        Returns:
            意图描述（一句话）
        """
        if not pv or len(pv) < 2:
            return ""

        try:
            # 检查 PV 中是否有进攻意图
            forward_moves = 0
            check_intent = False

            for i, move in enumerate(pv[:3]):
                if len(move) >= 4:
                    from_row = int(move[1])
                    to_row = int(move[3])
                    to_col = move[2]

                    # 黑方走法（索引 0, 2...），向红方推进
                    if i % 2 == 0:
                        if to_row < from_row:
                            forward_moves += 1
                        # 检查是否接近帅位
                        if to_row <= 2:
                            check_intent = True

                    # 检查是否涉及中路
                    if to_col in ['d', 'e', 'f']:
                        forward_moves += 0.5

            # 根据分析返回意图描述
            if check_intent:
                return "意图攻帅"
            elif forward_moves >= 2:
                return "酝酿进攻"
            elif forward_moves >= 1:
                return "调整部署"

            # 检查是否有兑子意图
            if len(pv) >= 2:
                move1 = pv[0]
                move2 = pv[1]
                if len(move1) >= 4 and len(move2) >= 4:
                    if move2[2:4] == move1[2:4]:
                        return "准备兑子"

        except (ValueError, IndexError):
            pass

        return ""

    def _guess_move_intent(self, move: Dict, player: str) -> str:
        """
        推测走棋意图（简单启发式）

        Args:
            move: 走棋信息
            player: 走棋方 ('red' 或 'black')

        Returns:
            意图描述（可选）
        """
        if not move:
            return ""

        piece_name = ""
        if move.get('moving_piece'):
            piece_name = move.get('moving_piece', {}).get('class_name', '').split('_')[-1]
        elif move.get('piece'):
            piece_name = move.get('piece', '').split('_')[-1]

        to_pos = move.get('to_pos') or move.get('to', '')
        captured = move.get('captured')

        # 吃子意图
        if captured:
            captured_name = self._piece_to_chinese(captured)
            if '将' in captured_name or '帅' in captured_name:
                return "威胁主将"
            elif '车' in captured_name:
                return "兑换主力"
            return ""

        # 位置意图
        if to_pos and len(to_pos) >= 2:
            col = to_pos[0]
            row = int(to_pos[1:])

            # 中路控制
            if col in ['d', 'e', 'f']:
                if piece_name == 'che':
                    return "控制中路"
                elif piece_name == 'pao':
                    return "中路施压"

            # 进攻意图（进入对方腹地）
            if player == 'red' and row >= 5:
                return "深入敌阵"
            if player == 'black' and row <= 4:
                return "逼近帅位"

            # 防守意图（退回己方区域）
            if player == 'red' and row <= 2:
                return "回防"
            if player == 'black' and row >= 8:
                return "巩固防线"

        return ""

    def _analyze_situation(self, ai_move: Dict) -> Dict:
        """
        分析当前局势（基于引擎评估分数）

        Args:
            ai_move: AI走棋信息，包含 score_cp

        Returns:
            局势分析结果
        """
        score_cp = ai_move.get('score_cp', 0) if ai_move else 0

        # 判断优势方（score_cp 从黑方视角）
        if score_cp > 300:
            advantage = "黑方大优"
        elif score_cp > 100:
            advantage = "黑方占优"
        elif score_cp > 30:
            advantage = "黑方略优"
        elif score_cp < -300:
            advantage = "红方大优"
        elif score_cp < -100:
            advantage = "红方占优"
        elif score_cp < -30:
            advantage = "红方略优"
        else:
            advantage = "均势"

        return {
            'score_cp': score_cp,
            'advantage': advantage,
            'is_checkmate': ai_move.get('is_checkmate', False) if ai_move else False,
            'mate_in': ai_move.get('mate_in')
        }

    def _describe_moves_enhanced(self, human_move: Dict, ai_move: Dict) -> str:
        """
        增强版走棋描述（简洁专业）

        使用象棋术语描述走法，而不是机械的坐标
        """
        desc = []

        if human_move:
            piece = self._piece_to_chinese(human_move.get('moving_piece', {}))
            from_pos = human_move.get('from_pos', '')
            to_pos = human_move.get('to_pos', '')
            captured = human_move.get('captured')

            # 红方棋子，传入 is_red=True
            move_str = f"红{piece}{self._describe_move_direction(from_pos, to_pos, is_red=True)}"
            if captured:
                captured_name = self._piece_to_chinese(captured)
                move_str += f"吃{captured_name}"
            desc.append(move_str)

        if ai_move:
            piece = self._piece_to_chinese_name(ai_move.get('piece', ''))
            from_pos = ai_move.get('from', '')
            to_pos = ai_move.get('to', '')
            captured = ai_move.get('captured')

            # 黑方棋子，传入 is_red=False
            move_str = f"黑{piece}{self._describe_move_direction(from_pos, to_pos, is_red=False)}"
            if captured:
                captured_name = self._piece_to_chinese(captured)
                move_str += f"吃{captured_name}"
            desc.append(move_str)

        return "；".join(desc)

    def _describe_move_direction(self, from_pos: str, to_pos: str, is_red: bool = True) -> str:
        """
        用象棋术语描述走法方向

        Args:
            from_pos: 起始位置，如 "e4"
            to_pos: 目标位置，如 "e5"
            is_red: 是否是红方棋子
                    红方在下方，"进"是 row 数字变大
                    黑方在上方，"进"是 row 数字变小

        Returns:
            走法描述，如 "进一"、"平二"、"退一"

        中国象棋规则：
            - 红方从下往上走是"进"（row 数字变大）
            - 黑方从上往下走是"进"（row 数字变小）
        """
        if not from_pos or not to_pos or len(from_pos) < 2 or len(to_pos) < 2:
            return "移动"

        from_col = from_pos[0]
        to_col = to_pos[0]
        from_row = int(from_pos[1:])
        to_row = int(to_pos[1:])

        row_diff = to_row - from_row
        col_diff = ord(to_col) - ord(from_col)

        # 判断走法类型
        if col_diff == 0:
            # 同一列：进/退（区分红黑方）
            if is_red:
                # 红方：row_diff > 0 是进（往上走）
                if row_diff > 0:
                    return f"进{abs(row_diff)}"
                elif row_diff < 0:
                    return f"退{abs(row_diff)}"
            else:
                # 黑方：row_diff < 0 是进（往下走）
                if row_diff < 0:
                    return f"进{abs(row_diff)}"
                elif row_diff > 0:
                    return f"退{abs(row_diff)}"
            return ""
        elif row_diff == 0:
            # 同一行：平
            return f"平{abs(col_diff)}"
        else:
            # 斜走（马、象等）：进/退
            if is_red:
                direction = "进" if row_diff > 0 else "退"
            else:
                direction = "进" if row_diff < 0 else "退"
            return direction

    def _call_api(self, system_prompt: str, user_prompt: str, max_tokens: int = None, temperature: float = None):
        """同步调用 API"""
        try:
            # 使用传入参数或默认配置
            max_tokens = max_tokens or config.LLM_MAX_TOKENS_COMMENTARY
            temperature = temperature or config.LLM_TEMPERATURE_COMMENTARY

            response = Generation.call(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=max_tokens,
                temperature=temperature,
                result_format='text'
            )
            return response
        except Exception as e:
            print(f"[DashScopeLLM] API调用异常: {e}")
            return None

    async def process_voice_command(
        self,
        transcript: str,
        game_state: Dict
    ) -> Dict:
        """处理语音命令"""

        prompt = get_command_prompt(transcript, game_state)

        try:
            response = await asyncio.to_thread(
                self._call_api,
                COMMAND_SYSTEM_PROMPT,
                prompt
            )

            if response and response.status_code == 200:
                result_text = response.output.text.strip()

                # 尝试解析 JSON 格式结果
                try:
                    # 尝试提取 JSON
                    if '{' in result_text and '}' in result_text:
                        json_str = result_text[result_text.find('{'):result_text.rfind('}')+1]
                        return json.loads(json_str)
                except json.JSONDecodeError:
                    pass

                return {
                    'action': 'unknown',
                    'params': {},
                    'response': result_text
                }

        except Exception as e:
            print(f"[DashScopeLLM] 处理命令失败: {e}")

        return {
            'action': 'error',
            'params': {},
            'response': '抱歉，我无法理解您的命令。'
        }

    def _describe_moves(self, human_move: Dict, ai_move: Dict) -> str:
        """描述走棋"""
        desc = []

        if human_move:
            piece = self._piece_to_chinese(human_move.get('moving_piece', {}))
            from_pos = human_move.get('from_pos', '')
            to_pos = human_move.get('to_pos', '')
            captured = human_move.get('captured')

            move_str = f"红方{piece}从{from_pos}移动到{to_pos}"
            if captured:
                captured_name = self._piece_to_chinese(captured)
                move_str += f"，吃掉黑方{captured_name}"
            desc.append(move_str)

        if ai_move:
            piece = self._piece_to_chinese_name(ai_move.get('piece', ''))
            from_pos = ai_move.get('from', '')
            to_pos = ai_move.get('to', '')

            move_str = f"黑方{piece}从{from_pos}移动到{to_pos}"

            # 添加 AI 吃子信息
            captured = ai_move.get('captured')
            if captured:
                captured_name = self._piece_to_chinese(captured)
                move_str += f"，吃掉红方{captured_name}"

            desc.append(move_str)

        return "；".join(desc)

    def _piece_to_chinese(self, piece_info: Dict) -> str:
        """棋子信息转中文"""
        if not piece_info:
            return "棋子"
        class_name = piece_info.get('class_name', '')
        return self._piece_to_chinese_name(class_name.split('_')[-1] if class_name else '')

    def _piece_to_chinese_name(self, piece_name: str) -> str:
        """棋子名称转中文"""
        mapping = {
            'shuai': '帅', 'jiang': '将',
            'shi': '仕', 'xiang': '相',
            'ma': '马', 'che': '车',
            'pao': '炮', 'bing': '兵', 'zu': '卒'
        }
        return mapping.get(piece_name, piece_name or '棋子')

    def _piece_value_tier(self, piece_info: Dict) -> str:
        """返回被吃棋子的价值等级：大子/中子/小子"""
        if not piece_info:
            return ""
        class_name = piece_info.get('class_name', '')
        piece_type = class_name.split('_')[-1] if class_name else ''
        if piece_type in ('che',):
            return "大子"
        elif piece_type in ('ma', 'pao'):
            return "中子"
        elif piece_type in ('bing', 'zu'):
            return "小子"
        return ""

    def _fallback_commentary(self, human_move: Dict, ai_move: Dict) -> str:
        """LLM 失败时的备用解说"""
        return self._describe_moves(human_move, ai_move) + "。"

    # ========== 对话生成方法 ==========

    async def generate_dialogue_response(
        self,
        transcript: str,
        history: List[Dict],
        game_context: Dict,
        character_system_prompt: str = None
    ) -> str:
        """
        生成对话回复

        Args:
            transcript: 用户输入文本
            history: 对话历史 [{"role": "user/assistant", "content": "..."}]
            game_context: 游戏上下文
            character_system_prompt: 角色系统提示词（可选）

        Returns:
            AI 回复文本
        """
        # 使用角色 Prompt 或默认
        system_prompt = character_system_prompt or DEFAULT_DIALOGUE_SYSTEM_PROMPT

        # 构建消息列表
        messages = build_dialogue_messages(
            system_prompt=system_prompt,
            history=history,
            current_input=transcript
        )

        try:
            response = await asyncio.to_thread(
                self._call_chat_api,
                messages,
                max_tokens=config.LLM_MAX_TOKENS_DIALOGUE,
                temperature=config.LLM_TEMPERATURE_DIALOGUE
            )

            if response and response.status_code == 200:
                return response.output.text.strip()
            else:
                error_msg = response.message if response else "未知错误"
                print(f"[DashScopeLLM] 对话生成失败: {error_msg}")
                return "抱歉，我暂时无法回答这个问题。"

        except Exception as e:
            print(f"[DashScopeLLM] 对话生成异常: {e}")
            return "抱歉，我遇到了一些问题。"

    def _call_chat_api(
        self,
        messages: List[Dict],
        max_tokens: int = None,
        temperature: float = None
    ):
        """
        调用对话 API

        Args:
            messages: 消息列表
            max_tokens: 最大 token 数
            temperature: 温度参数

        Returns:
            API 响应
        """
        return Generation.call(
            model=self.model,
            messages=messages,
            max_tokens=max_tokens or config.LLM_MAX_TOKENS,
            temperature=temperature or config.LLM_TEMPERATURE,
            result_format='text'
        )

    async def generate_character_commentary(
        self,
        board_state: Dict,
        human_move: Dict,
        ai_move: Dict,
        character_system_prompt: str
    ) -> str:
        """
        生成角色化解说

        Args:
            board_state: 棋盘状态
            human_move: 人类走棋
            ai_move: AI 走棋
            character_system_prompt: 角色系统提示词

        Returns:
            角色化的解说文本
        """
        move_desc = self._describe_moves(human_move, ai_move)
        prompt = get_character_commentary_prompt(
            move_description=move_desc,
            character_prompt=character_system_prompt
        )

        try:
            response = await asyncio.to_thread(
                self._call_api,
                character_system_prompt,
                prompt
            )

            if response and response.status_code == 200:
                return response.output.text.strip()
            else:
                return self._fallback_commentary(human_move, ai_move)

        except Exception as e:
            print(f"[DashScopeLLM] 角色解说失败: {e}")
            return self._fallback_commentary(human_move, ai_move)

    # ========== 会话模式方法 ==========

    async def generate_error_response(
        self,
        error_type: str,
        error_detail: str,
        character_name: str = "新手导师"
    ) -> str:
        """
        生成错误回复（友好提示用户）

        Args:
            error_type: 错误类型
            error_detail: 错误详情
            character_name: 角色名称

        Returns:
            友好的错误提示文本
        """
        from .prompts import get_error_response_prompt

        prompt = get_error_response_prompt(error_type, error_detail, character_name)

        try:
            response = await asyncio.to_thread(
                self._call_api,
                DEFAULT_DIALOGUE_SYSTEM_PROMPT,
                prompt
            )

            if response and response.status_code == 200:
                return response.output.text.strip()
            else:
                # 备用回复
                fallback_responses = {
                    "invalid_move": "这步棋不太对哦，请检查一下规则~",
                    "wrong_piece": "这是黑方的棋子，你只能走红方棋子哦~",
                    "no_move_detected": "我没看到你走棋呢，走好了再告诉我~",
                    "not_your_turn": "等等，现在轮到AI走了~",
                    "game_not_started": "游戏还没开始呢，先开始游戏吧~"
                }
                return fallback_responses.get(error_type, "遇到了一点问题，请重试~")

        except Exception as e:
            print(f"[DashScopeLLM] 错误回复生成失败: {e}")
            return "遇到了一点问题，请重试~"

    async def generate_move_commentary(
        self,
        user_move: Dict,
        ai_move: Dict,
        board_status: Dict,
        character_name: str = "新手导师"
    ) -> str:
        """
        生成走棋解说（AI 计算完成后立即使用）

        Args:
            user_move: 用户走棋信息
            ai_move: AI 走棋信息
            board_status: 棋盘状态
            character_name: 角色名称

        Returns:
            解说文本
        """
        from .prompts import get_move_commentary_prompt

        prompt = get_move_commentary_prompt(user_move, ai_move, board_status, character_name)

        try:
            response = await asyncio.to_thread(
                self._call_api,
                DEFAULT_DIALOGUE_SYSTEM_PROMPT,
                prompt
            )

            if response and response.status_code == 200:
                return response.output.text.strip()
            else:
                # 备用解说
                user_piece = user_move.get('piece', '棋子')
                ai_piece = ai_move.get('piece', '棋子')
                return f"你走了{user_piece}，我走{ai_piece}。"

        except Exception as e:
            print(f"[DashScopeLLM] 走棋解说生成失败: {e}")
            user_piece = user_move.get('piece', '棋子')
            ai_piece = ai_move.get('piece', '棋子')
            return f"你走了{user_piece}，我走{ai_piece}。"