"""
棋盘状态管理模块
"""
from typing import List, Dict, Optional, Tuple
import config
import json
import copy


class Piece:
    """棋子类"""

    def __init__(self, class_name: str, color: str, col: int = -1, row: int = -1,
                 center_x: float = None, center_y: float = None):
        self.class_name = class_name
        self.color = color  # 'red' 或 'black'
        self.col = col
        self.row = row
        self.center_x = center_x  # YOLO 检测的像素中心点 X
        self.center_y = center_y  # YOLO 检测的像素中心点 Y

    def to_dict(self) -> dict:
        return {
            'class_name': self.class_name,
            'color': self.color,
            'col': self.col,
            'row': self.row,
            'center_x': self.center_x,
            'center_y': self.center_y
        }

    @staticmethod
    def from_dict(data: dict) -> 'Piece':
        return Piece(
            class_name=data['class_name'],
            color=data['color'],
            col=data.get('col', -1),
            row=data.get('row', -1),
            center_x=data.get('center_x'),
            center_y=data.get('center_y')
        )


class BoardStateManager:
    """棋盘状态管理器"""

    def __init__(self):
        self.pieces: Dict[str, Piece] = {}  # 使用位置作为key: "a1", "b2"等
        self.history: List[Dict] = []  # 走棋历史

    def clear(self):
        """清空棋盘"""
        self.pieces.clear()

    def from_detections(self, detections: List[Dict]):
        """从检测结果更新棋盘状态"""
        self.clear()
        skipped = 0
        for det in detections:
            col = det.get('board_col')
            row = det.get('board_row')
            if col is not None and row is not None:
                pos = self._format_position(col, row)
                center = det.get('center', (None, None))
                # 将 numpy float32 转换为 Python float，确保可 JSON 序列化
                center_x = float(center[0]) if center and center[0] is not None else None
                center_y = float(center[1]) if center and center[1] is not None else None
                piece = Piece(
                    class_name=det['class_name'],
                    color=det['color'],
                    col=col,
                    row=row,
                    center_x=center_x,
                    center_y=center_y
                )
                self.pieces[pos] = piece
            else:
                skipped += 1
        if skipped > 0:
            print(f"[BoardState] 警告: 跳过了 {skipped} 个无法定位的棋子")

    def get_piece_at(self, col: int, row: int) -> Optional[Piece]:
        """获取指定位置的棋子"""
        pos = self._format_position(col, row)
        return self.pieces.get(pos)

    def get_piece_by_name(self, pos_name: str) -> Optional[Piece]:
        """通过位置名称获取棋子"""
        return self.pieces.get(pos_name)

    def set_piece(self, col: int, row: int, piece: Piece):
        """设置指定位置的棋子"""
        pos = self._format_position(col, row)
        piece.col = col
        piece.row = row
        self.pieces[pos] = piece

    def remove_piece(self, col: int, row: int) -> Optional[Piece]:
        """移除指定位置的棋子"""
        pos = self._format_position(col, row)
        return self.pieces.pop(pos, None)

    def move_piece(self, from_col: int, from_row: int, to_col: int, to_row: int) -> bool:
        """移动棋子"""
        piece = self.remove_piece(from_col, from_row)
        if piece:
            captured = self.remove_piece(to_col, to_row)
            self.set_piece(to_col, to_row, piece)

            # 记录历史
            self.history.append({
                'from': self._format_position(from_col, from_row),
                'to': self._format_position(to_col, to_row),
                'piece': piece.class_name,
                'captured': captured.class_name if captured else None
            })
            return True
        return False

    def compare(self, other: 'BoardStateManager') -> Dict:
        """比较两个棋盘状态，检测变化"""
        result = {
            'is_valid_move': False,
            'added': [],
            'removed': [],
            'changed': [],  # 位置不变但棋子变了（吃子）
            'moved': [],
            'from_pos': None,
            'to_pos': None,
            'captured': None
        }

        # 找出新增、移除和变化的位置
        self_positions = set(self.pieces.keys())
        other_positions = set(other.pieces.keys())

        added_positions = self_positions - other_positions
        removed_positions = other_positions - self_positions
        
        # 检查共同位置上的棋子变化（检测吃子）
        common_positions = self_positions & other_positions
        changed_count = 0
        for pos in common_positions:
            self_piece = self.pieces[pos]
            other_piece = other.pieces[pos]
            # 位置相同但棋子不同，说明发生了吃子
            if self_piece.class_name != other_piece.class_name or self_piece.color != other_piece.color:
                result['changed'].append({
                    'pos': pos,
                    'old_piece': other_piece.to_dict(),
                    'new_piece': self_piece.to_dict()
                })
                changed_count += 1
                print(f"[BoardState.compare] 检测到变化位置 {pos}: {other_piece.class_name}({other_piece.color}) -> {self_piece.class_name}({self_piece.color})")

        result['added'] = [self.pieces[pos].to_dict() for pos in added_positions]
        result['removed'] = [other.pieces[pos].to_dict() for pos in removed_positions]

        # 详细调试日志：打印具体位置变化
        if added_positions or removed_positions or result.get('changed'):
            print(f"[BoardState.compare] 详细变化：")
            for pos in added_positions:
                piece = self.pieces[pos]
                print(f"  + 新增: {pos} -> {piece.class_name}({piece.color})")
            for pos in removed_positions:
                piece = other.pieces[pos]
                print(f"  - 移除: {pos} -> {piece.class_name}({piece.color})")

        # 检查普通走棋（无吃子）
        if len(added_positions) == 1 and len(removed_positions) == 1:
            new_pos = list(added_positions)[0]
            old_pos = list(removed_positions)[0]

            added_piece = self.pieces[new_pos]
            removed_piece = other.pieces[old_pos]

            # 检查是否是同一颜色的棋子 (合法走棋)
            if added_piece.color == removed_piece.color:
                result['is_valid_move'] = True
                result['moved'] = [{
                    'from': old_pos,
                    'to': new_pos,
                    'piece': added_piece.class_name
                }]
                result['from_pos'] = old_pos
                result['to_pos'] = new_pos
                result['moving_piece'] = removed_piece.to_dict()  # 移动的棋子信息

        # 检查吃子情况：1 个位置移除（起始位）+ 1 个位置变化（目标位被吃）
        elif len(removed_positions) == 1 and len(result['changed']) == 1:
            old_pos = list(removed_positions)[0]
            changed_pos = result['changed'][0]['pos']
            moved_piece = other.pieces[old_pos]  # 移动的棋子（从 removed 位置，Piece 对象）
            captured_piece = result['changed'][0]['old_piece']  # 被吃的棋子（字典）
            new_piece = result['changed'][0]['new_piece']  # 新位置的棋子（字典）

            # 检查移动的棋子和新位置的棋子是否一致
            # new_piece 是字典，需要用 ['color'] 和 ['class_name']
            if moved_piece.color == new_piece['color'] and moved_piece.class_name == new_piece['class_name']:
                result['is_valid_move'] = True
                result['from_pos'] = old_pos
                result['to_pos'] = changed_pos
                result['captured'] = captured_piece
                result['moving_piece'] = moved_piece.to_dict()  # 移动的棋子信息

        # 检查吃子情况（旧逻辑保留）：added=1, removed=2
        elif len(added_positions) == 1 and len(removed_positions) == 2:
            new_pos = list(added_positions)[0]
            added_piece = self.pieces[new_pos]

            for old_pos in removed_positions:
                old_piece = other.pieces[old_pos]
                if old_piece.color == added_piece.color:
                    # 找到移动方
                    result['is_valid_move'] = True
                    result['from_pos'] = old_pos
                    result['to_pos'] = new_pos
                    result['moving_piece'] = old_piece.to_dict()  # 移动的棋子信息

                    # 找到被吃掉的棋子
                    for other_old_pos in removed_positions:
                        if other_old_pos != old_pos:
                            result['captured'] = other.pieces[other_old_pos].to_dict()

        return result

    def copy(self) -> 'BoardStateManager':
        """复制棋盘状态"""
        new_board = BoardStateManager()
        new_board.pieces = copy.deepcopy(self.pieces)
        new_board.history = copy.deepcopy(self.history)
        return new_board

    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            'pieces': {pos: piece.to_dict() for pos, piece in self.pieces.items()},
            'history': self.history
        }

    @staticmethod
    def from_dict(data: Dict) -> 'BoardStateManager':
        """从字典加载"""
        board = BoardStateManager()
        board.pieces = {pos: Piece.from_dict(p) for pos, p in data.get('pieces', {}).items()}
        board.history = data.get('history', [])
        return board

    def _format_position(self, col: int, row: int) -> str:
        """
        格式化位置为字符串

        坐标系统（与引擎一致）：
        - col: 0-8 对应 a-i
        - row: 0-9（红方底线 row 0，黑方底线 row 9）
        """
        col_names = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']
        if 0 <= col < 9 and 0 <= row <= 9:
            return f"{col_names[col]}{row}"
        return "??"

    def __str__(self) -> str:
        """字符串表示（显示时转换为用户习惯的 1-10 行号）"""
        lines = []
        col_names = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']

        lines.append("  " + " ".join(col_names))
        for row in range(9, -1, -1):  # 从9到0（黑方到红方）
            display_row = row + 1  # 显示为 1-10
            line = f"{display_row:2d}"
            for col in range(9):
                pos = self._format_position(col, row)
                piece = self.pieces.get(pos)
                if piece:
                    # 简写棋子名称
                    name = piece.class_name.split('_')[-1][:2]
                    line += f" {name:2s}"
                else:
                    line += " · "
            lines.append(line)

        return "\n".join(lines)

    def validate_standard_setup(self) -> Tuple[bool, List[str]]:
        """
        验证当前棋盘是否为标准开局阵型
        返回：(是否合法，错误信息列表)
        """
        errors = []

        # 1. 检查棋子总数
        if len(self.pieces) != 32:
            errors.append(f"棋子总数错误：检测到{len(self.pieces)}个，应为 32 个")

        # 2. 检查每个标准位置
        for pos, expected_class in config.STANDARD_SETUP.items():
            expected_name = class_to_chinese(expected_class)
            expected_color = expected_class.split('_')[0]
            
            if pos not in self.pieces:
                errors.append(f"缺少棋子：{pos.upper()} 位置应有{expected_name}")
            elif self.pieces[pos].class_name != expected_class:
                actual = class_to_chinese(self.pieces[pos].class_name)
                errors.append(f"棋子错误：{pos.upper()} 位置是{actual}，应为{expected_name}")
            # 颜色检查
            elif self.pieces[pos].color != expected_color:
                errors.append(f"颜色错误：{pos.upper()} 位置棋子颜色不对")

        # 3. 检查是否有额外棋子
        for pos in self.pieces:
            if pos not in config.STANDARD_SETUP:
                actual = class_to_chinese(self.pieces[pos].class_name)
                errors.append(f"多余棋子：{pos.upper()} 位置有{actual}")

        return len(errors) == 0, errors


def class_to_chinese(class_name: str) -> str:
    """将棋子类名转换为中文"""
    mapping = {
        'red_shuai': '红帅', 'red_shi': '红仕', 'red_xiang': '红相',
        'red_ma': '红马', 'red_che': '红车', 'red_pao': '红炮', 'red_bing': '红兵',
        'black_jiang': '黑将', 'black_shi': '黑士', 'black_xiang': '黑象',
        'black_ma': '黑马', 'black_che': '黑车', 'black_pao': '黑炮', 'black_zu': '黑卒',
    }
    return mapping.get(class_name, class_name)


# 中国象棋 FEN 格式棋子映射
# 红方用大写，黑方用小写
PIECE_TO_FEN = {
    'red_shuai': 'K',    # 帅
    'red_shi': 'A',      # 仕
    'red_xiang': 'B',    # 相
    'red_ma': 'N',       # 马
    'red_che': 'R',      # 车
    'red_pao': 'C',      # 炮
    'red_bing': 'P',     # 兵
    'black_jiang': 'k',  # 将
    'black_shi': 'a',    # 士
    'black_xiang': 'b',  # 象
    'black_ma': 'n',     # 马
    'black_che': 'r',    # 车
    'black_pao': 'c',    # 炮
    'black_zu': 'p',     # 卒
}


def to_fen(board_state: dict, current_player: str = 'red') -> str:
    """
    将棋盘状态转换为中国象棋 FEN 格式

    坐标系统（与引擎一致）：
    - row 0 = 红方底线，row 9 = 黑方底线
    - FEN 从上(黑方)到下(红方)，即 row 9 到 row 0

    Args:
        board_state: 棋盘状态字典，包含 'pieces' 键
        current_player: 当前走棋方 ('red' 或 'black')

    Returns:
        FEN 格式字符串
    """
    pieces = board_state.get('pieces', {})
    col_names = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']

    # 构建棋盘布局（从 row 9 到 row 0，即从黑方到红方）
    board_lines = []

    for row in range(9, -1, -1):  # 从9到0
        line = ''
        empty_count = 0

        for col in range(9):  # 从a到i
            pos = f"{col_names[col]}{row}"

            if pos in pieces:
                # 有棋子，先输出空格计数
                if empty_count > 0:
                    line += str(empty_count)
                    empty_count = 0

                # 输出棋子
                piece = pieces[pos]
                class_name = piece.get('class_name', '')
                fen_char = PIECE_TO_FEN.get(class_name, '?')
                line += fen_char
            else:
                empty_count += 1

        # 行末空格
        if empty_count > 0:
            line += str(empty_count)

        board_lines.append(line)

    # 合并棋盘布局
    board_fen = '/'.join(board_lines)

    # 当前走棋方（红方用 'w'，黑方用 'b'）
    turn = 'w' if current_player == 'red' else 'b'

    # 中国象棋 FEN 格式：棋盘布局 走棋方
    fen = f"{board_fen} {turn}"

    return fen

