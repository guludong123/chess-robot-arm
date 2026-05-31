"""
象棋AI模块 - 基于Minimax + Alpha-Beta剪枝的搜索引擎
支持使用 Pikafish UCI 引擎提升棋力
"""
from typing import List, Dict, Optional, Tuple
import random
import time
from config import AI_SEARCH_DEPTH, AI_TIME_LIMIT

# 引擎相关导入
try:
    from modules.uci_engine import get_engine
    ENGINE_AVAILABLE = True
except ImportError as e:
    ENGINE_AVAILABLE = False
    print(f"[ChessAI] 警告: UCI 引擎模块未加载 ({e})，将使用内置 Minimax")


class ChineseChessAI:
    """中国象棋AI - 支持引擎模式和 Minimax 模式"""

    # 使用引擎模式（默认开启，棋力更强）
    use_engine = True

    # 棋子基本分值
    PIECE_VALUES = {
        'shuai': 10000, 'jiang': 10000,  # 将帅
        'che': 900,                       # 车
        'ma': 400,                        # 马
        'pao': 450,                       # 炮
        'xiang': 200,                     # 相/象
        'shi': 200,                       # 士/仕
        'bing': 100, 'zu': 100,           # 兵/卒
    }

    # 位置价值表 (从红方视角，黑方需要翻转)
    # 车 - 中路价值更高
    ROOK_POSITION_VALUE = [
        [14, 14, 12, 18, 16, 18, 12, 14, 14],
        [16, 20, 18, 24, 26, 24, 18, 20, 16],
        [12, 12, 12, 18, 18, 18, 12, 12, 12],
        [12, 18, 16, 22, 22, 22, 16, 18, 12],
        [12, 14, 12, 18, 18, 18, 12, 14, 12],
        [12, 14, 12, 18, 18, 18, 12, 14, 12],
        [12, 18, 16, 22, 22, 22, 16, 18, 12],
        [12, 12, 12, 18, 18, 18, 12, 12, 12],
        [16, 20, 18, 24, 26, 24, 18, 20, 16],
        [14, 14, 12, 18, 16, 18, 12, 14, 14],
    ]

    # 马 - 中心位置价值更高
    HORSE_POSITION_VALUE = [
        [4, 8, 16, 12, 4, 12, 16, 8, 4],
        [4, 10, 28, 16, 8, 16, 28, 10, 4],
        [12, 14, 16, 20, 18, 20, 16, 14, 12],
        [8, 24, 18, 24, 20, 24, 18, 24, 8],
        [6, 16, 14, 18, 16, 18, 14, 16, 6],
        [6, 16, 14, 18, 16, 18, 14, 16, 6],
        [8, 24, 18, 24, 20, 24, 18, 24, 8],
        [12, 14, 16, 20, 18, 20, 16, 14, 12],
        [4, 10, 28, 16, 8, 16, 28, 10, 4],
        [4, 8, 16, 12, 4, 12, 16, 8, 4],
    ]

    # 炮 - 中路和底线价值更高
    CANNON_POSITION_VALUE = [
        [6, 4, 0, -10, -12, -10, 0, 4, 6],
        [2, 2, 0, -4, -14, -4, 0, 2, 2],
        [2, 2, 0, -10, -8, -10, 0, 2, 2],
        [0, 0, -2, 4, 10, 4, -2, 0, 0],
        [0, 0, 0, 2, 8, 2, 0, 0, 0],
        [0, 0, 0, 2, 8, 2, 0, 0, 0],
        [0, 0, -2, 4, 10, 4, -2, 0, 0],
        [2, 2, 0, -10, -8, -10, 0, 2, 2],
        [2, 2, 0, -4, -14, -4, 0, 2, 2],
        [6, 4, 0, -10, -12, -10, 0, 4, 6],
    ]

    # 兵/卒 - 过河后价值翻倍，前进价值更高
    PAWN_POSITION_VALUE = [
        [0, 3, 6, 9, 12, 9, 6, 3, 0],
        [18, 36, 56, 80, 120, 80, 56, 36, 18],
        [14, 20, 28, 40, 60, 40, 28, 20, 14],
        [10, 14, 20, 30, 40, 30, 20, 14, 10],
        [6, 10, 14, 20, 30, 20, 14, 10, 6],
        [2, 6, 8, 10, 16, 10, 8, 6, 2],
        [0, 0, 2, 6, 8, 6, 2, 0, 0],
        [0, 0, -4, 2, 4, 2, -4, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
    ]

    def __init__(self):
        self.current_player = 'red'
        self.search_depth = AI_SEARCH_DEPTH
        self.time_limit = AI_TIME_LIMIT
        self.nodes_searched = 0
        self.start_time = 0

    def get_valid_moves(self, board_state: Dict, player: str) -> List[Dict]:
        """获取所有合法走法"""
        valid_moves = []
        pieces = board_state.get('pieces', {})

        for pos, piece in pieces.items():
            if piece['color'] == player:
                moves = self._get_piece_moves(piece, pos, pieces)
                valid_moves.extend(moves)

        return valid_moves

    def _get_piece_moves(self, piece: Dict, pos: str, all_pieces: Dict) -> List[Dict]:
        """获取单个棋子的走法"""
        moves = []
        col_names = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']

        col = col_names.index(pos[0])
        row = int(pos[1:])

        piece_name = piece['class_name'].split('_')[-1]
        color = piece['color']

        if piece_name in ['shuai', 'jiang']:
            moves = self._get_king_moves(col, row, color, all_pieces)
        elif piece_name == 'shi':
            moves = self._get_advisor_moves(col, row, color, all_pieces)
        elif piece_name == 'xiang':
            moves = self._get_elephant_moves(col, row, color, all_pieces)
        elif piece_name == 'ma':
            moves = self._get_horse_moves(col, row, color, all_pieces)
        elif piece_name == 'che':
            moves = self._get_rook_moves(col, row, color, all_pieces)
        elif piece_name == 'pao':
            moves = self._get_cannon_moves(col, row, color, all_pieces)
        elif piece_name in ['bing', 'zu']:
            moves = self._get_pawn_moves(col, row, color, all_pieces)

        return moves

    def _get_king_moves(self, col: int, row: int, color: str, pieces: Dict) -> List[Dict]:
        """将/帅的走法"""
        moves = []
        col_names = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']

        # 坐标系统：row 0 = 红方底线，row 9 = 黑方底线
        if color == 'red':
            valid_rows = range(0, 3)  # row 0-2
            valid_cols = range(3, 6)
        else:
            valid_rows = range(7, 10)  # row 7-9
            valid_cols = range(3, 6)

        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]

        for dc, dr in directions:
            new_col = col + dc
            new_row = row + dr

            if new_col in valid_cols and new_row in valid_rows:
                new_pos = f"{col_names[new_col]}{new_row}"
                if self._can_move_to(new_pos, color, pieces):
                    moves.append({
                        'from': f"{col_names[col]}{row}",
                        'to': new_pos,
                        'piece': 'shuai' if color == 'red' else 'jiang'
                    })

        # 检查将帅对面（飞将）
        enemy_king = 'jiang' if color == 'red' else 'shuai'
        enemy_color = 'black' if color == 'red' else 'red'

        for pos, piece in pieces.items():
            if piece['class_name'].endswith(enemy_king):
                enemy_col = col_names.index(pos[0])
                enemy_row = int(pos[1:])

                # 如果在同一列
                if enemy_col == col:
                    # 检查中间是否有棋子
                    min_row, max_row = min(row, enemy_row), max(row, enemy_row)
                    blocked = False
                    for r in range(min_row + 1, max_row):
                        check_pos = f"{col_names[col]}{r}"
                        if check_pos in pieces:
                            blocked = True
                            break

                    if not blocked:
                        # 可以飞将吃掉对方将帅
                        moves.append({
                            'from': f"{col_names[col]}{row}",
                            'to': pos,
                            'piece': 'shuai' if color == 'red' else 'jiang'
                        })
                break

        return moves

    def _get_advisor_moves(self, col: int, row: int, color: str, pieces: Dict) -> List[Dict]:
        """士/仕的走法（斜走一格，限于九宫格内）"""
        moves = []
        col_names = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']

        # 坐标系统：row 0 = 红方底线，row 9 = 黑方底线
        # 士/仕只能在九宫格内斜走
        if color == 'red':
            valid_positions = [(3, 0), (5, 0), (4, 1), (3, 2), (5, 2)]
        else:
            valid_positions = [(3, 7), (5, 7), (4, 8), (3, 9), (5, 9)]

        current = (col, row)
        if current in valid_positions:
            for pos in valid_positions:
                if pos != current and abs(pos[0] - col) == 1 and abs(pos[1] - row) == 1:
                    new_pos = f"{col_names[pos[0]]}{pos[1]}"
                    if self._can_move_to(new_pos, color, pieces):
                        moves.append({
                            'from': f"{col_names[col]}{row}",
                            'to': new_pos,
                            'piece': 'shi'
                        })

        return moves

    def _get_elephant_moves(self, col: int, row: int, color: str, pieces: Dict) -> List[Dict]:
        """相/象的走法"""
        moves = []
        col_names = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']

        # 坐标系统：row 0 = 红方底线，row 9 = 黑方底线
        # 河界：row 4-5
        # 红象不能过河（row >= 5），黑象不能过河（row <= 4）
        if color == 'red' and row > 4:
            return moves
        if color == 'black' and row < 5:
            return moves

        directions = [(2, 2), (2, -2), (-2, 2), (-2, -2)]

        for dc, dr in directions:
            new_col = col + dc
            new_row = row + dr

            # 检查是否过河
            if color == 'red' and new_row > 4:
                continue
            if color == 'black' and new_row < 5:
                continue

            if 0 <= new_col < 9 and 0 <= new_row <= 9:
                eye_col = col + dc // 2
                eye_row = row + dr // 2
                eye_pos = f"{col_names[eye_col]}{eye_row}"
                if eye_pos not in pieces:
                    new_pos = f"{col_names[new_col]}{new_row}"
                    if self._can_move_to(new_pos, color, pieces):
                        moves.append({
                            'from': f"{col_names[col]}{row}",
                            'to': new_pos,
                            'piece': 'xiang'
                        })

        return moves

    def _get_horse_moves(self, col: int, row: int, color: str, pieces: Dict) -> List[Dict]:
        """马的走法"""
        moves = []
        col_names = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']

        directions = [
            (1, 2, 0, 1), (2, 1, 1, 0), (2, -1, 1, 0), (1, -2, 0, -1),
            (-1, -2, 0, -1), (-2, -1, -1, 0), (-2, 1, -1, 0), (-1, 2, 0, 1)
        ]

        for dc, dr, bc, br in directions:
            new_col = col + dc
            new_row = row + dr
            block_col = col + bc
            block_row = row + br

            if 0 <= new_col < 9 and 0 <= new_row <= 9:
                block_pos = f"{col_names[block_col]}{block_row}"
                if block_pos not in pieces:
                    new_pos = f"{col_names[new_col]}{new_row}"
                    if self._can_move_to(new_pos, color, pieces):
                        moves.append({
                            'from': f"{col_names[col]}{row}",
                            'to': new_pos,
                            'piece': 'ma'
                        })

        return moves

    def _get_rook_moves(self, col: int, row: int, color: str, pieces: Dict) -> List[Dict]:
        """车的走法"""
        moves = []
        col_names = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']

        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]

        for dc, dr in directions:
            for step in range(1, 10):
                new_col = col + dc * step
                new_row = row + dr * step

                if not (0 <= new_col < 9 and 0 <= new_row <= 9):
                    break

                new_pos = f"{col_names[new_col]}{new_row}"

                if new_pos in pieces:
                    if pieces[new_pos]['color'] != color:
                        moves.append({
                            'from': f"{col_names[col]}{row}",
                            'to': new_pos,
                            'piece': 'che'
                        })
                    break
                else:
                    moves.append({
                        'from': f"{col_names[col]}{row}",
                        'to': new_pos,
                        'piece': 'che'
                    })

        return moves

    def _get_cannon_moves(self, col: int, row: int, color: str, pieces: Dict) -> List[Dict]:
        """炮的走法"""
        moves = []
        col_names = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']

        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]

        for dc, dr in directions:
            skip_count = 0
            for step in range(1, 10):
                new_col = col + dc * step
                new_row = row + dr * step

                if not (0 <= new_col < 9 and 0 <= new_row <= 9):
                    break

                new_pos = f"{col_names[new_col]}{new_row}"

                if new_pos in pieces:
                    skip_count += 1
                    if skip_count == 2 and pieces[new_pos]['color'] != color:
                        moves.append({
                            'from': f"{col_names[col]}{row}",
                            'to': new_pos,
                            'piece': 'pao'
                        })
                        break
                    elif skip_count >= 2:
                        break
                elif skip_count == 0:
                    moves.append({
                        'from': f"{col_names[col]}{row}",
                        'to': new_pos,
                        'piece': 'pao'
                    })

        return moves

    def _get_pawn_moves(self, col: int, row: int, color: str, pieces: Dict) -> List[Dict]:
        """兵/卒的走法"""
        moves = []
        col_names = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']

        # 坐标系统：row 0 = 红方底线，row 9 = 黑方底线
        # 红兵向前（row增加），黑卒向前（row减少）
        # 过河后可以横走：红兵过河是 row >= 5，黑卒过河是 row <= 4
        if color == 'red':
            directions = [(0, 1)]  # 向前
            if row >= 5:  # 过河后可以横走
                directions.extend([(-1, 0), (1, 0)])
        else:
            directions = [(0, -1)]  # 向前
            if row <= 4:  # 过河后可以横走
                directions.extend([(-1, 0), (1, 0)])

        for dc, dr in directions:
            new_col = col + dc
            new_row = row + dr

            if 0 <= new_col < 9 and 0 <= new_row <= 9:
                new_pos = f"{col_names[new_col]}{new_row}"
                if self._can_move_to(new_pos, color, pieces):
                    piece_name = 'bing' if color == 'red' else 'zu'
                    moves.append({
                        'from': f"{col_names[col]}{row}",
                        'to': new_pos,
                        'piece': piece_name
                    })

        return moves

    def _can_move_to(self, pos: str, color: str, pieces: Dict) -> bool:
        """检查是否可以移动到目标位置"""
        if pos not in pieces:
            return True
        return pieces[pos]['color'] != color

    def _simulate_move(self, board_state: Dict, move: Dict) -> Dict:
        """模拟走棋"""
        new_state = {
            'pieces': dict(board_state.get('pieces', {}))
        }

        from_pos = move['from']
        to_pos = move['to']

        if from_pos in new_state['pieces']:
            piece = new_state['pieces'].pop(from_pos)
            new_state['pieces'][to_pos] = piece

        return new_state

    def _get_position_value(self, piece_name: str, col: int, row: int, color: str) -> int:
        """获取棋子位置价值"""
        col_names = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']

        # 坐标系统：row 0 = 红方底线，row 9 = 黑方底线
        # 位置价值表是从红方视角，row 0-9
        # 对于黑方，行需要翻转：翻转后 row = 9 - row
        if color == 'black':
            row = 9 - row

        # 行索引已经是 0-9，直接使用
        row_idx = row

        if piece_name == 'che':
            return self.ROOK_POSITION_VALUE[row_idx][col]
        elif piece_name == 'ma':
            return self.HORSE_POSITION_VALUE[row_idx][col]
        elif piece_name == 'pao':
            return self.CANNON_POSITION_VALUE[row_idx][col]
        elif piece_name in ['bing', 'zu']:
            return self.PAWN_POSITION_VALUE[row_idx][col]

        return 0

    def _is_king_exposed(self, board_state: Dict, color: str) -> bool:
        """检查将/帅是否暴露（将帅对面）"""
        pieces = board_state.get('pieces', {})
        col_names = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']

        # 找到己方将帅位置
        king_pos = None
        king_name = 'shuai' if color == 'red' else 'jiang'
        for pos, piece in pieces.items():
            if piece['class_name'].endswith(king_name):
                king_pos = pos
                break

        if not king_pos:
            return False

        # 找到对方将帅位置
        enemy_king_pos = None
        enemy_king_name = 'jiang' if color == 'red' else 'shuai'
        for pos, piece in pieces.items():
            if piece['class_name'].endswith(enemy_king_name):
                enemy_king_pos = pos
                break

        if not enemy_king_pos:
            return False

        # 检查是否在同一列
        king_col = col_names.index(king_pos[0])
        enemy_col = col_names.index(enemy_king_pos[0])

        if king_col != enemy_col:
            return False

        # 检查中间是否有棋子
        king_row = int(king_pos[1:])
        enemy_row = int(enemy_king_pos[1:])
        min_row, max_row = min(king_row, enemy_row), max(king_row, enemy_row)

        for r in range(min_row + 1, max_row):
            check_pos = f"{col_names[king_col]}{r}"
            if check_pos in pieces:
                return False

        return True

    def evaluate_position(self, board_state: Dict, player: str) -> int:
        """评估局面（增强版）"""
        score = 0
        pieces = board_state.get('pieces', {})
        col_names = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']

        for pos, piece in pieces.items():
            piece_name = piece['class_name'].split('_')[-1]
            color = piece['color']

            # 基本价值
            base_value = self.PIECE_VALUES.get(piece_name, 0)

            # 位置价值
            col = col_names.index(pos[0])
            row = int(pos[1:])
            pos_value = self._get_position_value(piece_name, col, row, color)

            # 过河兵价值加成
            river_bonus = 0
            # 坐标系统：row 0 = 红方底线，row 9 = 黑方底线
            # 河界：row 4-5，红兵过河是 row >= 5，黑卒过河是 row <= 4
            if piece_name == 'bing' and color == 'red' and row >= 5:
                river_bonus = 50
            elif piece_name == 'zu' and color == 'black' and row <= 4:
                river_bonus = 50

            total_value = base_value + pos_value + river_bonus

            if color == player:
                score += total_value
            else:
                score -= total_value

        # 将帅对面惩罚
        if self._is_king_exposed(board_state, player):
            score -= 500

        enemy = 'black' if player == 'red' else 'red'
        if self._is_king_exposed(board_state, enemy):
            score += 500

        return score

    def minimax(self, board_state: Dict, depth: int, alpha: float, beta: float,
                maximizing: bool, player: str) -> Tuple[int, Optional[Dict]]:
        """Minimax搜索 + Alpha-Beta剪枝"""
        self.nodes_searched += 1

        # 时间限制检查
        if time.time() - self.start_time > self.time_limit:
            return self.evaluate_position(board_state, player), None

        # 搜索深度到达或游戏结束
        if depth == 0:
            return self.evaluate_position(board_state, player), None

        current_player = player if maximizing else ('black' if player == 'red' else 'red')
        valid_moves = self.get_valid_moves(board_state, current_player)

        # 没有合法走法，游戏结束
        if not valid_moves:
            return self.evaluate_position(board_state, player), None

        # 走法排序：优先评估吃子走法，提高剪枝效率
        def move_score(move):
            to_pos = move['to']
            if to_pos in board_state.get('pieces', {}):
                captured = board_state['pieces'][to_pos]
                piece_name = captured['class_name'].split('_')[-1]
                return self.PIECE_VALUES.get(piece_name, 0)
            return 0

        valid_moves.sort(key=move_score, reverse=True)

        best_move = valid_moves[0] if valid_moves else None

        if maximizing:
            max_eval = float('-inf')
            for move in valid_moves:
                new_state = self._simulate_move(board_state, move)
                eval_score, _ = self.minimax(new_state, depth - 1, alpha, beta, False, player)

                if eval_score > max_eval:
                    max_eval = eval_score
                    best_move = move

                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    break  # Beta剪枝

            return max_eval, best_move
        else:
            min_eval = float('inf')
            for move in valid_moves:
                new_state = self._simulate_move(board_state, move)
                eval_score, _ = self.minimax(new_state, depth - 1, alpha, beta, True, player)

                if eval_score < min_eval:
                    min_eval = eval_score
                    best_move = move

                beta = min(beta, eval_score)
                if beta <= alpha:
                    break  # Alpha剪枝

            return min_eval, best_move

    def get_best_move(self, board_state: Dict, player: str) -> Optional[Dict]:
        """获取最佳走法（优先使用引擎，备用 Minimax）"""
        # 尝试使用引擎
        if self.use_engine and ENGINE_AVAILABLE:
            engine_move = self._get_best_move_from_engine(board_state, player)
            if engine_move:
                return engine_move
            # 引擎失败，回退到 Minimax
            print("[ChessAI] 引擎获取走法失败，回退到 Minimax")

        # 使用内置 Minimax 搜索
        self.nodes_searched = 0
        self.start_time = time.time()

        valid_moves = self.get_valid_moves(board_state, player)

        if not valid_moves:
            return None

        # 使用Minimax搜索
        _, best_move = self.minimax(
            board_state,
            self.search_depth,
            float('-inf'),
            float('inf'),
            True,
            player
        )

        # 如果搜索没有返回走法，随机选择一个
        if best_move is None:
            best_move = random.choice(valid_moves)

        return best_move

    def _get_best_move_from_engine(self, board_state: Dict, player: str) -> Optional[Dict]:
        """使用 Pikafish 引擎获取最佳走法"""
        try:
            import config
            from modules.board_state import to_fen

            # 获取引擎实例
            engine = get_engine(config.ENGINE_PATH)

            # 转换为 FEN 格式
            fen = to_fen(board_state, player)
            print(f"[ChessAI] FEN: {fen}")

            # 调试：打印棋盘关键信息
            pieces = board_state.get('pieces', {})
            red_shuai_pos = [pos for pos, p in pieces.items() if p.get('class_name') == 'red_shuai']
            black_jiang_pos = [pos for pos, p in pieces.items() if p.get('class_name') == 'black_jiang']
            print(f"[ChessAI] 棋子数: {len(pieces)}, 红帅: {red_shuai_pos}, 黑将: {black_jiang_pos}")

            # 设置局面
            engine.set_position(fen)

            # 获取走法（新版本返回字典）
            time_ms = config.ENGINE_TIME_LIMIT if hasattr(config, 'ENGINE_TIME_LIMIT') else 2000
            result = engine.get_best_move(time_ms=time_ms)

            move_str = result.get('move')

            if move_str and move_str != '(none)':
                # 解析走法（格式如 'e8e2'）
                # 坐标系统已统一：row 0-9（row 0 = 红方底线，row 9 = 黑方底线）
                # 直接使用引擎坐标，无需转换
                from_pos = move_str[:2]
                to_pos = move_str[2:4]

                print(f"[ChessAI] 引擎返回走法: {move_str} ({from_pos}->{to_pos})")
                print(f"[ChessAI] 引擎评估: score_cp={result.get('score_cp')}, is_checkmate={result.get('is_checkmate')}, mate_in={result.get('mate_in')}")

                # 打印当前棋盘上的棋子位置（帮助调试）
                pieces = board_state.get('pieces', {})
                print(f"[ChessAI] 当前棋盘棋子数: {len(pieces)}")

                # 检查目标位置是否有棋子
                to_piece = pieces.get(to_pos)
                if to_piece:
                    print(f"[ChessAI] 目标位置 {to_pos} 有棋子: {to_piece.get('class_name')} ({to_piece.get('color')})")

                # 获取移动的棋子信息
                from_piece = pieces.get(from_pos)

                # 检测吃子：目标位置是否有对方棋子
                captured_piece = board_state['pieces'].get(to_pos)
                captured = None
                if captured_piece and captured_piece.get('color') != player:
                    captured = captured_piece

                # 构建返回结果（包含 PV 信息）
                move_result = {
                    'from': from_pos,
                    'to': to_pos,
                    'piece': from_piece['class_name'].split('_')[-1] if from_piece else 'unknown',
                    'player': player,
                    'captured': captured,
                    'is_checkmate': result.get('is_checkmate', False),
                    'mate_in': result.get('mate_in'),
                    'score_cp': result.get('score_cp'),
                    'pv': result.get('pv', []),       # PV 序列（引擎预期的后续走法）
                    'depth': result.get('depth'),     # 搜索深度
                    'nodes': result.get('nodes')      # 搜索节点数
                }

                # 打印 PV 信息（用于调试解说）
                if result.get('pv'):
                    print(f"[ChessAI] PV 序列: {result.get('pv')[:5]}... (共 {len(result.get('pv', []))} 步)")

                if from_piece:
                    print(f"[ChessAI] 起始位置 {from_pos} 有棋子: {from_piece.get('class_name')} ({from_piece.get('color')})")
                    # 【引擎走法】直接信任，不做验证（引擎比内置规则更准确）
                    print(f"[ChessAI] 直接使用引擎走法（信任引擎）")
                    return move_result

                # 起始位置没有找到棋子，直接信任引擎走法
                print(f"[ChessAI] 起始位置 {from_pos} 未找到棋子，直接使用引擎走法")
                return move_result

            # 没有合法走法（被将死或困毙）
            if result.get('is_stalemate') or result.get('is_checkmate'):
                return {
                    'from': None,
                    'to': None,
                    'piece': None,
                    'player': player,
                    'is_checkmate': True,
                    'no_legal_moves': True
                }

            return None

        except Exception as e:
            print(f"[ChessAI] 引擎错误: {e}")
            import traceback
            traceback.print_exc()
            return None

    def set_difficulty(self, level: int):
        """设置难度等级 (1-20)"""
        if ENGINE_AVAILABLE:
            try:
                import config
                engine = get_engine(config.ENGINE_PATH)
                engine.set_difficulty(level)
                print(f"[ChessAI] 难度设置为: {level}")
            except Exception as e:
                print(f"[ChessAI] 设置难度失败: {e}")

    def is_valid_move(self, board_state: Dict, from_pos: str, to_pos: str) -> bool:
        """检查走法是否合法"""
        pieces = board_state.get('pieces', {})

        if from_pos not in pieces:
            return False

        piece = pieces[from_pos]
        moves = self._get_piece_moves(piece, from_pos, pieces)

        for move in moves:
            if move['to'] == to_pos:
                return True

        return False

    def get_current_score(self, board_state: Dict, player: str) -> Optional[int]:
        """
        获取当前局势分数（用于解说中的分数变化分析）

        Args:
            board_state: 棋盘状态
            player: 当前走棋方 ('red' 或 'black')

        Returns:
            局势分数（厘兵值），从黑方视角（正数黑方优，负数红方优）
            如果无法获取分数则返回 None
        """
        try:
            import config
            from modules.board_state import to_fen

            # 获取引擎实例
            engine = get_engine(config.ENGINE_PATH)

            # 转换为 FEN 格式
            fen = to_fen(board_state, player)

            # 设置局面
            engine.set_position(fen)

            # 使用较短时间获取评估分数（不需要最佳走法）
            result = engine.get_best_move(time_ms=500)

            score_cp = result.get('score_cp')
            return score_cp

        except Exception as e:
            print(f"[ChessAI] 获取局势分数失败: {e}")
            return None