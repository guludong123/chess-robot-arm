"""
测试将死检测修复 - 验证 score mate 解析逻辑
"""
import sys
sys.path.insert(0, '.')

from modules.uci_engine import get_engine
import config

def test_mate_score_parsing():
    """测试 score mate 解析"""
    print("=" * 50)
    print("测试 Pikafish score mate 解析逻辑")
    print("=" * 50)

    engine = get_engine(config.ENGINE_PATH)

    # 测试 1：标准开局局面
    print("\n--- 测试 1: 标准开局 ---")
    fen = "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR b"
    engine.set_position(fen)
    result = engine.get_best_move(time_ms=1000)

    print(f"走法: {result['move']}")
    print(f"score_cp: {result['score_cp']}")
    print(f"is_checkmate: {result['is_checkmate']}")
    print(f"mate_in: {result['mate_in']}")
    print(f"is_winning: {result.get('is_winning')}")
    print(f"is_losing: {result.get('is_losing')}")
    print(f"is_stalemate: {result['is_stalemate']}")

    # 测试 2：黑车直接面对红帅的将死局面
    # 黑车在 e1，红帅在 e0（黑车可以直接吃帅）
    print("\n--- 测试 2: 黑车直接面对红帅 ---")
    # FEN: 黑将在 e9，黑车在 e1（row 1），红帅在 e0（row 0）
    # 黑车可以直接吃红帅
    fen_checkmate = "4k4/9/9/9/9/9/9/9/4r4/4K4 b"
    engine.set_position(fen_checkmate)
    result = engine.get_best_move(depth=10)

    print(f"走法: {result['move']}")
    print(f"score_cp: {result['score_cp']}")
    print(f"is_checkmate: {result['is_checkmate']}")
    print(f"mate_in: {result['mate_in']}")
    print(f"is_winning: {result.get('is_winning')}")
    print(f"is_losing: {result.get('is_losing')}")

    # 验证：如果是将死局面，is_checkmate 应为 True
    if result['is_checkmate'] and result.get('is_winning'):
        print(f"[OK] 正确识别将死路径! mate_in={result['mate_in']} 步将死")
    else:
        print("[INFO] 当前局面未检测到将死路径")
        print(f"  走法: {result['move']} (黑车走法)")

    # 测试 3：红方即将被将死的局面
    print("\n--- 测试 3: 红方即将被将死 ---")
    # 红帅在 e0，黑车在 e1，红方走棋但无路可逃
    fen_losing = "4k4/9/9/9/9/9/9/9/4r4/4K4 w"
    engine.set_position(fen_losing)
    result = engine.get_best_move(depth=10)

    print(f"走法: {result['move']}")
    print(f"score_cp: {result['score_cp']}")
    print(f"is_checkmate: {result['is_checkmate']}")
    print(f"mate_in: {result['mate_in']}")
    print(f"is_winning: {result.get('is_winning')}")
    print(f"is_losing: {result.get('is_losing')}")

    # 验证：如果即将被将死，is_losing 应为 True
    if result.get('is_losing'):
        print(f"[OK] 正确识别即将被将死! mate_in={result['mate_in']} 步内被将死")
    else:
        print("[INFO] 当前局面状态")

    # 测试 4：困毙局面
    print("\n--- 测试 4: 困毙局面 ---")
    # 构造困毙：红帅无路可走，但未被将军
    fen_stalemate = "4k4/9/9/9/9/9/9/9/9/3K5 w"
    engine.set_position(fen_stalemate)
    result = engine.get_best_move(depth=10)

    print(f"走法: {result['move']}")
    print(f"is_stalemate: {result['is_stalemate']}")
    print(f"is_checkmate_received: {result.get('is_checkmate_received')}")

    if result['move'] is None or result['move'] == '(none)':
        if result['is_stalemate']:
            print("[OK] 正确识别困毙!")
        elif result.get('is_checkmate_received'):
            print("[WARN] 错误识别为被将死（实际是困毙）")
    else:
        print(f"[INFO] 有合法走法: {result['move']}，非困毙局面")

    print("\n" + "=" * 50)
    print("测试完成")
    print("=" * 50)

if __name__ == '__main__':
    test_mate_score_parsing()