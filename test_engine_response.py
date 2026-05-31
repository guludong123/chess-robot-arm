"""Pikafish 引擎响应时间测试 - v4
测试不同 movetime 下的引擎表现，以及不同 Skill Level 下的搜索深度差异"""
import sys
import os
import time
import statistics

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from modules.uci_engine import PikafishEngine
import config

TEST_FENS = [
    "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w",
    "r1ba1k2r/4a4/2n1b1n2/p3p3p/2p6/6P2/P1P1P3P/1CN1B4/9/R1BAK2NR w",
    "r1bakab1r/9/2n1c2c1/p3p1n1p/2p6/6P2/P1P1P3P/1CN1C4/9/R1BAKAB1R w",
    "rnbakabnr/9/1c4c2/p1p1C1p1p/9/9/P1P1P1P1P/1C7/9/RNBAKABNR w",
    "r1ba1k2r/4a4/2n1b1n2/p1N1p3p/2p6/9/P1P1P1P1P/1C4C2/9/R1BAKB2R w",
    "2bakab1r/9/2n1c2c1/p3p1n1p/2p3p2/9/P1P1P1P1P/1CN1C4/4A4/R1BAK1B1R w",
    "r1bak3r/4a4/2n1b1n2/p3p3p/2p3P2/9/P1P1P3P/1CN1B4/4A4/R1BAK3R w",
    "rnb1kab1r/4a4/2c1b1n2/p1p1p1p1p/9/2N6/P1P1P1P1P/1C4C2/9/R1BAKAB1R w",
    "r1bakab1r/9/1cn1c2c1/p3p1n1p/2p6/6P2/P1P1P3P/1C4C2/9/RNBAKAB1R w",
    "r1ba1kb1r/4a4/2n4n1/p1p1p1p1p/2c6/2B6/P1P1P1P1P/1C4C2/4A4/RN2KAB1R w",
]


def test_movetimes(engine):
    """测试不同思考时间下的引擎表现"""
    print("\n" + "=" * 70)
    print("TEST 1: 不同 movetime 下的引擎表现 (Skill Level = 10)")
    print("=" * 70)

    movetimes = [500, 1000, 1500, 2000]
    engine.set_difficulty(10)
    time.sleep(0.3)

    print(f"\n{'movetime':<12} {'实际耗时(ms)':<14} {'搜索深度':<10} {'搜索节点数':<14}")
    print("-" * 50)

    for mt in movetimes:
        times = []
        depths = []
        nodes_list = []
        for i in range(10):
            engine.set_position(TEST_FENS[i])
            start = time.perf_counter()
            result = engine.get_best_move(time_ms=mt)
            elapsed = (time.perf_counter() - start) * 1000
            times.append(elapsed)
            depths.append(result.get('depth', 0) or 0)
            nodes_list.append(result.get('nodes', 0) or 0)

        avg_t = statistics.mean(times)
        avg_d = statistics.mean(depths)
        avg_n = statistics.mean(nodes_list)
        print(f"{mt:<12} {avg_t:<14.0f} {avg_d:<10.1f} {avg_n:<14.0f}")


def test_skill_levels(engine):
    """测试不同 Skill Level 在固定 movetime 下的搜索深度差异"""
    print("\n" + "=" * 70)
    print("TEST 2: 不同 Skill Level 下的搜索深度 (movetime = 2000ms)")
    print("=" * 70)

    levels = [1, 5, 10, 15, 20]

    print(f"\n{'Level':<8} {'实际耗时(ms)':<14} {'搜索深度':<10} {'搜索节点数':<14}")
    print("-" * 46)

    for level in levels:
        engine.set_difficulty(level)
        time.sleep(0.3)

        times = []
        depths = []
        nodes_list = []
        for i in range(10):
            engine.set_position(TEST_FENS[i])
            start = time.perf_counter()
            result = engine.get_best_move(time_ms=2000)
            elapsed = (time.perf_counter() - start) * 1000
            times.append(elapsed)
            depths.append(result.get('depth', 0) or 0)
            nodes_list.append(result.get('nodes', 0) or 0)

        avg_t = statistics.mean(times)
        avg_d = statistics.mean(depths)
        avg_n = statistics.mean(nodes_list)
        print(f"{level:<8} {avg_t:<14.0f} {avg_d:<10.1f} {avg_n:<14.0f}")


def main():
    engine_path = config.ENGINE_PATH
    print(f"Engine: {engine_path}")

    engine = PikafishEngine(engine_path)

    test_movetimes(engine)
    test_skill_levels(engine)

    engine.quit()
    print("\nDone.")


if __name__ == '__main__':
    main()
