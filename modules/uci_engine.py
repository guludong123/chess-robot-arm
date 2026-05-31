"""
UCI 引擎接口模块 - 与 Pikafish 中国象棋引擎通信
"""
import subprocess
import threading
import queue
import time
import os
from typing import Optional, Tuple, Dict


class PikafishEngine:
    """Pikafish UCI 引擎接口"""

    def __init__(self, engine_path: str = 'engines/pikafish-avx2.exe'):
        """
        初始化引擎

        Args:
            engine_path: 引擎可执行文件路径
        """
        self.engine_path = engine_path
        self.process: Optional[subprocess.Popen] = None
        self.output_queue = queue.Queue()
        self.reader_thread: Optional[threading.Thread] = None
        self.is_ready = False
        self.difficulty = 10  # 默认难度 (1-20)

        # 启动引擎
        self._start_engine()

    def _start_engine(self):
        """启动引擎进程"""
        if not os.path.exists(self.engine_path):
            raise FileNotFoundError(f"引擎文件不存在: {self.engine_path}")

        # 获取引擎目录
        engine_dir = os.path.dirname(os.path.abspath(self.engine_path))

        # 启动进程
        self.process = subprocess.Popen(
            [self.engine_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=1,
            universal_newlines=True,
            cwd=engine_dir  # 设置工作目录
        )

        # 启动输出读取线程
        self.reader_thread = threading.Thread(target=self._read_output, daemon=True)
        self.reader_thread.start()

        # 初始化 UCI
        self._send_command('uci')
        self._wait_for_response('uciok', timeout=5.0)

        # 设置神经网络文件（使用 config 中的路径）
        try:
            import config
            if hasattr(config, 'ENGINE_NNUE_PATH') and os.path.exists(config.ENGINE_NNUE_PATH):
                self._send_command(f'setoption name EvalFile value {config.ENGINE_NNUE_PATH}')
                print(f"[Pikafish] 设置神经网络: {config.ENGINE_NNUE_PATH}")
        except ImportError:
            pass

        # 设置难度
        self.set_difficulty(self.difficulty)

        # 等待引擎就绪
        self._send_command('isready')
        self._wait_for_response('readyok', timeout=5.0)
        self.is_ready = True

        print(f"[Pikafish] 引擎已启动: {self.engine_path}")

    def _read_output(self):
        """读取引擎输出的线程"""
        while self.process and self.process.stdout:
            try:
                line = self.process.stdout.readline()
                if line:
                    line = line.strip()
                    self.output_queue.put(line)
                    # 打印调试信息
                    if line.startswith('bestmove') or line.startswith('info depth'):
                        print(f"[Pikafish] {line}")
            except Exception as e:
                print(f"[Pikafish] 读取输出错误: {e}")
                break

    def _send_command(self, command: str):
        """发送命令到引擎"""
        if self.process and self.process.stdin:
            self.process.stdin.write(command + '\n')
            self.process.stdin.flush()
            print(f"[Pikafish] 发送: {command}")

    def _wait_for_response(self, expected: str, timeout: float = 10.0) -> bool:
        """等待特定响应"""
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                line = self.output_queue.get(timeout=0.1)
                if line == expected or line.startswith(expected):
                    return True
            except queue.Empty:
                continue
        print(f"[Pikafish] 警告: 未收到预期响应 '{expected}'")
        return False

    def _get_response(self, timeout: float = 10.0) -> Optional[str]:
        """获取一条响应"""
        try:
            return self.output_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def set_position(self, fen: str):
        """
        设置棋盘局面

        Args:
            fen: 中国象棋 FEN 格式字符串
        """
        self._send_command(f'position fen {fen}')

    def set_position_from_moves(self, moves: list):
        """
        从起始局面和走法列表设置局面

        Args:
            moves: 走法列表，如 ['a1a3', 'b10c8']
        """
        moves_str = ' '.join(moves)
        self._send_command(f'position startpos moves {moves_str}')

    def get_best_move(self, time_ms: int = 2000, depth: int = None) -> Dict:
        """
        获取最佳走法（包含将死信息和PV序列）

        Args:
            time_ms: 思考时间（毫秒）
            depth: 搜索深度（可选，优先于时间）

        Returns:
            字典包含:
            - move: 走法字符串，如 'a1a3'，或 None
            - is_checkmate: 是否将死对方
            - mate_in: 几步将死（如果 is_checkmate）
            - score_cp: 分数（厘兵值，正数优势）
            - is_stalemate: 是否困毙（无合法走法但未被将死）
            - pv: PV序列（后续最佳走法列表），如 ['e2e4', 'e7e5', 'g1f3']
            - depth: 搜索深度
            - nodes: 搜索节点数
        """
        result = {
            'move': None,
            'is_checkmate': False,
            'mate_in': None,
            'score_cp': None,
            'is_stalemate': False,
            'pv': [],        # PV 序列（引擎预期的后续走法）
            'depth': None,   # 搜索深度
            'nodes': None    # 搜索节点数
        }

        if depth is not None:
            self._send_command(f'go depth {depth}')
        else:
            self._send_command(f'go movetime {time_ms}')

        # 等待 bestmove 响应
        start_time = time.time()
        timeout = max(time_ms / 1000 * 2, 10.0)  # 给引擎足够时间

        # 记录最深搜索的 info 行（用于获取完整 PV）
        best_info_line = None
        best_depth = 0

        while time.time() - start_time < timeout:
            line = self._get_response(timeout=0.5)
            if line:
                # 解析 info 行中的各种信息
                if line.startswith('info') and 'score' in line:
                    parts = line.split()

                    # 记录最深的 info 行
                    try:
                        depth_idx = parts.index('depth')
                        current_depth = int(parts[depth_idx + 1])
                        if current_depth > best_depth:
                            best_depth = current_depth
                            best_info_line = line
                    except (ValueError, IndexError):
                        pass

                    # 解析 score mate N 或 score cp X
                    # UCI 协议：
                    # - score mate N>0: 当前走棋方 N 步内将死对方
                    # - score mate -N<0: 当前走棋方 N 步内被将死
                    # - score mate 0: 当前已是将死状态
                    if 'score mate' in line:
                        try:
                            mate_idx = parts.index('mate')
                            mate_in = int(parts[mate_idx + 1])

                            if mate_in == 0:
                                # mate 0 = 当前已是将死状态
                                result['is_checkmate'] = True
                                result['mate_in'] = 0
                            elif mate_in > 0:
                                # mate N > 0 = 当前走棋方 N 步内将死对方
                                result['is_checkmate'] = True
                                result['mate_in'] = mate_in
                                result['is_winning'] = True
                            else:
                                # mate -N < 0 = 当前走棋方 N 步内被将死（败局）
                                # 这不是困毙！困毙是另一种情况
                                result['is_checkmate'] = False
                                result['mate_in'] = abs(mate_in)
                                result['is_losing'] = True
                        except (ValueError, IndexError):
                            pass
                    elif 'score cp' in line:
                        try:
                            cp_idx = parts.index('cp')
                            result['score_cp'] = int(parts[cp_idx + 1])
                        except (ValueError, IndexError):
                            pass

                # 解析 bestmove
                if line.startswith('bestmove'):
                    parts = line.split()
                    if len(parts) >= 2:
                        move = parts[1]
                        if move == '(none)':
                            # 无合法走法 - 根据上下文区分被将死和困毙
                            # 如果之前检测到 is_losing (mate -N)，说明是被将死
                            # 否则是困毙（无走法但未被将军）
                            if result.get('is_losing'):
                                result['is_checkmate_received'] = True  # 被将死
                                print(f"[Pikafish] 检测到被将死（无合法走法）")
                            else:
                                result['is_stalemate'] = True  # 困毙
                                print(f"[Pikafish] 检测到困毙（无合法走法且未被将军）")
                        else:
                            result['move'] = move
                            # 如果有将死路径，打印信息
                            if result.get('is_checkmate') and result.get('mate_in'):
                                print(f"[Pikafish] 检测到将死路径: mate_in={result['mate_in']}, is_winning={result.get('is_winning')}")

                    # 从最佳 info 行提取完整信息
                    if best_info_line:
                        self._parse_info_line(best_info_line, result)

                    return result

        print(f"[Pikafish] 警告: 获取走法超时")
        return result

    def _parse_info_line(self, line: str, result: Dict):
        """
        解析 info 行，提取 PV、depth、nodes 等信息

        Args:
            line: UCI info 行
            result: 结果字典（会被修改）
        """
        parts = line.split()

        try:
            # depth
            if 'depth' in parts:
                depth_idx = parts.index('depth')
                result['depth'] = int(parts[depth_idx + 1])

            # nodes
            if 'nodes' in parts:
                nodes_idx = parts.index('nodes')
                result['nodes'] = int(parts[nodes_idx + 1])

            # pv (Principal Variation - 后续最佳走法序列)
            if 'pv' in parts:
                pv_idx = parts.index('pv')
                # PV 在 pv 关键字之后，直到行尾
                pv_moves = parts[pv_idx + 1:]
                # 过滤掉可能的非走法关键字（如 'tbhits' 等）
                valid_pv = []
                for move in pv_moves:
                    # UCI 走法格式：4个字符（如 e2e4）或5个字符（带升变，如 e7e8q）
                    if len(move) >= 4 and move[0] in 'abcdefghi' and move[1] in '0123456789':
                        valid_pv.append(move)
                    else:
                        break  # 遇到非走法内容，停止
                result['pv'] = valid_pv
        except (ValueError, IndexError) as e:
            print(f"[Pikafish] 解析 info 行失败: {e}")

    def set_difficulty(self, level: int):
        """
        设置引擎难度

        Args:
            level: 难度等级 (1-20)，越高越强
        """
        if not 1 <= level <= 20:
            print(f"[Pikafish] 警告: 难度应在 1-20 范围，当前 {level}")
            level = max(1, min(20, level))

        self.difficulty = level

        # 通过 UCI 选项设置难度
        # Pikafish 支持 SkillLevel 选项
        self._send_command(f'setoption name SkillLevel value {level}')

    def stop(self):
        """停止当前计算"""
        self._send_command('stop')

    def quit(self):
        """退出引擎"""
        if self.process:
            self._send_command('quit')
            self.process.wait(timeout=5.0)
            self.process = None
            self.is_ready = False
            print("[Pikafish] 引擎已退出")

    def restart(self):
        """重启引擎"""
        self.quit()
        time.sleep(0.5)
        self._start_engine()

    def is_alive(self) -> bool:
        """检查引擎是否存活"""
        return self.process is not None and self.process.poll() is None

    def __del__(self):
        """析构时退出引擎"""
        try:
            self.quit()
        except:
            pass


# 全局引擎实例（懒加载）
_engine_instance: Optional[PikafishEngine] = None


def get_engine(engine_path: str = 'engines/pikafish-avx2.exe') -> PikafishEngine:
    """
    获取全局引擎实例

    Args:
        engine_path: 引擎路径

    Returns:
        PikafishEngine 实例
    """
    global _engine_instance

    if _engine_instance is None or not _engine_instance.is_alive():
        _engine_instance = PikafishEngine(engine_path)

    return _engine_instance


def restart_engine(engine_path: str = 'engines/pikafish-avx2.exe') -> PikafishEngine:
    """
    重启引擎

    Args:
        engine_path: 引擎路径

    Returns:
        新的 PikafishEngine 实例
    """
    global _engine_instance

    if _engine_instance:
        _engine_instance.quit()

    _engine_instance = PikafishEngine(engine_path)
    return _engine_instance