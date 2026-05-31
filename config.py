"""
中国象棋机器人系统配置文件
"""
import os
import cv2
from dotenv import load_dotenv

load_dotenv()

# ==================== 机械臂配置 ====================
SERIAL_PORT = 'COM12'
SERIAL_BAUD = 115200

# 机械臂运动参数
MOVE_ANGLES = {
    'home': [0.0, 0.0, 0.0],
    'ready': [25.55, 0.0, 15.24],
    'grab': [0.0, 14.32, 0.0]
}

# 放置区域坐标 (机械臂坐标系)
PLACE_COORDS = {
    'A': [269.02, -161.65, 51.42],   # A区 - 黑棋（被吃）
    'B': [146.8, -159.53, 50.44],    # B区 - 暂不使用
    'C': [248.52, 152.35, 53.45],    # C区 - 红棋（被吃）
    'D': [141.53, 148.67, 43.73],    # D区 - 暂不使用
}

# 机械臂运动高度参数
Z_HEIGHTS = {
    'safe': 10,
    'grab': -35,
    'place': -30
}

# ==================== 摄像头配置 ====================
CAMERA_INDEX = 1
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
FPS = 30

# 图像缩放参数 (用于YOLO检测)
FRAME_SCALE = 1.5

# ==================== YOLO模型配置 ====================
MODEL_PATH = r"F:\deeplearning\ultralytics-8.3.163\runs\detect\train15\weights\best.pt"

# 棋子类别定义
CHESS_CLASSES = {
    # 红棋
    'red_shuai': 0, 'red_shi': 1, 'red_xiang': 2,
    'red_ma': 3, 'red_che': 4, 'red_pao': 5, 'red_bing': 6,
    # 黑棋
    'black_jiang': 7, 'black_shi': 8, 'black_xiang': 9,
    'black_ma': 10, 'black_che': 11, 'black_pao': 12, 'black_zu': 13
}

# 颜色定义 (BGR格式)
COLORS = {
    'red': (0, 0, 255),
    'black': (0, 255, 0),
    'blue': (255, 0, 0),
    'yellow': (0, 255, 255),
    'purple': (255, 0, 255),
    'white': (255, 255, 255),
    'green': (0, 255, 0),
    'orange': (0, 165, 255),
}

# ==================== 棋子标定配置 ====================
# 使用棋子放置进行标定，4个角点透视变换
# 标定点对应：P0=右下, P1=左下, P2=左上, P3=右上
PIECE_CALIBRATION_POSITIONS = {
    'P0': {'robot': [324.76, -52.65], 'name': '画面右下角'},
    'P1': {'robot': [316.85, 84.23], 'name': '画面左下角'},
    'P2': {'robot': [197.98, 78.26], 'name': '画面左上角'},
    'P3': {'robot': [205.97, -60.15], 'name': '画面右上角'},
}

# 标定放置顺序：左上、右上、右下、左下（getPerspectiveTransform要求的顺序）
PIECE_CALIBRATION_ORDER = ['P2', 'P3', 'P0', 'P1']

# 棋子标定参数
PIECE_CALIBRATION_MIN_POINTS = 4  # 最少标定点数
PIECE_CALIBRATION_ERROR_THRESHOLD = 2.0  # 标定误差阈值 (mm)

# ==================== YOLO检测区域配置 ====================
# 棋盘检测ROI (像素坐标，相对于缩放后的图像)
# 需要根据实际摄像头画面调整，排除放置区域
BOARD_ROI = {
    'x_min': 550,    # 左边界
    'x_max': 1350,    # 右边界
    'y_min': 50,     # 上边界
    'y_max': 850,    # 下边界
}

# 棋盘旋转角度 (摄像头视角)
# 90: 顺时针旋转90度（红方在左，黑方在右）
BOARD_ROTATION = 90

# ==================== 标定配置 ====================
CALIBRATION_FILE = os.path.join(os.path.dirname(__file__), 'calibration_data.json')
REQUIRED_STABLE_FRAMES = 15
CALIBRATION_ERROR_THRESHOLD = 5.0  # mm

# ==================== 棋盘配置 ====================
# 中国象棋棋盘: 9列 x 10行
BOARD_COLS = 9   # a-i
BOARD_ROWS = 10  # 1-10

# 棋盘坐标名称
COL_NAMES = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']

# ==================== AI配置 ====================
# 搜索深度 (推荐3-4层，层数越高AI越强但计算越慢)
AI_SEARCH_DEPTH = 3

# 思考时间限制 (秒)
AI_TIME_LIMIT = 2.0

# ==================== 引擎配置 ====================
# Pikafish 引擎路径
ENGINE_PATH = os.path.join(os.path.dirname(__file__), 'engines', 'Pikafish.2026-01-02', 'Windows', 'pikafish-avx2.exe')

# Pikafish NNUE 神经网络文件路径
ENGINE_NNUE_PATH = os.path.join(os.path.dirname(__file__), 'engines', 'Pikafish.2026-01-02', 'pikafish.nnue')

# 引擎思考时间限制 (毫秒)
ENGINE_TIME_LIMIT = 2000

# 引擎难度等级 (1-20)，越高越强
ENGINE_DIFFICULTY = 10

# 是否使用引擎（True: 使用引擎, False: 使用内置 Minimax）
USE_ENGINE = True

# 走棋检测稳定帧数
MOVE_DETECTION_FRAMES = 15

# ==================== 检测平滑配置 ====================
# 解决 YOLO 检测类别抖动问题（如马一会认成马一会认成帅）
DETECTION_SMOOTH_ENABLED = True        # 是否启用检测平滑
DETECTION_SMOOTH_FRAMES = 8            # 平滑帧数（增加到8帧，提高稳定性）
DETECTION_SMOOTH_STRATEGY = 'weighted_vote'  # 平滑策略: weighted_vote(推荐), majority, highest_conf
DETECTION_MIN_HISTORY_SIZE = 3         # 最少历史帧数才开始平滑（增加到3帧）
DETECTION_CONF_THRESHOLD = 0.6         # 置信度阈值（提高到0.6，过滤低置信检测）

# 开局扫描配置 - 多次扫描取投票结果，提高开局识别准确性
GAME_START_SCAN_FRAMES = 10            # 开局扫描帧数 (建议 5-15)
GAME_START_SCAN_DELAY = 0.1            # 扫描间隔（秒）

# 游戏状态
GAME_STATES = ['waiting', 'calibrating', 'playing', 'paused', 'finished']

# 玩家
PLAYERS = ['red', 'black']

# ==================== 标准开局配置 ====================
# 中国象棋标准开局棋子位置
# 坐标系统（与引擎一致）：row 0 = 红方底线，row 9 = 黑方底线
STANDARD_SETUP = {
    # 黑方 (row 9)
    'a9': 'black_che', 'b9': 'black_ma', 'c9': 'black_xiang', 'd9': 'black_shi',
    'e9': 'black_jiang', 'f9': 'black_shi', 'g9': 'black_xiang', 'h9': 'black_ma',
    'i9': 'black_che',
    'b7': 'black_pao', 'h7': 'black_pao',
    'a6': 'black_zu', 'c6': 'black_zu', 'e6': 'black_zu', 'g6': 'black_zu', 'i6': 'black_zu',

    # 红方 (row 0)
    'a0': 'red_che', 'b0': 'red_ma', 'c0': 'red_xiang', 'd0': 'red_shi',
    'e0': 'red_shuai', 'f0': 'red_shi', 'g0': 'red_xiang', 'h0': 'red_ma',
    'i0': 'red_che',
    'b2': 'red_pao', 'h2': 'red_pao',
    'a3': 'red_bing', 'c3': 'red_bing', 'e3': 'red_bing', 'g3': 'red_bing', 'i3': 'red_bing',
}

# 开局模式配置
GAME_START_MODE_DEFAULT = 'standard'  # 默认开局模式: 'standard' 或 'custom'

# ==================== 语音交互配置 ====================
# 是否启用语音交互功能
VOICE_INTERACTION_ENABLED = True

# 自动生成解说（每次走棋完成后）
AUTO_COMMENTARY = True
AUTO_START_LISTENING = True  # 游戏开始后自动启动语音监听

# ========== LLM 配置 (阿里云 DashScope - Qwen) ==========
LLM_PROVIDER = 'dashscope'
DASHSCOPE_API_KEY = os.environ.get('DASHSCOPE_API_KEY', 'sk-b6549da5de3741b19a6562d55877164f')
DASHSCOPE_MODEL = 'qwen-turbo'  # qwen-turbo(快速) 或 qwen-plus(高质量)

# LLM 生成参数
LLM_MAX_TOKENS = 150           # 解说最大长度（备用）
LLM_MAX_TOKENS_COMMENTARY = 200  # 解说最大长度（生动模式）
LLM_TEMPERATURE = 0.8          # 创意度（提高以增加多样性）
LLM_TEMPERATURE_COMMENTARY = 0.9  # 解说温度（更高创意）

# ========== TTS 配置 (Edge TTS) ==========
TTS_PROVIDER = 'edge_tts'
TTS_VOICE = 'zh-CN-XiaoxiaoNeural'  # 中文女声
TTS_RATE = '+0%'          # 语速调整
TTS_VOLUME = '+0%'        # 音量调整

# ========== ASR 配置 ==========

# ASR 提供者选择
# 'dashscope_paraformer' - 阿里云 DashScope (云端，延迟约2秒)
# 'funasr' - FunASR 本地部署 (延迟200-500ms，需要GPU)
ASR_PROVIDER = 'funasr'

# DashScope ASR 配置 (云端)
DASHSCOPE_ASR_MODEL = 'paraformer-realtime-v2'

# FunASR 配置 (本地部署)
FUNASR_MODEL = 'paraformer-zh'  # 离线模型（准确率高）
# FUNASR_MODEL = 'paraformer-zh-streaming'  # 流式模型（实时性好，但准确率较低）
FUNASR_VAD_MODEL = 'fsmn-vad'  # VAD 模型
FUNASR_DEVICE = 'cuda'  # 'cuda' 或 'cpu'
FUNASR_CACHE_DIR = os.path.join(os.path.dirname(__file__), 'models', 'funasr')
FUNASR_CHUNK_SIZE = [5, 10, 5]  # 流式块大小 [左, 块, 右] 秒

# 通用 ASR 配置
ASR_SAMPLE_RATE = 16000           # 采样率
ASR_SILENCE_THRESHOLD = 1.2       # 静音检测阈值（秒）- 降低以加快响应
ASR_SILENCE_ENERGY_THRESHOLD = 300  # 静音能量阈值
ASR_TIMEOUT = 10.0                # 单次监听超时（秒）

# FunASR VAD 混合检测配置
ASR_VAD_ACCUMULATE_FRAMES = 3     # VAD 累积帧数 (~192ms，更快响应)
ASR_ENERGY_THRESHOLD = 0.0158      # 能量前端触发阈值 (RMS，降低灵敏度)
ASR_SPEECH_CONFIRM_COUNT = 2      # VAD 确认1次就开始（减少延迟）
ASR_VAD_END_COUNT = 4             # VAD 连续无语音次数就结束（防过早截断）

# ========== 对话配置 ==========
DIALOGUE_MODE_ENABLED = True      # 是否启用对话模式
DIALOGUE_MAX_HISTORY = 50         # 对话历史最大长度
LLM_MAX_TOKENS_DIALOGUE = 200     # 对话回复最大 token 数
LLM_TEMPERATURE_DIALOGUE = 0.8    # 对话温度（创意度）
DEFAULT_CHARACTER = 'novice_teacher'  # 默认角色

# ========== E2E 语音配置 (StepFun Realtime API) ==========
STEPFUN_API_KEY = os.environ.get('STEPFUN_API_KEY', '')
STEPFUN_MODEL = 'step-1o-audio'
STEPFUN_VOICE = 'qingchunshaonv'
STEPFUN_SAMPLE_RATE = 24000  # 24kHz（官方文档要求）
STEPFUN_INSTRUCTIONS = '你是象棋对弈助手，用简洁中文回复用户。'
STEPFUN_VAD_THRESHOLD = 0.6           # VAD 灵敏度 (0-1, 越低越灵敏)
STEPFUN_VAD_PREFIX_PADDING_MS = 200   # 语音前填充毫秒
STEPFUN_VAD_SILENCE_DURATION_MS = 600 # 静音判定毫秒（越短越快响应）

# 音频预处理（发送到 API 前）
AUDIO_NOISE_GATE_ENABLED = True       # 噪声门开关
AUDIO_NOISE_GATE_THRESHOLD = 0.0163    # 噪声门阈值（RMS，低于此值静音）
AUDIO_GAIN_NORMALIZE = True           # 增益归一化开关
AUDIO_GAIN_TARGET_RMS = 0.1          # 目标 RMS（归一化到此值）
AUDIO_GAIN_MAX = 10.0                 # 最大增益倍数（防止过度放大）
AUTO_CALIBRATE_NOISE_FLOOR = True     # 启动时自动测量噪底
