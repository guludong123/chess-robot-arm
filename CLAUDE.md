# CLAUDE.md

## 项目概述

基于计算机视觉的中国象棋对战机器人系统。人类走红棋，AI（Pikafish引擎）走黑棋，机械臂自动执行。

## 核心命令

```bash
python app.py                    # 启动应用（localhost:5000）
pip install -r requirements.txt  # 安装依赖
```

## 系统架构

```
app.py                 # Flask 主应用 + API 端点 + WebSocket
config.py              # 全局配置（机械臂/YOLO/AI/语音参数）
modules/
  ├── camera.py        # 摄像头管理
  ├── vision.py        # 视觉系统 - 棋子标定 + YOLO检测 + 四车定位
  ├── robot_arm.py     # 机械臂控制 (myCobot 280/320)
  ├── board_state.py   # 棋盘状态管理 + FEN格式 + Piece类
  ├── move_detector.py # 走棋检测
  ├── detection_smoother.py # YOLO检测平滑器
  ├── chess_ai.py      # 象棋 AI (Pikafish + Minimax备用)
  ├── uci_engine.py    # UCI 引擎通信（返回score_cp、将死信息）
  └── voice_interaction/
      ├── manager.py   # 语音交互管理器
      ├── state.py     # 状态管理
      ├── intent.py    # 意图识别（move/chat/cancel + 模糊匹配）
      ├── llm/         # LLM（DashScope Qwen）
      ├── tts/         # TTS（Edge TTS + 重试机制）
      ├── asr/         # ASR（FunASR本地 / DashScope云端）
      ├── character/   # 角色系统（base.py + presets.py + manager.py）
      └── dialogue/    # 对话管理（history.py + session.py）
engines/               # Pikafish 引擎
models/                # YOLO 模型 + FunASR 模型缓存
```

## 核心流程

**标定**：机械臂放棋子到4角 → YOLO检测 → 透视变换矩阵 → 保存 `calibration_data.json`。四车定位计算90个交叉点。

**坐标转换**：
- 棋盘位置判定：像素 → 最近交叉点 → 棋盘坐标 (col, row)
- 机械臂抓取：像素 → 透视变换 → 机械臂坐标 (rx, ry)
- 抓取用 YOLO 检测的实际中心点，棋子放歪也能抓

**对弈**：人类走棋 → 检测变化 → 验证合法性 → AI计算 → 机械臂执行。LLM解说与机械臂执行并行。

**走棋验证**：检查 `current_player=='red'`、棋子颜色为红、走法符合规则。

## 关键约定

- **Piece 对象**：`board_state.pieces` 存 `Piece` 对象，用 `piece.class_name` / `piece.color`，不是 dict
- **颜色**：red=人类红方，black=AI黑方
- **放置区**：被吃红棋→C区，被吃黑棋→A区
- **坐标系**：棋盘坐标 (a-i, 1-10)、像素坐标 (u,v)、机械臂坐标 (x,y,z) 三套，不要混用
- **引擎返回**：`move`(走法)、`score_cp`(厘兵值，正数黑优)、`is_checkmate`、`mate_in`、`is_stalemate`

## 踩坑记录

- numpy float32 不能 JSON 序列化，返回前端前转 `float()`
- 机械臂获取坐标用 `get_coords_info()`，不是 `get_coords()`
- YOLO 检测用全图+ROI 过滤，裁剪图像会降低识别率
- 标定点必须按顺序：左上→右上→右下→左下
- Pikafish 返回坐标可能有偏移，`_get_best_move_from_engine()` 自动尝试匹配
- 语音模块 import 失败不影响主功能（try/except 保护）
- 直接移动到目标点，不经过安全中转点
- `move_to()` 内部已等待到位，不需要额外 sleep

## 语音交互

**双角色系统**：解说角色（走棋后自动解说）和对话角色（语音交互）使用不同角色池。
- 解说角色：`professional`（男声）、`humorous`（女声）
- 对话角色：`novice_teacher`、`classical`、`humorous_player`、`sarcastic`
- 具体 TTS voice 和 prompt 在 `character/presets.py`

**ASR**：推荐 FunASR 本地（`paraformer-zh` + `fsmn-vad`，GPU加速）。备选 DashScope 云端。

**意图识别**：`move`（走棋）、`chat`（对话）、`cancel`（取消）。支持模糊匹配处理空格和同音字。

论文指导中尽量不用冒号双引号破折号