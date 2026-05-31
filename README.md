# 中国象棋对战机器人系统

基于计算机视觉的中国象棋对战机器人系统，实现人类与机械臂的对弈。

## 技术栈

- **后端**: Python + Flask
- **计算机视觉**: OpenCV + YOLOv11 (Ultralytics)
- **硬件控制**: pymycobot (myCobot机械臂)
- **AI引擎**: Pikafish (UCI中国象棋引擎) + Minimax备用
- **前端**: HTML5 + CSS3 + JavaScript (原生)

## 系统架构

```
chess_robot_arm/
├── app.py                 # Flask主应用
├── config.py              # 系统配置
├── requirements.txt       # Python依赖
├── modules/
│   ├── camera.py         # 摄像头管理
│   ├── vision.py         # 视觉系统(棋子标定+YOLO检测+四车定位)
│   ├── robot_arm.py      # 机械臂控制
│   ├── board_state.py    # 棋盘状态管理 + FEN转换
│   ├── move_detector.py  # 走棋检测
│   ├── chess_ai.py       # 象棋AI (Pikafish引擎 + Minimax)
│   └── uci_engine.py     # UCI引擎通信模块
├── engines/              # 引擎目录
│   └── Pikafish.2026-01-02/  # Pikafish引擎
├── templates/index.html   # 前端页面
├── static/               # 样式和脚本
└── calibration_data.json # 标定数据
```

## 安装步骤

```bash
pip install -r requirements.txt
python app.py
```

访问: http://localhost:5000

## 使用说明

### 1. 启动摄像头
点击"启动摄像头"按钮

### 2. 连接机械臂
点击"连接机械臂"，等待回零位

### 3. 棋子标定
1. 点击"开始标定"
2. 将棋子放到吸泵上，点击"放置并检测"
3. 依次完成 4 个标定点（左上→右上→右下→左下）
4. 标定误差应 < 2mm

### 4. 棋盘定位
1. 放置四个车在标准位置：a1, i1, a10, i10
2. 点击"扫描棋盘更新交叉点"
3. 系统自动计算90个交叉点坐标

### 5. 测试抓取
- 点击"测试抓取精度"
- 系统会抓取置信度最高的棋子放到放置区
- 红棋放置到C区，黑棋放置到A区

### 6. AI设置
- **难度等级**: 1-20级，越高越强
- **引擎模式**: 勾选使用Pikafish引擎，取消则使用内置Minimax

### 7. 开始游戏
1. 点击"开始游戏"
2. 系统验证标准开局（32个棋子）
3. 人类走红棋，AI走黑棋
4. 人类走棋后点击"AI走棋"
5. 系统检测→验证→AI计算→机械臂执行

## 放置区域说明

| 区域 | 用途 | 位置 |
|------|------|------|
| A区 | 黑棋（被吃） | 画面左下 |
| C区 | 红棋（被吃） | 画面右上 |
| B区 | 暂不使用 | - |
| D区 | 暂不使用 | - |

## 走棋规则

- 人类控制红方，AI控制黑方
- 人类只能移动红方棋子，系统会验证走棋权限
- 走棋必须符合中国象棋规则
- 在AI思考期间不能走棋

## AI引擎

### Pikafish引擎
- 基于 Stockfish 架构的中国象棋引擎
- 使用 UCI 协议通信
- 棋力：业余高手水平
- 支持难度调节（1-20级）

### 难度等级
| 等级 | 描述 |
|------|------|
| 1-3 | 新手 |
| 4-7 | 简单 |
| 8-12 | 普通 |
| 13-17 | 困难 |
| 18-20 | 大师 |

## 标准开局验证

游戏启动时自动验证：
- 棋子总数：32个（红黑各16个）
- 每个棋子在标准位置
- 棋子颜色正确

## API 端点

### 标定
| 端点 | 方法 | 描述 |
|------|------|------|
| `/api/calibration/start` | POST | 开始标定 |
| `/api/calibration/place` | POST | 放置标定点 |
| `/api/calibration/status` | GET | 获取标定状态 |
| `/api/calibration/reset` | POST | 重置标定 |
| `/api/calibration/test_grab` | POST | 测试抓取精度 |
| `/api/calibration/scan_board` | POST | 四车定位更新交叉点 |

### 游戏
| 端点 | 方法 | 描述 |
|------|------|------|
| `/api/game/start` | POST | 开始游戏 |
| `/api/game/stop` | POST | 停止游戏 |
| `/api/game/reset` | POST | 重置游戏 |
| `/api/game/state` | GET | 获取游戏状态 |
| `/api/game/move/ai` | POST | AI走棋 |
| `/api/game/difficulty` | GET/POST | 获取/设置AI难度 |

## 配置说明

```python
# 机械臂串口
SERIAL_PORT = 'COM12'

# 摄像头
CAMERA_INDEX = 1
FRAME_SCALE = 1.5

# 机械臂高度
Z_HEIGHTS = {'safe': 65.51, 'grab': -35, 'place': -30}

# 放置区域（A区黑棋，C区红棋）
PLACE_COORDS = {
    'A': [269.02, -161.65, 51.42],   # 黑棋
    'C': [248.52, 152.35, 53.45],    # 红棋
}

# YOLO检测区域（排除放置区，只检测棋盘）
BOARD_ROI = {
    'x_min': 550, 'x_max': 1350,
    'y_min': 50, 'y_max': 850,
}

# 棋子标定 (4点透视变换)
PIECE_CALIBRATION_ORDER = ['P2', 'P3', 'P0', 'P1']  # 左上→右上→右下→左下

# 棋盘旋转
BOARD_ROTATION = 90  # 摄像头顺时针旋转90度

# AI引擎配置
ENGINE_PATH = 'engines/Pikafish.2026-01-02/Windows/pikafish-avx2.exe'
ENGINE_NNUE_PATH = 'engines/Pikafish.2026-01-02/pikafish.nnue'
ENGINE_TIME_LIMIT = 2000  # 思考时间(毫秒)
ENGINE_DIFFICULTY = 10    # 难度等级(1-20)
USE_ENGINE = True         # 是否使用引擎
```

## 故障排除

### 摄像头无法打开
- 检查 `CAMERA_INDEX` 配置
- 检查摄像头权限

### 棋子标定失败
- 确保机械臂已连接、摄像头已启动
- 检查 YOLO 棋子检测是否正常

### 棋盘定位失败
- 确保四个车在正确位置 (a1, i1, a10, i10)
- 检查光照条件

### 抓取精度不够
- 重新标定
- 确认标定点坐标正确

### 机械臂连接失败
- 检查 `SERIAL_PORT` 配置
- 检查机械臂电源和驱动

### 点击"AI走棋"提示错误
- 确保已完成走棋
- 等待机械臂离开棋盘区域
- 检查是否走了黑方棋子（只能走红方）

### AI引擎加载失败
- 检查 `engines/` 目录下引擎文件是否完整
- 确保 `pikafish.nnue` 神经网络文件存在

## 注意事项

1. 标定误差控制在 2mm 以内
2. 确保光照均匀
3. 摄像头垂直对准棋盘
4. 机械臂速度默认60，避免过快震动
5. ROI区域需根据实际摄像头画面调整，实时画面会显示蓝色ROI框
6. 人类只能走红方棋子，AI走黑方

## 许可证

MIT License