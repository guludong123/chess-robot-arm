# 中国象棋对战机器人系统 - 环境配置指南

## 硬件要求

- **操作系统**：Windows 10/11
- **GPU**：NVIDIA 显卡 + CUDA 11.8（用于 YOLO 和 FunASR 加速）
- **摄像头**：USB 摄像头（用于棋盘识别）
- **机械臂**：myCobot 280 或 320（串口连接）
- **Pikafish 引擎**：已包含在 `engines/` 目录中

## 快速开始

### 1. 安装 Miniconda

下载地址：https://docs.conda.io/en/latest/miniconda.html

安装后打开 **Anaconda Prompt**。

### 2. 创建环境

```bash
conda env create -f environment.yml
conda activate chessrobot
```

### 3. 配置 API Key

复制模板并填入你自己的 key：

```bash
copy .env.example .env
```

编辑 `.env` 文件，填入：
- `DASHSCOPE_API_KEY`：阿里云 DashScope（用于 LLM 解说和云端 ASR）
- `STEPFUN_API_KEY`：StepFun（用于 E2E 语音，可选）

### 4. 检查硬件配置

编辑 `config.py`，修改以下配置匹配你的硬件：

```python
# 机械臂串口号（设备管理器中查看）
SERIAL_PORT = 'COM12'  # 改成你的串口号

# 摄像头索引（0=默认摄像头，多摄像头逐个试）
CAMERA_INDEX = 1
```

### 5. 启动

```bash
python app.py
```

访问 http://localhost:5000

## 常见问题

**Q: 没有 NVIDIA 显卡怎么办？**

修改 `requirements.txt` 中的 torch 相关行，去掉 `+cu118` 后缀，或执行：
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```
然后在 `config.py` 中将 `FUNASR_DEVICE` 改为 `'cpu'`。

**Q: 摄像头画面黑屏？**

检查 `config.py` 中的 `CAMERA_INDEX`，尝试改为 `1` 或 `2`。

**Q: 机械臂不响应？**

确认串口号正确，设备管理器中查看 COM 端口号。

**Q: FunASR 模型下载失败？**

模型已包含在 `models/funasr/` 目录中，无需额外下载。
