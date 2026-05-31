"""
摄像头管理模块
"""
import cv2
import threading
import time
import numpy as np
from typing import Optional, Callable
import config


class CameraManager:
    """摄像头管理器 - 处理视频流捕获"""

    def __init__(self, camera_index: int = config.CAMERA_INDEX):
        self.camera_index = camera_index
        self.cap: Optional[cv2.VideoCapture] = None
        self.frame: Optional[np.ndarray] = None
        self.running = False
        self.lock = threading.Lock()
        self.thread: Optional[threading.Thread] = None

    def start(self) -> bool:
        """启动摄像头"""
        if self.cap is not None:
            return True

        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            # 尝试备用摄像头
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                print("无法打开摄像头")
                self.cap = None
                return False

        # 设置分辨率
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
        self.cap.set(cv2.CAP_PROP_FPS, config.FPS)

        self.running = True
        self.thread = threading.Thread(target=self._capture_loop)
        self.thread.daemon = True
        self.thread.start()

        print(f"摄像头已启动，分辨率: {config.FRAME_WIDTH}x{config.FRAME_HEIGHT}")
        return True

    def stop(self):
        """停止摄像头"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2)
        if self.cap:
            self.cap.release()
            self.cap = None
        print("摄像头已停止")

    def _capture_loop(self):
        """视频捕获线程"""
        while self.running:
            if self.cap:
                ret, frame = self.cap.read()
                if ret:
                    with self.lock:
                        self.frame = frame.copy()
            time.sleep(0.01)

    def get_frame(self) -> Optional[np.ndarray]:
        """获取当前帧"""
        with self.lock:
            return self.frame.copy() if self.frame is not None else None

    def get_frame_scaled(self, scale: float = config.FRAME_SCALE) -> Optional[np.ndarray]:
        """获取缩放后的帧"""
        frame = self.get_frame()
        if frame is not None:
            return cv2.resize(frame, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        return None

    def is_running(self) -> bool:
        """检查摄像头是否运行中"""
        return self.running and self.cap is not None

    def get_frame_bytes(self, quality: int = 85) -> Optional[bytes]:
        """获取JPEG格式的帧字节数据"""
        frame = self.get_frame()
        if frame is not None:
            ret, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
            if ret:
                return jpeg.tobytes()
        return None
