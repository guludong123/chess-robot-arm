"""
YOLO 模型单帧推理时间测试
测试环境: RTX 3060
"""

import time
import cv2
import numpy as np
from ultralytics import YOLO

MODEL_PATH = r"F:\deeplearning\ultralytics-8.3.163\runs\detect\train15\weights\best.pt"
IMAGE_PATH = r"F:\chessrobotarm\static\grab_offset_1779354108.jpg"  # 用任意一张棋盘图片
NUM_WARMUP = 5
NUM_TEST = 100


def main():
    print(f"加载模型: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)

    # 读取测试图片
    image = cv2.imread(IMAGE_PATH)
    if image is None:
        print(f"图片读取失败，使用随机测试图")
        image = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
    else:
        print(f"测试图片: {IMAGE_PATH}, 尺寸: {image.shape}")

    # 预热
    print(f"\n预热 {NUM_WARMUP} 次...")
    for _ in range(NUM_WARMUP):
        model.predict(image, verbose=False, conf=0.5)
    print("预热完成")

    # 正式测试
    print(f"\n开始测试 {NUM_TEST} 次推理...")
    times = []
    for i in range(NUM_TEST):
        start = time.perf_counter()
        model.predict(image, verbose=False, conf=0.5)
        elapsed_ms = (time.perf_counter() - start) * 1000
        times.append(elapsed_ms)

    # 统计结果
    times = np.array(times)
    print("\n" + "=" * 40)
    print("推理时间统计结果")
    print("=" * 40)
    print(f"平均:   {times.mean():.2f} ms")
    print(f"中位数: {np.median(times):.2f} ms")
    print(f"标准差: {times.std():.2f} ms")
    print(f"最小:   {times.min():.2f} ms")
    print(f"最大:   {times.max():.2f} ms")
    print(f"P95:    {np.percentile(times, 95):.2f} ms")
    print(f"P99:    {np.percentile(times, 99):.2f} ms")
    print(f"FPS:    {1000 / times.mean():.1f}")
    print("=" * 40)


if __name__ == "__main__":
    main()
