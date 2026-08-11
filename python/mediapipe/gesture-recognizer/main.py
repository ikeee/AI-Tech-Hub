"""手势识别最简实现：基于 MediaPipe Tasks GestureRecognizer。

使用方法：
    python main.py <图片路径>

前置准备：
    下载 gesture_recognizer.task 模型文件并放置于本目录：
    https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task

依赖安装：pip install mediapipe
"""

import sys

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


def recognize(image_path: str) -> None:
    """读取图片并识别手势，输出手势类别（如 Thumb_Up、Victory）和置信度。"""
    base_options = python.BaseOptions(model_asset_path="gesture_recognizer.task")
    options = vision.GestureRecognizerOptions(
        base_options=base_options,
        num_hands=2,
    )

    with vision.GestureRecognizer.create_from_options(options) as recognizer:
        image = mp.Image.create_from_file(image_path)
        result = recognizer.recognize(image)

        if not result.gestures:
            print("未检测到手势")
            return

        for i, hand_gestures in enumerate(result.gestures, 1):
            top = hand_gestures[0]  # 置信度最高的手势
            print(f"手 {i}: 手势={top.category_name} 置信度={top.score:.4f}")


if __name__ == "__main__":
    recognize(sys.argv[1] if len(sys.argv) > 1 else "input.jpg")
