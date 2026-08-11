"""手势识别最简实现：基于 MediaPipe Tasks Gesture Recognizer。

使用方法：
    python main.py <图片路径>

输出：手部手势类别（None / Closed_Fist / Open_Palm / Pointing_Up / Thumb_Down / Thumb_Up / Victory / ILoveYou）。
模型文件：public/model/mediapipe/models/gesture_recognizer.task
"""

import sys
from pathlib import Path

import mediapipe as mp
from mediapipe.tasks.python import vision

MODEL_PATH = Path(__file__).resolve().parents[3] / "public" / "model" / "mediapipe" / "models" / "gesture_recognizer.task"


def recognize(image_path: str) -> None:
    options = vision.GestureRecognizerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=vision.RunningMode.IMAGE,
        num_hands=2,
    )

    with vision.GestureRecognizer.create_from_options(options) as recognizer:
        image = mp.Image.create_from_file(image_path)
        results = recognizer.recognize(image)

        if not results.gestures:
            print("未检测到手部")
            return

        for i, gestures in enumerate(results.gestures, 1):
            top = gestures[0]
            print(f"手 {i}: 手势={top.category_name} 置信度={top.score:.3f}")


if __name__ == "__main__":
    recognize(sys.argv[1] if len(sys.argv) > 1 else "input.jpg")
