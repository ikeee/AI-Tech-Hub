"""手部 21 关键点检测最简实现：基于 MediaPipe Tasks HandLandmarker。

使用方法：
    python main.py <图片路径>

前置准备：
    下载 hand_landmarker.task 模型文件并放置于本目录：
    https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task

依赖安装：pip install mediapipe
"""

import sys

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


def detect(image_path: str) -> None:
    """读取图片并检测手部 21 关键点，输出每只手的关键点坐标。"""
    base_options = python.BaseOptions(model_asset_path="hand_landmarker.task")
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=2,
    )

    with vision.HandLandmarker.create_from_options(options) as landmarker:
        image = mp.Image.create_from_file(image_path)
        result = landmarker.detect(image)

        if not result.hand_landmarks:
            print("未检测到手部")
            return

        for i, hand in enumerate(result.hand_landmarks, 1):
            # handedness 标签如 "Left" / "Right"
            label = result.handedness[i - 1][0].category_name
            print(f"手 {i} ({label}): 检测到 {len(hand)} 个关键点")
            for j, lm in enumerate(hand, 1):
                print(f"  点 {j}: x={lm.x:.4f} y={lm.y:.4f} z={lm.z:.4f}")


if __name__ == "__main__":
    detect(sys.argv[1] if len(sys.argv) > 1 else "input.jpg")
