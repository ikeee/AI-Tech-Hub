"""手部关键点最简实现：基于 MediaPipe Tasks Hand Landmarker。

使用方法：
    python main.py <图片路径>

输出：每只手 21 个关键点（归一化坐标）与左右手判定。
模型文件：public/model/mediapipe/models/hand_landmarker.task
"""

import sys
from pathlib import Path

import mediapipe as mp
from mediapipe.tasks.python import vision

MODEL_PATH = Path(__file__).resolve().parents[3] / "public" / "model" / "mediapipe" / "models" / "hand_landmarker.task"


def detect(image_path: str) -> None:
    options = vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=vision.RunningMode.IMAGE,
        num_hands=2,
    )

    with vision.HandLandmarker.create_from_options(options) as landmarker:
        image = mp.Image.create_from_file(image_path)
        results = landmarker.detect(image)

        if not results.hand_landmarks:
            print("未检测到手部")
            return

        for i, landmarks in enumerate(results.hand_landmarks, 1):
            handedness = results.handedness[i - 1][0].category_name if results.handedness else "?"
            print(f"手 {i} ({handedness}): {len(landmarks)} 个关键点")
            wrist = landmarks[0]
            print(f"  腕部: x={wrist.x:.4f}, y={wrist.y:.4f}")


if __name__ == "__main__":
    detect(sys.argv[1] if len(sys.argv) > 1 else "input.jpg")
