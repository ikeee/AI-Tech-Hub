"""整体检测最简实现：基于 MediaPipe Tasks Holistic Landmarker。

使用方法：
    python main.py <图片路径>

输出：同时检测人脸、左右手与人体姿态关键点数量。
模型文件：public/model/mediapipe/models/holistic_landmarker.task
"""

import sys
from pathlib import Path

import mediapipe as mp
from mediapipe.tasks.python import vision

MODEL_PATH = Path(__file__).resolve().parents[3] / "public" / "model" / "mediapipe" / "models" / "holistic_landmarker.task"


def detect(image_path: str) -> None:
    options = vision.HolisticLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=vision.RunningMode.IMAGE,
    )

    with vision.HolisticLandmarker.create_from_options(options) as landmarker:
        image = mp.Image.create_from_file(image_path)
        results = landmarker.detect(image)

        parts = {
            "人脸": len(results.face_landmarks),
            "姿态": len(results.pose_landmarks),
            "左手": len(results.left_hand_landmarks),
            "右手": len(results.right_hand_landmarks),
        }
        print("检测结果:", ", ".join(f"{k}={v}" for k, v in parts.items()))


if __name__ == "__main__":
    detect(sys.argv[1] if len(sys.argv) > 1 else "input.jpg")
