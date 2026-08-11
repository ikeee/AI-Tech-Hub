"""姿态估计最简实现：基于 MediaPipe Tasks Pose Landmarker。

使用方法：
    python main.py <图片路径>

输出：每人体 33 个姿态关键点（归一化坐标）。
模型文件：public/model/mediapipe/models/pose_landmarker_lite.task
"""

import sys
from pathlib import Path

import mediapipe as mp
from mediapipe.tasks.python import vision

MODEL_PATH = Path(__file__).resolve().parents[3] / "public" / "model" / "mediapipe" / "models" / "pose_landmarker_lite.task"


def detect(image_path: str) -> None:
    options = vision.PoseLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=vision.RunningMode.IMAGE,
        num_poses=1,
    )

    with vision.PoseLandmarker.create_from_options(options) as landmarker:
        image = mp.Image.create_from_file(image_path)
        results = landmarker.detect(image)

        if not results.pose_landmarks:
            print("未检测到人体")
            return

        for i, landmarks in enumerate(results.pose_landmarks, 1):
            print(f"人体 {i}: {len(landmarks)} 个关键点")
            nose = landmarks[0]
            print(f"  鼻子: x={nose.x:.4f}, y={nose.y:.4f}, z={nose.z:.4f}")


if __name__ == "__main__":
    detect(sys.argv[1] if len(sys.argv) > 1 else "input.jpg")
