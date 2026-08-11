"""人脸 478 关键点检测最简实现：基于 MediaPipe Tasks FaceLandmarker。

使用方法：
    python main.py <图片路径>

前置准备：
    下载 face_landmarker.task 模型文件并放置于本目录：
    https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task

依赖安装：pip install mediapipe
"""

import sys

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


def detect(image_path: str) -> None:
    """读取图片并检测人脸 478 关键点，输出关键点数量。"""
    base_options = python.BaseOptions(model_asset_path="face_landmarker.task")
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        num_faces=1,
    )

    with vision.FaceLandmarker.create_from_options(options) as landmarker:
        image = mp.Image.create_from_file(image_path)
        result = landmarker.detect(image)

        if not result.face_landmarks:
            print("未检测到人脸")
            return

        for i, face in enumerate(result.face_landmarks, 1):
            print(f"人脸 {i}: 检测到 {len(face)} 个关键点")
            # 仅展示前 3 个关键点
            for j, lm in enumerate(face[:3], 1):
                print(f"  点 {j}: x={lm.x:.4f} y={lm.y:.4f} z={lm.z:.4f}")


if __name__ == "__main__":
    detect(sys.argv[1] if len(sys.argv) > 1 else "input.jpg")
