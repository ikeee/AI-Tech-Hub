"""人脸检测最简实现：基于 MediaPipe Tasks Face Detector。

使用方法：
    python main.py <图片路径>

说明：
- model_selection=0 适用于 2 米以内的人脸（短距离）
- model_selection=1 适用于 5 米以内的人脸（长距离）
- 模型文件使用项目本地 public/model/mediapipe/models/ 下的 tflite

依赖安装：pip install mediapipe opencv-python
"""

import sys
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision

MODEL_PATH = Path(__file__).resolve().parents[3] / "public" / "model" / "mediapipe" / "models" / "blaze_face_short_range.tflite"


def detect(image_path: str) -> None:
    """读取图片并检测人脸，输出归一化坐标和置信度。"""
    options = vision.FaceDetectorOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=vision.RunningMode.IMAGE,
        min_detection_confidence=0.5,
    )

    with vision.FaceDetector.create_from_options(options) as detector:
        image = mp.Image.create_from_file(image_path)
        results = detector.detect(image)

        if not results.detections:
            print("未检测到人脸")
            return

        for i, det in enumerate(results.detections, 1):
            box = det.bounding_box
            score = det.categories[0].score if det.categories else 0.0
            print(f"人脸 {i}: 置信度={score:.3f}")
            print(f"  边界框: x={box.origin_x}, y={box.origin_y}, w={box.width}, h={box.height}")


if __name__ == "__main__":
    detect(sys.argv[1] if len(sys.argv) > 1 else "input.jpg")
