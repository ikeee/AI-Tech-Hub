"""目标检测最简实现：基于 MediaPipe Tasks Object Detector。

使用方法：
    python main.py <图片路径>

输出：检测到的目标类别、置信度与边界框。
模型文件：public/model/mediapipe/models/efficientdet_lite0.tflite
"""

import sys
from pathlib import Path

import mediapipe as mp
from mediapipe.tasks.python import vision

MODEL_PATH = Path(__file__).resolve().parents[3] / "public" / "model" / "mediapipe" / "models" / "efficientdet_lite0.tflite"


def detect(image_path: str) -> None:
    options = vision.ObjectDetectorOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=vision.RunningMode.IMAGE,
        max_results=5,
        score_threshold=0.5,
    )

    with vision.ObjectDetector.create_from_options(options) as detector:
        image = mp.Image.create_from_file(image_path)
        results = detector.detect(image)

        if not results.detections:
            print("未检测到目标")
            return

        for i, det in enumerate(results.detections, 1):
            cat = det.categories[0]
            box = det.bounding_box
            print(f"目标 {i}: {cat.category_name} 置信度={cat.score:.3f}")
            print(f"  边界框: x={box.origin_x}, y={box.origin_y}, w={box.width}, h={box.height}")


if __name__ == "__main__":
    detect(sys.argv[1] if len(sys.argv) > 1 else "input.jpg")
