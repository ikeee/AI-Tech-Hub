"""目标检测最简实现：基于 MediaPipe Tasks ObjectDetector。

使用方法：
    python main.py <图片路径>

前置准备：
    下载 efficientdet.tflite 模型文件并放置于本目录：
    https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/int8/latest/efficientdet_lite0.tflite

依赖安装：pip install mediapipe
"""

import sys

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


def detect(image_path: str) -> None:
    """读取图片并检测目标，输出类别、置信度和边界框。"""
    base_options = python.BaseOptions(model_asset_path="efficientdet.tflite")
    options = vision.ObjectDetectorOptions(
        base_options=base_options,
        score_threshold=0.5,
        max_results=5,
    )

    with vision.ObjectDetector.create_from_options(options) as detector:
        image = mp.Image.create_from_file(image_path)
        result = detector.detect(image)

        if not result.detections:
            print("未检测到目标")
            return

        for i, det in enumerate(result.detections, 1):
            cat = det.categories[0]
            box = det.bounding_box
            print(f"目标 {i}: 类别={cat.category_name} 置信度={cat.score:.4f}")
            print(f"  边界框: origin(x={box.origin_x}, y={box.origin_y}) "
                  f"size(w={box.width}, h={box.height})")


if __name__ == "__main__":
    detect(sys.argv[1] if len(sys.argv) > 1 else "input.jpg")
