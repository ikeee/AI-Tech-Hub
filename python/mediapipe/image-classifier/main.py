"""图像分类最简实现：基于 MediaPipe Tasks ImageClassifier。

使用方法：
    python main.py <图片路径>

前置准备：
    下载 efficientnet_lite0.tflite 模型文件并放置于本目录：
    https://storage.googleapis.com/mediapipe-models/image_classifier/efficientnet_lite0/float32/latest/efficientnet_lite0.tflite

依赖安装：pip install mediapipe
"""

import sys

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


def classify(image_path: str) -> None:
    """读取图片并分类，输出 Top-K 分类结果。"""
    base_options = python.BaseOptions(model_asset_path="efficientnet_lite0.tflite")
    options = vision.ImageClassifierOptions(
        base_options=base_options,
        max_results=5,  # Top-K
    )

    with vision.ImageClassifier.create_from_options(options) as classifier:
        image = mp.Image.create_from_file(image_path)
        result = classifier.classify(image)

        if not result.classifications:
            print("无分类结果")
            return

        top = result.classifications[0].categories
        print(f"Top-{len(top)} 分类结果:")
        for i, cat in enumerate(top, 1):
            print(f"  {i}. {cat.category_name} (置信度={cat.score:.4f})")


if __name__ == "__main__":
    classify(sys.argv[1] if len(sys.argv) > 1 else "input.jpg")
