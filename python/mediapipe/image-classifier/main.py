"""图像分类最简实现：基于 MediaPipe Tasks Image Classifier。

使用方法：
    python main.py <图片路径>

输出：Top-5 分类结果（类别 + 置信度）。
模型文件：public/model/mediapipe/models/efficientnet_lite0.tflite
"""

import sys
from pathlib import Path

import mediapipe as mp
from mediapipe.tasks.python import vision

MODEL_PATH = Path(__file__).resolve().parents[3] / "public" / "model" / "mediapipe" / "models" / "efficientnet_lite0.tflite"


def classify(image_path: str) -> None:
    options = vision.ImageClassifierOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=vision.RunningMode.IMAGE,
        max_results=5,
    )

    with vision.ImageClassifier.create_from_options(options) as classifier:
        image = mp.Image.create_from_file(image_path)
        results = classifier.classify(image)

        if not results.classifications:
            print("无分类结果")
            return

        for i, cat in enumerate(results.classifications[0].categories, 1):
            print(f"{i}. {cat.category_name} ({cat.score:.3f})")


if __name__ == "__main__":
    classify(sys.argv[1] if len(sys.argv) > 1 else "input.jpg")
