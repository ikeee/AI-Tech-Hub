"""图像嵌入最简实现：基于 MediaPipe Tasks Image Embedder。

使用方法：
    python main.py <图片A> <图片B>

输出：两张图片的嵌入向量余弦相似度（越接近 1 越相似）。
模型文件：public/model/mediapipe/models/mobilenet_v3_small.tflite
"""

import sys
from pathlib import Path

import mediapipe as mp
from mediapipe.tasks.python import vision

MODEL_PATH = Path(__file__).resolve().parents[3] / "public" / "model" / "mediapipe" / "models" / "mobilenet_v3_small.tflite"


def embed(image_path: str, l2_normalize: bool = True) -> tuple:
    options = vision.ImageEmbedderOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=vision.RunningMode.IMAGE,
        l2_normalize=l2_normalize,
    )

    with vision.ImageEmbedder.create_from_options(options) as embedder:
        image = mp.Image.create_from_file(image_path)
        result = embedder.embed(image)
        return result.embeddings[0]


def main() -> None:
    if len(sys.argv) < 3:
        print("用法: python main.py <图片A> <图片B>")
        sys.exit(1)
    e1 = embed(sys.argv[1])
    e2 = embed(sys.argv[2])
    similarity = vision.ImageEmbedder.cosine_similarity(e1, e2)
    print(f"余弦相似度: {similarity:.4f}")


if __name__ == "__main__":
    main()
