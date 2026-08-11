"""图像嵌入与相似度计算最简实现：基于 MediaPipe Tasks ImageEmbedder。

使用方法：
    python main.py <图片1> <图片2>

前置准备：
    下载 MobileNet 嵌入模型并放置于本目录（如 mobilenet_v3_small.tflite）：
    https://storage.googleapis.com/mediapipe-models/image_embedder/mobilenet_v3_small/float32/latest/mobilenet_v3_small.tflite

依赖安装：pip install mediapipe
"""

import sys

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


def embed_and_compare(image_path1: str, image_path2: str) -> None:
    """计算两张图片的嵌入向量并输出余弦相似度。"""
    base_options = python.BaseOptions(
        model_asset_path="mobilenet_v3_small.tflite"
    )
    options = vision.ImageEmbedderOptions(
        base_options=base_options,
        l2_normalize=True,  # 归一化后便于余弦相似度计算
    )

    with vision.ImageEmbedder.create_from_options(options) as embedder:
        img1 = mp.Image.create_from_file(image_path1)
        img2 = mp.Image.create_from_file(image_path2)

        emb1 = embedder.embed(img1)
        emb2 = embedder.embed(img2)

        # 使用静态方法计算余弦相似度
        similarity = vision.ImageEmbedder.cosine_similarity(
            emb1.embeddings[0], emb2.embeddings[0]
        )
        print(f"图片1: {image_path1}")
        print(f"图片2: {image_path2}")
        print(f"余弦相似度: {similarity:.4f}")


if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) < 2:
        print("用法: python main.py <图片1> <图片2>")
        sys.exit(1)
    embed_and_compare(args[0], args[1])
