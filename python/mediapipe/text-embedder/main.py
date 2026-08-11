"""MediaPipe 文本嵌入与相似度计算最简参考实现。

使用 TextEmbedder 将文本映射为向量，并计算两段文本的余弦相似度，
适用于语义检索、聚类、去重等场景。

前置准备：
    pip install mediapipe

模型文件：
    从 https://storage.googleapis.com/mediapipe-models/text_embedder/ 下载
    universal_sentence_encoder.tflite，重命名为 text_embedder.tflite 放到当前目录。
"""

from mediapipe.tasks import python
from mediapipe.tasks.python import text


def compute_similarity(text_a: str, text_b: str, model_path: str = "text_embedder.tflite") -> float:
    """计算两段文本的余弦相似度，返回 [-1, 1] 之间的浮点数。"""
    base_options = python.BaseOptions(model_asset_path=model_path)
    # l2_normalize=True 后可直接用余弦相似度度量
    options = text.TextEmbedderOptions(base_options=base_options, l2_normalize=True)
    embedder = text.TextEmbedder.create_from_options(options)

    embed_a = embedder.embed(text_a).embeddings[0]
    embed_b = embedder.embed(text_b).embeddings[0]

    # 使用静态方法计算余弦相似度
    similarity = text.TextEmbedder.cosine_similarity(embed_a, embed_b)
    return round(float(similarity), 4)


if __name__ == "__main__":
    a = "The cat is sitting on the mat."
    b = "A cat is on the mat."
    score = compute_similarity(a, b)
    print(f"cosine similarity: {score}")
