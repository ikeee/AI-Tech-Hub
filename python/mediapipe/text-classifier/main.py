"""MediaPipe 文本情感分类最简参考实现。

使用 TextClassifier 对输入文本进行情感分类（positive/negative），
适用于评论、反馈等情感分析场景。

前置准备：
    pip install mediapipe

模型文件：
    从 https://storage.googleapis.com/mediapipe-models/text_classifier/ 下载
    sentiment_classifier.tflite 放到当前目录。
"""

from mediapipe.tasks import python
from mediapipe.tasks.python import text


def classify_sentiment(text: str, model_path: str = "sentiment_classifier.tflite") -> list[dict]:
    """对输入文本进行情感分类，返回 [{label, score}]。"""
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = text.TextClassifierOptions(base_options=base_options)
    classifier = text.TextClassifier.create_from_options(options)

    result = classifier.classify(text)
    # result.classifications 可能为空列表
    if not result.classifications:
        return []
    categories = result.classifications[0].categories
    return [
        {"label": c.category_name, "score": round(c.score, 4)}
        for c in sorted(categories, key=lambda c: c.score, reverse=True)
    ]


if __name__ == "__main__":
    sample = "I really love this product, it works great!"
    results = classify_sentiment(sample)
    for item in results:
        print(f"{item['label']}: {item['score']}")
