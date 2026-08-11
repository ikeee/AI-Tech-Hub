"""MediaPipe 语言检测最简参考实现。

使用 LanguageDetector 检测输入文本的语言，返回语言代码和概率，
适用于多语言内容路由、翻译入口判断等场景。

前置准备：
    pip install mediapipe

模型文件：
    从 https://storage.googleapis.com/mediapipe-models/language_detector/ 下载
    language_detector.tflite 放到当前目录。
"""

from mediapipe.tasks import python
from mediapipe.tasks.python import text


def detect_language(text: str, model_path: str = "language_detector.tflite") -> list[dict]:
    """检测输入文本的语言，返回 [{language, probability}]。"""
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = text.LanguageDetectorOptions(base_options=base_options)
    detector = text.LanguageDetector.create_from_options(options)

    result = detector.detect(text)
    # result.languages 已按概率降序排列
    return [
        {"language": lang.language_code, "probability": round(lang.probability, 4)}
        for lang in result.languages
    ]


if __name__ == "__main__":
    sample = "Bonjour, comment allez-vous?"
    results = detect_language(sample)
    for item in results:
        print(f"{item['language']}: {item['probability']}")
