"""MediaPipe 音频事件分类（YAMNet）最简参考实现。

使用 AudioClassifier 读取 wav 文件并输出 Top-K 分类结果，
适用于环境音、人声、音乐等事件识别。

前置准备：
    pip install mediapipe scipy

模型文件：
    从 https://storage.googleapis.com/mediapipe-models/audio_classifier/yamnet/tflite/ 下载
    yamnet.tflite 放到当前目录。
"""

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import audio
from scipy.io import wavfile


def classify_audio(wav_path: str, model_path: str = "yamnet.tflite", top_k: int = 5) -> list[dict]:
    """读取 wav 文件，返回 Top-K 分类结果（类别名 + 置信度）。"""
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = audio.AudioClassifierOptions(base_options=base_options)
    classifier = audio.AudioClassifier.create_from_options(options)

    # 读取 wav 文件并构造 AudioData
    sample_rate, waveform = wavfile.read(wav_path)
    # 多声道时取平均，转为 float32 归一化
    if waveform.ndim > 1:
        waveform = waveform.mean(axis=1)
    audio_clip = mp.tasks.audio.AudioData.create_from_array(
        waveform.astype("float32"), sample_rate
    )

    result = classifier.classify(audio_clip)
    classifications = result.classifications[0]
    # 按置信度降序取 Top-K
    sorted_cats = sorted(classifications.categories, key=lambda c: c.score, reverse=True)
    return [{"category": c.category_name, "score": round(c.score, 4)} for c in sorted_cats[:top_k]]


if __name__ == "__main__":
    results = classify_audio("sample.wav")
    for item in results:
        print(f"{item['category']}: {item['score']}")
