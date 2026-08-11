"""声音迁移学习演示：YAMNet 特征提取 + KNN 分类。

对应浏览器端 Speech Commands + KNN 的 Python 参考。加载预训练
YAMNet 模型，对 16kHz 单声道 wav 提取 1024 维嵌入，用 KNN
完成少样本音频分类。

依赖安装：
    pip install tensorflow tensorflow-hub scikit-learn numpy

用法：
    python main.py
"""

import os

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from sklearn.neighbors import KNeighborsClassifier

YAMNET_URL = "https://tfhub.dev/google/yamnet/1"
DATA_DIR = "data"  # 组织：data/<类别名>/<音频>.wav
SR = 16000


def load_wav(path: str) -> np.ndarray:
    """读取 16kHz 单声道 wav，返回 float32 一维数组。"""
    wav, sr = tf.audio.decode_wav(tf.io.read_file(path))
    wav = tf.squeeze(wav, axis=-1).numpy().astype("float32")
    if sr != SR:
        wav = tf.signal.resample(wav, int(len(wav) * SR / sr)).numpy()
    return wav


def extract_embedding(model, wav: np.ndarray) -> np.ndarray:
    """YAMNet 输出多帧嵌入，取平均得到单条样本特征。"""
    scores, embeddings, _ = model(wav)
    return np.mean(embeddings.numpy(), axis=0)


def train_and_predict() -> None:
    yamnet = hub.KerasLayer(YAMNET_URL, trainable=False)

    X, y = [], []
    for cls in sorted(os.listdir(DATA_DIR)):
        cls_dir = os.path.join(DATA_DIR, cls)
        if not os.path.isdir(cls_dir):
            continue
        for fname in os.listdir(cls_dir):
            if fname.lower().endswith(".wav"):
                wav = load_wav(os.path.join(cls_dir, fname))
                X.append(extract_embedding(yamnet, wav))
                y.append(cls)

    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(np.stack(X), y)

    # 预测示例
    sample = extract_embedding(yamnet, load_wav("test.wav"))
    print("预测类别:", knn.predict(sample[None, ...])[0])


if __name__ == "__main__":
    train_and_predict()
