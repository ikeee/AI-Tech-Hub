"""图像迁移学习演示：MobileNetV2 特征提取 + KNN 分类。

对应浏览器端 MobileNet + KNN 的 Python 参考。加载预训练
MobileNetV2 作为特征提取器，对若干类别图片提取嵌入，用 KNN
完成少样本分类。

依赖安装：
    pip install tensorflow scikit-learn numpy pillow

用法：
    python main.py
"""

import numpy as np
import tensorflow as tf
from PIL import Image
from sklearn.neighbors import KNeighborsClassifier

# 假设数据组织：data/<类别名>/<图片>.jpg
DATA_DIR = "data"
IMG_SIZE = 224


def build_feature_extractor() -> tf.keras.Model:
    """加载 MobileNetV2，去掉分类头，输出 1280 维特征向量。"""
    base = tf.keras.applications.MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3), include_top=False, pooling="avg"
    )
    base.trainable = False
    return base


def load_image(path: str) -> np.ndarray:
    img = Image.open(path).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    arr = np.asarray(img, dtype="float32")
    return tf.keras.applications.mobilenet_v2.preprocess_input(arr)


def train_and_predict() -> None:
    extractor = build_feature_extractor()

    # 收集样本：每个子目录是一个类别
    import os
    X, y = [], []
    for cls in sorted(os.listdir(DATA_DIR)):
        cls_dir = os.path.join(DATA_DIR, cls)
        if not os.path.isdir(cls_dir):
            continue
        for fname in os.listdir(cls_dir):
            if fname.lower().endswith((".jpg", ".png", ".jpeg")):
                X.append(load_image(os.path.join(cls_dir, fname)))
                y.append(cls)

    feats = extractor.predict(np.stack(X), verbose=0)
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(feats, y)

    # 预测示例
    sample = load_image("test.jpg")[None, ...]
    feat = extractor.predict(sample, verbose=0)
    print("预测类别:", knn.predict(feat)[0])


if __name__ == "__main__":
    train_and_predict()
