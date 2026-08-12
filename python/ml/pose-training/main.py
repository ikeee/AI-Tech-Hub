"""
姿态迁移学习演示：关键点相对归一化 + KNN 分类（最简实现）。

对应浏览器端 MediaPipe PoseLandmarker + KNN 的 Python 参考：
以髋部中心为原点、肩宽为尺度做相对归一化，使同一动作在不同
距离/身位下特征一致，再用 KNN 完成少样本分类。

依赖安装：
    pip install scikit-learn numpy

输入格式（JSON）：
    {"samples": [{"class": "A", "landmarks": [[x, y, z] x 33]}, ...]}
    其中 landmarks 为 MediaPipe pose-landmarker 输出的归一化坐标。

用法：
    python main.py train.json test.json
"""

import json
import sys

import numpy as np
from sklearn.neighbors import KNeighborsClassifier


def normalize(landmarks):
    """髋部中心为原点、肩宽为尺度做相对归一化，返回 66 维特征。"""
    lhip, rhip = landmarks[23], landmarks[24]
    lsh, rsh = landmarks[11], landmarks[12]
    cx = (lhip[0] + rhip[0]) / 2
    cy = (lhip[1] + rhip[1]) / 2
    scale = float(np.hypot(lsh[0] - rsh[0], lsh[1] - rsh[1]))
    if scale < 1e-6:
        return None
    feat = []
    for x, y, _z in landmarks:
        feat.append((x - cx) / scale)
        feat.append((y - cy) / scale)
    return feat


def main() -> None:
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")
        except Exception:
            pass

    if len(sys.argv) < 3:
        print("用法: python main.py train.json test.json", flush=True)
        return

    with open(sys.argv[1], encoding="utf-8") as f:
        train = json.load(f)
    X, y = [], []
    for item in train["samples"]:
        feat = normalize(item["landmarks"])
        if feat is not None:
            X.append(feat)
            y.append(item["class"])
    if not X:
        print("训练集为空或无法提取姿态特征", flush=True)
        return

    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(np.asarray(X), y)

    with open(sys.argv[2], encoding="utf-8") as f:
        test = json.load(f)
    feat = normalize(test["landmarks"])
    if feat is None:
        print("测试样本无法提取姿态特征", flush=True)
        return
    print("预测类别:", knn.predict([feat])[0], flush=True)


if __name__ == "__main__":
    main()
