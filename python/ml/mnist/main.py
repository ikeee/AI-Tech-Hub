"""
MNIST 手写数字训练（Python 最简实现）。

对应浏览器端 TF.js 神经网络的参考实现：读取项目内 public/model/mnist/mnist.bin
（前 10000 张 28x28 灰度图 + 标签），用 sklearn MLPClassifier 训练并输出精度。

依赖安装：
    pip install scikit-learn numpy

用法：
    python main.py [num_samples]
"""

import os
import sys

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier


def main() -> None:
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")
        except Exception:
            pass

    num = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
    bin_path = os.path.join(os.path.dirname(__file__), "..", "..", "..", "public", "model", "mnist", "mnist.bin")
    bin_path = os.path.normpath(bin_path)
    with open(bin_path, "rb") as f:
        buf = f.read()
    n_total = (len(buf) - 784) // 785
    n = min(num, n_total)
    pixels = np.frombuffer(buf[: n * 784], dtype=np.uint8).reshape(n, 784) / 255.0
    labels = np.frombuffer(buf[n * 784 : n * 785], dtype=np.uint8)

    clf = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=10, random_state=42)
    clf.fit(pixels, labels)
    pred = clf.predict(pixels)
    print(f"样本数: {n}", flush=True)
    print(f"训练精度: {accuracy_score(labels, pred):.4f}", flush=True)


if __name__ == "__main__":
    main()
