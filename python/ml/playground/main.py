"""
神经网络游乐场（Python 最简实现）：sklearn MLPClassifier 在 2D 数据上训练。

对应浏览器端手写 BP 神经网络的参考实现。生成 circle/xor/gaussian/spiral
四类 2D 数据，训练多层感知机并输出精度与决策边界采样值。

依赖安装：
    pip install scikit-learn numpy

用法：
    python main.py [dataset] [hidden] [epochs]
    例如：python main.py xor "4,2" 300
"""

import sys

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier


def make_data(kind: str, n: int = 240, seed: int = 42):
    rng = np.random.default_rng(seed)
    X, y = [], []
    for i in range(n):
        if kind == "circle":
            angle = rng.uniform(0, 2 * np.pi)
            radius = rng.uniform(1, 3) if rng.random() < 0.5 else rng.uniform(3, 5)
            x = [np.cos(angle) * radius, np.sin(angle) * radius]
            label = 1 if radius < 2.5 else 0
        elif kind == "xor":
            qx, qy = rng.choice([-1, 1], 2)
            x = [qx * (1 + abs(rng.normal()) * 0.8), qy * (1 + abs(rng.normal()) * 0.8)]
            label = 1 if qx * qy < 0 else 0
        elif kind == "gaussian":
            cx = rng.choice([-2, 2])
            x = [cx + rng.normal() * 1.2, rng.normal() * 1.2]
            label = 1 if cx < 0 else 0
        else:  # spiral
            t = i / n
            angle = t * np.pi * 4
            radius = 1 + t * 4
            side = 1 if i % 2 == 0 else -1
            x = [np.cos(angle) * radius * side * 0.9, np.sin(angle) * radius * side * 0.9]
            label = 1 if i % 2 == 0 else 0
        X.append(x)
        y.append(label)
    return np.asarray(X), np.asarray(y)


def main() -> None:
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")
        except Exception:
            pass

    kind = sys.argv[1] if len(sys.argv) > 1 else "xor"
    hidden = [int(v) for v in sys.argv[2].split(",")] if len(sys.argv) > 2 else [4, 2]
    epochs = int(sys.argv[3]) if len(sys.argv) > 3 else 300

    X, y = make_data(kind)
    clf = MLPClassifier(hidden_layer_sizes=hidden, activation="tanh",
                        max_iter=epochs, random_state=42)
    clf.fit(X, y)
    print(f"数据集: {kind}, 隐藏层: {hidden}, 轮数: {epochs}", flush=True)
    print(f"训练精度: {accuracy_score(y, clf.predict(X)):.3f}", flush=True)

    # 决策边界采样（-6..6 网格）
    grid = np.meshgrid(np.linspace(-6, 6, 21), np.linspace(-6, 6, 21))
    pts = np.stack([g.ravel() for g in grid], axis=1)
    pred = clf.predict(pts).reshape(21, 21)
    print("决策边界网格(0/1):", flush=True)
    for row in pred.tolist():
        print("".join("1" if v else "." for v in row), flush=True)


if __name__ == "__main__":
    main()
