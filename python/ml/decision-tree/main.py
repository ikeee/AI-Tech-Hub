"""
决策树（Python 最简实现）：CART 分类树。

对应浏览器端手写 CART 的参考实现：生成 2D 数据，用 sklearn
DecisionTreeClassifier 训练并输出树结构与训练精度。

依赖安装：
    pip install scikit-learn numpy

用法：
    python main.py [dataset] [max_depth]
"""

import sys

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier, export_text


def make_data(kind: str, n: int = 240, seed: int = 42):
    rng = np.random.default_rng(seed)
    X, y = [], []
    for i in range(n):
        if kind == "circle":
            angle = rng.uniform(0, 2 * np.pi)
            radius = rng.uniform(1, 3) if rng.random() < 0.5 else rng.uniform(3, 5)
            X.append([np.cos(angle) * radius, np.sin(angle) * radius])
            y.append(1 if radius < 2.5 else 0)
        elif kind == "xor":
            qx, qy = rng.choice([-1, 1], 2)
            X.append([qx * (1 + abs(rng.normal()) * 0.8), qy * (1 + abs(rng.normal()) * 0.8)])
            y.append(1 if qx * qy < 0 else 0)
        else:
            cx = rng.choice([-2, 2])
            X.append([cx + rng.normal() * 1.2, rng.normal() * 1.2])
            y.append(1 if cx < 0 else 0)
    return np.asarray(X), np.asarray(y)


def main() -> None:
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")
        except Exception:
            pass

    kind = sys.argv[1] if len(sys.argv) > 1 else "circle"
    depth = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    X, y = make_data(kind)
    clf = DecisionTreeClassifier(max_depth=depth, random_state=42)
    clf.fit(X, y)
    print(f"数据集: {kind}, 最大深度: {depth}", flush=True)
    print(f"训练精度: {accuracy_score(y, clf.predict(X)):.4f}", flush=True)
    print(f"叶节点数: {clf.get_n_leaves()}", flush=True)
    print(export_text(clf, feature_names=["x", "y"]), flush=True)


if __name__ == "__main__":
    main()
