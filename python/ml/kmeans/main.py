"""
K-Means 聚类演示（Python 最简实现）。

对应浏览器端手写 K-Means 的参考实现：生成 3 簇高斯数据，
用 sklearn KMeans 聚类并输出质心与 inertia。

依赖安装：
    pip install scikit-learn numpy

用法：
    python main.py
"""

import sys

import numpy as np
from sklearn.cluster import KMeans


def main() -> None:
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")
        except Exception:
            pass

    rng = np.random.default_rng(42)
    centers = np.array([[-2.5, -2.0], [2.5, -2.0], [0.0, 3.0]])
    X = np.vstack([c + rng.normal(0, 1.2, (60, 2)) for c in centers])

    kmeans = KMeans(n_clusters=3, n_init=10, random_state=42)
    kmeans.fit(X)
    print(f"样本数: {X.shape[0]}", flush=True)
    print(f"质心: {kmeans.cluster_centers_.tolist()}", flush=True)
    print(f"inertia: {kmeans.inertia_:.4f}", flush=True)
    counts = np.bincount(kmeans.labels_)
    print(f"各簇样本数: {counts.tolist()}", flush=True)


if __name__ == "__main__":
    main()
