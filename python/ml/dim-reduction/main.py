"""
降维可视化（Dim Reduction）：上传 CSV（多列数值），用 PCA 或 t-SNE 降到 2D，
再用 K-Means 聚类着色，输出散点坐标与类别。

依赖安装：
    pip install pandas scikit-learn numpy

用法：
    python main.py data.csv params.json report.json
    params.json: {"method": "pca", "clusters": 3}
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler


def main() -> None:
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")
        except Exception:
            pass

    data_path, params_path, report_path = sys.argv[1], sys.argv[2], sys.argv[3]
    with open(params_path, encoding="utf-8") as f:
        params = json.load(f)

    df = pd.read_csv(data_path)
    X = df.select_dtypes(include=[np.number]).dropna().iloc[:2000]
    if X.shape[0] < 10 or X.shape[1] < 2:
        raise ValueError("需要至少 10 行、2 列数值数据")

    Xs = StandardScaler().fit_transform(X)
    method = params.get("method", "pca")
    if method == "tsne":
        embed = TSNE(n_components=2, perplexity=min(30, max(5, Xs.shape[0] // 5)), random_state=42)
        coords = embed.fit_transform(Xs)
        variance = []
    else:
        pca = PCA(n_components=2)
        coords = pca.fit_transform(Xs)
        variance = [round(float(v), 4) for v in pca.explained_variance_ratio_]

    n_clusters = min(int(params.get("clusters", 3)), max(1, Xs.shape[0]))
    kmeans = KMeans(n_clusters=n_clusters, n_init=5, random_state=42)
    labels = kmeans.fit_predict(coords).tolist()

    report = {
        "method": method,
        "points": [[round(float(a), 4), round(float(b), 4)] for a, b in coords],
        "labels": labels,
        "clusters": n_clusters,
        "variance": variance,
        "samples": int(Xs.shape[0]),
        "features": int(Xs.shape[1]),
    }

    Path(report_path).parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(report_path, flush=True)


if __name__ == "__main__":
    main()
