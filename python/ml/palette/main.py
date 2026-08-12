"""
图像主色调提取（Python 最简实现）：K-Means 聚类像素颜色。

对应浏览器端 K-Means 取色板的参考实现：用 sklearn KMeans 对图片像素
聚类，输出主色调（HEX）与占比。

依赖安装：
    pip install scikit-learn numpy pillow

用法：
    python main.py image.jpg [k]
"""

import sys

import numpy as np
from PIL import Image
from sklearn.cluster import KMeans


def main() -> None:
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")
        except Exception:
            pass

    if len(sys.argv) < 2:
        print("用法: python main.py image.jpg [k]", flush=True)
        return
    kk = int(sys.argv[2]) if len(sys.argv) > 2 else 5

    img = Image.open(sys.argv[1]).convert("RGB")
    img.thumbnail((400, 400))
    arr = np.asarray(img, dtype=float).reshape(-1, 3) / 255.0
    # 采样 ≤ 4000 点
    step = max(1, len(arr) // 4000)
    arr = arr[::step]

    kmeans = KMeans(n_clusters=kk, n_init=5, random_state=42)
    labels = kmeans.fit_predict(arr)
    counts = np.bincount(labels, minlength=kk)
    total = len(arr)
    for center, count in sorted(zip(kmeans.cluster_centers_, counts), key=lambda x: -x[1]):
        rgb = tuple(int(round(c * 255)) for c in center)
        hex_ = "#{:02x}{:02x}{:02x}".format(*rgb)
        print(f"{hex_}  rgb({rgb[0]}, {rgb[1]}, {rgb[2]})  {count / total * 100:.1f}%", flush=True)


if __name__ == "__main__":
    main()
