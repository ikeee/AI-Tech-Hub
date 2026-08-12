"""
文本迁移学习演示：TF-IDF 特征 + KNN 分类（最简实现）。

对应浏览器端 transformers.js 嵌入（MiniLM）+ KNN 的 Python 参考。
生产环境可把 TF-IDF 换成 sentence-transformers 嵌入以获得更好语义效果。

依赖安装：
    pip install scikit-learn

输入格式（JSON）：
    train.json: {"samples": [{"class": "A", "text": "..."}, ...]}
    test.json:  {"text": "..."}

用法：
    python main.py train.json test.json
"""

import json
import sys

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier


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
    texts = [item["text"] for item in train["samples"]]
    labels = [item["class"] for item in train["samples"]]
    if not texts:
        print("训练集为空", flush=True)
        return

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(texts)
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X, labels)

    with open(sys.argv[2], encoding="utf-8") as f:
        test = json.load(f)
    x = vectorizer.transform([test["text"]])
    print("预测类别:", knn.predict(x)[0], flush=True)
    print("置信度:", knn.predict_proba(x)[0].tolist(), flush=True)


if __name__ == "__main__":
    main()
