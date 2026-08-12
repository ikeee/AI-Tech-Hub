"""
多项式回归演示（Python 最简实现）。

对应浏览器端梯度下降拟合的参考实现：生成带噪声的二次数据，
用 PolynomialFeatures + LinearRegression 拟合并输出 R² 与系数。

依赖安装：
    pip install scikit-learn numpy

用法：
    python main.py
"""

import sys

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures


def main() -> None:
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")
        except Exception:
            pass

    rng = np.random.default_rng(42)
    X = rng.uniform(-5, 5, 200).reshape(-1, 1)
    y = 0.6 * X[:, 0] ** 2 - 0.8 * X[:, 0] + 1.2 + rng.normal(0, 2.2, 200)

    poly = PolynomialFeatures(degree=2)
    Xp = poly.fit_transform(X)
    model = LinearRegression().fit(Xp, y)
    pred = model.predict(Xp)

    print(f"样本数: {len(y)}", flush=True)
    print(f"R²: {r2_score(y, pred):.4f}", flush=True)
    print(f"系数: {model.coef_.tolist()}", flush=True)
    print(f"截距: {model.intercept_:.4f}", flush=True)


if __name__ == "__main__":
    main()
