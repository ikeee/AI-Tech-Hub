"""
推荐系统（MovieLens SVD）：矩阵分解做协同过滤推荐。

对应浏览器端无法实现（数据/训练在服务端）的参考实现：加载 MovieLens 100K
评分数据，用批量梯度下降训练 SVD（U @ V.T ≈ R），为指定用户推荐未看过的电影。
首次运行自动下载数据（约 2MB）并训练；模型缓存到 tmp/，之后秒级返回。

依赖安装：
    pip install numpy

用法（经 /api/python/run 调用）：
    python main.py --user 5
"""

import argparse
import json
import os
import sys
import urllib.request
import zipfile
import io
import shutil

import numpy as np

FEATURE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(FEATURE_DIR, "data")
TMP_ROOT = os.path.normpath(os.path.join(FEATURE_DIR, "..", "..", "..", "tmp"))
CACHE_PATH = os.path.join(TMP_ROOT, "svd_model.npz")


def load_data():
    ratings_path = os.path.join(DATA_DIR, "u.data")
    items_path = os.path.join(DATA_DIR, "u.item")
    if not os.path.exists(ratings_path):
        print("首次运行：下载 MovieLens 100K 数据…", flush=True)
        os.makedirs(DATA_DIR, exist_ok=True)
        data = urllib.request.urlopen(
            "https://files.grouplens.org/datasets/movielens/ml-100k.zip", timeout=120
        ).read()
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            z.extractall(os.path.join(DATA_DIR, "_tmp"))
        for name in ("u.data", "u.item"):
            shutil.copy(os.path.join(DATA_DIR, "_tmp", "ml-100k", name), os.path.join(DATA_DIR, name))
        shutil.rmtree(os.path.join(DATA_DIR, "_tmp"))

    ratings = np.loadtxt(ratings_path, delimiter="\t", dtype=int)[:, :3]
    titles = {}
    with open(items_path, encoding="latin-1") as f:
        for line in f:
            parts = line.split("|")
            try:
                titles[int(parts[0])] = parts[1]
            except Exception:
                pass
    return ratings, titles


def train_svd(ratings, n_factors=20, epochs=10, reg=0.1):
    """交替最小二乘（ALS）：交替固定 U/V 解正规方程，稳定不发散。"""
    n_users = int(ratings[:, 0].max())
    n_items = int(ratings[:, 1].max())
    R = np.zeros((n_users, n_items))
    for u, i, r in ratings:
        R[u - 1, i - 1] = r
    mask = (R != 0).astype(float)
    mu = float((R * mask).sum() / max(mask.sum(), 1))
    Rc = (R - mu) * mask
    U = np.random.default_rng(42).normal(0, 0.1, (n_users, n_factors))
    V = np.random.default_rng(7).normal(0, 0.1, (n_items, n_factors))
    for _ in range(epochs):
        # 更新用户因子
        for u in range(n_users):
            items = np.where(mask[u] > 0)[0]
            if len(items) == 0:
                continue
            Vr = V[items]
            A = Vr.T @ Vr + reg * np.eye(n_factors)
            U[u] = np.linalg.solve(A, Vr.T @ Rc[u, items])
        # 更新物品因子
        for i in range(n_items):
            users = np.where(mask[:, i] > 0)[0]
            if len(users) == 0:
                continue
            Ur = U[users]
            A = Ur.T @ Ur + reg * np.eye(n_factors)
            V[i] = np.linalg.solve(A, Ur.T @ Rc[users, i])
    pred = mu + U @ V.T
    rmse = float(np.sqrt(((R - pred) ** 2 * mask).sum() / max(mask.sum(), 1)))
    return U, V, rmse, mu, n_users, n_items


def get_model(ratings):
    if os.path.exists(CACHE_PATH):
        try:
            d = np.load(CACHE_PATH)
            return d["U"], d["V"], float(d["rmse"]), float(d["mu"]), int(d["n_users"]), int(d["n_items"])
        except Exception:
            pass
    print("正在训练 SVD（约 1-2 秒）…", flush=True)
    U, V, rmse, mu, n_users, n_items = train_svd(ratings)
    os.makedirs(TMP_ROOT, exist_ok=True)
    np.savez_compressed(CACHE_PATH, U=U, V=V, rmse=rmse, mu=mu, n_users=n_users, n_items=n_items)
    return U, V, rmse, mu, n_users, n_items


def main() -> None:
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")
        except Exception:
            pass

    parser = argparse.ArgumentParser()
    parser.add_argument("input", nargs="?", default="")
    parser.add_argument("--user", type=int, default=1)
    args = parser.parse_args()

    ratings, titles = load_data()
    U, V, rmse, mu, n_users, n_items = get_model(ratings)
    u = max(1, min(int(args.user), n_users))

    scores = mu + U[u - 1] @ V.T
    rated_items = set(int(i) for _, i, _ in ratings if int(_) == u)
    order = np.argsort(scores)[::-1]
    recommendations = []
    for i in order:
        mid = int(i) + 1
        if mid in rated_items:
            continue
        recommendations.append({
            "movie_id": mid,
            "title": titles.get(mid, f"Movie {mid}"),
            "score": round(float(scores[i]), 4),
        })
        if len(recommendations) >= 10:
            break

    rated_list = sorted(
        ({"movie_id": int(i), "title": titles.get(int(i), f"Movie {int(i)}"), "rating": float(r)}
         for uu, i, r in ratings if int(uu) == u),
        key=lambda x: -x["rating"],
    )[:5]

    result = {
        "user": u,
        "recommendations": recommendations,
        "rated": rated_list,
        "stats": {"users": n_users, "movies": n_items, "ratings": int(len(ratings)), "rmse": round(rmse, 4)},
    }
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
