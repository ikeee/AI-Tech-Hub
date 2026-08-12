"""
异常检测（Anomaly）：上传 CSV（两列数值特征），用 IsolationForest 找出异常点，
输出每个点的坐标、标签与异常分数，以及 2D 决策边界网格。

依赖安装：
    pip install pandas scikit-learn numpy

用法：
    python main.py data.csv params.json report.json
    params.json: {"x_col": "x", "y_col": "y", "contamination": 0.1}
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
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
    x_col = params.get("x_col") or df.columns[0]
    y_col = params.get("y_col") or df.columns[1]
    contamination = float(params.get("contamination", 0.1))
    if x_col not in df.columns or y_col not in df.columns:
        raise ValueError(f"列不存在: {x_col} / {y_col}")

    df = df[[x_col, y_col]].dropna().iloc[:5000]
    if len(df) < 10:
        raise ValueError("数据过小：至少需要 10 行")

    X = df[[x_col, y_col]].astype(float).to_numpy()
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    model = IsolationForest(n_estimators=100, contamination=contamination, random_state=42, n_jobs=-1)
    labels = model.fit_predict(Xs)  # 1 = 正常, -1 = 异常
    scores = model.decision_function(Xs)

    # 决策边界网格（归一化空间，-3..3）
    grid = np.meshgrid(np.linspace(-3, 3, 40), np.linspace(-3, 3, 40))
    pts = np.stack([g.ravel() for g in grid], axis=1)
    grid_pred = model.predict(pts).reshape(40, 40).tolist()
    grid_min = [-3.0, -3.0]
    grid_max = [3.0, 3.0]

    report = {
        "points": [[round(float(a), 4), round(float(b), 4)] for a, b in X],
        "labels": [int(v) for v in labels],
        "scores": [round(float(v), 4) for v in scores],
        "grid": grid_pred,
        "grid_min": grid_min,
        "grid_max": grid_max,
        "anomaly_count": int((labels == -1).sum()),
        "total": int(len(X)),
        "contamination": contamination,
    }

    Path(report_path).parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(report_path, flush=True)


if __name__ == "__main__":
    main()
