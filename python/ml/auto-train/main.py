"""
CSV 自动训练（AutoTrain）：上传 CSV 自动训练多个 scikit-learn 模型并输出对比报告。

支持分类与回归任务（默认按目标列取值自动判别，可用 task 参数覆盖）。
输出 JSON 报告：模型指标对比、混淆矩阵（分类）、特征重要性、测试集示例预测。

依赖安装：
    pip install pandas scikit-learn numpy

用法：
    python main.py data.csv params.json report.json
    params.json: {"target": "species", "task": "auto", "test_size": 0.2}
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR


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
    if df.shape[0] > 50000:
        df = df.head(50000)
    if df.shape[0] < 20 or df.shape[1] < 2:
        raise ValueError("数据过小：至少需要 20 行、2 列")

    target = params.get("target") or df.columns[-1]
    if target not in df.columns:
        raise ValueError(f"目标列不存在: {target}")

    # 只保留数值特征；目标列转成数值标签（分类）
    y_raw = df[target]
    X = df.drop(columns=[target]).select_dtypes(include=[np.number])
    if X.shape[1] == 0:
        raise ValueError("没有数值特征列，请检查 CSV（分类变量请先自行编码）")

    task = (params.get("task") or "auto").lower()
    if task == "auto":
        task = "classification" if y_raw.nunique() <= 12 else "regression"
    y = y_raw if task == "regression" else y_raw.astype("category").cat.codes
    classes = list(y_raw.astype("category").cat.categories) if task == "classification" else []

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=float(params.get("test_size", 0.2)), random_state=42
    )

    if task == "classification":
        models = {
            "逻辑回归": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
            "随机森林": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
            "支持向量机": make_pipeline(StandardScaler(), SVC()),
            "K 近邻": make_pipeline(StandardScaler(), KNeighborsClassifier()),
        }
    else:
        models = {
            "线性回归": make_pipeline(StandardScaler(), Ridge()),
            "随机森林": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
            "K 近邻": make_pipeline(StandardScaler(), KNeighborsRegressor()),
            "支持向量回归": make_pipeline(StandardScaler(), SVR()),
        }

    report = {
        "task": task,
        "samples": int(df.shape[0]),
        "features": int(X.shape[1]),
        "target": target,
        "classes": classes,
        "models": [],
    }

    best = None
    best_score = -1e9
    for name, model in models.items():
        try:
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            if task == "classification":
                acc = float(accuracy_score(y_test, pred))
                f1 = float(f1_score(y_test, pred, average="weighted", zero_division=0))
                pr = float(precision_score(y_test, pred, average="weighted", zero_division=0))
                rc = float(recall_score(y_test, pred, average="weighted", zero_division=0))
                score = acc
                metrics = {"accuracy": round(acc, 4), "f1": round(f1, 4), "precision": round(pr, 4), "recall": round(rc, 4)}
                if name == "随机森林":
                    report["confusion_matrix"] = {
                        "labels": classes,
                        "matrix": confusion_matrix(y_test, pred).tolist(),
                    }
            else:
                r2 = float(r2_score(y_test, pred))
                mae = float(mean_absolute_error(y_test, pred))
                mse = float(mean_squared_error(y_test, pred))
                score = r2
                metrics = {"r2": round(r2, 4), "mae": round(mae, 4), "mse": round(mse, 4)}
            report["models"].append({"name": name, **metrics})
            if score > best_score:
                best_score = score
                best = name
        except Exception as e:
            report["models"].append({"name": name, "error": str(e)})

    report["best"] = best

    # 特征重要性（随机森林）
    for name, model in models.items():
        if name == "随机森林":
            try:
                imp = model.feature_importances_
                total = float(imp.sum())
                report["feature_importance"] = [
                    {"name": str(col), "importance": round(float(v) / total, 4)}
                    for col, v in sorted(zip(X.columns, imp), key=lambda kv: -kv[1])
                ][:10]
            except Exception:
                pass

    # 测试集示例预测（前 10 条）
    try:
        best_model = models.get(best)
        if best_model:
            pred = best_model.predict(X_test[:10])
            rows = []
            for i in range(len(pred)):
                actual = classes[int(y_test.iloc[i])] if task == "classification" else round(float(y_test.iloc[i]), 4)
                predicted = classes[int(pred[i])] if task == "classification" else round(float(pred[i]), 4)
                rows.append({"actual": actual, "predicted": predicted})
            report["sample_predictions"] = rows
    except Exception:
        pass

    Path(report_path).parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(report_path, flush=True)


if __name__ == "__main__":
    main()
