"""
时间序列预测（Forecast）：上传 CSV（日期列 + 数值列），用 Holt-Winters
指数平滑拟合并外推 N 步，输出历史/预测/置信区间与回测指标。

依赖安装：
    pip install pandas statsmodels numpy

用法：
    python main.py data.csv params.json report.json
    params.json: {"date_col": "date", "value_col": "value", "horizon": 30}
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing


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
    date_col = params.get("date_col") or df.columns[0]
    value_col = params.get("value_col") or df.columns[1]
    horizon = int(params.get("horizon", 30))
    if date_col not in df.columns or value_col not in df.columns:
        raise ValueError(f"列不存在: {date_col} / {value_col}")

    df = df[[date_col, value_col]].dropna()
    df = df.iloc[:2000]  # 限制行数
    if len(df) < 12:
        raise ValueError("数据过小：至少需要 12 行")

    dates = pd.to_datetime(df[date_col]).dt.strftime("%Y-%m-%d").tolist()
    values = df[value_col].astype(float).tolist()
    series = pd.Series(values, dtype=float)

    # 回测：最后 10% 作为测试集
    test_n = max(1, len(series) // 10)
    train = series.iloc[:-test_n]
    test = series.iloc[-test_n:]

    try:
        model = ExponentialSmoothing(train, trend="add", damped_trend=True).fit()
    except Exception:
        model = ExponentialSmoothing(train, trend="add").fit()

    # 测试集预测（回测指标）
    backcast = model.forecast(test_n)
    pred_test = np.asarray(backcast, dtype=float)
    actual_test = np.asarray(test, dtype=float)
    mae = float(np.mean(np.abs(pred_test - actual_test)))
    mape = float(np.mean(np.abs((pred_test - actual_test) / (actual_test + 1e-9)))) * 100

    # 最终模型（全量）预测 horizon
    final_model = ExponentialSmoothing(series, trend="add", damped_trend=True).fit()
    fc = np.asarray(final_model.forecast(horizon), dtype=float)
    resid = np.asarray(final_model.resid, dtype=float)
    sigma = float(np.std(resid, ddof=1))
    fc_dates = pd.date_range(pd.to_datetime(dates[-1]), periods=horizon + 1)[1:].strftime("%Y-%m-%d").tolist()

    report = {
        "dates": dates,
        "history": [round(float(v), 4) for v in values],
        "forecast_dates": fc_dates,
        "forecast": [round(float(v), 4) for v in fc],
        "lower": [round(float(v - 1.96 * sigma), 4) for v in fc],
        "upper": [round(float(v + 1.96 * sigma), 4) for v in fc],
        "metrics": {"mae": round(mae, 4), "mape": round(mape, 2)},
        "horizon": horizon,
    }

    Path(report_path).parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(report_path, flush=True)


if __name__ == "__main__":
    main()
