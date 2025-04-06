import pandas as pd
import numpy as np
from prophet import Prophet
from xgboost import XGBRegressor
import matplotlib.pyplot as plt

def hybrid_prophet_xgb(df_raw, future_periods=1, plot_result=True):
    # สร้าง DataFrame สำหรับ Prophet
    df = pd.DataFrame({
        'ds': pd.date_range(start='2000-01-01', periods=len(df_raw)),
        'y': df_raw['numbers'].astype(float)
    })

    # ===== 1. Train Prophet =====
    model_p = Prophet()
    model_p.fit(df)

    future = model_p.make_future_dataframe(periods=future_periods)
    forecast = model_p.predict(future)

    # ===== 2. คำนวณ Residual =====
    merged = df.merge(forecast[['ds', 'yhat']], on='ds')
    merged['residual'] = merged['y'] - merged['yhat']

    # ===== 3. สร้างฟีเจอร์ย้อนหลังสำหรับ Residual (XGBoost) =====
    window_size = 10
    features = []
    targets = []
    for i in range(window_size, len(merged)):
        features.append(merged['residual'].values[i-window_size:i])
        targets.append(merged['residual'].values[i])

    X = pd.DataFrame(features)
    y = pd.Series(targets)

    # ===== 4. Train XGBoost =====
    model_xgb = XGBRegressor(n_estimators=100)
    model_xgb.fit(X, y)

    # ===== 5. ทำนาย Residual จาก XGBoost =====
    last_window = merged['residual'].values[-window_size:]
    pred_residual = model_xgb.predict([last_window])[0]

    # ===== 6. รวมค่าทำนาย =====
    prophet_pred = forecast.iloc[-1]['yhat']
    final_pred = prophet_pred + pred_residual

    # ===== 7. แสดงกราฟผลลัพธ์ =====
    if plot_result:
        actual = df['y'].values
        yhat = merged['yhat'].values
        pred_next = final_pred
        x = list(range(len(actual)))
        plt.figure(figsize=(10, 5))
        plt.plot(x, actual, label="Actual", marker='o')
        plt.plot(x, yhat, label="Prophet Forecast", linestyle='--')
        plt.plot(len(actual), pred_next, 'ro', label="Next Predicted")
        plt.title("Prophet + XGBoost Hybrid Prediction")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    return int(np.round(final_pred)) % 1000000
