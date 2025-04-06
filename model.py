from xgboost import XGBRegressor
from prophet import Prophet
import numpy as np
import pandas as pd

def train_and_predict(model_name, X=None, y=None, df_raw=None):
    if model_name == "XGBoost":
        model = XGBRegressor(n_estimators=100)
        model.fit(X, y)
        pred = model.predict(X.iloc[-1:].values)[0]
        return int(np.round(pred)) % 1000000

    elif model_name == "Prophet":
        df_prophet = pd.DataFrame({
            'ds': pd.date_range(start='2000-01-01', periods=len(df_raw)),
            'y': df_raw['numbers'].values
        })
        model = Prophet()
        model.fit(df_prophet)
        future = model.make_future_dataframe(periods=1)
        forecast = model.predict(future)
        pred = forecast.iloc[-1]['yhat']
        return int(np.round(pred)) % 1000000
