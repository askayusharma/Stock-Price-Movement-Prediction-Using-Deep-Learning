import pandas as pd
import numpy as np
from pmdarima import auto_arima
from statsmodels.tsa.arima.model import ARIMA

def arima_forecast(data: pd.DataFrame, stock_name: str, forecast_days: int = 30, holdout_days: int = 30, use_log: bool = False):
    """
    Fits ARIMA model and returns:
    - val_forecast: validation forecast values (array)
    - val_actual: actual values for holdout period (array)
    - future_forecast: out-of-sample forecast values (array)
    - future_lower: lower confidence bounds (array)
    - future_upper: upper confidence bounds (array)
    """
    df = data.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    df.set_index('date', inplace=True)
    
    close_series = df['close']

    # 1. Transform series if log is enabled
    ts = np.log(close_series) if use_log else close_series

    # 2. Validation Model Fit & Forecast
    if len(ts) > holdout_days:
        train_series = ts.iloc[:-holdout_days]
        val_actual_transformed = ts.iloc[-holdout_days:]
        
        try:
            stepwise_model = auto_arima(train_series, start_p=1, start_q=1,
                                         max_p=3, max_q=3, m=1,
                                         seasonal=False, trace=False,
                                         error_action='ignore', suppress_warnings=True)
            order = stepwise_model.order
        except Exception:
            order = (1, 1, 1)

        try:
            model = ARIMA(train_series, order=order)
            model_fit = model.fit()
            val_pred_res = model_fit.get_forecast(steps=holdout_days)
            val_pred_transformed = val_pred_res.predicted_mean.values
        except Exception:
            # Fallback to naive forecast
            val_pred_transformed = np.repeat(train_series.iloc[-1], holdout_days)
        
        # Inverse transform
        val_forecast = np.exp(val_pred_transformed) if use_log else val_pred_transformed
        val_actual = np.exp(val_actual_transformed.values) if use_log else val_actual_transformed.values
    else:
        val_forecast = np.repeat(close_series.iloc[-1], holdout_days)
        val_actual = close_series.iloc[-holdout_days:].values if len(close_series) >= holdout_days else close_series.values

    # 3. Future Forecasting (Fit on full dataset)
    try:
        stepwise_model_full = auto_arima(ts, start_p=1, start_q=1,
                                          max_p=3, max_q=3, m=1,
                                          seasonal=False, trace=False,
                                          error_action='ignore', suppress_warnings=True)
        order_full = stepwise_model_full.order
    except Exception:
        order_full = (1, 1, 1)

    try:
        model_full = ARIMA(ts, order=order_full)
        model_fit_full = model_full.fit()
        future_res = model_fit_full.get_forecast(steps=forecast_days)
        
        future_pred_transformed = future_res.predicted_mean.values
        conf_int = future_res.conf_int(alpha=0.05)
        future_lower_transformed = conf_int.iloc[:, 0].values
        future_upper_transformed = conf_int.iloc[:, 1].values
    except Exception:
        future_pred_transformed = np.repeat(ts.iloc[-1], forecast_days)
        future_lower_transformed = future_pred_transformed * 0.9
        future_upper_transformed = future_pred_transformed * 1.1

    # Inverse transform out-of-sample forecast
    if use_log:
        future_forecast = np.exp(future_pred_transformed)
        future_lower = np.exp(future_lower_transformed)
        future_upper = np.exp(future_upper_transformed)
    else:
        future_forecast = future_pred_transformed
        future_lower = future_lower_transformed
        future_upper = future_upper_transformed

    return val_forecast, val_actual, future_forecast, future_lower, future_upper
