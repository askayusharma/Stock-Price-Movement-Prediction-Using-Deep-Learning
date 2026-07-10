import pandas as pd
import numpy as np
from prophet import Prophet
import logging

# Suppress Prophet logging
logging.getLogger('prophet').setLevel(logging.WARNING)
logging.getLogger('cmdstanpy').setLevel(logging.WARNING)

def prophet_forecast(data: pd.DataFrame, stock_name: str, forecast_days: int = 30, holdout_days: int = 30, use_log: bool = False):
    """
    Fits Prophet model and returns:
    - val_forecast: validation forecast values (array)
    - val_actual: actual values for holdout period (array)
    - future_forecast: out-of-sample forecast values (array)
    - future_lower: lower confidence bounds (array)
    - future_upper: upper confidence bounds (array)
    """
    df = data.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    
    # 1. Transform series if log is enabled
    y_vals = np.log(df['close'].values) if use_log else df['close'].values
    prophet_df = pd.DataFrame({
        'ds': df['date'],
        'y': y_vals
    })

    # 2. Validation Split
    if len(df) > holdout_days:
        train_df = prophet_df.iloc[:-holdout_days]
        val_actual_transformed = prophet_df.iloc[-holdout_days:]['y'].values
        
        try:
            model = Prophet(daily_seasonality=False)
            model.fit(train_df)
            future = model.make_future_dataframe(periods=holdout_days, include_history=False)
            forecast = model.predict(future)
            val_pred_transformed = forecast['yhat'].values
        except Exception:
            val_pred_transformed = np.repeat(train_df['y'].iloc[-1], holdout_days)
            
        # Inverse transform
        val_forecast = np.exp(val_pred_transformed) if use_log else val_pred_transformed
        val_actual = np.exp(val_actual_transformed) if use_log else val_actual_transformed
    else:
        val_forecast = np.repeat(df['close'].iloc[-1], holdout_days)
        val_actual = df['close'].iloc[-holdout_days:].values if len(df) >= holdout_days else df['close'].values

    # 3. Future Forecasting (Fit on full dataset)
    try:
        model_full = Prophet(daily_seasonality=False)
        model_full.fit(prophet_df)
        future_full = model_full.make_future_dataframe(periods=forecast_days, include_history=False)
        forecast_full = model_full.predict(future_full)
        
        future_pred_transformed = forecast_full['yhat'].values
        future_lower_transformed = forecast_full['yhat_lower'].values
        future_upper_transformed = forecast_full['yhat_upper'].values
    except Exception:
        future_pred_transformed = np.repeat(prophet_df['y'].iloc[-1], forecast_days)
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
