import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler

class PyTorchLSTM(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=50, num_layers=2, output_dim=1):
        super(PyTorchLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

def train_and_forecast_lstm(train_series, forecast_periods, epochs=30, look_back=60):
    """
    Helper function to scale, build, train, and forecast using PyTorch LSTM.
    Returns scaling fitted object and forecasts (array).
    """
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(train_series.values.reshape(-1, 1))

    # Prepare datasets
    X, Y = [], []
    for i in range(look_back, len(scaled_data)):
        X.append(scaled_data[i - look_back:i, 0])
        Y.append(scaled_data[i, 0])

    if len(X) == 0:
        # Fallback if series is too short
        return scaler, np.repeat(train_series.iloc[-1], forecast_periods)

    X, Y = np.array(X), np.array(Y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    X_tensor = torch.tensor(X, dtype=torch.float32)
    Y_tensor = torch.tensor(Y, dtype=torch.float32).reshape(-1, 1)

    model = PyTorchLSTM(input_dim=1, hidden_dim=50, num_layers=2, output_dim=1)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

    # Train model
    model.train()
    batch_size = 32
    for epoch in range(epochs):
        permutation = torch.randperm(X_tensor.size(0))
        for i in range(0, X_tensor.size(0), batch_size):
            indices = permutation[i:i+batch_size]
            batch_x, batch_y = X_tensor[indices], Y_tensor[indices]

            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

    # Predict autoregressively
    model.eval()
    last_sequence = scaled_data[-look_back:]
    predictions = []

    with torch.no_grad():
        current_seq = torch.tensor(last_sequence.reshape(1, look_back, 1), dtype=torch.float32)
        for _ in range(forecast_periods):
            pred = model(current_seq)
            pred_val = pred.item()
            predictions.append(pred_val)
            # Update sequence by dropping first value and appending predicted value
            new_val = torch.tensor([[[pred_val]]], dtype=torch.float32)
            current_seq = torch.cat((current_seq[:, 1:, :], new_val), dim=1)

    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
    return scaler, predictions.flatten()

def lstm_forecast(data: pd.DataFrame, stock_name: str, forecast_days: int = 30, holdout_days: int = 30, epochs: int = 30, look_back: int = 60, use_log: bool = False):
    """
    Fits PyTorch LSTM and returns:
    - val_forecast: validation forecast values (array)
    - val_actual: actual values for holdout period (array)
    - future_forecast: out-of-sample forecast values (array)
    """
    df = data.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    df.set_index('date', inplace=True)
    
    close_series = df['close']
    
    # 1. Transform series if log is enabled
    train_series = np.log(close_series) if use_log else close_series

    # 2. Validation Split
    if len(train_series) > holdout_days + look_back:
        train_val_series = train_series.iloc[:-holdout_days]
        val_actual_transformed = train_series.iloc[-holdout_days:]
        
        _, val_pred_transformed = train_and_forecast_lstm(train_val_series, holdout_days, epochs, look_back)
        
        # Inverse transform validation values
        val_forecast = np.exp(val_pred_transformed) if use_log else val_pred_transformed
        val_actual = np.exp(val_actual_transformed.values) if use_log else val_actual_transformed.values
    else:
        val_forecast = np.repeat(close_series.iloc[-1], holdout_days)
        val_actual = close_series.iloc[-holdout_days:].values if len(close_series) >= holdout_days else close_series.values

    # 3. Future Forecasting (Fit on full dataset)
    _, future_pred_transformed = train_and_forecast_lstm(train_series, forecast_days, epochs, look_back)
    future_forecast = np.exp(future_pred_transformed) if use_log else future_pred_transformed

    return val_forecast, val_actual, future_forecast