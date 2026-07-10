import streamlit as st
import pandas as pd
import numpy as np
import os
import plotly.graph_objects as go
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf

from models.arima_model import arima_forecast
from models.sarima_model import sarima_forecast
from models.prophet_model import prophet_forecast
from models.lstm_model import lstm_forecast

st.set_page_config(page_title="📈 Quantitative Stock Forecasting Terminal", layout="wide")

# Custom Premium Styling
st.markdown("""
<style>
    body {
        background-color: #0d0f12;
        color: #e2e8f0;
    }
    .reportview-container {
        background: #0d0f12;
    }
    .main-title {
        font-family: 'Courier New', monospace;
        font-weight: 800;
        font-size: 2.8rem;
        background: linear-gradient(90deg, #00e5ff, #7b2cbf);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2px;
    }
    .subtitle {
        font-family: 'Inter', sans-serif;
        color: #94a3b8;
        font-size: 1.1rem;
        margin-bottom: 25px;
    }
    .metric-card {
        background-color: #151922;
        border: 1px solid #1e293b;
        border-radius: 8px;
        padding: 18px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
    }
    .metric-label {
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #64748b;
        font-weight: bold;
        margin-bottom: 4px;
    }
    .metric-val {
        font-size: 1.8rem;
        font-family: 'Courier New', monospace;
        color: #00f2fe;
        font-weight: bold;
    }
    div.stTabs [data-baseweb="tab-list"] {
        column-gap: 8px;
        border-bottom: 2px solid #1e293b;
    }
    div.stTabs [data-baseweb="tab"] {
        background-color: #131722;
        border: 1px solid #1e293b;
        border-radius: 6px 6px 0px 0px;
        padding: 10px 20px;
        color: #94a3b8;
        font-family: 'Courier New', monospace;
        font-weight: bold;
    }
    div.stTabs [aria-selected="true"] {
        background: linear-gradient(90deg, #00e5ff, #7b2cbf) !important;
        color: #0d0f12 !important;
        border: none;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">📈 QUANTITATIVE STOCK FORECASTING TERMINAL</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Quantitative Analytics & Time-Series Engineering Engine</div>', unsafe_allow_html=True)

# Load dataset options
DATA_DIR = "cleaned_data"
available_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.csv')]
stock_options = [f.replace('_cleaned.csv', '').upper() for f in available_files]

# Sidebar Configurations
st.sidebar.markdown("### ⚙️ TERMINAL CONFIGURATION")
selected_stock = st.sidebar.selectbox("Select Stock/Equity", stock_options, index=1 if "ICICI" in stock_options else 0)
forecast_days = st.sidebar.slider("Future Forecast Horizon (Days)", min_value=10, max_value=90, value=30, step=5)
holdout_days = st.sidebar.slider("Holdout Validation Period (Days)", min_value=10, max_value=90, value=30, step=5)

st.sidebar.markdown("### 🔬 MODEL TRAINING PARAMETERS")
use_log = st.sidebar.checkbox("Apply Log-Transformations (Stationarity)", value=True)
lstm_epochs = st.sidebar.slider("LSTM Epochs", min_value=5, max_value=50, value=20, step=5)
lstm_lookback = st.sidebar.slider("LSTM Lookback Window", min_value=20, max_value=100, value=60, step=10)

# Load selected stock data
data_path = os.path.join(DATA_DIR, selected_stock.lower() + "_cleaned.csv")
df = pd.read_csv(data_path)
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date')

# Caching forecast fitting
@st.cache_data
def compute_all_forecasts(stock_name, _data, f_days, h_days, log_transform, epochs, look_back):
    arima_res = arima_forecast(_data, stock_name, f_days, h_days, log_transform)
    sarima_res = sarima_forecast(_data, stock_name, f_days, h_days, log_transform)
    prophet_res = prophet_forecast(_data, stock_name, f_days, h_days, log_transform)
    lstm_res = lstm_forecast(_data, stock_name, f_days, h_days, epochs, look_back, log_transform)
    
    return {
        "ARIMA": arima_res,
        "SARIMA": sarima_res,
        "Prophet": prophet_res,
        "LSTM": lstm_res
    }

# Run modeling
with st.spinner("🚀 Running forecasting engines & optimization algorithms..."):
    results = compute_all_forecasts(selected_stock, df, forecast_days, holdout_days, use_log, lstm_epochs, lstm_lookback)

# Setup Dashboard Tabs
tabs = st.tabs(["📁 Overview & EDA", "🔬 Stationarity & Differencing", "📊 Performance Benchmarking", "🔮 Out-of-Sample Forecast"])

# Tab 1: Overview & EDA
with tabs[0]:
    st.subheader("📊 Stock Overview & Exploratory Analysis")
    
    # Metrics Panel
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">📅 Historical Horizon</div>
            <div class="metric-val">{df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">📊 Observations</div>
            <div class="metric-val">{len(df)} Days</div>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">💰 Last Close Price</div>
            <div class="metric-val">₹{df['close'].iloc[-1]:.2f}</div>
        </div>""", unsafe_allow_html=True)
    with col4:
        # Volatility as standard deviation of daily returns
        daily_returns = df['close'].pct_change().dropna()
        volatility = daily_returns.std() * np.sqrt(252) * 100 # Annualized volatility %
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">⚡ Annual Volatility</div>
            <div class="metric-val">{volatility:.2f}%</div>
        </div>""", unsafe_allow_html=True)

    st.write("")
    
    # Technical Indicators Selector
    st.markdown("### 📈 Interactive Stock Chart with Overlay Technical Indicators")
    show_ma20 = st.checkbox("Show 20-Day SMA", value=True)
    show_ma50 = st.checkbox("Show 50-Day SMA", value=False)
    show_bollinger = st.checkbox("Show Bollinger Bands (20-day, 2σ)", value=False)
    show_volatility = st.checkbox("Show Rolling Realized Volatility (20-day window)", value=False)

    # Plot Close Price & Indicators
    fig_overview = go.Figure()
    
    if show_volatility:
        # Volatility chart (separate axis or just separate plot)
        rolling_vol = df['close'].pct_change().rolling(20).std() * np.sqrt(252) * 100
        fig_overview.add_trace(go.Scatter(x=df['date'], y=rolling_vol, name='Rolling Volatility %', line=dict(color='#e056fd', width=2)))
        fig_overview.update_layout(yaxis_title="Annualized Volatility %")
    else:
        fig_overview.add_trace(go.Scatter(x=df['date'], y=df['close'], name='Close Price', line=dict(color='#00e5ff', width=2)))
        
        if show_ma20:
            df['MA20'] = df['close'].rolling(20).mean()
            fig_overview.add_trace(go.Scatter(x=df['date'], y=df['MA20'], name='20-Day SMA', line=dict(color='#ff9f43', width=1.5, dash='dash')))
        if show_ma50:
            df['MA50'] = df['close'].rolling(50).mean()
            fig_overview.add_trace(go.Scatter(x=df['date'], y=df['MA50'], name='50-Day SMA', line=dict(color='#10ac84', width=1.5, dash='dash')))
        if show_bollinger:
            ma20 = df['close'].rolling(20).mean()
            sd20 = df['close'].rolling(20).std()
            upper_band = ma20 + 2 * sd20
            lower_band = ma20 - 2 * sd20
            fig_overview.add_trace(go.Scatter(x=df['date'], y=upper_band, name='Upper Bollinger Band', line=dict(color='#576574', width=1, dash='dot')))
            fig_overview.add_trace(go.Scatter(x=df['date'], y=lower_band, name='Lower Bollinger Band', line=dict(color='#576574', width=1, dash='dot'), fill='tonexty', fillcolor='rgba(87, 101, 116, 0.1)'))
        fig_overview.update_layout(yaxis_title="Price (₹)")

    fig_overview.update_layout(
        title=f"Historical Price Series Analysis for {selected_stock}",
        xaxis_title="Date",
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(showgrid=True, gridcolor='#1e293b'),
        yaxis=dict(showgrid=True, gridcolor='#1e293b'),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig_overview, use_container_width=True)

    # Show raw dataset preview
    st.markdown("#### 📁 Raw Equities Data Feed")
    st.dataframe(df.tail(10), use_container_width=True)

# Tab 2: Stationarity & Differencing
with tabs[1]:
    st.subheader("🔬 Time-Series Stationarity & Differencing Analysis")
    st.markdown("""
    Mathematical forecasting models (like ARIMA and SARIMA) require time-series data to be **stationary** (constant mean, variance, and autocorrelation over time).
    Use this panel to identify the mathematical transformation required to achieve stationarity.
    """)

    col_ts_sel, col_ts_txt = st.columns([1, 2])
    with col_ts_sel:
        ts_transformation = st.radio(
            "Select Time-Series Representation:",
            [
                "Raw Price Series", 
                "Log-Transformed Series", 
                "1st Difference (Raw Series)", 
                "1st Difference (Log Series)",
                "Seasonal Difference (lag 5, Raw Series)"
            ]
        )
    with col_ts_txt:
        st.info(fr"""
        **Mathematical transformation formulas**:
        * **Raw Price**: \( Y_t \)
        * **Log Transform**: \( \ln(Y_t) \) (Stabilizes exponential growth and variance)
        * **1st Difference**: \( Y_t - Y_{{t-1}} \) (Removes trend, stabilizing mean)
        * **1st Difference of Log**: \( \ln(Y_t) - \ln(Y_{{t-1}}) \) (Approximates daily compounding returns)
        * **Seasonal Difference**: \( Y_t - Y_{{t-5}} \) (Removes weekly/business-day seasonality)
        """)

    # Get transformed series
    close_vals = df['close']
    if ts_transformation == "Log-Transformed Series":
        y_trans = np.log(close_vals)
    elif ts_transformation == "1st Difference (Raw Series)":
        y_trans = close_vals.diff()
    elif ts_transformation == "1st Difference (Log Series)":
        y_trans = np.log(close_vals).diff()
    elif ts_transformation == "Seasonal Difference (lag 5, Raw Series)":
        y_trans = close_vals.diff(5)
    else:
        y_trans = close_vals.copy()

    # Drop NaNs for statistics and plots
    y_trans_clean = y_trans.dropna()

    # Statistical Tests
    st.markdown("### 📊 Econometric Stationarity Diagnostic Tests")
    col_adf, col_kpss = st.columns(2)
    
    with col_adf:
        st.markdown("**Augmented Dickey-Fuller (ADF) Test** (H0: Series is non-stationary / has unit root)")
        try:
            adf_test = adfuller(y_trans_clean)
            adf_stat, adf_p = adf_test[0], adf_test[1]
            critical_values = adf_test[4]
            is_stationary_adf = adf_p < 0.05
            
            st.metric("ADF Test Statistic", f"{adf_stat:.4f}")
            st.metric("ADF p-value", f"{adf_p:.4e}")
            if is_stationary_adf:
                st.success("✅ Stationary (Reject H0 at 5% level)")
            else:
                st.error("❌ Non-Stationary (Fail to Reject H0)")
        except Exception as e:
            st.warning(f"ADF Test error: {e}")

    with col_kpss:
        st.markdown("**Kwiatkowski-Phillips-Schmidt-Shin (KPSS) Test** (H0: Series is stationary)")
        try:
            # Suppress warning for low/high p-values in statsmodels KPSS
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                kpss_test = kpss(y_trans_clean, regression='c')
            kpss_stat, kpss_p = kpss_test[0], kpss_test[1]
            is_stationary_kpss = kpss_p > 0.05
            
            st.metric("KPSS Test Statistic", f"{kpss_stat:.4f}")
            st.metric("KPSS p-value", f"{kpss_p:.4f}")
            if is_stationary_kpss:
                st.success("✅ Stationary (Fail to Reject H0 at 5% level)")
            else:
                st.error("❌ Non-Stationary (Reject H0)")
        except Exception as e:
            st.warning(f"KPSS Test error: {e}")

    # Plot Rolling Statistics & ACF/PACF
    st.markdown("### 📈 Visual Diagnostics (Rolling Stats & Autocorrelations)")
    col_roll, col_acf_pacf = st.columns(2)
    
    with col_roll:
        st.markdown("**Rolling Mean & Standard Deviation**")
        roll_df = pd.DataFrame({'val': y_trans})
        roll_df['Rolling Mean'] = roll_df['val'].rolling(20).mean()
        roll_df['Rolling Std'] = roll_df['val'].rolling(20).std()
        
        fig_roll = go.Figure()
        fig_roll.add_trace(go.Scatter(x=df['date'], y=roll_df['val'], name='Transformed Series', line=dict(color='#4b6584', width=1)))
        fig_roll.add_trace(go.Scatter(x=df['date'], y=roll_df['Rolling Mean'], name='20-Day Rolling Mean', line=dict(color='#fed330', width=2)))
        fig_roll.add_trace(go.Scatter(x=df['date'], y=roll_df['Rolling Std'], name='20-Day Rolling Std', line=dict(color='#eb3b5a', width=2)))
        fig_roll.update_layout(
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(showgrid=True, gridcolor='#1e293b'),
            yaxis=dict(showgrid=True, gridcolor='#1e293b'),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0)
        )
        st.plotly_chart(fig_roll, use_container_width=True)

    with col_acf_pacf:
        st.markdown("**ACF & PACF Plots** (Correlation structures)")
        try:
            acf_vals = acf(y_trans_clean, nlags=30, fft=True)
            pacf_vals = pacf(y_trans_clean, nlags=30, method='ols')
            lags = list(range(len(acf_vals)))
            
            fig_acf_pacf = go.Figure()
            fig_acf_pacf.add_trace(go.Bar(x=lags, y=acf_vals, name='ACF', marker_color='#00d2d3'))
            fig_acf_pacf.add_trace(go.Bar(x=lags, y=pacf_vals, name='PACF', marker_color='#ff9f43'))
            
            # Add 95% confidence bands
            ci = 1.96 / np.sqrt(len(y_trans_clean))
            fig_acf_pacf.add_shape(type="line", x0=0, y0=ci, x1=30, y1=ci, line=dict(color="red", width=1.5, dash="dash"))
            fig_acf_pacf.add_shape(type="line", x0=0, y0=-ci, x1=30, y1=-ci, line=dict(color="red", width=1.5, dash="dash"))
            
            fig_acf_pacf.update_layout(
                xaxis_title="Lag",
                yaxis_title="Correlation Coefficient",
                template="plotly_dark",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                xaxis=dict(showgrid=True, gridcolor='#1e293b'),
                yaxis=dict(showgrid=True, gridcolor='#1e293b'),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig_acf_pacf, use_container_width=True)
        except Exception as e:
            st.warning(f"Error computing ACF/PACF: {e}")

# Tab 3: Performance Benchmarking (Validation)
with tabs[2]:
    st.subheader("📊 Comparative Performance Benchmarking (Holdout Period)")
    st.markdown(f"""
    Validation mode fits the models on the historical dataset excluding the last **{holdout_days} days**.
    The models then forecast this holdout period, and we benchmark the forecasts against the *real observed market prices* to evaluate out-of-sample metrics.
    """)

    # Extract validation metrics
    validation_metrics = []
    
    # Store validation series for plotting
    val_series_plot = {}
    
    # Actual values
    # Take the last N elements
    val_actual = results["ARIMA"][1]
    val_dates = df['date'].iloc[-holdout_days:].values
    
    for model_name, res in results.items():
        val_forecast = res[0]
        
        # Calculate evaluation metrics
        rmse = np.sqrt(mean_squared_error(val_actual, val_forecast))
        mae = mean_absolute_error(val_actual, val_forecast)
        mape = np.mean(np.abs((val_actual - val_forecast) / val_actual)) * 100
        error_variance = np.var(val_actual - val_forecast)
        r2 = r2_score(val_actual, val_forecast)
        
        validation_metrics.append({
            "Model": model_name,
            "RMSE (Root Mean Sq. Error)": rmse,
            "MAE (Mean Absolute Error)": mae,
            "MAPE (Mean Abs. % Error)": mape,
            "Error Variance": error_variance,
            "R-squared Score": r2
        })
        val_series_plot[model_name] = val_forecast

    # Convert to df
    metrics_df = pd.DataFrame(validation_metrics).sort_values("RMSE (Root Mean Sq. Error)")
    
    # Highlight Best Model
    best_model = metrics_df.iloc[0]["Model"]
    
    st.markdown(f"### 🏆 Model Diagnostics & Statistics Table")
    st.dataframe(
        metrics_df.style.format({
            "RMSE (Root Mean Sq. Error)": "{:.2f}",
            "MAE (Mean Absolute Error)": "{:.2f}",
            "MAPE (Mean Abs. % Error)": "{:.2f}%",
            "Error Variance": "{:.2f}",
            "R-squared Score": "{:.4f}"
        }).highlight_min(subset=["RMSE (Root Mean Sq. Error)", "MAE (Mean Absolute Error)", "MAPE (Mean Abs. % Error)", "Error Variance"], color="#154f30")
          .highlight_max(subset=["R-squared Score"], color="#154f30"),
        use_container_width=True
    )
    
    st.success(f"💡 **Recommended Model**: Based on out-of-sample holdout validation, the **{best_model}** model minimizes prediction error variance and RMSE for {selected_stock}.")

    # Plot validation curves
    st.markdown("### 📈 Validation Forecast vs. Real Market Prices")
    
    # Historical tail to display (e.g. 90 days before holdout)
    tail_len = 90
    hist_tail_df = df.iloc[-(holdout_days + tail_len): -holdout_days]
    
    fig_val = go.Figure()
    fig_val.add_trace(go.Scatter(x=hist_tail_df['date'], y=hist_tail_df['close'], name='Historical Context', line=dict(color='#4b6584', width=1.5)))
    fig_val.add_trace(go.Scatter(x=val_dates, y=val_actual, name='Real Observed Prices (Holdout)', line=dict(color='#00e5ff', width=3)))
    
    colors = {
        "ARIMA": "#ff9f43",
        "SARIMA": "#10ac84",
        "Prophet": "#ee5253",
        "LSTM": "#9b59b6"
    }
    
    for model_name, val_forecast in val_series_plot.items():
        fig_val.add_trace(go.Scatter(x=val_dates, y=val_forecast, name=f"{model_name} Forecast", line=dict(color=colors[model_name], width=1.5, dash='dash')))

    fig_val.update_layout(
        title=f"Holdout Validation Analysis for {selected_stock}",
        xaxis_title="Date",
        yaxis_title="Price (₹)",
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(showgrid=True, gridcolor='#1e293b'),
        yaxis=dict(showgrid=True, gridcolor='#1e293b'),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig_val, use_container_width=True)

# Tab 4: Out-of-Sample Forecasting
with tabs[3]:
    st.subheader(f"🔮 Out-of-Sample Future Projections ({forecast_days}-Day Horizon)")
    st.markdown(f"""
    This panel fits the time-series models on the **entire** historical dataset and projects stock prices for the next **{forecast_days} days** into the future.
    """)

    # Prepare future dates (business days or regular days)
    last_date = df['date'].iloc[-1]
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_days)
    
    # Plot out-of-sample forecast
    # We display the last 90 days of history for visualization clarity
    tail_df = df.tail(90)
    
    fig_future = go.Figure()
    fig_future.add_trace(go.Scatter(x=tail_df['date'], y=tail_df['close'], name='Historical Price', line=dict(color='#00e5ff', width=2)))
    
    # We will gather forecasting values for CSV download
    forecast_csv_data = {'Date': future_dates.strftime('%Y-%m-%d')}
    
    for model_name, res in results.items():
        future_forecast = res[2]
        fig_future.add_trace(go.Scatter(x=future_dates, y=future_forecast, name=f"{model_name} Forecast", line=dict(color=colors[model_name], width=2)))
        forecast_csv_data[f"{model_name}_Forecast"] = future_forecast
        
        # Plot confidence intervals if available and not LSTM
        if model_name in ["ARIMA", "SARIMA", "Prophet"]:
            lower_bound = res[3]
            upper_bound = res[4]
            
            # Add transparent shaded area for confidence intervals
            fig_future.add_trace(go.Scatter(
                x=list(future_dates) + list(future_dates)[::-1],
                y=list(upper_bound) + list(lower_bound)[::-1],
                fill='toself',
                fillcolor=f"rgba({int(colors[model_name][1:3], 16)}, {int(colors[model_name][3:5], 16)}, {int(colors[model_name][5:7], 16)}, 0.08)",
                line=dict(color='rgba(255,255,255,0)'),
                hoverinfo="skip",
                name=f"{model_name} 95% Confidence Interval",
                showlegend=False
            ))
            forecast_csv_data[f"{model_name}_CI_Lower"] = lower_bound
            forecast_csv_data[f"{model_name}_CI_Upper"] = upper_bound

    fig_future.update_layout(
        title=f"Multi-Model Out-of-Sample Price Forecasts for {selected_stock}",
        xaxis_title="Date",
        yaxis_title="Price (₹)",
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(showgrid=True, gridcolor='#1e293b'),
        yaxis=dict(showgrid=True, gridcolor='#1e293b'),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig_future, use_container_width=True)

    # Download CSV Section
    st.markdown("### 📅 Export Future Quant Projections")
    forecast_export_df = pd.DataFrame(forecast_csv_data)
    
    col_dl_sel, col_dl_btn = st.columns([1, 1])
    with col_dl_sel:
        export_model = st.selectbox(
            "Select Model Forecast to Export:",
            ["Best Model (Recommended: " + best_model + ")", "ARIMA", "SARIMA", "Prophet", "LSTM"]
        )
    
    # Resolve actual model selected
    selected_export_model = best_model if "Recommended" in export_model else export_model
    
    # Filter columns to export
    cols_to_export = ['Date', f"{selected_export_model}_Forecast"]
    if f"{selected_export_model}_CI_Lower" in forecast_export_df.columns:
        cols_to_export.append(f"{selected_export_model}_CI_Lower")
        cols_to_export.append(f"{selected_export_model}_CI_Upper")
        
    export_sub_df = forecast_export_df[cols_to_export]
    
    with col_dl_btn:
        st.write("")
        st.write("")
        st.download_button(
            label=f"📅 Download {selected_export_model} Forecast CSV",
            data=export_sub_df.to_csv(index=False),
            file_name=f"{selected_stock}_{selected_export_model.lower()}_forecast_{forecast_days}d.csv",
            mime='text/csv'
        )
        
    st.markdown("#### Preview Export Dataset")
    st.dataframe(export_sub_df, use_container_width=True)

