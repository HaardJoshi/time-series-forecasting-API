import streamlit as st
import pandas as pd
import requests
import plotly.graph_objects as go
from datetime import datetime
import os

# --- Page Configuration ---
st.set_page_config(
    page_title="Stock Forecast App",
    page_icon="📈",
    layout="wide"
)

# --- API URLs ---
# Get the backend host from an env var, defaulting to localhost
API_HOST = os.getenv("API_BACKEND_HOST", "127.0.0.1")
API_PREDICT_URL = f"http://{API_HOST}:8000/predict"
API_HISTORICAL_URL = f"http://{API_HOST}:8000/historical-data"

# --- Helper Functions ---
def get_forecast(ticker: str, days: int):
    """Calls the FastAPI backend to get a forecast."""
    params = {"ticker": ticker, "days": days}
    try:
        response = requests.get(API_PREDICT_URL, params=params, timeout=300)
        response.raise_for_status() 
        return response.json()
    except requests.exceptions.HTTPError as err:
        try:
            return {"error": err.response.json().get('detail', 'Unknown error')}
        except:
            return {"error": f"HTTP error: {err}"}
    except requests.exceptions.ConnectionError:
        return {"error": "Connection Error: Could not connect to the API. Is it running?"}
    except requests.exceptions.Timeout:
        return {"error": "Timeout: The request took too long. The model might be training."}
    except Exception as e:
        return {"error": f"An unknown error occurred: {e}"}

def get_historical_data(ticker: str):
    """Calls the FastAPI backend to get historical data."""
    params = {"ticker": ticker}
    try:
        response = requests.get(API_HISTORICAL_URL, params=params, timeout=300)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return {"error": f"Could not fetch historical data: {e}"}

# --- App Layout ---
st.title("📈 Stock Price Forecaster")
st.write("Enter a stock ticker and select the number of days to forecast.")

# --- User Inputs ---
ticker = st.text_input("Stock Ticker", value="SPY", help="e.g., AAPL, MSFT, GOOG").upper()
days = st.slider("Days to Forecast", min_value=7, max_value=365, value=30, step=1)

# --- Forecast Button and Logic ---
if st.button("Get Forecast", type="primary"):
    if not ticker:
        st.error("Please enter a stock ticker.")
    else:
        # Show a spinner while fetching BOTH data sources
        with st.spinner(f"Fetching data and forecast for {ticker}..."):
            forecast_data = get_forecast(ticker, days)
            historical_data = get_historical_data(ticker)
        
        # Check for errors in both requests
        if "error" in forecast_data:
            st.error(f"Forecast Error: {forecast_data['error']}")
        elif "error" in historical_data:
            st.error(f"Historical Data Error: {historical_data['error']}")
        else:
            st.success(f"Data for {ticker} processed successfully!")
            
            # --- Create DataFrames ---
            forecast_df = pd.DataFrame(forecast_data)
            forecast_df['ds'] = pd.to_datetime(forecast_df['ds'])
            
            historical_df = pd.DataFrame(historical_data)
            historical_df['Date'] = pd.to_datetime(historical_df['Date'])
            
            # --- Display Results in Tabs ---
            tab1, tab2 = st.tabs(["📈 Forecast Plot", "🕯️ Historical Candlestick"])

            with tab1:
                st.subheader(f"Forecast for {days} Days")
                # Plotly Forecast Chart
                fig_forecast = go.Figure()
                fig_forecast.add_trace(go.Scatter(
                    x=forecast_df['ds'], 
                    y=forecast_df['yhat'], 
                    mode='lines', 
                    name='Forecast (yhat)',
                    line=dict(color='orange')
                ))
                fig_forecast.add_trace(go.Scatter(
                    x=forecast_df['ds'], y=forecast_df['yhat_upper'],
                    mode='lines', line=dict(width=0), showlegend=False
                ))
                fig_forecast.add_trace(go.Scatter(
                    x=forecast_df['ds'], y=forecast_df['yhat_lower'],
                    mode='lines', line=dict(width=0), fill='tonexty',
                    fillcolor='rgba(255, 165, 0, 0.2)', showlegend=False
                ))
                fig_forecast.update_layout(
                    title=f"{ticker} Forecast for {days} Days",
                    xaxis_title="Date",
                    yaxis_title="Forecasted Price"
                )
                st.plotly_chart(fig_forecast, use_container_width=True)
                
                # Data Table
                st.subheader("Forecast Data")
                st.dataframe(forecast_df.set_index('ds').style.format("{:.2f}"))

            with tab2:
                st.subheader("Historical Candlestick Chart")
                # Plotly Candlestick Chart
                fig_candle = go.Figure(data=[go.Candlestick(
                    x=historical_df['Date'],
                    open=historical_df['Open'],
                    high=historical_df['High'],
                    low=historical_df['Low'],
                    close=historical_df['Close'],
                    name="Candlestick"
                )])
                fig_candle.update_layout(
                    title=f"{ticker} Historical Price",
                    xaxis_title="Date",
                    yaxis_title="Price",
                    xaxis_rangeslider_visible=True # Add a range slider
                )
                st.plotly_chart(fig_candle, use_container_width=True)