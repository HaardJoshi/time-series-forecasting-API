import streamlit as st
import pandas as pd
import requests
import plotly.graph_objects as go
from datetime import datetime

# --- Page Configuration ---
st.set_page_config(
    page_title="Stock Forecast App",
    page_icon="📈",
    layout="wide"
)

# --- API URL ---
# This is the address of your FastAPI backend
API_URL = "http://127.0.0.1:8000/predict"

# --- Helper Function ---
def get_forecast(ticker: str, days: int):
    """
    Calls the FastAPI backend to get a forecast.
    """
    params = {"ticker": ticker, "days": days}
    try:
        response = requests.get(API_URL, params=params, timeout=300) # 5 min timeout
        response.raise_for_status() # Raise an error for bad responses (4xx or 5xx)
        return response.json()
    except requests.exceptions.HTTPError as err:
        # Try to parse the error detail from the API response
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
        # Show a spinner while fetching data
        with st.spinner(f"Fetching forecast for {ticker}... (This might take a moment if it's a new ticker)"):
            data = get_forecast(ticker, days)
        
        if "error" in data:
            st.error(f"Error: {data['error']}")
        else:
            st.success(f"Forecast for {ticker} generated successfully!")
            
            # --- Display Results ---
            forecast_df = pd.DataFrame(data)
            forecast_df['ds'] = pd.to_datetime(forecast_df['ds'])
            
            # 1. Plotly Chart
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=forecast_df['ds'], 
                y=forecast_df['yhat'], 
                mode='lines', 
                name='Forecast (yhat)',
                line=dict(color='orange')
            ))
            fig.add_trace(go.Scatter(
                x=forecast_df['ds'], y=forecast_df['yhat_upper'],
                mode='lines', line=dict(width=0), showlegend=False
            ))
            fig.add_trace(go.Scatter(
                x=forecast_df['ds'], y=forecast_df['yhat_lower'],
                mode='lines', line=dict(width=0), fill='tonexty',
                fillcolor='rgba(255, 165, 0, 0.2)', showlegend=False
            ))
            fig.update_layout(
                title=f"{ticker} Forecast for {days} Days",
                xaxis_title="Date",
                yaxis_title="Forecasted Price"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # 2. Data Table
            st.subheader("Forecast Data")
            st.dataframe(forecast_df.set_index('ds').style.format("{:.2f}"))