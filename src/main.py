import uvicorn
from fastapi import FastAPI, HTTPException
from prophet import Prophet
from prophet.serialize import model_from_json
from pathlib import Path
import pandas as pd

# Import from our other src files
from .model_training import train_and_save_model
from .data_ingestion import fetch_data
from .utils import read_config



# --- Configuration ---
app = FastAPI(title="Dynamic Stock Forecast API", version="2.0")

ROOT_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT_DIR / "models"
DATA_DIR = ROOT_DIR / "data"
CONFIG_PATH = ROOT_DIR / "config.yaml"

# This is a cache to hold loaded models in memory
model_cache = {}

# --- Model Loading Logic ---
def get_model_path(ticker: str) -> Path:
    """Gets the path for a given ticker's model file."""
    return MODELS_DIR / f"prophet_model_{ticker}.json"

def load_model(ticker: str):
    """
    Loads a model into the cache. If it doesn't exist,
    it triggers training.
    """
    if ticker in model_cache:
        print(f"Model for {ticker} found in cache.")
        return model_cache[ticker]
    
    model_path = get_model_path(ticker)
    
    if not model_path.exists():
        print(f"No model found for {ticker}. Starting training...")
        success = train_and_save_model(ticker)
        if not success:
            raise HTTPException(status_code=404, detail=f"Could not train model for ticker {ticker}. Ticker may be invalid.")
    
    try:
        print(f"Loading model from {model_path}...")
        with open(model_path, 'r') as fin:
            model = model_from_json(fin.read())
        model_cache[ticker] = model # Save to cache
        print("Model loaded successfully.")
        return model
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading model for {ticker}: {e}")

# --- API Endpoints ---
@app.get("/")
def read_root():
    return {"message": "Welcome to the Dynamic Stock Forecasting API."}

@app.get("/predict")
def predict_forecast(ticker: str, days: int = 7):
    """
    Generates a future forecast for the specified ticker and days.
    """
    try:
        model = load_model(ticker.upper())
        
        # Create a future dataframe
        future_df = model.make_future_dataframe(periods=days, freq='D')
        forecast = model.predict(future_df)
        
        # Extract and return the relevant part
        response_data = forecast.tail(days)[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
        response_data['ds'] = response_data['ds'].dt.strftime('%Y-%m-%d')
        
        return response_data.to_dict('records')
        
    except HTTPException as e:
        # Re-raise HTTPException to return proper error codes
        raise e
    except Exception as e:
        return HTTPException(status_code=500, detail=f"An error occurred during prediction: {e}")


@app.get("/historical-data")
def get_historical(ticker: str):
    """
    Fetches and returns historical OHLCV data for a ticker.
    """
    ticker = ticker.upper()
    try:
        # 1. Ensure data file exists and is up-to-date
        config = read_config(CONFIG_PATH)
        params = config['data_ingestion']
        DATA_PATH = DATA_DIR / f"{ticker}_data.csv"
        
        fetch_data(
            ticker=ticker,
            start_date=params['start_date'],
            end_date=params['end_date'],
            output_path=DATA_PATH
        )

        # 2. Load and clean data
        df = pd.read_csv(
            DATA_PATH, 
            index_col=0, 
            parse_dates=True, 
            date_format='%Y-%m-%d'
        )
        
        # Clean all necessary columns
        for col in ['Open', 'High', 'Low', 'Close']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df.dropna(subset=['Close'], inplace=True) 

        # Reset index to make 'Date' a column for JSON
        df.reset_index(inplace=True)
        # Robust fix: The new date column is the first one, let's rename it.
        df.rename(columns={df.columns[0]: 'Date'}, inplace=True)

        # 3. Return as JSON records
        return df.to_dict('records')

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching historical data: {e}")
    
# This allows running the app directly
if __name__ == "__main__":
    # Pre-load the SPY model on startup
    load_model("SPY") 
    uvicorn.run(app, host="127.0.0.1", port=8000)
    