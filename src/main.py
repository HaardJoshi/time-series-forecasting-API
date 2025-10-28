import uvicorn
from fastapi import FastAPI, HTTPException
from prophet import Prophet
from prophet.serialize import model_from_json
from pathlib import Path
import atexit
import httpx

# Import our new training function
from model_training import train_and_save_model

# --- Configuration ---
app = FastAPI(title="Dynamic Stock Forecast API", version="2.0")

ROOT_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT_DIR / "models"

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

# This allows running the app directly
if __name__ == "__main__":
    # Pre-load the SPY model on startup
    load_model("SPY") 
    uvicorn.run(app, host="127.0.0.1", port=8000)