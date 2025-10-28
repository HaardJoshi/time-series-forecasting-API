import uvicorn
from fastapi import FastAPI
from prophet import Prophet
from prophet.serialize import model_from_json
from pathlib import Path

# --- Configuration ---
# Create a FastAPI app instance
app = FastAPI(title="SPY Forecast API", version="1.0")

# Define the path to our trained model
ROOT_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT_DIR / "models"
MODEL_PATH = MODELS_DIR / "prophet_model.json"

# --- Model Loading ---
# This block will run once when the API starts
try:
    print(f"Loading model from {MODEL_PATH}...")
    with open(MODEL_PATH, 'r') as fin:
        model = model_from_json(fin.read())
    print("Model loaded successfully.")
except FileNotFoundError:
    print(f"Error: Model file not found at {MODEL_PATH}")
    model = None
except Exception as e:
    print(f"An error occurred while loading the model: {e}")
    model = None

# --- API Endpoints ---
@app.get("/")
def read_root():
    """
    A simple root endpoint to check if the API is running.
    """
    return {"message": "Welcome to the SPY Forecasting API. Go to /predict to get a forecast."}

@app.get("/predict")
def predict_forecast(days: int = 7):
    """
    Generates a future forecast for the specified number of days.
    
    Query Parameter:
    - days (int): The number of days to forecast into the future. Default is 7.
    """
    if model is None:
        return {"error": "Model not loaded. Please train the model first."}

    try:
        # Create a future dataframe for the specified number of days
        future_df = model.make_future_dataframe(periods=days, freq='D')
        
        # Generate the forecast
        forecast = model.predict(future_df)
        
        # Extract and return the relevant part of the forecast
        # We return the last 'days' rows
        response_data = forecast.tail(days)[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
        
        # Convert to a more JSON-friendly format
        response_data['ds'] = response_data['ds'].dt.strftime('%Y-%m-%d')
        
        return response_data.to_dict('records')

    except Exception as e:
        return {"error": f"An error occurred during prediction: {e}"}

# This allows running the app directly using "python src/main.py"
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)