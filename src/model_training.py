import pandas as pd
from prophet import Prophet
from pathlib import Path
import json
from prophet.serialize import model_to_json

# Import our existing functions
from utils import read_config
from data_ingestion import fetch_data

# Define the paths
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"
CONFIG_PATH = ROOT_DIR / "config.yaml"

def train_model():
    """
    Trains the Prophet model on the full dataset and saves it to a file.
    """
    print("Starting model training process...")
    
    # 1. Load config
    config = read_config(CONFIG_PATH)
    params = config['data_ingestion']
    
    # 2. Fetch/Update data
    DATA_PATH = DATA_DIR / f"{params['ticker']}_data.csv"
    fetch_data(
        ticker=params['ticker'],
        start_date=params['start_date'],
        end_date=params['end_date'],
        output_path=DATA_PATH
    )
    
    # 3. Load and clean data (same logic as notebook)
    print(f"Loading data from {DATA_PATH}...")
    df = pd.read_csv(
        DATA_PATH, 
        index_col=0, 
        parse_dates=True, 
        date_format='%Y-%m-%d'
    )
    df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
    df.dropna(subset=['Close'], inplace=True)
    
    # 4. Prepare data for Prophet
    prophet_df = pd.DataFrame({'ds': df.index, 'y': df['Close']})
    
    # 5. Train the model
    print("Training Prophet model on the full dataset...")
    model = Prophet(yearly_seasonality=True, daily_seasonality=False)
    model.fit(prophet_df)
    print("Model training complete.")
    
    # 6. Save the model
    MODELS_DIR.mkdir(exist_ok=True) # Ensure the 'models' directory exists
    model_path = MODELS_DIR / "prophet_model.json"
    
    with open(model_path, 'w') as fout:
        fout.write(model_to_json(model)) # Save model as a JSON file
        
    print(f"Model successfully saved to {model_path}")

if __name__ == "__main__":
    train_model()