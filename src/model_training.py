import pandas as pd
from prophet import Prophet
from pathlib import Path
from prophet.serialize import model_to_json
from utils import read_config
from data_ingestion import fetch_data

# Define paths
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"
CONFIG_PATH = ROOT_DIR / "config.yaml"

def train_and_save_model(ticker: str) -> bool:
    """
    Trains a Prophet model for a specific ticker and saves it to a file.
    Returns True if successful, False otherwise.
    """
    print(f"Starting model training process for {ticker}...")
    
    try:
        # 1. Load config
        config = read_config(CONFIG_PATH)
        params = config['data_ingestion']
        
        # 2. Fetch/Update data
        DATA_PATH = DATA_DIR / f"{ticker}_data.csv"
        fetch_data(
            ticker=ticker,
            start_date=params['start_date'],
            end_date=params['end_date'],
            output_path=DATA_PATH
        )
        
        # 3. Load and clean data
        print(f"Loading data from {DATA_PATH}...")
        df = pd.read_csv(
            DATA_PATH, 
            index_col=0, 
            parse_dates=True, 
            date_format='%Y-%m-%d'
        )
        df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
        df.dropna(subset=['Close'], inplace=True)
        
        if df.empty:
            print(f"No data found for {ticker} after cleaning. Aborting training.")
            return False

        # 4. Prepare data for Prophet
        prophet_df = pd.DataFrame({'ds': df.index, 'y': df['Close']})
        
        # 5. Train the model
        print(f"Training Prophet model for {ticker}...")
        model = Prophet(yearly_seasonality=True, daily_seasonality=False)
        model.fit(prophet_df)
        print("Model training complete.")
        
        # 6. Save the model
        MODELS_DIR.mkdir(exist_ok=True)
        model_path = MODELS_DIR / f"prophet_model_{ticker}.json"
        
        with open(model_path, 'w') as fout:
            fout.write(model_to_json(model))
            
        print(f"Model successfully saved to {model_path}")
        return True
        
    except Exception as e:
        print(f"An error occurred during training for {ticker}: {e}")
        return False

if __name__ == "__main__":
    # You can run this file directly to pre-train a common model
    train_and_save_model("SPY")
    train_and_save_model("AAPL")