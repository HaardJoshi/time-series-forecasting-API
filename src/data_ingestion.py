import yfinance as yf
import pandas as pd
from pathlib import Path
from utils import read_config

def fetch_data(ticker: str, start_date: str, end_date: str, output_path: Path):
    """
    Fetches historical stock data from Yahoo Finance and saves it to a CSV file.

    Args:
        ticker (str): The stock ticker symbol (e.g., 'SPY').
        start_date (str): The start date for the data in 'YYYY-MM-DD' format.
        end_date (str): The end date for the data in 'YYYY-MM-DD' format.
        output_path (Path): The file path to save the downloaded data.
    """
    try:
        print(f"Fetching data for {ticker} from {start_date} to {end_date}...")
        data = yf.download(ticker, start=start_date, end=end_date)
        
        if data.empty:
            print(f"Warning: No data found for ticker {ticker}. It may be invalid.")
            return

        # Ensure the output directory exists before saving
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save the data to the specified path
        data.to_csv(output_path)
        print(f"Data successfully saved to {output_path}")

    except Exception as e:
        print(f"An error occurred: {e}")

# This block allows us to test the function directly by running this script
if __name__ == '__main__':
    # Read configuration from the YAML file
    config = read_config()
    
    if config:
        # Get parameters from the 'data_ingestion' section of the config
        params = config['data_ingestion']
        TICKER = params['ticker']
        START_DATE = params['start_date']
        END_DATE = params['end_date']
        
        # Define the output path
        OUTPUT_PATH = Path(__file__).resolve().parent.parent / "data" / f"{TICKER}_data.csv"
        
        # Call the function with parameters from the config file
        fetch_data(ticker=TICKER, start_date=START_DATE, end_date=END_DATE, output_path=OUTPUT_PATH)