import pandas as pd
import gdown
import logging
import json
import os

# --- Logging Setup (reused for consistency) ---
class JsonFormatter(logging.Formatter):
    """Custom formatter to output logs in JSON format."""
    def format(self, record):
        log_record = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "message": record.getMessage(),
            "module": record.module,
            "funcName": record.funcName,
            "lineno": record.lineno,
        }
        if record.exc_info:
            log_record['exc_info'] = self.formatException(record.exc_info)
        return json.dumps(log_record)

def setup_logging():
    """Sets up JSON logging to a file for this specific question."""
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if not logger.handlers:
        handler = logging.FileHandler('debug_q4.log.json', mode='w')
        formatter = JsonFormatter()
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger

# --- Main Solution ---

def download_data(file_id, output_path, logger):
    """Downloads data from Google Drive if the file doesn't already exist."""
    if os.path.exists(output_path):
        logger.info(f"Data file '{output_path}' already exists. Skipping download.")
        return True
    
    logger.info(f"Downloading data file to '{output_path}'...")
    try:
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, output_path, quiet=False)
        logger.info("Download complete.")
        return True
    except Exception as e:
        logger.error("Failed to download data file.", exc_info=True)
        print(f"Error downloading data: {e}")
        return False

def solve_question_4():
    """
    Solves Question 4: Simple RSI-Based Trading Strategy.
    """
    logger = setup_logging()
    logger.info("Starting solution for Question 4: RSI-Based Trading Strategy.")

    # Step 1 & 2: Download precomputed data if it doesn't exist
    file_id = "1grCTCzMZKY5sJRtdbLVCXg8JXA8VPyg-"
    data_path = "data.parquet"
    if not download_data(file_id, data_path, logger):
        return

    try:
        df = pd.read_parquet(data_path, engine="pyarrow")
        logger.info(f"Successfully loaded data from '{data_path}'. Shape: {df.shape}")
    except Exception as e:
        logger.error(f"Failed to read parquet file '{data_path}'.", exc_info=True)
        print(f"Error reading data file: {e}")
        return

    # Step 3 & 4: RSI Strategy Setup and Filtering
    rsi_threshold = 25
    start_date = '2000-01-01'
    end_date = '2025-06-01'
    
    logger.info(f"Filtering data for RSI < {rsi_threshold} between {start_date} and {end_date}.")
    
    df['Date'] = pd.to_datetime(df['Date'])

    selected_df = df[
        (df['rsi'] < rsi_threshold) &
        (df['Date'] >= start_date) &
        (df['Date'] <= end_date)
    ].copy()

    # Verification step from homework
    num_trades = len(selected_df)
    logger.info(f"Found {num_trades} trades matching the criteria.")
    if num_trades != 1568:
        logger.warning(f"Expected 1568 trades, but found {num_trades}. The source data or filters may have changed.")
        print(f"\n[WARNING] Expected 1568 trades, but found {num_trades}. The result might differ from the expected answer.")

    # Step 5: Calculate Net Profit
    investment_per_trade = 1000
    
    # Ensure we only calculate profit on trades where future growth is known
    selected_df.dropna(subset=['growth_future_30d'], inplace=True)
    
    net_income = investment_per_trade * (selected_df['growth_future_30d'] - 1).sum()
    net_income_thousands = net_income / 1000

    logger.info(f"Final Answer (in $K): {net_income_thousands:.2f}")

    print("\n" + "="*60)
    print(f"Total net profit from the RSI strategy (in $ thousands): {net_income_thousands:.2f}")
    print("="*60 + "\n")

if __name__ == "__main__":
    solve_question_4()