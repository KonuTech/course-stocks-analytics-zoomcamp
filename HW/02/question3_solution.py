import pandas as pd
import numpy as np
import requests
import yfinance as yf
from io import StringIO
import logging
import json
import re

# --- Logging Setup (reused from previous questions for consistency) ---
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
        handler = logging.FileHandler('debug_q3.log.json', mode='w')
        formatter = JsonFormatter()
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger

# --- Data Fetching and Processing Functions ---

def get_2024_ipo_tickers(logger):
    """
    Fetches the list of tickers for companies that had an IPO in the first 5 months of 2024.
    It also filters out any companies that do not have a valid IPO price listed.
    """
    url = "https://stockanalysis.com/ipos/2024/"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    logger.info(f"Fetching IPO data from {url}")
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        df = pd.read_html(StringIO(response.text))[0]
        
        # Clean 'IPO Price' column and filter out entries without a valid price.
        # This addresses the issue of skewed results from companies with missing IPO data.
        original_count = len(df)
        # First, clean the 'IPO Price' column to handle non-numeric characters (e.g., '$').
        df['IPO Price'] = df['IPO Price'].astype(str).str.replace(r'[^\d.]', '', regex=True)
        df['IPO Price'] = pd.to_numeric(df['IPO Price'], errors='coerce') # Convert cleaned strings to numbers
        df.dropna(subset=['IPO Price'], inplace=True)
        filtered_count = len(df)
        if (original_count - filtered_count) > 0:
            logger.info(f"Filtered out {original_count - filtered_count} companies with no valid IPO price.")
        
        # Add a check to see if the DataFrame is empty after filtering. This prevents silent failures.
        if df.empty:
            logger.warning("DataFrame is empty after filtering for valid IPO prices. No tickers will be returned.")
            print("\n[CRITICAL] No companies with a valid IPO price were found. The script cannot continue.")
            return []
        
        df['IPO Date'] = pd.to_datetime(df['IPO Date'])
        filtered_df = df[df['IPO Date'] < '2024-06-01'].copy()
        tickers = filtered_df['Symbol'].tolist()
        logger.info(f"Found {len(tickers)} tickers with valid IPO prices for IPOs before June 1, 2024.")
        return tickers
    except Exception as e:
        logger.error("Failed to fetch or process IPO tickers.", exc_info=True)
        print(f"Error getting IPO tickers: {e}")
        return []

def download_and_calculate_growth(tickers, logger):
    """
    Downloads OHLCV data and calculates 1-12 month future growth for each stock.
    """
    if not tickers:
        logger.error("Ticker list is empty. Aborting.")
        return pd.DataFrame()

    logger.info(f"Downloading OHLCV data for {len(tickers)} tickers...")
    try:
        # Download data for a sufficient period to calculate 12 months of future growth.
        # The problem is set in June 2025, so we have enough data for IPOs from early 2024.
        data = yf.download(tickers, start='2024-01-01', end='2025-06-07', auto_adjust=False)
        if data.empty:
            logger.warning("yfinance download returned an empty DataFrame.")
            return pd.DataFrame()
        
        df = data.stack(level=1).rename_axis(['Date', 'Ticker']).reset_index()
        logger.info(f"Successfully downloaded and reshaped data. Shape: {df.shape}")

        df = df.sort_values(by=['Ticker', 'Date'])

        # Calculate future growth for 1 to 12 months
        logger.info("Calculating future growth columns (1m to 12m)...")
        for month in range(1, 13):
            trading_days = month * 21
            # Use groupby().shift() with a negative period to look into the future
            future_price = df.groupby('Ticker')['Close'].shift(-trading_days)
            df[f'future_growth_{month}m'] = future_price / df['Close']
        
        logger.info("Finished calculating all future growth columns.")
        return df

    except Exception as e:
        logger.error("Failed to download or process data.", exc_info=True)
        print(f"Error during data download/processing: {e}")
        return pd.DataFrame()

def solve_question_3():
    """
    Solves Question 3: ‘Fixed Months Holding Strategy’.
    """
    logger = setup_logging()
    logger.info("Starting solution for Question 3.")
    print("[INFO] Starting solution for Question 3: Fixed Months Holding Strategy.")

    # Step 1: Get tickers and download data with future growth columns
    print("[INFO] Fetching list of IPO tickers...")
    tickers = get_2024_ipo_tickers(logger)
    print(f"[INFO] Found {len(tickers)} valid tickers.")

    if not tickers:
        print("[INFO] Ticker list is empty. Exiting script.")
        return

    print("[INFO] Downloading historical stock data. This may take a moment...")
    stocks_df = download_and_calculate_growth(tickers, logger)
    print(f"[INFO] Data download complete. DataFrame shape: {stocks_df.shape}")
    if stocks_df.empty:
        print("Could not retrieve stock data. Aborting.")
        print("\nNOTE: This script simulates a scenario in June 2025. "
              "If run before this date, future growth data may be incomplete.")
        return

    # Step 2 & 3: Determine the first trading day for each ticker and get its performance data.
    # We use idxmin() to efficiently find the index of the first entry for each ticker.
    logger.info("Determining the first trading day for each ticker to analyze its future growth.")
    first_day_df = stocks_df.loc[stocks_df.groupby('Ticker')['Date'].idxmin()].copy()
    
    logger.info(f"Successfully isolated the first trading day for {first_day_df.shape[0]} tickers.")

    # Step 4: Isolate future growth columns and compute descriptive statistics
    growth_cols = [f'future_growth_{m}m' for m in range(1, 13)]
    first_day_growth_df = first_day_df[growth_cols]
    
    logger.info("Computing descriptive statistics for future growth from the first trading day.")
    stats = first_day_growth_df.describe()
    print("\nDescriptive Statistics for Future Growth from First Trading Day:")
    print(stats)
    logger.info(f"\n{stats.to_string()}")

    # Step 5: Determine the best holding period based on mean growth
    logger.info("Determining the optimal holding period based on mean future growth.")
    mean_growth = stats.loc['mean']
    
    if mean_growth.isnull().all():
        logger.error("Mean growth could not be calculated. This is likely due to lack of future data.")
        print("\nCould not calculate mean growth. Aborting analysis. (This is expected if run before June 2025).")
        return

    optimal_period_col = mean_growth.idxmax()
    max_avg_growth = mean_growth.max()
    
    # Extract the number of months from the column name (e.g., 'future_growth_3m' -> 3)
    optimal_months = int(re.search(r'\d+', optimal_period_col).group())

    logger.info(f"Optimal holding period is {optimal_months} months with a mean growth of {max_avg_growth:.4f}.")

    print("\n" + "="*60)
    print("Mean Future Growth by Holding Period (Months):")
    print(mean_growth.round(4))
    print("\n" + "="*60)
    print(f"The optimal number of months to hold is: {optimal_months}")
    print(f"This period yields the highest average growth of: {max_avg_growth:.4f}")
    print("="*60 + "\n")

    # Additional verification as per homework instructions
    second_highest_growth = mean_growth.drop(optimal_period_col).max()
    uplift = (max_avg_growth / second_highest_growth - 1) * 100
    print(f"This represents an uplift of {uplift:.2f}% over the next best period.")
    if max_avg_growth < 1:
        print("As expected, the average return is still less than 1 (a net loss on average).")
    else:
        print("Note: The average return is >= 1 (a net gain on average).")

if __name__ == "__main__":
    solve_question_3()