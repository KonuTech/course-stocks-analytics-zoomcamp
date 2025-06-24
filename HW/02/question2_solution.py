import pandas as pd
import numpy as np
import requests
import yfinance as yf
from io import StringIO
import logging
import json

# --- Logging Setup (copied from Q1 for consistency) ---
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
    """Sets up JSON logging to a file."""
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if not logger.handlers:
        handler = logging.FileHandler('debug_q2.log.json', mode='w')
        formatter = JsonFormatter()
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger

# --- Main Solution ---

def get_2024_ipo_tickers(logger):
    """
    Fetches the list of tickers for companies that had an IPO in the first 5 months of 2024.
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
        
        # Clean and filter data
        df['IPO Date'] = pd.to_datetime(df['IPO Date'])
        filtered_df = df[df['IPO Date'] < '2024-06-01'].copy()
        
        tickers = filtered_df['Symbol'].tolist()
        logger.info(f"Found {len(tickers)} tickers for IPOs before June 1, 2024.")
        
        # Verification from homework
        if len(tickers) != 75:
            logger.warning(f"Expected 75 tickers, but found {len(tickers)}. The source data may have changed.")
        
        return tickers
    except Exception as e:
        logger.error("Failed to fetch or process IPO tickers.", exc_info=True)
        print(f"Error getting IPO tickers: {e}")
        return []

def download_and_process_data(tickers, logger):
    """
    Downloads OHLCV data for the given tickers and calculates required metrics.
    """
    if not tickers:
        logger.error("Ticker list is empty. Aborting download.")
        return pd.DataFrame()

    logger.info(f"Downloading OHLCV data for {len(tickers)} tickers...")
    try:
        # Download data up to the required analysis date.
        data = yf.download(tickers, start='2024-01-01', end='2025-06-07', auto_adjust=False)
        if data.empty:
            logger.warning("yfinance download returned an empty DataFrame.")
            return pd.DataFrame()
        
        # Reshape data from wide to long format for easier processing
        df = data.stack(level=1).rename_axis(['Date', 'Ticker']).reset_index()
        logger.info(f"Successfully downloaded and reshaped data. Shape: {df.shape}")

        # Sort values to ensure correct rolling/shift calculations per ticker
        df = df.sort_values(by=['Ticker', 'Date'])

        # Calculate volatility (annualized 30-day rolling std dev of Close price)
        df['volatility'] = df.groupby('Ticker')['Close'].transform(
            lambda x: x.rolling(window=30).std()
        ) * np.sqrt(252)
        
        # Calculate growth over 252 trading days (historical ratio: P_today / P_252_days_ago)
        df['growth_252d'] = df['Close'] / df.groupby('Ticker')['Close'].shift(252)
        
        # Calculate Sharpe Ratio using the formula exactly as specified in the homework
        # Note: A standard formula would be ((growth - 1) - risk_free_rate) / volatility.
        # We are following the homework's literal instruction.
        df['Sharpe'] = (df['growth_252d'] - 0.045) / df['volatility']
        
        logger.info("Calculated volatility, growth_252d, and Sharpe ratio.")
        return df

    except Exception as e:
        logger.error("Failed to download or process OHLCV data.", exc_info=True)
        print(f"Error during data download/processing: {e}")
        return pd.DataFrame()

def solve_question_2():
    """
    Solves Question 2: Median Sharpe Ratio for 2024 IPOs.
    """
    logger = setup_logging()
    logger.info("Starting solution for Question 2: Median Sharpe Ratio.")

    tickers = get_2024_ipo_tickers(logger)
    if not tickers:
        return

    stocks_df = download_and_process_data(tickers, logger)
    if stocks_df.empty:
        print("Could not retrieve sufficient stock data. Aborting.")
        print("\nNOTE: This script simulates a scenario in June 2025. "
              "Since yfinance cannot fetch future data, the analysis will likely yield no results "
              "if run before this date. The code logic correctly follows the homework instructions for that scenario.")
        return

    # Filter for the specific trading day for analysis
    analysis_date = '2025-06-06'
    logger.info(f"Filtering data for analysis date: {analysis_date}")
    daily_snapshot_df = stocks_df[stocks_df['Date'] == analysis_date].copy()

    if daily_snapshot_df.empty:
        logger.warning(f"No data available for the analysis date {analysis_date}. This is expected if run before this date.")
        print(f"\nNo data available for {analysis_date}. Cannot perform final analysis.")
        return

    # Perform descriptive statistics as per homework
    logger.info("Descriptive statistics for the daily snapshot:")
    stats = daily_snapshot_df[['growth_252d', 'Sharpe']].describe()
    logger.info(f"\n{stats.to_string()}")
    print("\nDescriptive Statistics for 2025-06-06:")
    print(stats)

    # Verification checks from homework
    growth_count = daily_snapshot_df['growth_252d'].notna().sum()
    logger.info(f"Found {growth_count} stocks with defined 'growth_252d'.")
    if growth_count != 71:
        logger.warning(f"Expected 71 stocks with growth_252d, but found {growth_count}.")
        print(f"\nWarning: Expected 71 stocks with growth_252d, but found {growth_count}.")

    median_growth = daily_snapshot_df['growth_252d'].median()
    logger.info(f"Median 'growth_252d' is {median_growth:.4f}.")
    print(f"Median 'growth_252d' is {median_growth:.4f} (Expected ~0.75)")

    # Final Answer
    median_sharpe = daily_snapshot_df['Sharpe'].median()
    logger.info(f"Final Answer: Median Sharpe Ratio is {median_sharpe:.4f}")
    
    print("\n" + "="*50)
    print(f"The median Sharpe ratio for these stocks is: {median_sharpe:.4f}")
    print("="*50 + "\n")

    # Additional analysis: Top 10 comparison
    logger.info("Performing additional analysis: Top 10 comparison.")
    top_10_growth = daily_snapshot_df.sort_values('growth_252d', ascending=False).head(10)
    top_10_sharpe = daily_snapshot_df.sort_values('Sharpe', ascending=False).head(10)

    print("Top 10 Companies by 252-day Growth:")
    print(top_10_growth[['Ticker', 'growth_252d']])
    print("\nTop 10 Companies by Sharpe Ratio:")
    print(top_10_sharpe[['Ticker', 'Sharpe', 'growth_252d', 'volatility']])
    
    growth_tickers = set(top_10_growth['Ticker'])
    sharpe_tickers = set(top_10_sharpe['Ticker'])
    
    print("\nDo you observe the same top 10 companies?")
    if growth_tickers == sharpe_tickers:
        print("Answer: Yes, the top 10 companies are the SAME when sorting by growth vs. Sharpe ratio.")
        logger.info("The top 10 companies are the same for both metrics.")
    else:
        print("Answer: No, the top 10 companies are DIFFERENT when sorting by growth vs. Sharpe ratio.")
        logger.info("The top 10 companies are different for the two metrics.")

if __name__ == "__main__":
    solve_question_2()