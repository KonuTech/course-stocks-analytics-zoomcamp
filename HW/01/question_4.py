# question_4.py
"""
Question 4: Analyze Amazon's stock reaction to positive earnings surprises.
Calculate the median 2-day percentage change in stock prices following positive earnings surprises.
"""
import pandas as pd
import numpy as np
import yfinance as yf
import logging
from typing import Dict, Tuple
from datetime import datetime
import re

logger = logging.getLogger(__name__)

def clean_eps_value(value):
    """
    Clean EPS values that may contain '???' characters or other formatting issues.
    """
    if pd.isna(value) or value == '-' or value == '':
        return np.nan
    
    # Convert to string and clean
    str_value = str(value)
    
    # Replace '???' with empty string and extract numeric part
    cleaned = re.sub(r'\?+', '', str_value)
    
    try:
        return float(cleaned)
    except (ValueError, TypeError):
        return np.nan

def load_earnings_data(file_path: str = 'ha1_Amazon.csv') -> pd.DataFrame:
    """
    Load and clean Amazon earnings data from CSV file.
    
    Args:
        file_path: Path to the CSV file
        
    Returns:
        Cleaned DataFrame with earnings data
    """
    try:
        # Read CSV with semicolon delimiter as specified in the question
        df = pd.read_csv(file_path, delimiter=';')
        
        print(f"Raw data loaded: {len(df)} rows")
        print(f"Columns: {list(df.columns)}")
        
        # Clean column names (remove any extra spaces)
        df.columns = df.columns.str.strip()
        
        # Convert earnings date to datetime
        df['Earnings Date'] = pd.to_datetime(df['Earnings Date'], errors='coerce')
        
        # Clean EPS values
        df['EPS Estimate Clean'] = df['EPS Estimate'].apply(clean_eps_value)
        df['Reported EPS Clean'] = df['Reported EPS'].apply(clean_eps_value)
        
        # Clean surprise percentage
        df['Surprise (%) Clean'] = pd.to_numeric(df['Surprise (%)'], errors='coerce')
        
        # Filter out rows with missing essential data and future dates
        current_date = datetime.now()
        df_clean = df[
            (df['Earnings Date'].notna()) &
            (df['Earnings Date'] < current_date) &
            (df['Reported EPS Clean'].notna()) &
            (df['EPS Estimate Clean'].notna()) &
            (df['Surprise (%) Clean'].notna())
        ].copy()
        
        # Sort by earnings date
        df_clean = df_clean.sort_values('Earnings Date').reset_index(drop=True)
        
        print(f"Clean data after filtering: {len(df_clean)} rows")
        if len(df_clean) > 0:
            print(f"Date range: {df_clean['Earnings Date'].min().strftime('%Y-%m-%d')} to {df_clean['Earnings Date'].max().strftime('%Y-%m-%d')}")
        
        return df_clean
        
    except FileNotFoundError:
        raise FileNotFoundError(f"CSV file {file_path} not found. Please ensure the file exists in the current directory.")
    except Exception as e:
        raise Exception(f"Error reading {file_path}: {e}")

def calculate_2day_returns(price_data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate 2-day percentage changes for all historical dates.
    For each sequence of 3 consecutive trading days (Day 1, Day 2, Day 3),
    compute the return as Close_Day3 / Close_Day1 - 1.
    
    Args:
        price_data: DataFrame with stock price data
        
    Returns:
        DataFrame with 2-day returns added
    """
    price_data = price_data.copy()
    
    # Calculate 2-day return: Close_Day3 / Close_Day1 - 1
    # This means we're looking 2 days ahead from each day
    price_data['Close_2Days_Later'] = price_data['Close'].shift(-2)
    price_data['2Day_Return'] = (price_data['Close_2Days_Later'] / price_data['Close']) - 1
    price_data['2Day_Return_Pct'] = price_data['2Day_Return'] * 100.0
    
    return price_data

def question_4_amazon_earnings_analysis(csv_file_path: str = 'ha1_Amazon.csv') -> Dict:
    """
    Analyze 2-day stock returns following positive earnings surprises for Amazon.
    Calculate the median 2-day percentage change in stock prices following positive earnings surprises.

    Args:
        csv_file_path: Path to the CSV file containing earnings data
        
    Returns:
        Dictionary of earnings reaction metrics.
    """
    print("QUESTION 4: Amazon Earnings Surprise Analysis")
    print("=" * 60)
    print("Calculate the median 2-day percentage change in stock prices following positive earnings surprises")
    print()

    # Step 1: Load earnings data from CSV
    print("Step 1: Loading earnings data from CSV...")
    earnings_df = load_earnings_data(csv_file_path)
    
    if earnings_df.empty:
        raise ValueError("No valid earnings data available")

    # Step 2: Download complete historical price data using yfinance
    print("Step 2: Downloading complete historical price data using yfinance...")
    logger.info("Downloading Amazon stock data")
    
    # Get date range from earnings data and extend it
    start_date = earnings_df['Earnings Date'].min() - pd.Timedelta(days=30)
    end_date = max(earnings_df['Earnings Date'].max() + pd.Timedelta(days=30), datetime.now())
    
    amzn_data = yf.download('AMZN', start=start_date, end=end_date, 
                           progress=False, auto_adjust=True)
    
    if amzn_data.empty:
        raise ValueError("Failed to download AMZN data")

    # Handle MultiIndex columns if present
    if isinstance(amzn_data.columns, pd.MultiIndex):
        amzn_data.columns = amzn_data.columns.droplevel(1)

    print(f"Downloaded stock data: {len(amzn_data)} trading days")
    print(f"Stock data range: {amzn_data.index.min().strftime('%Y-%m-%d')} to {amzn_data.index.max().strftime('%Y-%m-%d')}")

    # Step 3: Calculate 2-day percentage changes for all historical dates
    print("Step 3: Calculating 2-day percentage changes for all historical dates...")
    amzn_data = calculate_2day_returns(amzn_data)
    
    # Remove rows where we can't calculate 2-day returns (last 2 days)
    valid_returns = amzn_data['2Day_Return_Pct'].dropna()
    
    print(f"Total trading days with valid 2-day returns: {len(valid_returns)}")

    # Calculate baseline statistics for all historical dates
    median_all = float(valid_returns.median())
    mean_all = float(valid_returns.mean())
    std_all = float(valid_returns.std())

    print(f"Baseline Statistics (All Historical Dates):")
    print(f"  Median 2-day return: {median_all:.2f}%")
    print(f"  Mean 2-day return: {mean_all:.2f}%")
    print(f"  Standard deviation: {std_all:.2f}%")
    print()

    # Step 4: Identify positive earnings surprises
    print("Step 4: Identifying positive earnings surprises...")
    positive_surprises = earnings_df[
        (earnings_df['Reported EPS Clean'] > earnings_df['EPS Estimate Clean']) |
        (earnings_df['Surprise (%) Clean'] > 0)
    ].copy()

    print(f"Total earnings announcements: {len(earnings_df)}")
    print(f"Positive earnings surprises: {len(positive_surprises)}")
    print()

    # Step 5: Calculate 2-day percentage changes following positive earnings surprises
    print("Step 5: Calculating 2-day percentage changes following positive earnings surprises...")
    
    returns_after_positive = []
    matched_events = []
    unmatched_events = []
    
    for _, row in positive_surprises.iterrows():
        earnings_date = row['Earnings Date']
        
        # Find the trading day on or after the earnings date (Day 2 in the sequence)
        # We want the return starting from this day
        future_dates = amzn_data.index[amzn_data.index >= earnings_date]
        
        if not future_dates.empty:
            trading_date = future_dates[0]  # Day 2 (earnings announcement day or next trading day)
            
            if trading_date in amzn_data.index:
                return_value = amzn_data.loc[trading_date, '2Day_Return_Pct']
                
                if not pd.isna(return_value):
                    returns_after_positive.append(float(return_value))
                    matched_events.append({
                        'earnings_date': earnings_date,
                        'trading_date': trading_date,
                        'reported_eps': row['Reported EPS Clean'],
                        'estimated_eps': row['EPS Estimate Clean'],
                        'surprise_pct': row['Surprise (%) Clean'],
                        '2day_return_pct': float(return_value),
                        'close_day1': float(amzn_data.loc[trading_date, 'Close']),
                        'close_day3': float(amzn_data.loc[trading_date, 'Close_2Days_Later']) if not pd.isna(amzn_data.loc[trading_date, 'Close_2Days_Later']) else None
                    })
                else:
                    unmatched_events.append((earnings_date, "No 2-day return data available"))
            else:
                unmatched_events.append((earnings_date, "Trading date not found"))
        else:
            unmatched_events.append((earnings_date, "No future trading dates"))

    if not returns_after_positive:
        raise ValueError("No valid earnings dates matched with 2-day return data")

    print(f"Successfully matched {len(matched_events)} positive earnings events with 2-day return data")
    if unmatched_events:
        print(f"Could not match {len(unmatched_events)} events:")
        for date, reason in unmatched_events[:5]:  # Show first 5
            print(f"  {date.strftime('%Y-%m-%d')}: {reason}")
        if len(unmatched_events) > 5:
            print(f"  ... and {len(unmatched_events) - 5} more")
    print()

    # Calculate statistics for positive earnings surprises
    median_positive = float(np.median(returns_after_positive))
    mean_positive = float(np.mean(returns_after_positive))
    std_positive = float(np.std(returns_after_positive))

    print(f"Results for Positive Earnings Surprises:")
    print(f"  Number of events: {len(returns_after_positive)}")
    print(f"  Median 2-day return: {median_positive:.2f}%")
    print(f"  Mean 2-day return: {mean_positive:.2f}%")
    print(f"  Standard deviation: {std_positive:.2f}%")
    print()

    # Step 6: Compare with baseline
    print("Step 6: Comparison to All Historical Dates:")
    median_difference = median_positive - median_all
    mean_difference = mean_positive - mean_all
    
    print(f"  Median difference: {median_difference:.2f}% (positive surprise vs all days)")
    print(f"  Mean difference: {mean_difference:.2f}% (positive surprise vs all days)")
    
    if median_difference > 0:
        print(f"  → Positive surprises show {median_difference:.2f}% higher median returns")
    else:
        print(f"  → Positive surprises show {abs(median_difference):.2f}% lower median returns")
    print()

    # Statistical significance test
    try:
        from scipy import stats
        if len(returns_after_positive) > 1:
            t_stat, p_value = stats.ttest_1samp(returns_after_positive, median_all)
            print(f"Statistical test (t-test vs baseline median): p-value = {p_value:.4f}")
            if p_value < 0.05:
                print("  → Statistically significant difference at 5% level")
            else:
                print("  → No statistically significant difference at 5% level")
            print()
    except ImportError:
        print("Statistical test skipped (scipy not available)")

    # Show top examples
    print("Top 5 Positive Surprise Events by 2-Day Return:")
    sorted_events = sorted(matched_events, key=lambda x: x['2day_return_pct'], reverse=True)
    for i, event in enumerate(sorted_events[:5], 1):
        eps_surprise = event['reported_eps'] - event['estimated_eps']
        print(f"{i}. {event['earnings_date'].strftime('%Y-%m-%d')}: "
              f"EPS {event['reported_eps']:.2f} vs {event['estimated_eps']:.2f} "
              f"(+{eps_surprise:.2f}, {event['surprise_pct']:.1f}%), "
              f"2-day return: {event['2day_return_pct']:.2f}%")
    print()

    # Return distribution
    positive_count = sum(1 for r in returns_after_positive if r > 0)
    negative_count = len(returns_after_positive) - positive_count
    
    print(f"Return Distribution After Positive Surprises:")
    print(f"  Positive returns: {positive_count} ({positive_count/len(returns_after_positive)*100:.1f}%)")
    print(f"  Negative returns: {negative_count} ({negative_count/len(returns_after_positive)*100:.1f}%)")

    return {
        'baseline_median': median_all,
        'baseline_mean': mean_all,
        'baseline_std': std_all,
        'positive_median': median_positive,
        'positive_mean': mean_positive,
        'positive_std': std_positive,
        'total_positive_events': len(returns_after_positive),
        'median_difference': median_difference,
        'mean_difference': mean_difference,
        'sample_events': sorted_events[:5],
        'positive_return_rate': positive_count/len(returns_after_positive),
        'total_earnings_events': len(earnings_df),
        'positive_surprises_count': len(positive_surprises),
        'matched_events': len(matched_events),
        'unmatched_events': len(unmatched_events)
    }

if __name__ == "__main__":
    try:
        result = question_4_amazon_earnings_analysis()
        print(f"\n" + "="*60)
        print(f"FINAL ANSWER")
        print(f"="*60)
        print(f"MEDIAN 2-DAY PERCENTAGE CHANGE FOLLOWING POSITIVE EARNINGS SURPRISES:")
        print(f"{result['positive_median']:.2f}%")
        print(f"")
        print(f"Additional Statistics:")
        print(f"  Total positive surprise events analyzed: {result['matched_events']}")
        print(f"  Baseline median (all historical dates): {result['baseline_median']:.2f}%")
        print(f"  Outperformance vs baseline: {result['median_difference']:.2f}%")
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        print(f"Error: {e}")
