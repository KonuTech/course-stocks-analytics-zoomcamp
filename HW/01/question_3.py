# question_3.py
"""
Question 3: Analyze historical market corrections (>5% drawdown) in S&P 500.
"""
import yfinance as yf
import numpy as np
import pandas as pd
import logging
from typing import Dict

logger = logging.getLogger(__name__)

def question_3_market_corrections() -> Dict:
    """
    Analyze drawdowns in the S&P 500 since 1950.
    
    Returns:
        Dictionary with duration percentiles and top corrections.
    """
    # Download data with explicit parameters to avoid warnings
    df = yf.download('^GSPC', start='1950-01-01', end='2025-05-31', 
                     progress=False, auto_adjust=True)
    
    if df.empty:
        raise ValueError("Failed to download S&P 500 data")
    
    # Debug: Print column information
    print(f"DataFrame shape: {df.shape}")
    print(f"Column names: {list(df.columns)}")
    print(f"Column types: {type(df.columns)}")
    
    # Handle MultiIndex columns if present
    if isinstance(df.columns, pd.MultiIndex):
        print("MultiIndex detected, flattening...")
        # Drop the ticker level (level 1), keep the price type level (level 0)
        df.columns = df.columns.droplevel(1)
        print(f"Columns after flattening: {list(df.columns)}")
    
    # Find the appropriate close price column
    close_col = None
    possible_close_cols = ['Close', 'Adj Close', 'close', 'adj close', 'adj_close']
    
    for col in possible_close_cols:
        if col in df.columns:
            close_col = col
            break
    
    if close_col is None:
        # If no standard close column found, use the first available price column
        price_cols = [col for col in df.columns if any(word in col.lower() for word in ['close', 'price'])]
        if price_cols:
            close_col = price_cols[0]
        else:
            # Last resort: use any numeric column
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                close_col = numeric_cols[-1]  # Often the last column is close/adj close
            else:
                raise ValueError(f"No suitable price column found. Available columns: {list(df.columns)}")
    
    print(f"Using column '{close_col}' as close price")
    
    # Extract the close price series
    close_prices = df[close_col]
    
    # Remove any NaN values
    close_prices = close_prices.dropna()
    
    if close_prices.empty:
        raise ValueError("No valid price data after removing NaN values")
    
    # Create a new DataFrame with just the close prices
    df = pd.DataFrame({'Close': close_prices})
    
    # Calculate rolling maximum and identify all-time highs
    df['Rolling_Max'] = df['Close'].expanding().max()
    df['Is_ATH'] = (df['Close'] == df['Rolling_Max'])
    
    # Get dates where new ATHs were reached
    ath_dates = df[df['Is_ATH']].index.tolist()
    
    if len(ath_dates) < 2:
        raise ValueError("Insufficient ATH data points")
    
    corrections = []
    
    # Analyze periods between ATHs
    for i in range(len(ath_dates) - 1):
        start_date = ath_dates[i]
        next_ath_date = ath_dates[i + 1]
        
        # Get segment between ATHs
        segment = df.loc[start_date:next_ath_date].copy()
        
        if len(segment) < 2:
            continue
        
        # Calculate drawdown
        high_price = segment['Close'].iloc[0]
        low_price = segment['Close'].min()
        drawdown_pct = ((high_price - low_price) / high_price) * 100
        
        # Only consider corrections > 5%
        if drawdown_pct > 5:
            # Find the date of the lowest point
            low_date = segment[segment['Close'] == low_price].index[0]
            duration_days = (low_date - start_date).days
            
            corrections.append({
                'Start': start_date,
                'End': low_date,
                'Duration_Days': duration_days,
                'Drawdown_Pct': float(drawdown_pct),
                'High': float(high_price),
                'Low': float(low_price)
            })
    
    if not corrections:
        raise ValueError("No drawdowns > 5% found")
    
    # Convert to DataFrame for analysis
    df_corr = pd.DataFrame(corrections)
    durations = df_corr['Duration_Days'].values
    
    # Calculate percentiles
    p25 = float(np.percentile(durations, 25))
    p50 = float(np.percentile(durations, 50))
    p75 = float(np.percentile(durations, 75))
    
    print(f"Total corrections found: {len(df_corr)}")
    print(f"Median correction duration: {p50:.1f} days")
    print(f"25th percentile duration: {p25:.1f} days")
    print(f"75th percentile duration: {p75:.1f} days")
    
    return {
        'total_corrections': len(df_corr),
        'median_duration_days': p50,
        'percentile_25': p25,
        'percentile_75': p75,
        'top_10_corrections': df_corr.nlargest(10, 'Drawdown_Pct').to_dict('records'),
        'all_corrections': df_corr.to_dict('records')
    }

if __name__ == "__main__":
    try:
        result = question_3_market_corrections()
        print("\nTop 5 corrections by magnitude:")
        for i, corr in enumerate(result['top_10_corrections'][:5], 1):
            print(f"{i}. {corr['Start'].strftime('%Y-%m-%d')}: "
                  f"{corr['Drawdown_Pct']:.1f}% drawdown, "
                  f"{corr['Duration_Days']} days duration")
    except Exception as e:
        print(f"Error: {e}")
