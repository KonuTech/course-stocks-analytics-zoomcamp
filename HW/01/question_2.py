# question_2.py
"""
Question 2: Compare YTD performance of global indices.
"""
import yfinance as yf
import logging
import warnings
from typing import Dict, Optional
import pandas as pd

# Suppress pandas FutureWarnings for cleaner output
warnings.filterwarnings('ignore', category=FutureWarning)

logger = logging.getLogger(__name__)

def question_2_ytd_performance() -> Dict[str, Optional[float]]:
    """
    Calculate YTD return from Jan 1 to May 1, 2025 for major indices.

    Returns:
        Dictionary of results including return values and errors.
    """
    indexes = {
        'United States (S&P 500)': '^GSPC',
        'China (Shanghai Composite)': '000001.SS',
        'Hong Kong (Hang Seng)': '^HSI',
        'Australia (S&P/ASX 200)': '^AXJO',
        'India (Nifty 50)': '^NSEI',
        'Canada (S&P/TSX)': '^GSPTSE',
        'Germany (DAX)': '^GDAXI',
        'United Kingdom (FTSE 100)': '^FTSE',
        'Japan (Nikkei 225)': '^N225',
        'Mexico (IPC)': '^MXX',
        'Brazil (Ibovespa)': '^BVSP'
    }

    start_date = '2025-01-01'
    end_date = '2025-05-01'

    ytd_returns = {}
    failed = []
    sp500_return = None

    print(f"Calculating YTD returns from {start_date} to {end_date}")
    print("=" * 60)

    for name, symbol in indexes.items():
        try:
            logger.info(f"Downloading data for {name} ({symbol})")
            df = yf.download(symbol, start=start_date, end=end_date, progress=False, auto_adjust=True)
            
            if df.empty:
                logger.warning(f"No data returned for {name} ({symbol})")
                failed.append(name)
                continue

            # Fix the FutureWarning by using .iloc[0] to get scalar values
            start_price = df['Close'].iloc[0]
            end_price = df['Close'].iloc[-1]
            
            # Calculate return as percentage
            ret_pct = ((end_price - start_price) / start_price) * 100
            
            # Convert to float using .item() method for pandas scalars
            ret_float = float(ret_pct) if hasattr(ret_pct, 'item') else ret_pct
            
            ytd_returns[name] = ret_float
            print(f"{name}: {ret_float:.2f}%")

            # Store S&P 500 return for comparison
            if 'S&P 500' in name:
                sp500_return = ret_float
                
        except Exception as e:
            logger.error(f"Error downloading {name} ({symbol}): {str(e)}")
            failed.append(name)
            print(f"{name}: Failed to download data")

    # Count how many indexes outperformed S&P 500
    outperformers = []
    if sp500_return is not None:
        for name, return_val in ytd_returns.items():
            if 'S&P 500' not in name and return_val > sp500_return:
                outperformers.append(name)

    print("\n" + "=" * 60)
    print("SUMMARY:")
    print(f"S&P 500 YTD Return: {sp500_return:.2f}%" if sp500_return else "S&P 500: Data not available")
    print(f"Indexes outperforming S&P 500: {len(outperformers)} out of {len(indexes) - 1}")
    
    if outperformers:
        print("\nIndexes with BETTER YTD returns than S&P 500:")
        positive_outperformers = []
        negative_but_better = []
        
        for name in outperformers:
            return_val = ytd_returns[name]
            if return_val > 0:
                positive_outperformers.append((name, return_val))
            else:
                negative_but_better.append((name, return_val))
        
        if positive_outperformers:
            print("  Positive returns:")
            for name, ret in positive_outperformers:
                print(f"    • {name}: +{ret:.2f}%")
        
        if negative_but_better:
            print("  Negative but better than S&P 500:")
            for name, ret in negative_but_better:
                print(f"    • {name}: {ret:.2f}% (lost less)")
                
        print(f"\nNote: 'Better performance' means higher return.")
        print(f"Even negative returns can outperform if they lose less money.")
    
    if failed:
        print(f"\nFailed downloads: {', '.join(failed)}")

    # Find best and worst performers
    if ytd_returns:
        best_performer = max(ytd_returns.items(), key=lambda x: x[1])
        worst_performer = min(ytd_returns.items(), key=lambda x: x[1])
        
        print(f"\nBest performer: {best_performer[0]} ({best_performer[1]:.2f}%)")
        print(f"Worst performer: {worst_performer[0]} ({worst_performer[1]:.2f}%)")

    return {
        'sp500_return': sp500_return,
        'ytd_returns': ytd_returns,
        'outperformers': outperformers,
        'outperformers_count': len(outperformers),
        'failed_downloads': failed,
        'total_indexes': len(indexes),
        'best_performer': best_performer if ytd_returns else None,
        'worst_performer': worst_performer if ytd_returns else None
    }


def compare_index_performance(reference_index: str = 'United States (S&P 500)') -> None:
    """
    Compare all indexes against a reference index.
    
    Args:
        reference_index: Name of the reference index for comparison
    """
    result = question_2_ytd_performance()
    
    if not result['ytd_returns']:
        print("No data available for comparison")
        return
    
    ref_return = result['ytd_returns'].get(reference_index)
    if ref_return is None:
        print(f"Reference index '{reference_index}' not found in results")
        return
    
    print(f"\nPerformance vs {reference_index} ({ref_return:.2f}%):")
    print("-" * 50)
    
    for name, return_val in sorted(result['ytd_returns'].items(), key=lambda x: x[1], reverse=True):
        if name != reference_index:
            diff = return_val - ref_return
            status = "↑" if diff > 0 else "↓"
            print(f"{name}: {return_val:.2f}% ({status}{abs(diff):.2f}pp)")


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO, 
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    try:
        result = question_2_ytd_performance()
        
        # Additional analysis
        print("\n" + "=" * 60)
        compare_index_performance()
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        print(f"Error: {e}")
