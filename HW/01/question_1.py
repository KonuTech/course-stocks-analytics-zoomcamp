# question_1.py
"""
Question 1: Analyze S&P 500 additions by year.
"""
import pandas as pd
import logging
from typing import Dict
from datetime import datetime

logger = logging.getLogger(__name__)

def question_1_sp500_additions() -> Dict:
    """
    Identify the year with the highest number of S&P 500 additions.

    Returns:
        A dictionary with year of max additions, count, and summary stats.
    """
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    logger.info(f"Fetching S&P 500 company data from {url}")
    
    try:
        # Use pandas.read_html to scrape the data
        tables = pd.read_html(url)
        logger.info(f"Found {len(tables)} tables on the page")
    except Exception as e:
        logger.error(f"Failed to fetch data from Wikipedia: {e}")
        raise ValueError(f"Could not retrieve data from {url}: {e}")

    if not tables:
        raise ValueError("No tables found at Wikipedia URL")

    # The first table contains the S&P 500 companies
    df = tables[0]
    logger.info(f"DataFrame shape: {df.shape}")
    logger.info(f"Columns: {list(df.columns)}")
    
    # Find the date added column - it might have different names
    date_col = None
    for col in df.columns:
        col_lower = str(col).lower()
        if 'date' in col_lower and 'added' in col_lower:
            date_col = col
            break
    
    if date_col is None:
        # Try alternative column names
        for col in df.columns:
            col_lower = str(col).lower()
            if 'added' in col_lower or 'date' in col_lower:
                date_col = col
                break
    
    if date_col is None:
        logger.error(f"Available columns: {list(df.columns)}")
        raise ValueError("Could not find date added column")

    logger.info(f"Using date column: '{date_col}'")
    
    # Clean and convert the date column
    df_clean = df.copy()
    df_clean['Date_Added_Clean'] = pd.to_datetime(df_clean[date_col], errors='coerce')
    df_clean['Year_Added'] = df_clean['Date_Added_Clean'].dt.year
    
    # Filter out invalid years and exclude 1957 (founding year)
    current_year = datetime.now().year
    valid_mask = (
        df_clean['Year_Added'].notna() & 
        (df_clean['Year_Added'] != 1957) &
        (df_clean['Year_Added'] >= 1958) &  # Start from 1958
        (df_clean['Year_Added'] <= current_year)
    )
    
    valid_df = df_clean[valid_mask].copy()
    logger.info(f"Valid entries after filtering: {len(valid_df)}")
    
    if len(valid_df) == 0:
        raise ValueError("No valid addition dates found after filtering")

    # Count additions by year
    yearly_counts = valid_df['Year_Added'].value_counts().sort_index()
    logger.info(f"Years with additions: {len(yearly_counts)}")
    
    # Find the year with maximum additions
    max_year = int(yearly_counts.idxmax())
    max_count = int(yearly_counts.max())
    
    # Count companies with 20+ years in the index
    cutoff_year = current_year - 20
    companies_20_plus = len(valid_df[valid_df['Year_Added'] <= cutoff_year])
    
    # Additional statistics
    avg_additions_per_year = yearly_counts.mean()
    median_additions_per_year = yearly_counts.median()
    
    result = {
        'max_additions_year': max_year,
        'max_additions_count': max_count,
        'companies_20_plus_years': companies_20_plus,
        'total_valid_additions': len(valid_df),
        'years_with_additions': len(yearly_counts),
        'avg_additions_per_year': round(avg_additions_per_year, 2),
        'median_additions_per_year': float(median_additions_per_year),
        'yearly_additions': yearly_counts.to_dict(),
        'cutoff_year_for_20_plus': cutoff_year,
        'data_last_updated': datetime.now().isoformat()
    }

    # Print results
    print(f"\n=== S&P 500 Additions Analysis ===")
    print(f"Year with most additions: {max_year} ({max_count} companies)")
    print(f"Companies with 20+ years in index: {companies_20_plus}")
    print(f"Total companies analyzed: {len(valid_df)}")
    print(f"Years with additions: {len(yearly_counts)}")
    print(f"Average additions per year: {avg_additions_per_year:.2f}")
    print(f"Median additions per year: {median_additions_per_year}")
    
    # Show top 5 years
    top_5_years = yearly_counts.nlargest(5)
    print(f"\nTop 5 years with most additions:")
    for year, count in top_5_years.items():
        print(f"  {int(year)}: {count} companies")

    return result


def debug_wikipedia_table():
    """
    Debug function to inspect the Wikipedia table structure.
    """
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    try:
        tables = pd.read_html(url)
        print(f"Number of tables found: {len(tables)}")
        
        for i, table in enumerate(tables[:3]):  # Check first 3 tables
            print(f"\n--- Table {i} ---")
            print(f"Shape: {table.shape}")
            print(f"Columns: {list(table.columns)}")
            if len(table) > 0:
                print("First few rows:")
                print(table.head(3))
                
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO, 
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    try:
        # Run the main analysis
        result = question_1_sp500_additions()
        
        print(f"\n=== ANSWER ===")
        print(f"Year with highest number of additions: {result['max_additions_year']}")
        print(f"Number of additions in that year: {result['max_additions_count']}")
        print(f"Companies with 20+ years in index: {result['companies_20_plus_years']}")
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        print(f"\nRunning debug to inspect table structure...")
        debug_wikipedia_table()