import pandas as pd
import os
import gdown
from sklearn.metrics import precision_score

def load_and_prepare_data(file_path: str = 'data.parquet'):
    """
    Loads the dataset, downloads if necessary, and performs initial
    feature engineering.

    Args:
        file_path (str): The path to the .parquet file.

    Returns:
        pd.DataFrame: The loaded and prepared DataFrame.
    """
    # Ensure the data file exists, download if not (using ID from HW2)
    if not os.path.exists(file_path):
        print(f"File '{file_path}' not found. Downloading...")
        file_id = "1grCTCzMZKY5sJRtdbLVCXg8JXA8VPyg-"
        gdown.download(f"https://drive.google.com/uc?id={file_id}", file_path, quiet=False)
        print("Download complete.")

    # Load the dataset
    print(f"Reading data from '{file_path}'...")
    df = pd.read_parquet(file_path, engine='pyarrow')

    # Ensure 'Date' is a datetime object
    df['Date'] = pd.to_datetime(df['Date'])
    
    return df

def solve_question_2():
    """
    Solves Question 2 of Homework 3: calculates precision for new hand rules.
    """
    df = load_and_prepare_data()

    # --- Suggested Steps from Homework ---

    # Define target and features needed for the rules
    target = 'is_positive_growth_30d_future'
    rule_features = ['DGS10', 'DGS5', 'FEDFUNDS']

    # Clean data by dropping rows where target or rule features are NaN
    df_clean = df.dropna(subset=[target] + rule_features).copy()
    df_clean[target] = df_clean[target].astype(int)

    # 1. Define and apply the two new 'hand' rules
    # pred3_manual_dgs10_5
    df_clean['pred3'] = ((df_clean['DGS10'] <= 4) & (df_clean['DGS5'] <= 1)).astype(int)
    
    # pred4_manual_dgs10_fedfunds
    df_clean['pred4'] = ((df_clean['DGS10'] > 4) & (df_clean['FEDFUNDS'] <= 4.795)).astype(int)

    # 2. Isolate the TEST set using the specified split date
    test_start_date = '2021-08-20'
    test_df = df_clean[df_clean['Date'] >= test_start_date].copy()
    
    if test_df.empty:
        print(f"Error: No data available for the test period ({test_start_date} onwards).")
        return

    # 3. Compute precision for each new rule on the TEST set
    y_true_test = test_df[target]
    
    # Calculate precision for pred3 and pred4, handling cases with no positive predictions
    precision3 = precision_score(y_true_test, test_df['pred3'], zero_division=0)
    precision4 = precision_score(y_true_test, test_df['pred4'], zero_division=0)

    # Identify the best precision score among the new rules
    best_new_precision = max(precision3, precision4)

    print("\n" + "="*50)
    print("                Question 2 Answer")
    print("="*50)
    print(f"The precision score for the best of the new predictions (pred3 or pred4) is: {best_new_precision:.3f}")

if __name__ == "__main__":
    solve_question_2()