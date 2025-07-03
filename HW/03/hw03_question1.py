import pandas as pd
import os
import gdown

def solve_question_1(file_path: str = 'data.parquet'):
    """
    Solves Question 1 of Homework 3: finds the highest absolute correlation
    between a month_wom dummy variable and the target.
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

    # --- Suggested Steps from Homework ---

    # 1. Compute week of the month
    # Ensure 'Date' is a datetime object first
    df['Date'] = pd.to_datetime(df['Date'])
    df['week_of_month'] = (df['Date'].dt.day - 1) // 7 + 1

    # 2. Create a new string variable for month and week-of-month
    df['month_wom'] = df['Date'].dt.strftime('%B') + '_w' + df['week_of_month'].astype(str)

    # 3. Define categorical features including the new one
    categorical_features = ['Month', 'Weekday', 'Ticker', 'ticker_type', 'month_wom']
    target = 'is_positive_growth_30d_future'

    # Clean data by dropping rows where the target is NaN, as we need it for correlation
    df_clean = df.dropna(subset=[target]).copy()
    df_clean[target] = df_clean[target].astype(int)

    # 4. Use pandas.get_dummies()
    df_dummies = pd.get_dummies(df_clean, columns=categorical_features, drop_first=False)

    # 5. Compute correlation with the target variable
    correlation_matrix = df_dummies.corr(numeric_only=True)
    correlations_with_target = correlation_matrix[target]

    # 6. Filter correlation results for 'month_wom' dummies
    month_wom_corr = correlations_with_target[correlations_with_target.index.str.startswith('month_wom_')]

    # 7. Create a new column for absolute correlation
    corr_df = pd.DataFrame(month_wom_corr).rename(columns={target: 'correlation'})
    corr_df['abs_corr'] = corr_df['correlation'].abs()

    # 8. Sort by 'abs_corr' in descending order
    sorted_corr = corr_df.sort_values(by='abs_corr', ascending=False)

    # 9. Identify and report the highest absolute correlation value
    most_correlated_variable = sorted_corr.index[0]
    highest_abs_corr_value = sorted_corr['abs_corr'].iloc[0]

    print("\n" + "="*50)
    print("                Question 1 Answer")
    print("="*50)
    print(f"The most correlated dummy variable is: '{most_correlated_variable}'")
    print(f"The absolute correlation value is: {highest_abs_corr_value:.3f}")

if __name__ == "__main__":
    solve_question_1()