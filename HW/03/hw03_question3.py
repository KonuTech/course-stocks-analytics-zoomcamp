import pandas as pd
import os
import gdown
from sklearn.tree import DecisionTreeClassifier
import numpy as np

def load_and_prepare_data(file_path: str = 'data.parquet'):
    """
    Loads the dataset, downloads if necessary, and performs initial
    feature engineering.

    Args:
        file_path (str): The path to the .parquet file.

    Returns:
        pd.DataFrame: The loaded and prepared DataFrame.
    """
    if not os.path.exists(file_path):
        print(f"File '{file_path}' not found. Downloading...")
        file_id = "1grCTCzMZKY5sJRtdbLVCXg8JXA8VPyg-"
        gdown.download(f"https://drive.google.com/uc?id={file_id}", file_path, quiet=False)
        print("Download complete.")

    print(f"Reading data from '{file_path}'...")
    df = pd.read_parquet(file_path, engine='pyarrow')
    df['Date'] = pd.to_datetime(df['Date'])
    return df

def add_all_predictions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds all 'hand' rule predictions (pred0-pred4) to the DataFrame.
    """
    # Hand rules from the course notebook (pred0-pred2)
    df['pred0'] = (df['cci'] > 200).astype(int)
    df['pred1'] = (df['growth_30d'] > 1).astype(int)
    df['pred2'] = ((df['growth_30d'] > 1) & (df['growth_snp500_30d'] > 1)).astype(int)
    
    # New hand rules from Question 2 (pred3-pred4)
    df['pred3'] = ((df['DGS10'] <= 4) & (df['DGS5'] <= 1)).astype(int)
    df['pred4'] = ((df['DGS10'] > 4) & (df['FEDFUNDS'] <= 4.795)).astype(int)
    
    return df

def solve_question_3():
    """
    Solves Question 3: Finds unique correct predictions from a Decision Tree.
    """
    df = load_and_prepare_data()
    df = add_all_predictions(df)

    # --- Step 1: Train the Decision Tree and Generate Predictions ---

    # Feature Engineering from Q1
    df['week_of_month'] = (df['Date'].dt.day - 1) // 7 + 1
    df['month_wom'] = df['Date'].dt.strftime('%B') + '_w' + df['week_of_month'].astype(str)

    # Define features and target
    target = 'is_positive_growth_30d_future'
    categorical_features = ['Month', 'Weekday', 'Ticker', 'ticker_type', 'month_wom']
    
    # Identify numerical features by excluding non-feature columns
    cols_to_exclude = [
        'Date', 'Ticker', 'Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 
        'Stock Splits', 'growth_future_1d', 'growth_future_5d', 'growth_future_30d', 
        'is_positive_growth_30d_future', 'Name', 'GICS Sector', 'GICS Sub-Industry'
    ]
    numerical_features = [col for col in df.columns if df[col].dtype in ['int64', 'float64'] and col not in cols_to_exclude and not col.startswith('pred')]

    # Prepare data for modeling
    features_to_use = numerical_features + categorical_features
    df_model = df.dropna(subset=[target] + features_to_use).copy() 

    # Replace infinite values with NaN, as they can cause errors in modeling.
    # This happens sometimes with technical indicators (e.g., division by zero).
    df_model.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Final clean to remove any rows that had infinite values
    df_model.dropna(subset=[target] + features_to_use, inplace=True)
    df_model[target] = df_model[target].astype(int)

    # Create dummies and define X and y
    X = pd.get_dummies(df_model[features_to_use], columns=categorical_features, drop_first=True)
    y = df_model[target]

    # Define data splits
    train_end_date = '2017-10-25'
    validation_end_date = '2021-08-19'
    test_start_date = '2021-08-20'

    # Combine TRAIN and VALIDATION sets for fitting
    X_train_val = X[df_model['Date'] <= validation_end_date]
    y_train_val = y[df_model['Date'] <= validation_end_date]

    # Initialize and fit the classifier
    clf10 = DecisionTreeClassifier(max_depth=10, random_state=42)
    print("Training Decision Tree with max_depth=10...")
    clf10.fit(X_train_val, y_train_val)

    # Predict on the entire dataset
    df_model['pred5_clf_10'] = clf10.predict(X)

    # --- Step 2: Identify Unique Correct Predictions ---

    # Check if pred5 is correct
    is_correct_pred5 = (df_model['pred5_clf_10'] == df_model[target])

    # Check if all hand rules are incorrect
    hand_preds = ['pred0', 'pred1', 'pred2', 'pred3', 'pred4']
    all_hand_rules_incorrect = True
    for pred_col in hand_preds:
        all_hand_rules_incorrect &= (df_model[pred_col] != df_model[target])

    # Combine conditions
    df_model['only_pred5_is_correct'] = (is_correct_pred5 & all_hand_rules_incorrect)

    # --- Step 3: Count Unique Correct Predictions on the TEST Set ---

    # Filter for the TEST dataset
    test_df = df_model[df_model['Date'] >= test_start_date].copy()

    # Convert boolean to integer and count
    unique_correct_count = test_df['only_pred5_is_correct'].astype(int).sum()

    print("\n" + "="*60)
    print("                       Question 3 Answer")
    print("="*60)
    print(f"Total records in the TEST set where only pred5_clf_10 is correct: {unique_correct_count}")
    print("="*60)

if __name__ == "__main__":
    solve_question_3()