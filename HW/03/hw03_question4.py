import pandas as pd
import os
import gdown
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_score
import numpy as np
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.tree import plot_tree

def load_and_prepare_data(file_path: str = 'data.parquet'):
    """
    Loads the dataset, downloads if necessary, and performs initial
    feature engineering.
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

def solve_question_4():
    """
    Solves Question 4: Hyperparameter tuning for a Decision Tree.
    """
    df = load_and_prepare_data()

    # --- Data Preparation (similar to Q3) ---

    # Feature Engineering from Q1
    df['week_of_month'] = (df['Date'].dt.day - 1) // 7 + 1
    df['month_wom'] = df['Date'].dt.strftime('%B') + '_w' + df['week_of_month'].astype(str)

    # Define features and target
    target = 'is_positive_growth_30d_future'
    categorical_features = ['Month', 'Weekday', 'Ticker', 'ticker_type', 'month_wom']
    
    cols_to_exclude = [
        'Date', 'Ticker', 'Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 
        'Stock Splits', 'growth_future_1d', 'growth_future_5d', 'growth_future_30d', 
        'is_positive_growth_30d_future', 'Name', 'GICS Sector', 'GICS Sub-Industry'
    ]
    numerical_features = [col for col in df.columns if df[col].dtype in ['int64', 'float64'] and col not in cols_to_exclude and not col.startswith('pred')]

    # Prepare data for modeling
    features_to_use = numerical_features + categorical_features
    df_model = df.dropna(subset=[target] + features_to_use).copy()

    # Clean infinite values
    df_model.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_model.dropna(subset=[target] + features_to_use, inplace=True)
    df_model[target] = df_model[target].astype(int)

    # Create dummies and define X and y
    X = pd.get_dummies(df_model[features_to_use], columns=categorical_features, drop_first=True)
    y = df_model[target]

    # Define data splits
    validation_end_date = '2021-08-19'
    test_start_date = '2021-08-20'

    # Create train/validation and test sets
    X_train_val = X[df_model['Date'] <= validation_end_date]
    y_train_val = y[df_model['Date'] <= validation_end_date]
    X_test = X[df_model['Date'] >= test_start_date]
    y_test = y[df_model['Date'] >= test_start_date]

    # --- Report Generation Setup ---
    now = datetime.now()
    reports_dir = "reports"
    os.makedirs(reports_dir, exist_ok=True)
    report_filename = f"tuning_report_{now.strftime('%Y%m%d_%H%M%S')}.md"
    report_path = os.path.join(reports_dir, report_filename)

    with open(report_path, 'w') as report_file:
        report_file.write(f"# Decision Tree Tuning Report\n\n")
        report_file.write(f"**Run Date:** {now.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        # --- Recursive Feature Elimination with Cross-Validation (RFECV) ---
        report_file.write("## 1. Feature Selection (RFECV)\n\n")
        print("\nPerforming Recursive Feature Elimination with Cross-Validation (RFECV)...")
        
        estimator_for_rfecv = DecisionTreeClassifier(random_state=42)
        cv_strategy = StratifiedKFold(n_splits=5)
        
        rfecv = RFECV(
            estimator=estimator_for_rfecv,
            step=1,
            cv=cv_strategy,
            scoring="precision",
            min_features_to_select=20,
            n_jobs=-1,
            verbose=1
        )
        
        print("Fitting RFECV... This may take a while depending on your system.")
        rfecv.fit(X_train_val, y_train_val)
        print("RFECV fitting complete.")
        
        print(f"RFECV selected {rfecv.n_features_} features out of {X_train_val.shape[1]}.")
        report_file.write(f"RFECV selected **{rfecv.n_features_}** features out of {X_train_val.shape[1]}.\n\n")
        
        X_train_val_selected = X_train_val.loc[:, rfecv.support_]
        X_test_selected = X_test.loc[:, rfecv.support_]

        selected_features_list = X_train_val_selected.columns.tolist()
        report_file.write("### Selected Features for Tuning:\n\n")
        report_file.write("```\n")
        for feature in selected_features_list:
            report_file.write(f"- {feature}\n")
        report_file.write("```\n\n")

        # --- Hyperparameter Tuning Loop ---
        plots_dir = "tree_visualizations"
        os.makedirs(plots_dir, exist_ok=True)

        print("\nStarting hyperparameter tuning for max_depth on selected features...")
        print(f"Tree visualizations will be saved in the '{plots_dir}' directory.")
        print("-" * 50)

        report_file.write("## 2. Hyperparameter Tuning Results\n\n")
        report_file.write("All trees were trained using the feature set selected by RFECV.\n\n")
        report_file.write("| Depth | Test Precision |\n")
        report_file.write("|:-----:|:--------------:|\n")

        depths_to_check = range(1, 21)
        precision_scores = []

        for depth in depths_to_check:
            clf = DecisionTreeClassifier(max_depth=depth, random_state=42)
            clf.fit(X_train_val_selected, y_train_val)

            y_pred_test = clf.predict(X_test_selected)

            precision = precision_score(y_test, y_pred_test, zero_division=0)
            precision_scores.append(precision)
            
            print(f"  max_depth = {depth:2d}  |  Test Precision = {precision:.4f}")
            report_file.write(f"| {depth:^5} | {precision:^14.4f} |\n")

            plt.figure(figsize=(40, 20))
            plot_tree(clf,
                      filled=True,
                      feature_names=selected_features_list,
                      class_names=['Negative', 'Positive'],
                      max_depth=3,
                      fontsize=10)
            plt.title(f"Decision Tree (Trained with max_depth={depth}, Visualized to depth=3)", fontsize=20)
            
            plot_filename = os.path.join(plots_dir, f'tree_trained_depth_{depth:02d}.png')
            plt.savefig(plot_filename)
            plt.close()

        # --- Identify Optimal Depth & Final Reporting ---
        best_precision = max(precision_scores)
        best_max_depth = depths_to_check[precision_scores.index(best_precision)]

        report_file.write("\n## 3. Final Result\n\n")
        report_file.write(f"- **Optimal `max_depth`**: {best_max_depth}\n")
        report_file.write(f"- **Best Test Precision**: {best_precision:.4f}\n")

        print("-" * 50)
        print("\n" + "="*50)
        print("                Question 4 Answer")
        print("="*50)
        print(f"The optimal tree depth (max_depth) is: {best_max_depth}")
        print(f"This depth achieved the highest precision on the test set: {best_precision:.4f}")
        print("="*50)

    print(f"\nDetailed report saved to: {report_path}")

if __name__ == "__main__":
    solve_question_4()