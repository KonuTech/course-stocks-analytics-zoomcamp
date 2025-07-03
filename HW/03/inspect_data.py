import pandas as pd
import os

def inspect_parquet_file(file_path: str):
    """
    Reads a Parquet file and prints the first and last 5 rows.

    This function assumes the Parquet file can be read into a pandas DataFrame.
    It requires 'pandas' and 'pyarrow' or 'fastparquet' to be installed.

    Args:
        file_path (str): The path to the .parquet file.
    """
    # Check if the file exists at the given path
    if not os.path.exists(file_path):
        print(f"Error: The file was not found at '{file_path}'")
        print("Please make sure the 'data.parquet' file is in the same directory as this script.")
        return

    try:
        # Read the Parquet file. The engine 'pyarrow' is often used for performance.
        print(f"Reading data from '{file_path}'...")
        df = pd.read_parquet(file_path, engine='pyarrow')

        print("\n" + "="*40)
        print("          Top 5 Rows")
        print("="*40)
        print(df.head())

        print("\n" + "="*40)
        print("         Bottom 5 Rows")
        print("="*40)
        print(df.tail())
        print("\n" + "="*40)

    except Exception as e:
        print(f"An error occurred while processing the file: {e}")

if __name__ == "__main__":
    # The homework instructions suggest the file is named 'data.parquet'
    # We assume it is located in the same directory as this script.
    parquet_file_name = 'data.parquet'
    inspect_parquet_file(parquet_file_name)