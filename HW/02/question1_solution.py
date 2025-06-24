import pandas as pd
import re
import requests
import logging
import json
from io import StringIO

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
        handler = logging.FileHandler('debug.log.json', mode='w')
        formatter = JsonFormatter()
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger

def solve_question_1():
    logger = setup_logging()
    logger.info("Starting solution for Question 1: Withdrawn IPOs by Company Type.")

    url = "https://stockanalysis.com/ipos/withdrawn/"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)
        logger.info(f"Successfully fetched data from {url}")
        # Use StringIO to wrap the HTML text, addressing a FutureWarning in pandas.
        dfs = pd.read_html(StringIO(response.text))
        df = dfs[0] # Assuming the first table is the correct one
        logger.info(f"Successfully parsed HTML table into DataFrame. Shape: {df.shape}")
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to load data from URL: {url}", exc_info=True)
        print(f"Error loading data from URL: {e}")
        return
    except (KeyError, IndexError) as e:
        logger.error("Failed to parse the table from the HTML.", exc_info=True)
        print(f"Error: Could not find or parse the expected table from the URL. The website structure may have changed. Details: {e}")
        return
    
    # The column name on the website can change. Rename 'Company Name' to 'Company' for consistency.
    if 'Company Name' in df.columns:
        df = df.rename(columns={'Company Name': 'Company'})
        logger.info("Renamed column 'Company Name' to 'Company'.")


    # 2. Create a new column called `Company Class`
    def get_company_class(company_name):
        if pd.isna(company_name):
            return 'Other'
        name = str(company_name).lower()
        # Order of matching is crucial as per instructions
        if re.search(r'\bacquisition corp\b|\bacquisition corporation\b', name):
            return "Acq.Corp"
        elif re.search(r'\binc\b|\bincorporated\b', name):
            return "Inc"
        elif re.search(r'\bgroup\b', name):
            return "Group"
        elif re.search(r'\bltd\b|\blimited\b', name):
            return "Limited"
        elif re.search(r'\bholdings\b', name):
            return "Holdings"
        else:
            return "Other"

    try:
        df['Company Class'] = df['Company'].apply(get_company_class)
    except KeyError:
        err_msg = "The required 'Company' or 'Company Name' column was not found in the DataFrame."
        logger.error(f"{err_msg} Available columns are: {list(df.columns)}")
        print(f"ERROR: {err_msg} Available columns are: {list(df.columns)}")
        return
    logger.info("Created 'Company Class' column.")

    # 3. Define a new field `Avg. price` by parsing the `Price Range` field
    def parse_price_range(price_range_str):
        if pd.isna(price_range_str) or str(price_range_str).strip() == '-':
            return None
        
        # Remove '$' sign and any leading/trailing whitespace
        price_range_str = str(price_range_str).replace('$', '').strip()
        
        if '-' in price_range_str:
            try:
                low, high = map(float, price_range_str.split('-'))
                return (low + high) / 2
            except ValueError:
                return None # Handle cases where conversion to float fails
        else:
            try:
                return float(price_range_str)
            except ValueError:
                return None # Handle cases where conversion to float fails

    df['Avg. price'] = df['Price Range'].apply(parse_price_range)
    logger.info("Created 'Avg. price' column by parsing 'Price Range'.")

    # 4. Convert `Shares Offered` to numeric, clean missing or invalid values.
    # Remove commas and convert to numeric, coercing errors to NaN
    df['Shares Offered'] = df['Shares Offered'].astype(str).str.replace(',', '', regex=False)
    df['Shares Offered'] = pd.to_numeric(df['Shares Offered'], errors='coerce')

    # 5. Create a new column: `Withdrawn Value = Shares Offered * Avg Price`
    logger.info("Cleaned and converted 'Shares Offered' column to numeric.")
    df['Withdrawn Value'] = df['Shares Offered'] * df['Avg. price']
    logger.info("Calculated 'Withdrawn Value' column.")

    # Log the number of non-null values found. This is more robust than a hardcoded check.
    withdrawn_value_count = df['Withdrawn Value'].count()
    logger.info(f"Found {withdrawn_value_count} non-null 'Withdrawn Value' entries.")

    # 6. Group by `Company Class` and calculate total withdrawn value.
    total_withdrawn_by_class = df.groupby('Company Class')['Withdrawn Value'].sum()
    logger.info("Grouped by 'Company Class' and summed 'Withdrawn Value'.")
    
    # 7. Answer: Which class had the highest total value of withdrawals?
    if not total_withdrawn_by_class.empty:
        highest_withdrawal_class = total_withdrawn_by_class.idxmax()
        highest_withdrawal_value = total_withdrawn_by_class.max()

        # Convert to millions for the final answer
        highest_withdrawal_value_millions = highest_withdrawal_value / 1_000_000

        result = {
            "highest_class": highest_withdrawal_class,
            "total_value_millions": highest_withdrawal_value_millions
        }
        log_message = f"Determined the highest withdrawal class. Result: {result}"
        logger.info(log_message)

        print(f"The company class with the highest total withdrawn IPO value is: {highest_withdrawal_class}")
        print(f"Total withdrawn IPO value for this class: ${highest_withdrawal_value_millions:,.2f} million")
    else:
        logger.error("No withdrawn value data available to determine the highest class.")
        print("No withdrawn value data available to determine the highest class.")

if __name__ == "__main__":
    solve_question_1()