# master.py
"""
Master script to run all finance homework question modules.
"""
from question_1 import question_1_sp500_additions
from question_2 import question_2_ytd_performance
from question_3 import question_3_market_corrections
from question_4 import question_4_amazon_earnings_analysis

import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('finance_homework.log')
    ]
)
logger = logging.getLogger(__name__)

def main():
    print("FINANCE HOMEWORK - 2025 COHORT")
    print("=" * 80)
    print("Starting comprehensive analysis with logging and error handling...")
    print("=" * 80)

    all_results = {}
    questions = [
        ("Question 1", question_1_sp500_additions),
        ("Question 2", question_2_ytd_performance),
        ("Question 3", question_3_market_corrections),
        ("Question 4", question_4_amazon_earnings_analysis)
    ]

    for name, func in questions:
        try:
            print(f"\n{name}")
            print("-" * 60)
            result = func()
            all_results[name] = result
            print(f"{name} COMPLETED SUCCESSFULLY")
        except Exception as e:
            logger.error(f"Error in {name}: {str(e)}")
            all_results[name] = {"error": str(e)}
            print(f"{name} ERROR - {e}")

    print("\n" + "=" * 80)
    print("HOMEWORK SUMMARY")
    print("=" * 80)
    for name, result in all_results.items():
        if 'error' in result:
            print(f"{name.upper()}: ERROR - {result['error']}")
        else:
            print(f"{name.upper()}: COMPLETED SUCCESSFULLY")

if __name__ == "__main__":
    main()

