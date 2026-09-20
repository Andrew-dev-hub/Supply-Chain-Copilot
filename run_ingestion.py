"""
Step 7: Module 1 — Generic ingestion pipeline (Haiku schema detection).

Usage:
    py run_ingestion.py data/demand_data.csv
    py run_ingestion.py path/to/your_file.xlsx
    py run_ingestion.py path/to/your_file.csv --no-interactive

The normalised output is saved to data/ingested_<filename>.csv
and can be passed directly to run_forecast.py.
"""

import sys
import argparse
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, str(Path(__file__).parent))

from src.ingestion.pipeline import IngestionPipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("file", help="Path to CSV or Excel file")
    parser.add_argument("--no-interactive", action="store_true",
                        help="Accept detected mapping without prompting")
    parser.add_argument("--no-llm", action="store_true",
                        help="Force rule-based detection (skip Haiku)")
    args = parser.parse_args()

    pipeline = IngestionPipeline(use_llm=not args.no_llm)
    df, mapping = pipeline.run(args.file, interactive=not args.no_interactive)

    # Save normalised output
    stem = Path(args.file).stem
    out_path = f"data/ingested_{stem}.csv"
    Path("data").mkdir(exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"\nNormalised dataset saved: {out_path}")
    print(df.head(5).to_string(index=False))
    print(f"\nReady to use in the pipeline:")
    print(f"  py run_forecast.py  (edit DATA_PATH = '{out_path}')")


if __name__ == "__main__":
    main()
