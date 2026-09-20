"""
Entry point for Step 1: Forecasting Engine on a fixed synthetic dataset.

Generates (or loads) a weekly demand dataset, runs three forecasting models
(WMA, ETS, SARIMA), prints accuracy metrics, flags significant deviations,
and saves charts to output/.
"""

import sys
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd

from src.forecasting.data_generator import generate_demand_dataset
from src.forecasting.pipeline import ForecastingPipeline
from src.forecasting.visualizer import plot_forecast, plot_model_comparison

DATA_PATH = "data/demand_data.csv"
OUTPUT_DIR = "output/forecasts"
TEST_WEEKS = 13            # ~1 quarter held out for evaluation
DEVIATION_THRESHOLD = 20.0 # flag weeks where |error| > 20%


def main():
    # --- 1. Data preparation ---
    if Path(DATA_PATH).exists():
        print(f"Loading existing dataset from {DATA_PATH}")
        df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    else:
        print("Generating synthetic demand dataset...")
        df = generate_demand_dataset(n_weeks=104, n_skus=3)
        Path(DATA_PATH).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(DATA_PATH, index=False)
        print(f"Saved {len(df)} rows to {DATA_PATH}")

    print(f"\nDataset shape : {df.shape}")
    print(df.head(6).to_string(index=False))
    print()

    # --- 2. Run forecasting pipeline ---
    pipeline = ForecastingPipeline(
        test_weeks=TEST_WEEKS,
        deviation_threshold_pct=DEVIATION_THRESHOLD,
    )
    print("Running forecasting pipeline...\n")
    results = pipeline.run(df)

    # --- 3. Print results summary ---
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    best = pipeline.best_model(results)

    for sku, res_list in results.items():
        print(f"\n--- SKU: {sku} ---")
        for r in res_list:
            print(r.summary())
            if not r.deviations.empty:
                print("  Flagged deviations:")
                print(r.deviations[["date", "actual", "forecast", "pct_error"]].to_string(index=False))
            print()

        winner = best.get(sku)
        if winner:
            print(f"  >> Best model for {sku}: {winner.model_name}  (MAPE={winner.mape:.2f}%)")

    # --- 4. Save charts ---
    print("\nGenerating charts...")
    df_indexed = df.set_index("date")

    for sku, res_list in results.items():
        sku_train = df_indexed[df_indexed["sku"] == sku]["demand"].sort_index()

        for r in res_list:
            model_tag = r.model_name.split("(")[0].replace(" ", "_")
            plot_forecast(
                r,
                train_series=sku_train.iloc[: -TEST_WEEKS],
                output_path=f"{OUTPUT_DIR}/{sku}_{model_tag}.png",
                show=False,
            )

        if len(res_list) > 1:
            plot_model_comparison(
                res_list,
                output_path=f"{OUTPUT_DIR}/{sku}_model_comparison.png",
                show=False,
            )

    print(f"\nAll charts saved to {OUTPUT_DIR}/")
    print("\nStep 1 complete.")


if __name__ == "__main__":
    main()
