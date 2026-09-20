"""
Step 2: Resource Optimisation (Module 3).

Loads the forecast produced in Step 1, then:
  A) Runs the LP with hardcoded constraints (no API key needed)
  B) Optionally parses constraints from natural language via Claude API

Usage:
    py run_optimization.py              # hardcoded constraints
    py run_optimization.py --llm        # LLM-parsed constraints (needs ANTHROPIC_API_KEY)
"""

import sys
import os
import argparse
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd

from src.forecasting.data_generator import generate_demand_dataset
from src.forecasting.pipeline import ForecastingPipeline
from src.optimization.constraints import WorkforceConstraints
from src.optimization.pipeline import OptimizationPipeline
from src.optimization.visualizer import plot_allocation, plot_cost_breakdown

DATA_PATH = "data/demand_data.csv"
OUTPUT_DIR = "output/optimization"

# --- Hardcoded constraints (used when --llm is not passed) ---
# Scenario: a logistics warehouse processing ~500 units/week on the main SKU
# Each worker handles 50 units/week; 20 workers max; 900 EUR/worker/week
HARDCODED_CONSTRAINTS = WorkforceConstraints(
    capacity_per_worker=50.0,
    cost_per_worker=900.0,
    max_workers=20,
    budget_total=175_000.0,   # 13 weeks * ~13 workers avg * 900 EUR + some margin
    min_service_level=0.92,
    unmet_demand_penalty=4_000.0,
    max_ramp_up=3,
    max_ramp_down=3,
    currency="EUR",
)

# --- Natural-language constraint description (used with --llm) ---
NL_CONSTRAINTS = """
We run a logistics warehouse with at most 20 workers available each week.
Each worker can process 50 units per week.
Labour costs 900 EUR per worker per week.
Our total staffing budget for the 13-week planning horizon is 175,000 EUR.
We must cover at least 92% of forecasted demand.
We cannot hire or release more than 3 workers between consecutive weeks.
"""


def get_forecast(sku: str = "SKU-A") -> tuple:
    """Re-run the forecasting pipeline and return the best forecast for *sku*."""
    if Path(DATA_PATH).exists():
        df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    else:
        df = generate_demand_dataset(n_weeks=104, n_skus=3)
        Path(DATA_PATH).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(DATA_PATH, index=False)

    pipeline = ForecastingPipeline(test_weeks=13, deviation_threshold_pct=20.0)
    print("Running forecasting pipeline...")
    results = pipeline.run(df)
    best = pipeline.best_model(results)

    if sku not in best:
        sku = next(iter(best))
        print(f"[INFO] Using SKU {sku}")

    result = best[sku]
    print(f"  Best model for {sku}: {result.model_name}  MAPE={result.mape:.2f}%\n")
    return result.test_dates, result.forecast, sku


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm", action="store_true", help="Use Claude API to parse constraints")
    parser.add_argument("--sku", default="SKU-A", help="SKU to optimise (default: SKU-A)")
    args = parser.parse_args()

    # 1. Get forecast
    dates, demand, sku = get_forecast(sku=args.sku)

    # 2. Get constraints
    if args.llm:
        print("Parsing constraints from natural language via Claude API...")
        from src.optimization.llm_parser import parse_constraints_from_text
        constraints = parse_constraints_from_text(NL_CONSTRAINTS, horizon_weeks=len(dates))
        print("  Constraints extracted by LLM:")
    else:
        print("Using hardcoded constraints.")
        constraints = HARDCODED_CONSTRAINTS

    print(constraints.summary())
    print()

    # 3. Run optimisation
    opt_pipeline = OptimizationPipeline(constraints=constraints, integer=True)
    print(f"Solving workforce scheduling LP for {sku}...")
    result = opt_pipeline.run(forecast_dates=dates, forecast_demand=demand, sku=sku)

    # 4. Print results
    print("\n" + "=" * 65)
    print("OPTIMISATION RESULTS")
    print("=" * 65)
    print(result.summary())

    print("\nWeekly allocation table:")
    df_out = result.to_dataframe()
    print(df_out.to_string(index=False, float_format="%.1f"))

    # 5. Save charts
    print("\nGenerating charts...")
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    plot_allocation(result, output_path=f"{OUTPUT_DIR}/{sku}_allocation.png", show=False)
    plot_cost_breakdown(result, output_path=f"{OUTPUT_DIR}/{sku}_cost_breakdown.png", show=False)

    # 6. Save allocation table to CSV
    csv_path = f"{OUTPUT_DIR}/{sku}_allocation.csv"
    df_out.to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path}")

    print(f"\nAll outputs saved to {OUTPUT_DIR}/")
    print("\nStep 2 complete.")


if __name__ == "__main__":
    main()
