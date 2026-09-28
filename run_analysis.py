"""
Step 3: Deviation detection and sensitivity analysis (Module 2+3 bridge).

Chains the forecasting and optimisation pipelines, then:
  - builds a per-week deviation report (forecast error vs plan coverage gap)
  - runs a 3-axis sensitivity analysis (demand shocks, budget cuts, headcount loss)
  - saves charts and CSV exports to output/analysis/
"""

import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd

from src.forecasting.data_generator import generate_demand_dataset
from src.forecasting.pipeline import ForecastingPipeline
from src.optimization.constraints import WorkforceConstraints
from src.optimization.pipeline import OptimizationPipeline
from src.analysis.deviation import build_deviation_report
from src.analysis.sensitivity import SensitivityAnalysis
from src.analysis.visualizer import plot_deviation_report, plot_sensitivity

DATA_PATH = "data/demand_data.csv"
OUTPUT_DIR = "output/analysis"
SKU = "SKU-A"

CONSTRAINTS = WorkforceConstraints(
    capacity_per_worker=50.0,
    cost_per_worker=900.0,
    max_workers=20,
    budget_total=175_000.0,
    min_service_level=0.92,
    unmet_demand_penalty=4_000.0,
    max_ramp_up=3,
    max_ramp_down=3,
    currency="EUR",
)


def main():
    # ── 1. Forecast ────────────────────────────────────────────────────
    if Path(DATA_PATH).exists():
        df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    else:
        df = generate_demand_dataset(n_weeks=156, n_skus=3)
        Path(DATA_PATH).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(DATA_PATH, index=False)

    print("Running forecasting pipeline...")
    fc_pipeline = ForecastingPipeline(test_weeks=13, deviation_threshold_pct=20.0)
    fc_results = fc_pipeline.run(df)
    best = fc_pipeline.best_model(fc_results)
    fc_result = best[SKU]
    print(f"  Best model ({SKU}): {fc_result.model_name}  MAPE={fc_result.mape:.2f}%\n")

    # ── 2. Optimise ────────────────────────────────────────────────────
    print("Running optimisation pipeline...")
    opt_pipeline = OptimizationPipeline(constraints=CONSTRAINTS, integer=True)
    alloc_result = opt_pipeline.run(
        forecast_dates=fc_result.test_dates,
        forecast_demand=fc_result.forecast,
        sku=SKU,
    )
    print(f"  Solver status: {alloc_result.solver_status}  "
          f"Service level: {alloc_result.service_level:.1%}  "
          f"Cost: {alloc_result.total_cost:,.0f} EUR\n")

    # ── 3. Deviation report ────────────────────────────────────────────
    print("Building deviation report...")
    report = build_deviation_report(fc_result, alloc_result)

    print("\n" + "=" * 65)
    print("DEVIATION REPORT")
    print("=" * 65)
    print(report.summary())

    # ── 4. Sensitivity analysis ────────────────────────────────────────
    print("\n" + "=" * 65)
    print("SENSITIVITY ANALYSIS")
    print("=" * 65)
    sa = SensitivityAnalysis(
        base_demand=fc_result.forecast,
        base_constraints=CONSTRAINTS,
    )
    tables = sa.run_all()

    for axis, tdf in tables.items():
        print(f"\n--- {axis.capitalize()} sensitivity ---")
        print(
            tdf[["scenario", "total_cost", "service_level", "avg_workers", "status"]]
            .assign(
                total_cost=tdf["total_cost"].map("{:,.0f}".format),
                service_level=tdf["service_level"].map("{:.1%}".format),
                avg_workers=tdf["avg_workers"].map("{:.1f}".format),
            )
            .to_string(index=False)
        )

    # ── 5. Save outputs ────────────────────────────────────────────────
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    print("\nGenerating charts...")
    plot_deviation_report(report, output_path=f"{OUTPUT_DIR}/{SKU}_deviation.png", show=False)
    plot_sensitivity(tables, currency="EUR", output_path=f"{OUTPUT_DIR}/{SKU}_sensitivity.png", show=False)

    print("Saving CSV exports...")
    report.df.to_csv(f"{OUTPUT_DIR}/{SKU}_deviation.csv", index=False)
    for axis, tdf in tables.items():
        tdf.to_csv(f"{OUTPUT_DIR}/{SKU}_sensitivity_{axis}.csv", index=False)
        print(f"  Saved: {OUTPUT_DIR}/{SKU}_sensitivity_{axis}.csv")

    print(f"\nAll outputs saved to {OUTPUT_DIR}/")
    print("\nStep 3 complete.")


if __name__ == "__main__":
    main()
