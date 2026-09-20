"""
Step 4: LLM interpretation layer and scenario mode (Module 4).

Chains forecast → optimisation → deviation/sensitivity → LLM interpretation.
Produces:
  - Executive summary (LLM-generated business narrative)
  - Robustness report (auto scenarios tested pass/fail)
  - Interactive Q&A loop (Ctrl+C or empty input to exit)

Requires ANTHROPIC_API_KEY (in .env or environment).
"""

import sys
import os
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
from src.analysis.deviation import build_deviation_report
from src.analysis.sensitivity import SensitivityAnalysis
from src.interpretation.context_builder import build_context
from src.interpretation.interpreter import SupplyChainInterpreter
from src.interpretation.scenarios import run_scenarios, robustness_report, to_dataframe

DATA_PATH = "data/demand_data.csv"
OUTPUT_DIR = "output/interpretation"
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


# ── Pipeline runner (used by Q&A for what-if re-runs) ────────────────
def make_rerun_fn(base_demand: np.ndarray, base_constraints: WorkforceConstraints):
    def rerun(axis: str, delta: float) -> dict:
        demand = base_demand.copy()
        c = base_constraints

        if axis == "demand":
            demand = demand * (1 + delta)
        elif axis == "budget":
            c = WorkforceConstraints(
                capacity_per_worker=c.capacity_per_worker,
                cost_per_worker=c.cost_per_worker,
                max_workers=c.max_workers,
                budget_total=c.budget_total * delta,
                min_service_level=c.min_service_level,
                unmet_demand_penalty=c.unmet_demand_penalty,
                max_ramp_up=c.max_ramp_up,
                max_ramp_down=c.max_ramp_down,
                currency=c.currency,
            )
        elif axis == "headcount":
            new_max = max(1, c.max_workers - int(delta))
            c = WorkforceConstraints(
                capacity_per_worker=c.capacity_per_worker,
                cost_per_worker=c.cost_per_worker,
                max_workers=new_max,
                budget_total=c.budget_total,
                min_service_level=c.min_service_level,
                unmet_demand_penalty=c.unmet_demand_penalty,
                max_ramp_up=c.max_ramp_up,
                max_ramp_down=c.max_ramp_down,
                currency=c.currency,
            )

        opt = OptimizationPipeline(constraints=c, integer=True)
        result = opt.run(
            forecast_dates=pd.date_range("2024-09-30", periods=len(demand), freq="W-MON"),
            forecast_demand=demand,
            sku=SKU,
        )
        return {
            "cost": result.total_cost,
            "service_level": result.service_level,
            "avg_workers": float(result.workers.mean()),
            "status": result.solver_status,
        }
    return rerun


def main():
    # ── 1. Run upstream pipeline ──────────────────────────────────────
    if Path(DATA_PATH).exists():
        df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    else:
        df = generate_demand_dataset(n_weeks=104, n_skus=3)
        Path(DATA_PATH).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(DATA_PATH, index=False)

    print("Running forecasting pipeline...")
    fc_pipeline = ForecastingPipeline(test_weeks=13, deviation_threshold_pct=20.0)
    fc_results = fc_pipeline.run(df)
    best = fc_pipeline.best_model(fc_results)
    fc_result = best[SKU]
    print(f"  {SKU}: {fc_result.model_name}  MAPE={fc_result.mape:.2f}%")

    print("Running optimisation pipeline...")
    opt_pipeline = OptimizationPipeline(constraints=CONSTRAINTS, integer=True)
    alloc_result = opt_pipeline.run(fc_result.test_dates, fc_result.forecast, sku=SKU)
    print(f"  Status: {alloc_result.solver_status}  SL: {alloc_result.service_level:.1%}  "
          f"Cost: {alloc_result.total_cost:,.0f} EUR")

    print("Running deviation & sensitivity analysis...")
    dev_report = build_deviation_report(fc_result, alloc_result)
    sa = SensitivityAnalysis(base_demand=fc_result.forecast, base_constraints=CONSTRAINTS)
    sensitivity_tables = sa.run_all()

    context = build_context(fc_result, alloc_result, dev_report, sensitivity_tables)

    # ── 2. LLM layer ─────────────────────────────────────────────────
    try:
        interpreter = SupplyChainInterpreter()
        # Probe the key immediately with a tiny call before going further
        interpreter._call("Reply with the single word OK.", max_tokens=5)
    except EnvironmentError as e:
        print(f"\n[WARNING] {e}")
        print("LLM features disabled. Set ANTHROPIC_API_KEY in .env to enable them.")
        print("\n--- Context that would be sent to the LLM ---\n")
        print(context)
        return
    except Exception as e:
        if "401" in str(e) or "authentication" in str(e).lower() or "invalid" in str(e).lower():
            print(f"\n[WARNING] API key invalid or not yet set in .env: {e}")
            print("LLM features disabled. Replace the placeholder in .env with your real key.")
            print("\n--- Context that would be sent to the LLM ---\n")
            print(context)
        else:
            raise
        return

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    # ── 3. Executive summary ─────────────────────────────────────────
    print("\nGenerating executive summary...")
    summary = interpreter.executive_summary(context)
    print("\n" + "=" * 65)
    print("EXECUTIVE SUMMARY")
    print("=" * 65)
    print(summary)

    summary_path = f"{OUTPUT_DIR}/{SKU}_executive_summary.txt"
    Path(summary_path).write_text(summary, encoding="utf-8")
    print(f"\n  Saved: {summary_path}")

    # ── 4. Auto scenarios + robustness report ────────────────────────
    print("\nGenerating test scenarios...")
    scenarios = interpreter.generate_scenarios(context, n=5)
    print(f"  {len(scenarios)} scenarios proposed by LLM:")
    for s in scenarios:
        print(f"    [{s['expected_risk'].upper():6s}] {s['name']}: {s['description']}")

    print("\nRunning scenarios through pipeline...")
    scenario_results = run_scenarios(scenarios, fc_result.forecast, CONSTRAINTS)
    report_text = robustness_report(scenario_results, currency="EUR")

    print("\n" + "=" * 65)
    print("ROBUSTNESS REPORT")
    print("=" * 65)
    print(report_text)

    report_path = f"{OUTPUT_DIR}/{SKU}_robustness_report.txt"
    Path(report_path).write_text(report_text, encoding="utf-8")

    scenarios_csv = f"{OUTPUT_DIR}/{SKU}_scenarios.csv"
    to_dataframe(scenario_results).to_csv(scenarios_csv, index=False)
    print(f"  Saved: {report_path}")
    print(f"  Saved: {scenarios_csv}")

    # ── 5. Interactive Q&A ───────────────────────────────────────────
    rerun_fn = make_rerun_fn(fc_result.forecast, CONSTRAINTS)
    print("\n" + "=" * 65)
    print("INTERACTIVE Q&A (press Enter twice or Ctrl+C to exit)")
    print("=" * 65)
    print("Examples:")
    print("  What happens if demand rises by 25%?")
    print("  Which weeks are at risk?")
    print("  What if our budget is cut by 20%?")
    print("  What are the main recommendations?\n")

    qa_log = []
    while True:
        try:
            question = input("Your question: ").strip()
        except (KeyboardInterrupt, EOFError):
            break
        if not question:
            break

        print("\nThinking...", flush=True)
        answer = interpreter.answer(context, question, rerun_fn=rerun_fn)
        print(f"\nAnswer:\n{answer}\n")
        qa_log.append({"question": question, "answer": answer})

    if qa_log:
        qa_path = f"{OUTPUT_DIR}/{SKU}_qa_log.txt"
        with open(qa_path, "w", encoding="utf-8") as f:
            for entry in qa_log:
                f.write(f"Q: {entry['question']}\n\nA: {entry['answer']}\n\n{'─'*60}\n\n")
        print(f"Q&A log saved: {qa_path}")

    print("\nStep 4 complete.")


if __name__ == "__main__":
    main()
