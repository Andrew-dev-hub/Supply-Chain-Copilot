"""
Step 6: AI-automated documentation (Sonnet).

Chains the full pipeline, then generates:
  - Architecture overview
  - Methodology section
  - Results & Key Findings
  - Full README.md

All saved to output/documentation/ and README.md at project root.
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
from src.interpretation.context_builder import build_context
from src.documentation.generator import DocumentationGenerator

DATA_PATH = "data/demand_data.csv"
OUTPUT_DIR = "output/documentation"
SKU = "SKU-A"

CONSTRAINTS = WorkforceConstraints(
    capacity_per_worker=50.0, cost_per_worker=900.0, max_workers=20,
    budget_total=175_000.0, min_service_level=0.92, unmet_demand_penalty=4_000.0,
    max_ramp_up=3, max_ramp_down=3, currency="EUR",
)


def main():
    # ── Run pipeline to get context ───────────────────────────────────
    if Path(DATA_PATH).exists():
        df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    else:
        df = generate_demand_dataset(n_weeks=156, n_skus=3)
        df.to_csv(DATA_PATH, index=False)

    print("Running pipeline to gather context...")
    fc_pipe = ForecastingPipeline(test_weeks=13)
    fc_results = fc_pipe.run(df)
    fc = fc_pipe.best_model(fc_results)[SKU]

    opt = OptimizationPipeline(constraints=CONSTRAINTS)
    alloc = opt.run(fc.test_dates, fc.forecast, sku=SKU)

    dev = build_deviation_report(fc, alloc)
    sa = SensitivityAnalysis(base_demand=fc.forecast, base_constraints=CONSTRAINTS)
    sensitivity = sa.run_all()
    context = build_context(fc, alloc, dev, sensitivity)

    # ── Generate documentation ────────────────────────────────────────
    print("\nGenerating documentation with Sonnet...")
    gen = DocumentationGenerator()
    docs = gen.generate_all(context)

    # ── Save outputs ──────────────────────────────────────────────────
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    section_files = {
        "architecture": f"{OUTPUT_DIR}/architecture.md",
        "methodology":  f"{OUTPUT_DIR}/methodology.md",
        "results":      f"{OUTPUT_DIR}/results.md",
    }
    for key, path in section_files.items():
        Path(path).write_text(docs[key], encoding="utf-8")
        print(f"  Saved: {path}")

    # README at project root
    readme_path = "README.md"
    Path(readme_path).write_text(docs["readme"], encoding="utf-8")
    print(f"  Saved: {readme_path}")

    # Full combined doc
    combined = "\n\n---\n\n".join([
        f"# Architecture\n\n{docs['architecture']}",
        f"# Methodology\n\n{docs['methodology']}",
        f"# Results\n\n{docs['results']}",
    ])
    combined_path = f"{OUTPUT_DIR}/technical_documentation.md"
    Path(combined_path).write_text(combined, encoding="utf-8")
    print(f"  Saved: {combined_path}")

    print("\n--- README preview (first 30 lines) ---")
    for line in docs["readme"].splitlines()[:30]:
        print(line.encode("ascii", errors="replace").decode("ascii"))
    print("...")

    print("\nStep 6 complete.")


if __name__ == "__main__":
    main()
