"""
Serialises pipeline outputs into a compact, LLM-readable context block.
Keeps numbers precise but avoids dumping raw DataFrames verbatim.
"""

from __future__ import annotations

import pandas as pd

from src.forecasting.pipeline import ForecastResult
from src.optimization.pipeline import AllocationResult
from src.analysis.deviation import DeviationReport


def build_context(
    fc_result: ForecastResult,
    alloc_result: AllocationResult,
    dev_report: DeviationReport,
    sensitivity_tables: dict[str, pd.DataFrame],
) -> str:
    c = alloc_result.constraints
    lines = []

    # ── Forecast summary ──────────────────────────────────────────────
    lines += [
        "=== FORECASTING (Module 2) ===",
        f"SKU: {fc_result.sku}",
        f"Model: {fc_result.model_name}",
        f"Test period: {fc_result.test_dates[0].date()} to {fc_result.test_dates[-1].date()} ({len(fc_result.test_dates)} weeks)",
        f"MAPE: {fc_result.mape:.2f}%  |  MAE: {fc_result.mae:.1f}  |  RMSE: {fc_result.rmse:.1f}",
        f"Total forecasted demand (test horizon): {fc_result.forecast.sum():,.0f} units",
        f"Significant forecast deviations (>20%): {len(fc_result.deviations)}",
        "",
    ]

    # ── Optimisation summary ──────────────────────────────────────────
    lines += [
        "=== OPTIMISATION (Module 3) ===",
        f"Solver status: {alloc_result.solver_status}",
        f"Total cost: {alloc_result.total_cost:,.0f} {c.currency}  "
        f"(budget: {c.budget_total:,.0f} {c.currency})",
        f"Service level: {alloc_result.service_level:.1%}  (target: {c.min_service_level:.0%})",
        f"Workers: avg {alloc_result.workers.mean():.1f}  "
        f"min {alloc_result.workers.min():.0f}  max {alloc_result.workers.max():.0f}  "
        f"(hard cap: {c.max_workers})",
        f"Total unmet demand: {alloc_result.unmet_demand.sum():,.0f} units",
        f"Baseline flat cost: {alloc_result.baseline_flat_cost:,.0f} {c.currency}  "
        f"(savings: {alloc_result.baseline_flat_cost - alloc_result.total_cost:+,.0f})",
        "",
    ]

    # ── Constraints ───────────────────────────────────────────────────
    lines += [
        "=== CONSTRAINTS ===",
        f"Capacity per worker per week: {c.capacity_per_worker:.0f} units",
        f"Cost per worker per week: {c.cost_per_worker:.0f} {c.currency}",
        f"Max workers: {c.max_workers}",
        f"Budget: {c.budget_total:,.0f} {c.currency}",
        f"Min service level: {c.min_service_level:.0%}",
        f"Ramp-up limit: {c.max_ramp_up if c.max_ramp_up is not None else 'none'} workers/week",
        f"Ramp-down limit: {c.max_ramp_down if c.max_ramp_down is not None else 'none'} workers/week",
        "",
    ]

    # ── Deviation report ──────────────────────────────────────────────
    lines += [
        "=== DEVIATION REPORT (Module 2+3 bridge) ===",
        f"At-risk weeks (plan capacity < actual demand): {dev_report.weeks_at_risk}",
        f"Max coverage gap: {dev_report.max_coverage_gap:,.0f} units",
    ]
    if not dev_report.df[dev_report.df["plan_gap"] > 0].empty:
        at_risk = dev_report.df[dev_report.df["plan_gap"] > 0][
            ["date", "actual", "forecast", "plan_capacity", "plan_gap", "forecast_error_pct"]
        ]
        lines.append("At-risk week details:")
        for _, row in at_risk.iterrows():
            lines.append(
                f"  {row['date'].date()}  actual={row['actual']:.0f}  "
                f"forecast={row['forecast']:.0f}  capacity={row['plan_capacity']:.0f}  "
                f"gap={row['plan_gap']:.0f}  err={row['forecast_error_pct']:+.1f}%"
            )
    lines.append("")

    # ── Sensitivity summary ───────────────────────────────────────────
    lines.append("=== SENSITIVITY ANALYSIS ===")
    for axis, tdf in sensitivity_tables.items():
        lines.append(f"[{axis.upper()}]")
        for _, row in tdf.iterrows():
            feasible_tag = "" if row["feasible"] else "  *** INFEASIBLE ***"
            lines.append(
                f"  {row['scenario']:35s}  cost={row['total_cost']:>9,.0f} {c.currency}  "
                f"SL={row['service_level']:.1%}  workers={row['avg_workers']:.1f}{feasible_tag}"
            )
        lines.append("")

    return "\n".join(lines)
