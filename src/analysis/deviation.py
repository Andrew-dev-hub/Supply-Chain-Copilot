"""
Deviation analysis: bridges the forecasting and optimization layers.

Computes three complementary views:
  1. Forecast accuracy deviations (actual vs forecast, from module 2)
  2. Plan coverage gaps (forecast demand vs planned capacity)
  3. Risk weeks: periods where forecast uncertainty may break the plan
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.forecasting.pipeline import ForecastResult
from src.optimization.pipeline import AllocationResult


@dataclass
class DeviationReport:
    sku: str
    df: pd.DataFrame   # one row per week, all metrics combined

    # Aggregate KPIs
    weeks_at_risk: int          # plan capacity < actual demand (if actuals known)
    max_coverage_gap: float     # worst-case units uncovered by plan
    forecast_mape: float
    plan_service_level: float

    def summary(self) -> str:
        lines = [
            f"SKU: {self.sku}",
            f"Forecast MAPE            : {self.forecast_mape:.2f}%",
            f"Plan service level       : {self.plan_service_level:.1%}",
            f"Weeks where plan < actual: {self.weeks_at_risk}",
            f"Max coverage gap         : {self.max_coverage_gap:,.0f} units",
        ]
        at_risk = self.df[self.df["plan_gap"] > 0]
        if not at_risk.empty:
            lines.append("\nAt-risk weeks (planned capacity < actual demand):")
            lines.append(
                at_risk[["date", "actual", "forecast", "plan_capacity", "plan_gap", "forecast_error_pct"]]
                .to_string(index=False, float_format="%.1f")
            )
        return "\n".join(lines)


def build_deviation_report(
    forecast_result: ForecastResult,
    alloc_result: AllocationResult,
) -> DeviationReport:
    """
    Merge forecast and allocation data into a unified deviation report.

    forecast_result : output of ForecastingPipeline (has actual + forecast)
    alloc_result    : output of OptimizationPipeline (has workers + capacity)
    """
    c = alloc_result.constraints
    dates = pd.DatetimeIndex(forecast_result.test_dates)

    plan_capacity = alloc_result.workers * c.capacity_per_worker
    forecast_error_pct = np.where(
        forecast_result.actual != 0,
        (forecast_result.forecast - forecast_result.actual) / forecast_result.actual * 100,
        np.nan,
    )
    # Gap = how many units the plan fails to cover vs the ACTUAL demand
    plan_gap = np.maximum(forecast_result.actual - plan_capacity, 0)

    df = pd.DataFrame({
        "date": dates,
        "actual": forecast_result.actual,
        "forecast": forecast_result.forecast,
        "forecast_error_pct": forecast_error_pct,
        "workers": alloc_result.workers,
        "plan_capacity": plan_capacity,
        "plan_gap": plan_gap,
        "plan_unmet": alloc_result.unmet_demand,
        "cost": alloc_result.cost_per_period,
    })

    return DeviationReport(
        sku=forecast_result.sku,
        df=df,
        weeks_at_risk=int((plan_gap > 0).sum()),
        max_coverage_gap=float(plan_gap.max()),
        forecast_mape=forecast_result.mape,
        plan_service_level=alloc_result.service_level,
    )
