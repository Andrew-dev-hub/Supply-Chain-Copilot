"""
OptimizationPipeline: takes a demand forecast and constraints,
runs the LP solver, and returns a structured AllocationResult.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .constraints import WorkforceConstraints
from .model import build_baseline, solve_workforce_lp


@dataclass
class AllocationResult:
    sku: str
    periods: pd.DatetimeIndex
    demand: np.ndarray
    # Optimized plan
    workers: np.ndarray
    covered_demand: np.ndarray
    unmet_demand: np.ndarray
    cost_per_period: np.ndarray
    total_cost: float
    service_level: float
    solver_status: str
    # Baselines
    baseline_flat_workers: np.ndarray
    baseline_flat_cost: float
    baseline_reactive_workers: np.ndarray
    baseline_reactive_cost: float

    constraints: WorkforceConstraints

    def summary(self) -> str:
        c = self.constraints
        total_demand = self.demand.sum()
        savings_vs_flat = self.baseline_flat_cost - self.total_cost
        savings_vs_reactive = self.baseline_reactive_cost - self.total_cost

        lines = [
            f"SKU              : {self.sku}",
            f"Periods          : {self.periods[0].date()} to {self.periods[-1].date()} ({len(self.periods)} weeks)",
            f"Total demand     : {total_demand:,.0f} units",
            f"Solver status    : {self.solver_status}",
            "",
            "--- Optimised Plan ---",
            f"Total cost       : {self.total_cost:,.0f} {c.currency}",
            f"Service level    : {self.service_level:.1%}",
            f"Avg workers/week : {self.workers.mean():.1f}  (min {self.workers.min():.0f}, max {self.workers.max():.0f})",
            f"Total unmet dem. : {self.unmet_demand.sum():,.0f} units",
            "",
            "--- Baseline Comparison ---",
            f"Flat staffing cost    : {self.baseline_flat_cost:,.0f} {c.currency}  "
            f"(savings: {savings_vs_flat:+,.0f})",
            f"Reactive staffing cost: {self.baseline_reactive_cost:,.0f} {c.currency}  "
            f"(savings: {savings_vs_reactive:+,.0f})",
        ]
        return "\n".join(lines)

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "date": self.periods,
                "demand": self.demand,
                "workers_optimised": self.workers,
                "covered_demand": self.covered_demand,
                "unmet_demand": self.unmet_demand,
                "cost": self.cost_per_period,
                "workers_flat": self.baseline_flat_workers,
                "workers_reactive": self.baseline_reactive_workers,
            }
        )


class OptimizationPipeline:
    """
    Runs the workforce scheduling LP on a forecast and returns an AllocationResult.

    Parameters
    ----------
    constraints : WorkforceConstraints (hardcoded or LLM-parsed)
    integer     : use integer workers (MIP) vs LP relaxation
    """

    def __init__(self, constraints: WorkforceConstraints, integer: bool = True):
        self.constraints = constraints
        self.integer = integer

    def run(
        self,
        forecast_dates: pd.DatetimeIndex,
        forecast_demand: np.ndarray,
        sku: str = "ALL",
    ) -> AllocationResult:
        demand = np.asarray(forecast_demand, dtype=float)
        c = self.constraints

        workers, unmet, status = solve_workforce_lp(demand, c, integer=self.integer)

        covered = demand - unmet
        cost_per_period = workers * c.cost_per_worker
        total_cost = float(cost_per_period.sum())
        service_level = float(covered.sum() / demand.sum()) if demand.sum() > 0 else 0.0

        flat_w = build_baseline(demand, c, mode="flat")
        flat_cost = float((flat_w * c.cost_per_worker).sum())

        reactive_w = build_baseline(demand, c, mode="reactive")
        reactive_cost = float((reactive_w * c.cost_per_worker).sum())

        return AllocationResult(
            sku=sku,
            periods=forecast_dates,
            demand=demand,
            workers=workers,
            covered_demand=covered,
            unmet_demand=unmet,
            cost_per_period=cost_per_period,
            total_cost=total_cost,
            service_level=service_level,
            solver_status=status,
            baseline_flat_workers=flat_w,
            baseline_flat_cost=flat_cost,
            baseline_reactive_workers=reactive_w,
            baseline_reactive_cost=reactive_cost,
            constraints=c,
        )
