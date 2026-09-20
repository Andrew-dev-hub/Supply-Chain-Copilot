"""
Sensitivity analysis: re-runs the LP across a grid of perturbed inputs
and records how KPIs (cost, service level, workers) respond.

Three axes of analysis:
  A. Demand shocks:       demand * (1 + delta) for delta in [-30%..+30%]
  B. Budget reduction:    budget * factor for factor in [0.5..1.0]
  C. Headcount loss:      max_workers reduced by 0..8 workers
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd

from src.optimization.constraints import WorkforceConstraints
from src.optimization.model import solve_workforce_lp


@dataclass
class ScenarioResult:
    label: str
    axis: str           # "demand" | "budget" | "headcount"
    delta: float        # perturbation value (%, absolute reduction, etc.)
    total_cost: float
    service_level: float
    avg_workers: float
    solver_status: str
    feasible: bool


class SensitivityAnalysis:
    """
    Runs the LP solver across scenario grids and returns tidy DataFrames.

    Parameters
    ----------
    base_demand      : baseline demand array (T,)
    base_constraints : baseline WorkforceConstraints
    demand_deltas    : list of demand shock percentages, e.g. [-0.3,-0.2,...,0.3]
    budget_factors   : list of budget multipliers, e.g. [0.5, 0.6, ..., 1.0]
    headcount_losses : list of integer worker reductions, e.g. [0,2,4,6,8]
    """

    def __init__(
        self,
        base_demand: np.ndarray,
        base_constraints: WorkforceConstraints,
        demand_deltas: List[float] | None = None,
        budget_factors: List[float] | None = None,
        headcount_losses: List[int] | None = None,
    ):
        self.demand = np.asarray(base_demand, float)
        self.c = base_constraints
        self.demand_deltas = demand_deltas or [-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3]
        self.budget_factors = budget_factors or [0.50, 0.60, 0.70, 0.80, 0.90, 1.00]
        self.headcount_losses = headcount_losses or [0, 2, 4, 6, 8]

    # ------------------------------------------------------------------ #
    #  Internal helper                                                     #
    # ------------------------------------------------------------------ #
    def _run_one(self, demand: np.ndarray, constraints: WorkforceConstraints,
                 label: str, axis: str, delta: float) -> ScenarioResult:
        try:
            workers, unmet, status = solve_workforce_lp(demand, constraints, integer=True)
            covered = demand - unmet
            total_cost = float((workers * constraints.cost_per_worker).sum())
            service_level = float(covered.sum() / demand.sum()) if demand.sum() > 0 else 0.0
            feasible = status == "Optimal"
        except Exception:
            total_cost = float("nan")
            service_level = float("nan")
            workers = np.zeros(len(demand))
            status = "Error"
            feasible = False

        return ScenarioResult(
            label=label,
            axis=axis,
            delta=delta,
            total_cost=total_cost,
            service_level=service_level,
            avg_workers=float(np.mean(workers)),
            solver_status=status,
            feasible=feasible,
        )

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #
    def run_demand_sensitivity(self) -> pd.DataFrame:
        """Shock demand by demand_deltas and record KPIs."""
        rows = []
        for delta in self.demand_deltas:
            d = self.demand * (1 + delta)
            label = f"Demand {delta:+.0%}"
            rows.append(self._run_one(d, self.c, label, "demand", delta))
        return self._to_df(rows)

    def run_budget_sensitivity(self) -> pd.DataFrame:
        """Reduce total budget by budget_factors and record KPIs."""
        rows = []
        for factor in self.budget_factors:
            c2 = WorkforceConstraints(
                capacity_per_worker=self.c.capacity_per_worker,
                cost_per_worker=self.c.cost_per_worker,
                max_workers=self.c.max_workers,
                budget_total=self.c.budget_total * factor,
                min_service_level=self.c.min_service_level,
                unmet_demand_penalty=self.c.unmet_demand_penalty,
                max_ramp_up=self.c.max_ramp_up,
                max_ramp_down=self.c.max_ramp_down,
                currency=self.c.currency,
            )
            label = f"Budget x{factor:.0%}"
            rows.append(self._run_one(self.demand, c2, label, "budget", factor))
        return self._to_df(rows)

    def run_headcount_sensitivity(self) -> pd.DataFrame:
        """Remove headcount_losses workers from max_workers and record KPIs."""
        rows = []
        for loss in self.headcount_losses:
            new_max = max(1, self.c.max_workers - loss)
            c2 = WorkforceConstraints(
                capacity_per_worker=self.c.capacity_per_worker,
                cost_per_worker=self.c.cost_per_worker,
                max_workers=new_max,
                budget_total=self.c.budget_total,
                min_service_level=self.c.min_service_level,
                unmet_demand_penalty=self.c.unmet_demand_penalty,
                max_ramp_up=self.c.max_ramp_up,
                max_ramp_down=self.c.max_ramp_down,
                currency=self.c.currency,
            )
            label = f"Max workers = {new_max} (-{loss})"
            rows.append(self._run_one(self.demand, c2, label, "headcount", float(loss)))
        return self._to_df(rows)

    def run_all(self) -> dict[str, pd.DataFrame]:
        print("  Running demand sensitivity...")
        demand_df = self.run_demand_sensitivity()
        print("  Running budget sensitivity...")
        budget_df = self.run_budget_sensitivity()
        print("  Running headcount sensitivity...")
        headcount_df = self.run_headcount_sensitivity()
        return {"demand": demand_df, "budget": budget_df, "headcount": headcount_df}

    @staticmethod
    def _to_df(rows: List[ScenarioResult]) -> pd.DataFrame:
        return pd.DataFrame([
            {
                "scenario": r.label,
                "delta": r.delta,
                "total_cost": r.total_cost,
                "service_level": r.service_level,
                "avg_workers": r.avg_workers,
                "status": r.solver_status,
                "feasible": r.feasible,
            }
            for r in rows
        ])
