"""
Auto scenario runner: takes LLM-generated scenario specs,
runs each through the optimisation LP, and produces a pass/fail robustness report.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd

from src.optimization.constraints import WorkforceConstraints
from src.optimization.model import solve_workforce_lp


@dataclass
class ScenarioRunResult:
    name: str
    axis: str
    delta: float
    description: str
    expected_risk: str
    total_cost: float
    service_level: float
    avg_workers: float
    solver_status: str
    passed: bool        # True if service level >= min_service_level


def run_scenarios(
    scenarios: List[dict],
    base_demand: np.ndarray,
    base_constraints: WorkforceConstraints,
) -> List[ScenarioRunResult]:
    results = []
    for spec in scenarios:
        axis = spec["axis"]
        delta = float(spec["delta"])
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

        try:
            workers, unmet, status = solve_workforce_lp(demand, c, integer=True)
            covered = demand - unmet
            total_cost = float((workers * c.cost_per_worker).sum())
            service_level = float(covered.sum() / demand.sum()) if demand.sum() > 0 else 0.0
        except Exception as e:
            workers = np.zeros(len(demand))
            total_cost = float("nan")
            service_level = 0.0
            status = f"Error: {e}"

        results.append(ScenarioRunResult(
            name=spec["name"],
            axis=axis,
            delta=delta,
            description=spec["description"],
            expected_risk=spec.get("expected_risk", "medium"),
            total_cost=total_cost,
            service_level=service_level,
            avg_workers=float(np.mean(workers)),
            solver_status=status,
            passed=service_level >= base_constraints.min_service_level,
        ))

    return results


def robustness_report(results: List[ScenarioRunResult], currency: str = "EUR") -> str:
    passed = [r for r in results if r.passed]
    failed = [r for r in results if not r.passed]
    lines = [
        f"Robustness Report — {len(results)} scenarios tested",
        f"  Passed: {len(passed)} / {len(results)}",
        f"  Failed: {len(failed)} / {len(results)}",
        "",
    ]
    for status_label, group in [("PASSED", passed), ("FAILED", failed)]:
        if not group:
            continue
        lines.append(f"--- {status_label} ---")
        for r in group:
            lines.append(
                f"  [{r.expected_risk.upper():6s}] {r.name}"
            )
            lines.append(f"           {r.description}")
            lines.append(
                f"           SL={r.service_level:.1%}  cost={r.total_cost:,.0f} {currency}  "
                f"workers={r.avg_workers:.1f}  solver={r.solver_status}"
            )
        lines.append("")
    return "\n".join(lines)


def to_dataframe(results: List[ScenarioRunResult]) -> pd.DataFrame:
    return pd.DataFrame([
        {
            "scenario": r.name,
            "axis": r.axis,
            "delta": r.delta,
            "description": r.description,
            "expected_risk": r.expected_risk,
            "total_cost": r.total_cost,
            "service_level": r.service_level,
            "avg_workers": r.avg_workers,
            "solver_status": r.solver_status,
            "passed": r.passed,
        }
        for r in results
    ])
