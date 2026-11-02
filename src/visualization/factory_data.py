"""
3D factory replay: turns the forecast + staffing plans into a week-by-week JSON
that the Three.js page (factory3d.html) animates.

Both factories receive the ACTUAL demand of the test horizon:
  - "Optimised plan" staffs with the LP workers (built on the forecast)
  - "Flat plan" staffs the same headcount every week (sized on the forecast peak)
so the replay shows what each plan would really have delivered.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.forecasting.pipeline import ForecastResult
from src.optimization.pipeline import AllocationResult

TEMPLATE_PATH = Path(__file__).with_name("factory3d.html")

# A week is "tight" (amber) when the plan uses at least this share of its capacity
TIGHT_UTILISATION = 0.95
# The optimised plan is "budget bound" when it spends at least this share of the budget
BUDGET_BOUND_SHARE = 0.99
# Target number of boxes on screen for the busiest week
MAX_BOXES = 24


def _week_cause(actual, forecast, workers, capacity, unmet, c, budget_bound):
    """Plain-language reason for a week's status."""
    if unmet <= 0:
        utilisation = actual / capacity if capacity > 0 else 0.0
        if utilisation >= TIGHT_UTILISATION:
            return f"Tight week: {utilisation:.0%} of capacity used"
        return "All demand served"
    if capacity >= forecast:
        # The plan covered its forecast, so the shortfall comes from the forecast error
        return f"Forecast too low by {actual - forecast:,.0f} units"
    if workers >= c.max_workers:
        return f"Headcount ceiling reached ({c.max_workers} workers)"
    if budget_bound:
        return "Budget ceiling: the plan could not staff the full forecast"
    return "Plan accepted unmet demand (cheaper than staffing)"


def _plan_weeks(name, key, workers, actual, forecast, c, budget_bound):
    workers = np.asarray(workers, dtype=float)
    capacity = workers * c.capacity_per_worker
    served = np.minimum(actual, capacity)
    unmet = actual - served
    cost = workers * c.cost_per_worker
    cum_cost = np.cumsum(cost)

    weeks = []
    for i in range(len(actual)):
        utilisation = actual[i] / capacity[i] if capacity[i] > 0 else 0.0
        if unmet[i] > 0:
            status = "red"
        elif utilisation >= TIGHT_UTILISATION:
            status = "amber"
        else:
            status = "green"
        weeks.append({
            "workers": int(round(workers[i])),
            # Workers actually needed for the demand served; the rest are idle
            "busy": int(min(np.ceil(served[i] / c.capacity_per_worker), workers[i])),
            "capacity": float(capacity[i]),
            "served": float(served[i]),
            "unmet": float(unmet[i]),
            "utilisation": float(utilisation),
            "cost": float(cost[i]),
            "cum_cost": float(cum_cost[i]),
            "status": status,
            "cause": _week_cause(actual[i], forecast[i], workers[i], capacity[i], unmet[i], c,
                                 budget_bound),
        })
    return {
        "name": name,
        "key": key,
        "total_cost": float(cum_cost[-1]) if len(cum_cost) else 0.0,
        "service_level": float(served.sum() / actual.sum()) if actual.sum() > 0 else 0.0,
        "weeks": weeks,
    }


def build_factory_data(fc: ForecastResult, alloc: AllocationResult) -> dict:
    """Week-by-week replay data for the optimised and flat plans."""
    c = alloc.constraints
    actual = np.asarray(fc.actual, dtype=float)
    forecast = np.asarray(fc.forecast, dtype=float)
    budget_bound = alloc.total_cost >= BUDGET_BOUND_SHARE * c.budget_total

    peak = max(float(actual.max()), 1.0)
    unit_per_box = max(1, int(np.ceil(peak / MAX_BOXES)))

    return {
        "sku": str(alloc.sku),
        "currency": c.currency,
        "capacity_per_worker": float(c.capacity_per_worker),
        "max_workers": int(c.max_workers),
        "budget_total": float(c.budget_total),
        "unit_per_box": unit_per_box,
        "weeks": [
            {"date": str(d.date()), "actual": float(a), "forecast": float(f)}
            for d, a, f in zip(fc.test_dates, actual, forecast)
        ],
        "plans": [
            _plan_weeks("Optimised plan", "opt", alloc.workers, actual, forecast, c, budget_bound),
            _plan_weeks("Flat plan", "flat", alloc.baseline_flat_workers, actual, forecast, c,
                        False),
        ],
    }


def render_factory_html(data: dict) -> str:
    """Inject the replay data into the Three.js page."""
    payload = json.dumps(data).replace("</", "<\\/")  # never close the <script> early
    return TEMPLATE_PATH.read_text(encoding="utf-8").replace("__FACTORY_DATA__", payload)
