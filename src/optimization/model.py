"""
PuLP-based workforce scheduling LP.

Objective: minimise (labour cost + unmet-demand penalty)
           subject to coverage, headcount, budget, and ramp constraints.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pulp

from .constraints import WorkforceConstraints


def solve_workforce_lp(
    demand: np.ndarray,
    constraints: WorkforceConstraints,
    integer: bool = True,
) -> Tuple[np.ndarray, np.ndarray, str]:
    """
    Solve the staffing LP/MIP for a demand vector.

    Parameters
    ----------
    demand     : forecasted demand per period, shape (T,)
    constraints: WorkforceConstraints instance
    integer    : if True use integer workers (MIP); False gives LP relaxation

    Returns
    -------
    workers       : array(T,) allocated headcount per period
    unmet_demand  : array(T,) unmet demand per period
    status        : solver status string
    """
    T = len(demand)
    c = constraints

    total_demand = float(demand.sum())

    def build_and_solve(enforce_service: bool):
        prob = pulp.LpProblem("WorkforceScheduling", pulp.LpMinimize)
        var_cat = "Integer" if integer else "Continuous"

        # Decision variables
        w = [pulp.LpVariable(f"w_{t}", lowBound=0, upBound=c.max_workers, cat=var_cat) for t in range(T)]
        s = [pulp.LpVariable(f"s_{t}", lowBound=0, cat="Continuous") for t in range(T)]

        # Objective: minimise labour cost + unmet demand penalty
        prob += (
            pulp.lpSum(c.cost_per_worker * w[t] for t in range(T))
            + pulp.lpSum(c.unmet_demand_penalty * s[t] for t in range(T))
        )

        for t in range(T):
            # Coverage: workers * capacity + slack >= demand
            prob += w[t] * c.capacity_per_worker + s[t] >= demand[t], f"coverage_{t}"

        # Total budget
        prob += pulp.lpSum(c.cost_per_worker * w[t] for t in range(T)) <= c.budget_total, "budget"

        # Minimum service level: total unmet demand <= (1 - SL) * total demand
        if enforce_service:
            max_unmet = (1 - c.min_service_level) * total_demand
            prob += pulp.lpSum(s[t] for t in range(T)) <= max_unmet, "service_level"

        # Ramp-up / ramp-down constraints
        for t in range(1, T):
            if c.max_ramp_up is not None:
                prob += w[t] - w[t - 1] <= c.max_ramp_up, f"ramp_up_{t}"
            if c.max_ramp_down is not None:
                prob += w[t - 1] - w[t] <= c.max_ramp_down, f"ramp_down_{t}"

        prob.solve(pulp.PULP_CBC_CMD(msg=0))
        return prob, w, s

    prob, w, s = build_and_solve(enforce_service=True)
    status = pulp.LpStatus[prob.status]

    if prob.status == -1:
        # The minimum service level cannot be met within budget / headcount / ramp limits.
        # Values of an infeasible solve are meaningless (they can be negative), so re-solve
        # without the service-level constraint (unmet demand is then only penalised) and
        # keep the "Infeasible" status so callers know the target was missed.
        prob, w, s = build_and_solve(enforce_service=False)

    if prob.status != 1:
        # Fallback: just enough workers each period ignoring budget
        workers_arr = np.ceil(demand / c.capacity_per_worker).clip(0, c.max_workers)
        unmet_arr = np.maximum(demand - workers_arr * c.capacity_per_worker, 0)
        return workers_arr, unmet_arr, f"Fallback ({pulp.LpStatus[prob.status]})"

    workers_arr = np.array([pulp.value(w[t]) or 0.0 for t in range(T)])
    unmet_arr = np.array([pulp.value(s[t]) or 0.0 for t in range(T)])

    if integer:
        workers_arr = np.round(workers_arr).astype(int)

    return workers_arr, unmet_arr, status


def build_baseline(
    demand: np.ndarray,
    constraints: WorkforceConstraints,
    mode: str = "flat",
) -> np.ndarray:
    """
    Build a naive baseline allocation for comparison.

    mode='flat'     : same headcount every week (based on peak demand)
    mode='reactive' : exactly enough workers each week (no smoothing, ignores budget)
    """
    if mode == "flat":
        peak = float(demand.max())
        flat_workers = int(np.ceil(peak / constraints.capacity_per_worker))
        flat_workers = min(flat_workers, constraints.max_workers)
        return np.full(len(demand), flat_workers, dtype=float)
    elif mode == "reactive":
        return np.minimum(
            np.ceil(demand / constraints.capacity_per_worker),
            constraints.max_workers,
        )
    else:
        raise ValueError(f"Unknown baseline mode: {mode!r}")
