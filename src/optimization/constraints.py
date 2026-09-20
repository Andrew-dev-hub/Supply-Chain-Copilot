"""WorkforceConstraints: typed container for all planning parameters."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class WorkforceConstraints:
    """
    Parameters governing the workforce scheduling optimisation.

    capacity_per_worker : units a single worker can process per planning period
    cost_per_worker     : labour cost per worker per planning period (same currency)
    max_workers         : maximum headcount available each period
    budget_total        : total labour budget over the whole horizon
    min_service_level   : minimum fraction of demand that must be covered (0-1)
    unmet_demand_penalty: cost per unit of unmet demand (drives the objective trade-off)
    max_ramp_up         : max increase in workers between consecutive periods (None = no limit)
    max_ramp_down       : max decrease in workers between consecutive periods (None = no limit)
    currency            : label used in reporting (default "EUR")
    """

    capacity_per_worker: float
    cost_per_worker: float
    max_workers: int
    budget_total: float
    min_service_level: float = 0.90
    unmet_demand_penalty: float = 5000.0
    max_ramp_up: Optional[int] = None
    max_ramp_down: Optional[int] = None
    currency: str = "EUR"

    def __post_init__(self):
        assert 0 < self.min_service_level <= 1, "min_service_level must be in (0, 1]"
        assert self.capacity_per_worker > 0, "capacity_per_worker must be positive"
        assert self.cost_per_worker > 0, "cost_per_worker must be positive"
        assert self.max_workers > 0, "max_workers must be positive"
        assert self.budget_total > 0, "budget_total must be positive"

    def summary(self) -> str:
        lines = [
            "Workforce Constraints",
            f"  Capacity / worker / period : {self.capacity_per_worker:.0f} units",
            f"  Cost / worker / period     : {self.cost_per_worker:.0f} {self.currency}",
            f"  Max workers available      : {self.max_workers}",
            f"  Total budget               : {self.budget_total:,.0f} {self.currency}",
            f"  Min service level          : {self.min_service_level:.0%}",
            f"  Unmet demand penalty       : {self.unmet_demand_penalty:.0f} {self.currency}/unit",
        ]
        if self.max_ramp_up is not None:
            lines.append(f"  Max ramp-up / period       : {self.max_ramp_up} workers")
        if self.max_ramp_down is not None:
            lines.append(f"  Max ramp-down / period     : {self.max_ramp_down} workers")
        return "\n".join(lines)
