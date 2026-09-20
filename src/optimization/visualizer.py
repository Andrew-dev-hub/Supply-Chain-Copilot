"""Charts for allocation results."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np

from .pipeline import AllocationResult


def plot_allocation(
    result: AllocationResult,
    output_path: Optional[str] = None,
    show: bool = True,
) -> plt.Figure:
    c = result.constraints
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), gridspec_kw={"height_ratios": [2, 2, 1]})
    fig.suptitle(
        f"Workforce Allocation — {result.sku}  |  Status: {result.solver_status}",
        fontsize=13, fontweight="bold",
    )

    dates = result.periods
    fmt = mdates.DateFormatter("%b %Y")
    loc = mdates.MonthLocator()

    # --- Panel 1: Demand coverage ---
    ax = axes[0]
    ax.fill_between(dates, result.demand, alpha=0.15, color="#1565c0", label="Forecasted demand")
    ax.plot(dates, result.demand, color="#1565c0", linewidth=1.5)
    ax.plot(dates, result.covered_demand, color="#2e7d32", linewidth=2,
            marker="o", markersize=3, label="Covered demand (optimised)")
    if result.unmet_demand.sum() > 0:
        ax.fill_between(dates, result.covered_demand, result.demand,
                        alpha=0.4, color="#e53935", label="Unmet demand")
    ax.set_ylabel("Units")
    ax.legend(fontsize=9, loc="upper left")
    ax.xaxis.set_major_formatter(fmt)
    ax.xaxis.set_major_locator(loc)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.set_title(f"Service level: {result.service_level:.1%}", fontsize=10, loc="right")

    # --- Panel 2: Workers comparison ---
    ax2 = axes[1]
    width = 3.5
    ax2.bar(dates - np.timedelta64(4, "D"), result.baseline_flat_workers,
            width=width, color="#90a4ae", alpha=0.7, label="Baseline (flat)")
    ax2.bar(dates, result.baseline_reactive_workers,
            width=width, color="#ffb74d", alpha=0.7, label="Baseline (reactive)")
    ax2.bar(dates + np.timedelta64(4, "D"), result.workers,
            width=width, color="#1565c0", alpha=0.9, label="Optimised")
    ax2.axhline(result.constraints.max_workers, color="red", linestyle="--",
                linewidth=1, label=f"Max workers ({result.constraints.max_workers})")
    ax2.set_ylabel("Workers")
    ax2.legend(fontsize=9, loc="upper right")
    ax2.xaxis.set_major_formatter(fmt)
    ax2.xaxis.set_major_locator(loc)
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=30, ha="right")
    ax2.grid(axis="y", linestyle=":", alpha=0.5)

    # --- Panel 3: Cumulative cost ---
    ax3 = axes[2]
    cum_opt = np.cumsum(result.cost_per_period)
    cum_flat = np.cumsum(result.baseline_flat_workers * c.cost_per_worker)
    cum_reactive = np.cumsum(result.baseline_reactive_workers * c.cost_per_worker)
    ax3.plot(dates, cum_opt / 1000, color="#1565c0", linewidth=2, label=f"Optimised ({result.total_cost/1000:.0f}k)")
    ax3.plot(dates, cum_flat / 1000, color="#90a4ae", linewidth=1.5,
             linestyle="--", label=f"Flat ({result.baseline_flat_cost/1000:.0f}k)")
    ax3.plot(dates, cum_reactive / 1000, color="#ffb74d", linewidth=1.5,
             linestyle=":", label=f"Reactive ({result.baseline_reactive_cost/1000:.0f}k)")
    ax3.axhline(c.budget_total / 1000, color="red", linestyle="-.", linewidth=1,
                label=f"Budget ({c.budget_total/1000:.0f}k)")
    ax3.set_ylabel(f"Cumulative cost (k{c.currency})")
    ax3.legend(fontsize=8, loc="upper left")
    ax3.xaxis.set_major_formatter(fmt)
    ax3.xaxis.set_major_locator(loc)
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=30, ha="right")
    ax3.grid(axis="y", linestyle=":", alpha=0.5)

    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {output_path}")
    if show:
        plt.show()
    return fig


def plot_cost_breakdown(
    result: AllocationResult,
    output_path: Optional[str] = None,
    show: bool = True,
) -> plt.Figure:
    """Grouped bar: optimised vs flat vs reactive total cost."""
    c = result.constraints
    labels = ["Optimised", "Flat baseline", "Reactive baseline"]
    costs = [result.total_cost, result.baseline_flat_cost, result.baseline_reactive_cost]
    colors = ["#1565c0", "#90a4ae", "#ffb74d"]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(labels, [v / 1000 for v in costs], color=colors, alpha=0.85, width=0.5)
    ax.bar_label(bars, fmt=lambda v: f"{v:.0f}k {c.currency}", padding=4, fontsize=10)
    ax.axhline(c.budget_total / 1000, color="red", linestyle="--", linewidth=1.2,
               label=f"Budget ({c.budget_total/1000:.0f}k)")
    ax.set_ylabel(f"Total cost (k{c.currency})")
    ax.set_title(f"Cost comparison — {result.sku}")
    ax.legend(fontsize=9)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.set_ylim(0, max(costs) / 1000 * 1.25)
    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {output_path}")
    if show:
        plt.show()
    return fig
