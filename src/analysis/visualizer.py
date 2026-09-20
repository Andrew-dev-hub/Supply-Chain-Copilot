"""Charts for deviation and sensitivity analysis."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

from .deviation import DeviationReport


def plot_deviation_report(
    report: DeviationReport,
    output_path: Optional[str] = None,
    show: bool = True,
) -> plt.Figure:
    df = report.df
    dates = pd.to_datetime(df["date"])
    fmt = mdates.DateFormatter("%b %Y")
    loc = mdates.MonthLocator()

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), gridspec_kw={"height_ratios": [2, 2, 1]})
    fig.suptitle(
        f"Deviation & Coverage Report — {report.sku}",
        fontsize=13, fontweight="bold",
    )

    # Panel 1: actual vs forecast vs plan capacity
    ax = axes[0]
    ax.plot(dates, df["actual"], color="#1565c0", lw=2, marker="o", ms=4, label="Actual demand")
    ax.plot(dates, df["forecast"], color="#e53935", lw=1.8, ls="--", label="Forecast")
    ax.step(dates, df["plan_capacity"], color="#2e7d32", lw=2, where="post", label="Plan capacity")
    gap_mask = df["plan_gap"] > 0
    if gap_mask.any():
        ax.fill_between(
            dates, df["plan_capacity"], df["actual"],
            where=gap_mask, alpha=0.35, color="#e53935", label="Coverage gap (plan < actual)",
        )
    ax.set_ylabel("Units")
    ax.legend(fontsize=9, loc="upper right")
    ax.xaxis.set_major_formatter(fmt); ax.xaxis.set_major_locator(loc)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")
    ax.grid(axis="y", ls=":", alpha=0.5)

    # Panel 2: forecast error % with at-risk coloring
    ax2 = axes[1]
    colors = ["#e53935" if gap > 0 else "#1565c0" for gap in df["plan_gap"]]
    ax2.bar(dates, df["forecast_error_pct"], width=5, color=colors, alpha=0.75)
    ax2.axhline(0, color="black", lw=0.8)
    ax2.axhline(20, color="orange", ls="--", lw=0.9, label="+20% threshold")
    ax2.axhline(-20, color="orange", ls="--", lw=0.9)
    ax2.set_ylabel("Forecast error (%)")
    ax2.set_ylim(-50, 50)
    ax2.xaxis.set_major_formatter(fmt); ax2.xaxis.set_major_locator(loc)
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=30, ha="right")
    ax2.grid(axis="y", ls=":", alpha=0.5)
    red_patch = mpatches.Patch(color="#e53935", alpha=0.75, label="At-risk week (plan gap > 0)")
    blue_patch = mpatches.Patch(color="#1565c0", alpha=0.75, label="Safe week")
    ax2.legend(handles=[red_patch, blue_patch, mpatches.Patch(color="orange", label="±20% threshold")],
               fontsize=8, loc="upper right")

    # Panel 3: workers per week
    ax3 = axes[2]
    ax3.bar(dates, df["workers"], width=5, color="#546e7a", alpha=0.8, label="Workers assigned")
    ax3.set_ylabel("Workers")
    ax3.xaxis.set_major_formatter(fmt); ax3.xaxis.set_major_locator(loc)
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=30, ha="right")
    ax3.grid(axis="y", ls=":", alpha=0.5)
    ax3.legend(fontsize=9)

    plt.tight_layout()
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {output_path}")
    if show:
        plt.show()
    return fig


def plot_sensitivity(
    tables: dict[str, pd.DataFrame],
    currency: str = "EUR",
    output_path: Optional[str] = None,
    show: bool = True,
) -> plt.Figure:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Sensitivity Analysis", fontsize=13, fontweight="bold")

    configs = [
        ("demand",    "Demand shock",          "delta",  "Demand change (%)",       lambda x: f"{x:+.0%}"),
        ("budget",    "Budget factor",          "delta",  "Budget (% of baseline)",  lambda x: f"{x:.0%}"),
        ("headcount", "Headcount loss (workers)", "delta", "Workers removed",         lambda x: f"{x:.0f}"),
    ]

    for ax, (key, title, xcol, xlabel, xfmt) in zip(axes, configs):
        df = tables[key]
        x = df[xcol].values

        ax2 = ax.twinx()

        cost_vals = df["total_cost"].values / 1000
        svc_vals = df["service_level"].values * 100

        line1 = ax.plot(x, cost_vals, color="#1565c0", lw=2, marker="o", ms=5, label=f"Cost (k{currency})")
        line2 = ax2.plot(x, svc_vals, color="#e53935", lw=2, marker="s", ms=5,
                         linestyle="--", label="Service level (%)")

        ax2.axhline(90, color="#e53935", ls=":", lw=0.8, alpha=0.6)
        ax2.set_ylim(0, 110)
        ax2.set_ylabel("Service level (%)", color="#e53935", fontsize=9)
        ax2.tick_params(axis="y", colors="#e53935")

        infeasible = df[~df["feasible"]]
        if not infeasible.empty:
            ax.axvspan(infeasible[xcol].min() - 0.02, infeasible[xcol].max() + 0.02,
                       alpha=0.12, color="red", label="Infeasible")

        ax.set_title(title, fontsize=10)
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_ylabel(f"Total cost (k{currency})", color="#1565c0", fontsize=9)
        ax.tick_params(axis="y", colors="#1565c0")

        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax.legend(lines, labels, fontsize=8, loc="upper left")
        ax.grid(ls=":", alpha=0.4)

        # Custom x-tick labels
        ax.set_xticks(x)
        ax.set_xticklabels([xfmt(v) for v in x], rotation=35, ha="right", fontsize=8)

    plt.tight_layout()
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {output_path}")
    if show:
        plt.show()
    return fig
