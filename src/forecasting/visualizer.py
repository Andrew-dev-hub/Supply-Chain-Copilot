"""Plot forecast vs actual for a ForecastResult."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

from .pipeline import ForecastResult


def plot_forecast(
    result: ForecastResult,
    train_series: Optional[pd.Series] = None,
    output_path: Optional[str] = None,
    show: bool = True,
) -> plt.Figure:
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), gridspec_kw={"height_ratios": [3, 1]})
    fig.suptitle(
        f"Forecast vs Actual — {result.sku} | {result.model_name}",
        fontsize=13,
        fontweight="bold",
    )

    ax = axes[0]
    if train_series is not None:
        ax.plot(
            train_series.index, train_series.values,
            color="#90a4ae", linewidth=1.2, label="Historical (train)", alpha=0.7,
        )
        ax.axvline(result.train_end, color="#546e7a", linestyle="--", linewidth=1, label="Train / Test split")

    ax.plot(result.test_dates, result.actual, color="#1565c0", linewidth=2, marker="o",
            markersize=3, label="Actual (test)")
    ax.plot(result.test_dates, result.forecast, color="#e53935", linewidth=2,
            linestyle="--", marker="x", markersize=4, label="Forecast")

    # Highlight significant deviations
    if not result.deviations.empty:
        ax.scatter(
            result.deviations["date"], result.deviations["actual"],
            color="orange", zorder=5, s=60, label=f"Deviation >{result._threshold}%",
        )

    ax.set_ylabel("Demand (units)")
    ax.legend(loc="upper left", fontsize=9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")
    ax.grid(axis="y", linestyle=":", alpha=0.5)

    # Bottom panel: percentage error
    ax2 = axes[1]
    pct_err = np.where(
        result.actual != 0,
        (result.forecast - result.actual) / result.actual * 100,
        np.nan,
    )
    ax2.bar(result.test_dates, pct_err, width=5, color=["#e53935" if e > 0 else "#1565c0" for e in pct_err], alpha=0.7)
    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.axhline(result._threshold, color="orange", linestyle="--", linewidth=0.8, label=f"+{result._threshold}%")
    ax2.axhline(-result._threshold, color="orange", linestyle="--", linewidth=0.8, label=f"-{result._threshold}%")
    ax2.set_ylabel("% Error")
    ax2.set_ylim(-60, 60)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax2.xaxis.set_major_locator(mdates.MonthLocator())
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=30, ha="right")
    ax2.grid(axis="y", linestyle=":", alpha=0.5)
    ax2.legend(fontsize=8, loc="upper right")

    metrics_text = f"MAPE={result.mape:.1f}%  MAE={result.mae:.0f}  RMSE={result.rmse:.0f}"
    fig.text(0.99, 0.01, metrics_text, ha="right", fontsize=9, color="#555")

    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {output_path}")

    if show:
        plt.show()

    return fig


def plot_model_comparison(
    results: List[ForecastResult],
    output_path: Optional[str] = None,
    show: bool = True,
) -> plt.Figure:
    """Bar chart comparing MAPE across models for a single SKU."""
    names = [r.model_name.split("(")[0] for r in results]
    mapes = [r.mape for r in results]

    fig, ax = plt.subplots(figsize=(8, 4))
    colors = ["#1565c0", "#2e7d32", "#e53935", "#f57c00"]
    bars = ax.barh(names, mapes, color=colors[: len(names)], alpha=0.8)
    ax.bar_label(bars, fmt="%.1f%%", padding=4, fontsize=10)
    ax.set_xlabel("MAPE (%)")
    ax.set_title(f"Model Comparison — {results[0].sku}" if results else "Model Comparison")
    ax.set_xlim(0, max(mapes) * 1.25 if mapes else 1)
    ax.grid(axis="x", linestyle=":", alpha=0.5)
    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {output_path}")

    if show:
        plt.show()

    return fig
