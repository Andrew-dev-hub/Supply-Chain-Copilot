"""
ForecastingPipeline: orchestrates train/test split, model fitting, evaluation,
and deviation detection for a single demand time series.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .metrics import compute_mape, compute_mae, compute_rmse, detect_deviations
from .models import ExponentialSmoothing, MovingAverage, SARIMAModel


@dataclass
class ForecastResult:
    model_name: str
    sku: str
    train_end: pd.Timestamp
    test_dates: pd.DatetimeIndex
    actual: np.ndarray
    forecast: np.ndarray
    mape: float
    mae: float
    rmse: float
    deviations: pd.DataFrame

    def summary(self) -> str:
        lines = [
            f"Model : {self.model_name}",
            f"SKU   : {self.sku}",
            f"Test period: {self.test_dates[0].date()} to {self.test_dates[-1].date()} "
            f"({len(self.test_dates)} weeks)",
            f"MAPE  : {self.mape:.2f}%",
            f"MAE   : {self.mae:.1f}",
            f"RMSE  : {self.rmse:.1f}",
            f"Significant deviations (>{self._threshold}%): {len(self.deviations)}",
        ]
        return "\n".join(lines)

    # Set by pipeline after construction
    _threshold: float = field(default=20.0, repr=False)


class ForecastingPipeline:
    """
    Runs multiple forecasting models on a demand time series and returns
    structured results including accuracy metrics and deviation flags.

    Parameters
    ----------
    test_weeks : int
        Number of weeks held out for evaluation.
    deviation_threshold_pct : float
        Percentage error above which a week is flagged as a significant deviation.
    models : list | None
        Model instances to evaluate. Defaults to WMA, ETS, and SARIMA.
    """

    def __init__(
        self,
        test_weeks: int = 13,
        deviation_threshold_pct: float = 20.0,
        models: Optional[List] = None,
    ):
        self.test_weeks = test_weeks
        self.deviation_threshold_pct = deviation_threshold_pct
        self.models = models or [
            MovingAverage(window=12),
            ExponentialSmoothing(seasonal_periods=52),
            SARIMAModel(order=(1, 1, 1), seasonal_order=(1, 1, 1, 52)),
        ]

    def run(
        self,
        df: pd.DataFrame,
        date_col: str = "date",
        demand_col: str = "demand",
        sku_col: Optional[str] = "sku",
    ) -> Dict[str, List[ForecastResult]]:
        """
        Run the pipeline on *df*.

        Returns a dict mapping SKU name → list of ForecastResult (one per model).
        If sku_col is None or absent, treats the whole series as a single SKU.
        """
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.sort_values(date_col)

        if sku_col and sku_col in df.columns:
            skus = df[sku_col].unique()
        else:
            df["_sku"] = "ALL"
            sku_col = "_sku"
            skus = ["ALL"]

        results: Dict[str, List[ForecastResult]] = {}

        for sku in skus:
            subset = df[df[sku_col] == sku].set_index(date_col)[demand_col]
            subset = subset.sort_index().asfreq("W-MON", method="ffill")

            if len(subset) <= self.test_weeks:
                print(f"[WARNING] SKU {sku}: not enough data ({len(subset)} rows). Skipping.")
                continue

            train = subset.iloc[: -self.test_weeks]
            test = subset.iloc[-self.test_weeks :]

            sku_results = []
            for model in self.models:
                try:
                    model.fit(train)
                    forecast = model.predict(self.test_weeks)
                    actual = test.values

                    mape = compute_mape(actual, forecast)
                    mae = compute_mae(actual, forecast)
                    rmse = compute_rmse(actual, forecast)
                    deviations = detect_deviations(
                        test.index.to_series(),
                        actual,
                        forecast,
                        threshold_pct=self.deviation_threshold_pct,
                    )

                    result = ForecastResult(
                        model_name=repr(model),
                        sku=str(sku),
                        train_end=train.index[-1],
                        test_dates=test.index,
                        actual=actual,
                        forecast=forecast,
                        mape=mape,
                        mae=mae,
                        rmse=rmse,
                        deviations=deviations,
                    )
                    result._threshold = self.deviation_threshold_pct
                    sku_results.append(result)
                    print(f"  [{sku}] {repr(model):60s}  MAPE={mape:.2f}%")

                except Exception as exc:
                    print(f"  [{sku}] {repr(model)} FAILED: {exc}")

            results[str(sku)] = sku_results

        return results

    def best_model(self, results: Dict[str, List[ForecastResult]]) -> Dict[str, ForecastResult]:
        """Returns the model with the lowest MAPE for each SKU."""
        best = {}
        for sku, res_list in results.items():
            valid = [r for r in res_list if not np.isnan(r.mape)]
            if valid:
                best[sku] = min(valid, key=lambda r: r.mape)
        return best
