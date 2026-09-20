"""
Forecasting models: Weighted Moving Average, Holt-Winters ETS, SARIMA.

Each model follows the same interface:
    fit(train: pd.Series) -> self
    predict(horizon: int) -> np.ndarray
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing as HWExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX


class MovingAverage:
    """Weighted moving average baseline (more recent weeks get higher weight)."""

    def __init__(self, window: int = 12):
        self.window = window
        self._last_values: Optional[np.ndarray] = None

    def fit(self, train: pd.Series) -> "MovingAverage":
        self._last_values = train.values[-self.window :]
        return self

    def predict(self, horizon: int) -> np.ndarray:
        if self._last_values is None:
            raise RuntimeError("Call fit() before predict().")
        weights = np.arange(1, len(self._last_values) + 1, dtype=float)
        weights /= weights.sum()
        level = float(np.dot(weights, self._last_values))
        return np.full(horizon, level)

    def __repr__(self) -> str:
        return f"MovingAverage(window={self.window})"


class ExponentialSmoothing:
    """Holt-Winters triple exponential smoothing (additive trend + seasonality)."""

    def __init__(self, seasonal_periods: int = 52, trend: str = "add", seasonal: str = "add"):
        self.seasonal_periods = seasonal_periods
        self.trend = trend
        self.seasonal = seasonal
        self._fitted = None

    def fit(self, train: pd.Series) -> "ExponentialSmoothing":
        # Need at least 2 full seasonal cycles for additive seasonality
        sp = self.seasonal_periods
        if len(train) < 2 * sp:
            sp = None  # fall back to simple exponential smoothing
            seasonal = None
        else:
            seasonal = self.seasonal

        model = HWExponentialSmoothing(
            train,
            trend=self.trend,
            seasonal=seasonal,
            seasonal_periods=sp,
            initialization_method="estimated",
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._fitted = model.fit(optimized=True)
        return self

    def predict(self, horizon: int) -> np.ndarray:
        if self._fitted is None:
            raise RuntimeError("Call fit() before predict().")
        return self._fitted.forecast(horizon).values

    def __repr__(self) -> str:
        return f"ExponentialSmoothing(seasonal_periods={self.seasonal_periods})"


class SARIMAModel:
    """Seasonal ARIMA (SARIMA) via statsmodels SARIMAX."""

    def __init__(
        self,
        order: tuple = (1, 1, 1),
        seasonal_order: tuple = (1, 1, 1, 52),
    ):
        self.order = order
        self.seasonal_order = seasonal_order
        self._fitted = None

    def fit(self, train: pd.Series) -> "SARIMAModel":
        # Use a simpler seasonal order if not enough data
        sp = self.seasonal_order[3]
        if len(train) < 2 * sp:
            seasonal_order = (0, 0, 0, 0)
        else:
            seasonal_order = self.seasonal_order

        model = SARIMAX(
            train,
            order=self.order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._fitted = model.fit(disp=False)
        return self

    def predict(self, horizon: int) -> np.ndarray:
        if self._fitted is None:
            raise RuntimeError("Call fit() before predict().")
        forecast = self._fitted.forecast(steps=horizon)
        return np.maximum(forecast.values, 0)

    def __repr__(self) -> str:
        return f"SARIMAModel(order={self.order}, seasonal_order={self.seasonal_order})"
