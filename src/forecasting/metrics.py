"""Forecast accuracy metrics and deviation detection."""

import numpy as np
import pandas as pd


def compute_mape(actual: np.ndarray, forecast: np.ndarray) -> float:
    """Mean Absolute Percentage Error; skips zero-actual rows to avoid division by zero."""
    actual, forecast = np.asarray(actual, float), np.asarray(forecast, float)
    mask = actual != 0
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs((actual[mask] - forecast[mask]) / actual[mask])) * 100)


def compute_mae(actual: np.ndarray, forecast: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(actual, float) - np.asarray(forecast, float))))


def compute_rmse(actual: np.ndarray, forecast: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(actual, float) - np.asarray(forecast, float)) ** 2)))


def detect_deviations(
    dates: pd.Series,
    actual: np.ndarray,
    forecast: np.ndarray,
    threshold_pct: float = 20.0,
) -> pd.DataFrame:
    """
    Returns rows where |actual - forecast| / actual exceeds threshold_pct %.
    threshold_pct: e.g. 20.0 means flag deviations larger than 20 %.
    """
    actual = np.asarray(actual, float)
    forecast = np.asarray(forecast, float)
    pct_error = np.where(actual != 0, np.abs(actual - forecast) / actual * 100, np.nan)

    df = pd.DataFrame(
        {
            "date": dates.values,
            "actual": actual,
            "forecast": forecast,
            "abs_error": np.abs(actual - forecast),
            "pct_error": pct_error,
        }
    )
    return df[df["pct_error"] > threshold_pct].reset_index(drop=True)
