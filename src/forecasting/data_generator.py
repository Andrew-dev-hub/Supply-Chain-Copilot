"""Generates a synthetic weekly demand dataset for a logistics/retail scenario."""

import numpy as np
import pandas as pd


def generate_demand_dataset(
    start_date: str = "2023-01-02",
    n_weeks: int = 104,
    n_skus: int = 3,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Produces a weekly demand time series with trend, seasonality, and noise.

    Returns a DataFrame with columns: date, sku, demand.
    """
    rng = np.random.default_rng(seed)

    sku_configs = {
        "SKU-A": {"base": 500, "trend": 1.5, "seasonal_amp": 120, "noise_std": 30},
        "SKU-B": {"base": 300, "trend": 0.5, "seasonal_amp": 60, "noise_std": 20},
        "SKU-C": {"base": 800, "trend": -0.8, "seasonal_amp": 200, "noise_std": 50},
    }
    sku_names = list(sku_configs.keys())[:n_skus]

    dates = pd.date_range(start=start_date, periods=n_weeks, freq="W-MON")
    records = []

    for sku in sku_names:
        cfg = sku_configs[sku]
        t = np.arange(n_weeks)

        # Linear trend
        trend = cfg["trend"] * t

        # Yearly seasonality (52-week cycle) with a Q4 peak
        week_of_year = (dates.isocalendar().week.values - 1) / 52
        seasonality = cfg["seasonal_amp"] * np.sin(2 * np.pi * week_of_year - np.pi / 2)

        # Additional short holiday spike around week 48-52
        holiday_mask = (dates.isocalendar().week.values >= 48).astype(float)
        holiday_spike = 80 * holiday_mask * rng.uniform(0.7, 1.3, n_weeks)

        # Gaussian noise
        noise = rng.normal(0, cfg["noise_std"], n_weeks)

        demand = cfg["base"] + trend + seasonality + holiday_spike + noise
        demand = np.maximum(demand, 0).round(1)

        for i, d in enumerate(dates):
            records.append({"date": d, "sku": sku, "demand": demand[i]})

    return pd.DataFrame(records).sort_values(["sku", "date"]).reset_index(drop=True)


if __name__ == "__main__":
    df = generate_demand_dataset()
    out = "data/demand_data.csv"
    df.to_csv(out, index=False)
    print(f"Saved {len(df)} rows to {out}")
    print(df.groupby("sku")["demand"].describe().round(1))
