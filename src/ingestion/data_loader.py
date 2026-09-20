"""Load a CSV or Excel file and normalise it to the standard pipeline format."""

from __future__ import annotations

import io
import re
from typing import Optional

import pandas as pd

from .schema_detector import ColumnMapping


_WEEK_COL = re.compile(r"^(?:W|Week_?|Wk_?)(\d+)$", re.IGNORECASE)


def unpivot_wide(df: pd.DataFrame, start_date: str = "2020-01-06") -> pd.DataFrame:
    """
    Convert a wide table (one row per product, one column per week: W0, W1, ...)
    to long format with columns: date, demand, sku.

    The file carries no calendar dates, so week N is dated start_date + N weeks
    (a Monday, matching the pipeline's W-MON frequency). Returns df unchanged if
    it doesn't look wide. Derived columns (MIN, MAX, Normalized *) are dropped.
    """
    week_cols = [c for c in df.columns if _WEEK_COL.match(str(c))]
    if len(week_cols) < 8:
        return df

    id_col = next((c for c in df.columns if c not in week_cols), None)
    if id_col is None:
        return df

    long = df[[id_col] + week_cols].melt(id_vars=id_col, var_name="week", value_name="demand")
    offsets = long["week"].map(lambda c: int(_WEEK_COL.match(str(c)).group(1)))
    long["date"] = pd.Timestamp(start_date) + pd.to_timedelta(offsets * 7, unit="D")
    long = long.rename(columns={id_col: "sku"})
    return long[["date", "demand", "sku"]].sort_values(["sku", "date"]).reset_index(drop=True)


def load_file(uploaded_file) -> pd.DataFrame:
    """
    Accept a Streamlit UploadedFile (or a file path string) and return a raw DataFrame.
    Supports CSV and Excel (.xlsx / .xls). Wide weekly tables are unpivoted.
    """
    return unpivot_wide(_read_raw(uploaded_file))


# Personal data that must never reach the pipeline, the UI or the LLM prompts.
_PII = re.compile(r"(e-?mail|password|first ?name|last ?name|fname|lname|street|zip ?code)", re.IGNORECASE)
_ENCODINGS = ["utf-8", "utf-8-sig", "latin-1"]


def _drop_pii(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(columns=[c for c in df.columns if _PII.search(str(c))])


def _read_csv_bytes(content: bytes) -> pd.DataFrame:
    for enc in _ENCODINGS:
        for sep in [",", ";", "	"]:
            try:
                df = pd.read_csv(io.BytesIO(content), sep=sep, encoding=enc, low_memory=False)
            except UnicodeDecodeError:
                break  # wrong encoding: try the next one
            except Exception:
                continue
            if len(df.columns) > 1:
                return df
    return pd.read_csv(io.BytesIO(content), encoding="latin-1")


def _read_raw(uploaded_file) -> pd.DataFrame:
    if hasattr(uploaded_file, "name"):
        name = uploaded_file.name.lower()
        content = uploaded_file.read()
        if name.endswith((".xlsx", ".xls")):
            df = pd.read_excel(io.BytesIO(content))
        else:
            df = _read_csv_bytes(content)
    else:
        path = str(uploaded_file)
        if path.endswith((".xlsx", ".xls")):
            df = pd.read_excel(path)
        else:
            with open(path, "rb") as f:
                df = _read_csv_bytes(f.read())
    return _drop_pii(df)


def _needs_weekly_aggregation(out: pd.DataFrame) -> bool:
    keys = ["date", "sku"] if "sku" in out.columns else ["date"]
    if out.duplicated(keys).any():
        return True
    gaps = out.drop_duplicates("date")["date"].sort_values().diff().dropna()
    return len(gaps) > 0 and gaps.median() < pd.Timedelta(days=6)


def _to_weekly(out: pd.DataFrame) -> pd.DataFrame:
    """
    Sum transactions into Monday-start weeks (per sku), filling empty weeks with 0.
    Drops the partial first/last week and any trailing weeks below 50% of the
    median weekly total (an export cut-off, not a real demand collapse).
    """
    lo, hi = out["date"].min().normalize(), out["date"].max().normalize()
    sku = out["sku"] if "sku" in out.columns else "ALL"
    week = out["date"].dt.normalize() - pd.to_timedelta(out["date"].dt.weekday, unit="D")
    wk = (out.assign(week=week, sku=sku)
             .groupby(["sku", "week"])["demand"].sum()
             .unstack("sku", fill_value=0)
             .asfreq("W-MON", fill_value=0))

    wk = wk[(wk.index >= lo) & (wk.index + pd.Timedelta(days=6) <= hi)]  # full weeks only
    total = wk.sum(axis=1)
    if len(total) > 8:
        floor = 0.5 * total.median()
        while len(total) > 8 and total.iloc[-1] < floor:
            total = total.iloc[:-1]
        wk = wk.loc[total.index]

    long = wk.stack().rename("demand").reset_index()
    long.columns = ["date", "sku", "demand"]
    if "sku" not in out.columns:
        long = long.drop(columns="sku")
    return long


def apply_mapping(df: pd.DataFrame, mapping: ColumnMapping) -> pd.DataFrame:
    """
    Normalise the raw DataFrame to columns: date, demand[, sku].
    Parses dates, coerces demand to float, drops rows with NaN in key columns.
    Transaction-level or daily data is aggregated to weekly totals.
    """
    out = pd.DataFrame()
    out["date"]   = pd.to_datetime(df[mapping.date_col], errors="coerce")
    out["demand"] = pd.to_numeric(df[mapping.demand_col], errors="coerce")

    if mapping.sku_col:
        out["sku"] = df[mapping.sku_col].astype(str)

    out = out.dropna(subset=["date", "demand"]).reset_index(drop=True)
    if out.empty:
        raise ValueError(
            f"No usable rows: could not parse '{mapping.date_col}' as dates and "
            f"'{mapping.demand_col}' as numbers. Check the column mapping."
        )
    if _needs_weekly_aggregation(out):
        out = _to_weekly(out)

    return out.sort_values("date").reset_index(drop=True)


def usable_skus(df: pd.DataFrame, min_weeks: int = 60) -> tuple[list[str], int]:
    """
    SKUs with enough active (demand > 0) weeks to be forecast, largest volume first.

    min_weeks is capped at 75% of the longest series so short datasets (e.g. 52
    weeks) still keep their best SKUs. Returns (usable skus, number hidden).
    If nothing qualifies, every SKU is returned so the user isn't locked out.
    """
    weeks = df[df["demand"] > 0].groupby("sku").size()
    all_skus = df.groupby("sku")["demand"].sum().sort_values(ascending=False).index.tolist()
    threshold = min(min_weeks, int(0.75 * weeks.max())) if len(weeks) else min_weeks
    keep = [s for s in all_skus if weeks.get(s, 0) >= threshold]
    if not keep:
        return all_skus, 0
    return keep, len(all_skus) - len(keep)
