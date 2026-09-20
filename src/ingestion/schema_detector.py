"""
Schema detection for user-uploaded datasets.

Two modes:
  detect_schema_rules() — keyword + type inference, no API key required
  detect_schema_llm()   — Claude analyses headers + sample rows (needs ANTHROPIC_API_KEY)

Both return a ColumnMapping that the user can review and override in the dashboard.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd


@dataclass
class ColumnMapping:
    date_col: str
    demand_col: str
    sku_col: Optional[str] = None        # None = no grouping column
    confidence: dict = field(default_factory=dict)  # col → "high" | "medium" | "low"
    warnings: list[str] = field(default_factory=list)


# ── keyword banks ─────────────────────────────────────────────────────
_DATE_KEYWORDS  = ["date", "week", "time", "period", "month", "day", "year", "dt", "timestamp"]
_DEMAND_KEYWORDS = ["demand", "qty", "quantity", "volume", "units", "sales", "orders",
                    "forecast", "actual", "consumption", "throughput"]
# Volume-like names beat money/other measures; these names are never a demand signal.
_DEMAND_STRONG  = ["demand", "qty", "quantity", "volume", "units"]
_DEMAND_EXCLUDE = ["days", "id", "price", "ratio", "rate", "profit", "discount", "latitude",
                   "longitude", "zip", "risk", "status"]
_SKU_KEYWORDS   = ["sku", "product", "item", "ref", "code", "id", "article",
                   "category", "site", "location", "warehouse", "store"]


def _score(col: str, keywords: list[str]) -> int:
    col_lower = col.lower()
    return sum(kw in col_lower for kw in keywords)


def _demand_score(col: str) -> int:
    col_lower = col.lower()
    if any(re.search(rf"(^|[^a-z]){kw}([^a-z]|$)", col_lower) for kw in _DEMAND_EXCLUDE):
        return -1
    return _score(col, _DEMAND_KEYWORDS) + 2 * _score(col, _DEMAND_STRONG)


def _date_score(col: str) -> int:
    # "order date" is the demand event; "shipping date" is a downstream timestamp.
    return _score(col, _DATE_KEYWORDS) + (1 if "order" in col.lower() else 0)


def detect_schema_rules(df: pd.DataFrame) -> ColumnMapping:
    """
    Infer date / demand / sku columns using column names and dtypes.
    Always returns a mapping even if uncertain — confidence flags tell the UI
    which fields to highlight for user review.
    """
    cols = list(df.columns)
    warnings = []

    # ── 1. Date column ────────────────────────────────────────────────
    # Prefer columns already parsed as datetime, then keyword score
    datetime_cols = [c for c in cols if pd.api.types.is_datetime64_any_dtype(df[c])]
    if datetime_cols:
        date_col = datetime_cols[0]
        date_conf = "high"
    else:
        # Try to coerce each column and pick the best keyword match
        parseable = []
        for c in cols:
            if pd.api.types.is_numeric_dtype(df[c]):
                continue  # numbers would "parse" as epoch timestamps
            try:
                sample = df[c].dropna().iloc[:20]
                if len(sample) and pd.to_datetime(sample, errors="coerce").notna().mean() > 0.9:
                    parseable.append(c)
            except Exception:
                pass
        scored = sorted(parseable, key=_date_score, reverse=True)
        if scored:
            date_col = scored[0]
            date_conf = "high" if _score(date_col, _DATE_KEYWORDS) > 0 else "medium"
        elif cols:
            date_col = cols[0]
            date_conf = "low"
            warnings.append(f"No obvious date column found — defaulting to '{date_col}'.")
        else:
            raise ValueError("Dataset has no columns.")

    # ── 2. Demand column ─────────────────────────────────────────────
    numeric_cols = [c for c in cols if c != date_col
                    and pd.api.types.is_numeric_dtype(df[c])]
    if not numeric_cols:
        # Try to coerce
        numeric_cols = [c for c in cols if c != date_col
                        and pd.to_numeric(df[c], errors="coerce").notna().mean() > 0.8]

    scored_num = sorted(numeric_cols, key=_demand_score, reverse=True)
    if scored_num and _demand_score(scored_num[0]) > 0:
        demand_col = scored_num[0]
        demand_conf = "high"
    elif scored_num:
        demand_col = scored_num[0]
        demand_conf = "medium"
    elif numeric_cols:
        demand_col = numeric_cols[0]
        demand_conf = "low"
        warnings.append(f"No obvious demand column found — defaulting to '{demand_col}'.")
    else:
        raise ValueError("No numeric column found to use as demand.")

    # ── 3. SKU / grouping column (optional) ─────────────────────────
    remaining = [c for c in cols if c not in (date_col, demand_col)]
    cat_cols = [c for c in remaining
                if not pd.api.types.is_numeric_dtype(df[c])
                and not pd.api.types.is_datetime64_any_dtype(df[c])]
    scored_cat = sorted(cat_cols, key=lambda c: _score(c, _SKU_KEYWORDS), reverse=True)

    # Integer identifiers (item=1..50, store=1..10): named like a SKU, few distinct values
    id_cols = [c for c in remaining
               if pd.api.types.is_integer_dtype(df[c])
               and _score(c, _SKU_KEYWORDS) > 0
               and 2 <= df[c].nunique() <= 500]
    # Finest product-like grain first (item/product/sku over store/site), then more values
    id_cols.sort(key=lambda c: (_score(c, ["sku", "item", "product", "article"]), df[c].nunique()),
                 reverse=True)

    if scored_cat and _score(scored_cat[0], _SKU_KEYWORDS) > 0:
        sku_col = scored_cat[0]
        sku_conf = "high"
    elif id_cols:
        sku_col = id_cols[0]
        sku_conf = "medium"
        warnings.append(f"Using numeric column '{sku_col}' as SKU identifier "
                        f"({df[sku_col].nunique()} distinct values) — other columns are summed.")
    elif cat_cols:
        # Only use a categorical column as SKU if it has reasonable cardinality
        for c in cat_cols:
            n_unique = df[c].nunique()
            if 2 <= n_unique <= 200:
                sku_col = c
                sku_conf = "medium"
                break
        else:
            sku_col = None
            sku_conf = "low"
    else:
        sku_col = None
        sku_conf = "low"

    if sku_col is None:
        warnings.append("No grouping column detected — treating dataset as a single series.")

    return ColumnMapping(
        date_col=date_col,
        demand_col=demand_col,
        sku_col=sku_col,
        confidence={date_col: date_conf, demand_col: demand_conf,
                    **(({sku_col: sku_conf}) if sku_col else {})},
        warnings=warnings,
    )


def detect_schema_llm(df: pd.DataFrame, api_key: str) -> ColumnMapping:
    """
    Ask Claude to identify date / demand / sku columns from headers + sample rows.
    Falls back to detect_schema_rules() on any failure.
    """
    import anthropic

    sample = df.head(5).to_csv(index=False)
    prompt = f"""You are a data analyst. Identify the correct columns in this CSV dataset for a supply chain demand forecasting pipeline.

Column names: {list(df.columns)}

Sample rows (first 5):
{sample}

Reply ONLY with a JSON object (no markdown):
{{
  "date_col": "<column name for dates/periods>",
  "demand_col": "<column name for demand/quantity/volume>",
  "sku_col": "<column name for product/SKU/site grouping, or null if none>",
  "reasoning": "<one sentence explaining your choices>"
}}

Rules:
- date_col must contain dates, weeks, or time periods
- demand_col must be numeric and represent a quantity (sales, orders, units, etc.)
- sku_col is optional — only set it if there is clearly a grouping/product column
- Use exact column names from the list above
"""

    client = anthropic.Anthropic(api_key=api_key)
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=300,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = response.content[0].text.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return detect_schema_rules(df)

    cols = list(df.columns)
    date_col   = data.get("date_col") if data.get("date_col") in cols else None
    demand_col = data.get("demand_col") if data.get("demand_col") in cols else None
    sku_col    = data.get("sku_col") if data.get("sku_col") in cols else None

    if not date_col or not demand_col:
        return detect_schema_rules(df)

    return ColumnMapping(
        date_col=date_col,
        demand_col=demand_col,
        sku_col=sku_col,
        confidence={date_col: "high", demand_col: "high",
                    **(({sku_col: "high"}) if sku_col else {})},
        warnings=[f"LLM reasoning: {data.get('reasoning', '')}"],
    )
