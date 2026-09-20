"""
IngestionPipeline: standalone Module 1.

Loads any CSV/Excel file, runs LLM-enhanced (Haiku) schema detection,
presents the mapping for user confirmation (CLI), then returns a
normalised DataFrame ready for the forecasting pipeline.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import pandas as pd

from .data_loader import load_file, apply_mapping
from .schema_detector import ColumnMapping, detect_schema_llm, detect_schema_rules


class IngestionPipeline:
    """
    Parameters
    ----------
    use_llm   : use Haiku for schema detection (falls back to rule-based if False or no key)
    api_key   : override ANTHROPIC_API_KEY env var
    """

    def __init__(self, use_llm: bool = True, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self.use_llm = use_llm and bool(self.api_key)

    def run(self, file_path: str, interactive: bool = True) -> tuple[pd.DataFrame, ColumnMapping]:
        """
        Load *file_path*, detect schema, optionally prompt user to confirm,
        and return (normalised_df, mapping).
        """
        print(f"Loading: {file_path}")
        raw = load_file(file_path)
        print(f"  Shape: {raw.shape}  |  Columns: {list(raw.columns)}")

        # Schema detection
        if self.use_llm:
            print("  Detecting schema with Haiku...")
            try:
                mapping = detect_schema_llm(raw, self.api_key)
                method = "LLM (Haiku)"
            except Exception as e:
                print(f"  LLM detection failed ({e}), falling back to rule-based.")
                mapping = detect_schema_rules(raw)
                method = "rule-based (fallback)"
        else:
            print("  Detecting schema (rule-based)...")
            mapping = detect_schema_rules(raw)
            method = "rule-based"

        print(f"  Method: {method}")
        self._print_mapping(mapping)

        # Interactive confirmation
        if interactive:
            mapping = self._confirm_mapping(mapping, raw)

        df = apply_mapping(raw, mapping)
        print(f"  Normalised shape: {df.shape}")

        if df.empty:
            raise ValueError("Mapping produced an empty DataFrame. Check column selection.")

        return df, mapping

    # ── CLI helpers ───────────────────────────────────────────────────

    def _print_mapping(self, mapping: ColumnMapping):
        conf_icon = {"high": "[OK]", "medium": "[~] ", "low": "[?] "}
        print("  Proposed mapping:")
        for field, col in [("date", mapping.date_col), ("demand", mapping.demand_col),
                            ("sku", mapping.sku_col or "(none)")]:
            icon = conf_icon.get(mapping.confidence.get(col, "low"), "[?] ")
            print(f"    {icon} {field:8s} -> {col}")
        for w in mapping.warnings:
            print(f"  [WARN] {w}")

    def _confirm_mapping(self, mapping: ColumnMapping, raw: pd.DataFrame) -> ColumnMapping:
        print("\n  Press Enter to accept, or type a new column name to override.")
        cols = list(raw.columns)

        date_col = self._prompt("date column", mapping.date_col, cols)
        demand_col = self._prompt("demand column", mapping.demand_col, cols)
        sku_input = self._prompt("SKU/group column (leave blank for none)",
                                 mapping.sku_col or "", cols + [""])
        sku_col = sku_input if sku_input else None

        return ColumnMapping(date_col=date_col, demand_col=demand_col, sku_col=sku_col)

    @staticmethod
    def _prompt(label: str, default: str, valid: list[str]) -> str:
        while True:
            val = input(f"  {label} [{default}]: ").strip()
            if val == "":
                return default
            if val in valid:
                return val
            print(f"    Not a valid column. Choose from: {valid}")
