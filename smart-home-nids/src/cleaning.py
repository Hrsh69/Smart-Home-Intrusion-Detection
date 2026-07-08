from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict

import numpy as np
import pandas as pd

from .utils import (
    coerce_numeric,
    drop_impossible_numeric_values,
    replace_inf_with_nan,
    standardize_column_name,
)


@dataclass
class CleaningReport:
    initial_rows: int
    final_rows: int
    duplicates_removed: int
    empty_rows_removed: int
    rows_removed_missing: int
    columns_dropped: list[str] = field(default_factory=list)
    missing_values_before: Dict[str, int] = field(default_factory=dict)
    missing_values_after: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "initial_rows": self.initial_rows,
            "final_rows": self.final_rows,
            "duplicates_removed": self.duplicates_removed,
            "empty_rows_removed": self.empty_rows_removed,
            "rows_removed_missing": self.rows_removed_missing,
            "columns_dropped": self.columns_dropped,
            "missing_values_before": self.missing_values_before,
            "missing_values_after": self.missing_values_after,
        }


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [standardize_column_name(c) for c in out.columns]
    return out


def remove_constant_columns(df: pd.DataFrame, logger: logging.Logger) -> tuple[pd.DataFrame, list[str]]:
    nunique = df.nunique(dropna=False)
    const_cols = nunique[nunique <= 1].index.tolist()
    if const_cols:
        logger.info("Dropping %d constant columns.", len(const_cols))
        return df.drop(columns=const_cols), const_cols
    return df, []


def clean_data(df: pd.DataFrame, label_col: str, logger: logging.Logger) -> tuple[pd.DataFrame, CleaningReport]:
    """
    Clean CIC IoT-2023 tabular data for ML.

    Key NIDS-specific cleaning:
    - Remove exact duplicates (common when merging multiple CSVs)
    - Replace inf with NaN (feature extraction artifacts)
    - Remove empty rows and rows missing label
    - Coerce numeric columns; drop rows with impossible negative values for counts/sizes/durations
    - Drop constant columns (non-informative)
    """
    df0 = df
    report = CleaningReport(
        initial_rows=int(df0.shape[0]),
        final_rows=int(df0.shape[0]),
        duplicates_removed=0,
        empty_rows_removed=0,
        rows_removed_missing=0,
    )

    df1 = df0.drop_duplicates()
    report.duplicates_removed = int(df0.shape[0] - df1.shape[0])

    # Remove completely empty rows (all NaN)
    df2 = df1.dropna(axis=0, how="all")
    report.empty_rows_removed = int(df1.shape[0] - df2.shape[0])

    df3 = replace_inf_with_nan(df2)

    # Standardize column names early so downstream modules are stable.
    df3 = standardize_columns(df3)
    label_col_std = standardize_column_name(label_col)
    if label_col_std not in df3.columns:
        raise KeyError(f"Label column '{label_col}' not found after standardization. Found={list(df3.columns)[:10]}...")

    # Missing values snapshot
    report.missing_values_before = df3.isna().sum().astype(int).to_dict()

    # Drop rows missing label
    before = df3.shape[0]
    df4 = df3.dropna(subset=[label_col_std])
    report.rows_removed_missing += int(before - df4.shape[0])

    # Coerce numerics
    numeric_cols = df4.select_dtypes(include=[np.number]).columns.tolist()
    non_numeric_cols = [c for c in df4.columns if c not in numeric_cols]
    df5 = coerce_numeric(df4, numeric_cols)

    # Drop rows with impossible values in numeric columns (negative counts/sizes/durations)
    df6 = drop_impossible_numeric_values(df5, numeric_cols=numeric_cols)

    # Drop constant columns
    df7, const_cols = remove_constant_columns(df6, logger)
    report.columns_dropped.extend(const_cols)

    # Final missing snapshot
    report.missing_values_after = df7.isna().sum().astype(int).to_dict()

    # Strategy: keep NaNs for now; later we will impute with median (numerical) / most_frequent (categorical)
    # because many CIC flow features can be missing depending on protocol.

    report.final_rows = int(df7.shape[0])
    logger.info(
        "Cleaning complete. Rows %d -> %d (dupes=%d, empty=%d, missing_removed=%d).",
        report.initial_rows,
        report.final_rows,
        report.duplicates_removed,
        report.empty_rows_removed,
        report.rows_removed_missing,
    )
    return df7, report

