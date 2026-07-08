from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

from .utils import is_timestamp_column, likely_identifier_column


@dataclass(frozen=True)
class FeatureSelectionResult:
    selected_features: list[str]
    rf_importances: pd.DataFrame  # columns: feature, importance
    mi_scores: pd.DataFrame  # columns: feature, mi
    dropped_columns: list[str]


def _drop_unwanted_columns(
    df: pd.DataFrame,
    label_col: str,
    drop_timestamp_columns: bool,
    logger: logging.Logger,
) -> tuple[pd.DataFrame, list[str]]:
    drop_cols: list[str] = []
    for c in df.columns:
        if c == label_col:
            continue
        if likely_identifier_column(c):
            drop_cols.append(c)
        elif drop_timestamp_columns and is_timestamp_column(c):
            drop_cols.append(c)

    drop_cols = sorted(set(drop_cols))
    if drop_cols:
        logger.info("Dropping %d identifier/timestamp columns before selection.", len(drop_cols))
        return df.drop(columns=drop_cols), drop_cols
    return df, []


def select_features(
    df: pd.DataFrame,
    label_col: str,
    logger: logging.Logger,
    n_keep: int,
    correlation_threshold: float,
    rf_estimators: int,
    top_k_rf: int,
    top_k_mi: int,
    sample_rows: int | None = None,
    drop_timestamp_columns: bool = True,
) -> FeatureSelectionResult:
    """
    Automatic feature selection for NIDS tabular data.

    Process:
    - Drop IDs/flow identifiers and timestamp-like columns
    - Keep only numeric columns for RF/MI (protocol indicator columns in this dataset are numeric)
    - Drop highly correlated features (|corr| >= threshold)
    - Rank features using Random Forest importance AND Mutual Information
    - Select ~15–20 best features (configurable)
    """
    df2, dropped = _drop_unwanted_columns(df, label_col, drop_timestamp_columns, logger)

    # Keep numeric features only for selection
    X = df2.drop(columns=[label_col], errors="ignore")
    y = df2[label_col]

    # Subsample for very large datasets (RF/MI on millions of rows is unnecessarily slow for ranking).
    if sample_rows is not None and df2.shape[0] > sample_rows:
        try:
            frac = float(sample_rows / df2.shape[0])
            df2 = df2.groupby(y.astype(str), group_keys=False).apply(
                lambda g: g.sample(max(1, int(len(g) * frac)), random_state=42)
            )
            X = df2.drop(columns=[label_col], errors="ignore")
            y = df2[label_col]
            logger.info("Feature selection sampling: using %d rows (stratified).", int(df2.shape[0]))
        except Exception as exc:  # noqa: BLE001
            df2 = df2.sample(n=sample_rows, random_state=42)
            X = df2.drop(columns=[label_col], errors="ignore")
            y = df2[label_col]
            logger.warning("Stratified sampling failed (%s). Used random sample=%d rows.", exc, sample_rows)

    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    Xn = X[numeric_cols].copy()

    # Impute numerics for selection (RF/MI cannot handle NaNs)
    imputer = SimpleImputer(strategy="median")
    Xn_imp = pd.DataFrame(imputer.fit_transform(Xn), columns=numeric_cols)

    # Correlation pruning
    corr = Xn_imp.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop_corr = [c for c in upper.columns if any(upper[c] >= correlation_threshold)]
    if to_drop_corr:
        logger.info("Dropping %d highly correlated numeric features.", len(to_drop_corr))
    X_sel = Xn_imp.drop(columns=to_drop_corr, errors="ignore")

    # Encode y for sklearn
    le = LabelEncoder()
    y_enc = le.fit_transform(y.astype(str))

    # Random Forest importance
    rf = RandomForestClassifier(
        n_estimators=rf_estimators,
        random_state=42,
        n_jobs=1,
        class_weight="balanced_subsample",
    )
    rf.fit(X_sel, y_enc)
    rf_df = (
        pd.DataFrame({"feature": X_sel.columns, "importance": rf.feature_importances_})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

    # Mutual information
    mi = mutual_info_classif(X_sel, y_enc, random_state=42)
    mi_df = pd.DataFrame({"feature": X_sel.columns, "mi": mi}).sort_values("mi", ascending=False).reset_index(drop=True)

    top_rf = rf_df.head(top_k_rf)["feature"].tolist()
    top_mi = mi_df.head(top_k_mi)["feature"].tolist()

    # Combine rankings: prefer features scoring well on both signals
    combined = list(dict.fromkeys(top_rf + top_mi))  # preserve order, de-dupe
    selected = combined[:n_keep]

    dropped_all = sorted(set(dropped + to_drop_corr))
    logger.info("Selected %d features for modeling.", len(selected))
    return FeatureSelectionResult(
        selected_features=selected,
        rf_importances=rf_df,
        mi_scores=mi_df,
        dropped_columns=dropped_all,
    )

