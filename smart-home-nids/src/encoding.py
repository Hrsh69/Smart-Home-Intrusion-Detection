from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

from .utils import pick_device_column


@dataclass(frozen=True)
class EncodingArtifacts:
    protocol_encoder_path: Optional[Path]
    device_encoder_path: Optional[Path]
    encoded_df: pd.DataFrame
    protocol_column: Optional[str]
    device_column: Optional[str]


def _is_low_cardinality(series: pd.Series, max_unique: int = 50) -> bool:
    return int(series.nunique(dropna=True)) <= max_unique


def detect_protocol_column(df: pd.DataFrame) -> Optional[str]:
    """
    Attempt to detect a 'protocol' categorical column.

    CIC exports vary; sometimes it's a numeric protocol id, sometimes a string.
    Here we look for common names.
    """
    candidates = ("protocol", "protocol_type", "proto", "protocoltype")
    cols = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c in cols:
            return cols[c]
    return None


def fit_categorical_encoders(
    df: pd.DataFrame,
    models_dir: Path,
    device_column_candidates: tuple[str, ...],
    logger: logging.Logger,
) -> EncodingArtifacts:
    """
    Encode categorical columns and labels.

    Required saved artifacts:
    - models/label_encoder.pkl
    - models/protocol_encoder.pkl (if protocol column exists)
    - models/device_encoder.pkl (if device column exists)
    """
    models_dir.mkdir(parents=True, exist_ok=True)
    out = df.copy()

    protocol_col = detect_protocol_column(out)
    protocol_encoder_path: Optional[Path] = None
    if protocol_col and protocol_col in out.columns:
        # Only treat as categorical if it's not too high-cardinality
        if out[protocol_col].dtype == "object" or _is_low_cardinality(out[protocol_col], max_unique=256):
            enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
            out[[protocol_col]] = enc.fit_transform(out[[protocol_col]].astype(str).fillna("Unknown"))
            protocol_encoder_path = models_dir / "protocol_encoder.pkl"
            joblib.dump(enc, protocol_encoder_path)
            logger.info("Encoded protocol column '%s'.", protocol_col)

    device_col = pick_device_column(out, device_column_candidates)
    device_encoder_path: Optional[Path] = None
    if device_col and device_col in out.columns:
        if out[device_col].dtype == "object" or _is_low_cardinality(out[device_col], max_unique=2048):
            enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
            out[[device_col]] = enc.fit_transform(out[[device_col]].astype(str).fillna("Unknown"))
            device_encoder_path = models_dir / "device_encoder.pkl"
            joblib.dump(enc, device_encoder_path)
            logger.info("Encoded device column '%s'.", device_col)

    # Auto-encode any remaining object/category columns with a generic ordinal encoding.
    # (Not required to be saved individually per prompt; protocol/device/label are explicitly saved.)
    cat_cols = out.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    if cat_cols:
        generic_enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        out[cat_cols] = generic_enc.fit_transform(out[cat_cols].astype(str).fillna("Unknown"))
        joblib.dump(generic_enc, models_dir / "categorical_encoder.pkl")
        logger.info("Encoded %d additional categorical columns.", len(cat_cols))

    out = out.replace([np.inf, -np.inf], np.nan)

    return EncodingArtifacts(
        protocol_encoder_path=protocol_encoder_path,
        device_encoder_path=device_encoder_path,
        encoded_df=out,
        protocol_column=protocol_col,
        device_column=device_col,
    )


def transform_categorical_columns(
    df: pd.DataFrame,
    models_dir: Path,
    protocol_column: Optional[str],
    device_column: Optional[str],
) -> pd.DataFrame:
    """
    Transform categorical columns using previously saved encoders.
    """
    out = df.copy()

    protocol_path = models_dir / "protocol_encoder.pkl"
    if protocol_column and protocol_column in out.columns and protocol_path.exists():
        enc = joblib.load(protocol_path)
        out[[protocol_column]] = enc.transform(out[[protocol_column]].astype(str).fillna("Unknown"))

    device_path = models_dir / "device_encoder.pkl"
    if device_column and device_column in out.columns and device_path.exists():
        enc = joblib.load(device_path)
        out[[device_column]] = enc.transform(out[[device_column]].astype(str).fillna("Unknown"))

    generic_path = models_dir / "categorical_encoder.pkl"
    cat_cols = out.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    if cat_cols and generic_path.exists():
        enc = joblib.load(generic_path)
        out[cat_cols] = enc.transform(out[cat_cols].astype(str).fillna("Unknown"))

    return out.replace([np.inf, -np.inf], np.nan)

