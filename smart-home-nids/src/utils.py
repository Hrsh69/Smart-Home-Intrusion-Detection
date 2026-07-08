from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd


def setup_logging(log_dir: Path, name: str = "smart_home_nids") -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        fmt = logging.Formatter(
            fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(fmt)
        logger.addHandler(ch)

        fh = logging.FileHandler(log_dir / "preprocessing.log")
        fh.setLevel(logging.INFO)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


def ensure_dirs(*dirs: Path) -> None:
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)


def standardize_column_name(col: str) -> str:
    s = str(col).strip().lower()
    s = s.replace("/", "_").replace("-", "_").replace(" ", "_")
    s = "".join(ch for ch in s if ch.isalnum() or ch == "_")
    s = "_".join(part for part in s.split("_") if part)
    return s


def safe_div(n: pd.Series | np.ndarray, d: pd.Series | np.ndarray, eps: float = 1e-9):
    return n / (d + eps)


def is_timestamp_column(name: str) -> bool:
    n = name.lower()
    return any(k in n for k in ("timestamp", "time", "date", "start", "end"))


def likely_identifier_column(name: str) -> bool:
    n = name.lower()
    return any(
        k in n
        for k in (
            "id",
            "flow_id",
            "flowid",
            "src_ip",
            "dst_ip",
            "source_ip",
            "destination_ip",
            "src_port",
            "dst_port",
            "mac",
            "uuid",
        )
    )


def coerce_numeric(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def replace_inf_with_nan(df: pd.DataFrame) -> pd.DataFrame:
    out = df.replace([np.inf, -np.inf], np.nan)
    return out


def drop_impossible_numeric_values(df: pd.DataFrame, numeric_cols: list[str]) -> pd.DataFrame:
    """
    Drop rows with impossible numeric values for network flow features.

    Rationale: CIC flow stats should not contain negative sizes, negative counts,
    or negative durations.
    """
    out = df.copy()
    for c in numeric_cols:
        lc = c.lower()
        if any(k in lc for k in ("duration", "time", "iat", "length", "size", "count", "number", "rate", "sum", "min", "max", "avg", "std", "variance", "radius", "weight", "covariance", "magnitude")):
            out = out[out[c].isna() | (out[c] >= 0)]
    return out


def memory_usage_mb(df: pd.DataFrame) -> float:
    return float(df.memory_usage(deep=True).sum() / (1024 * 1024))


def imbalance_ratio(y: pd.Series) -> float:
    vc = y.value_counts(dropna=False)
    if vc.empty:
        return math.inf
    return float(vc.max() / max(1, vc.min()))


def pick_device_column(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    cols = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols:
            return cols[cand.lower()]
    return None

