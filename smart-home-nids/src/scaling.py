from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler


@dataclass(frozen=True)
class ScalingArtifacts:
    scaler_path: Path
    scaler_name: str
    feature_columns: list[str]
    X_scaled: pd.DataFrame


def fit_transform_scaler(
    X: pd.DataFrame,
    models_dir: Path,
    scaler_name: str,
    logger: logging.Logger,
) -> ScalingArtifacts:
    """
    Scale numerical features and save scaler.

    Prefer RobustScaler (median/IQR) because CIC traffic features can be heavy-tailed
    (bursty DDoS, scans, and irregular IoT traffic generate extreme values).
    """
    models_dir.mkdir(parents=True, exist_ok=True)

    feature_columns = X.columns.tolist()
    imputer = SimpleImputer(strategy="median")
    X_imp = pd.DataFrame(imputer.fit_transform(X), columns=feature_columns)
    joblib.dump(imputer, models_dir / "numeric_imputer.pkl")

    scalers = {
        "standard": StandardScaler(),
        "robust": RobustScaler(with_centering=True, with_scaling=True, quantile_range=(25.0, 75.0)),
        "minmax": MinMaxScaler(),
    }
    if scaler_name not in scalers:
        raise ValueError(f"Unknown scaler '{scaler_name}'. Choose from {list(scalers)}")

    scaler = scalers[scaler_name]
    X_scaled = pd.DataFrame(scaler.fit_transform(X_imp), columns=feature_columns)

    scaler_path = models_dir / "scaler.pkl"
    joblib.dump(scaler, scaler_path)

    logger.info(
        "Scaler selected: %s. Rationale: %s",
        scaler_name,
        "RobustScaler handles heavy-tailed traffic/outliers best"
        if scaler_name == "robust"
        else "Selected via config (RobustScaler recommended for NIDS outliers)",
    )

    X_scaled = X_scaled.replace([np.inf, -np.inf], np.nan)
    return ScalingArtifacts(
        scaler_path=scaler_path,
        scaler_name=scaler_name,
        feature_columns=feature_columns,
        X_scaled=X_scaled,
    )

