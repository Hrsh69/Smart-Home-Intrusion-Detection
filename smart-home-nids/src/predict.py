"""Prediction engine for Smart Home NIDS.

Loads the real trained artifacts and provides a predict_single() interface
that mirrors the training preprocessing pipeline exactly:

    raw dict → impute (numeric_imputer.pkl) → scale (scaler.pkl)
             → predict (rf_model.pkl) → decode (label_encoder.pkl)

Feature order is read from selected_features.pkl at load time, so the
predictor automatically adapts if the model is retrained with different
features — no code changes required.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np
import pandas as pd

logger = logging.getLogger("smart_home_nids.predict")

# ── Severity classification ──────────────────────────────────────────────────
SEVERITY_MAP: dict[str, str] = {
    "BENIGN": "Info",
    "Recon": "Low",
    "Spoofing": "Medium",
    "BruteForce": "High",
    "WebAttack": "High",
    "DoS": "High",
    "DDoS": "Critical",
    "Mirai": "Critical",
    "Malware": "Critical",
}

SEVERITY_ORDER: dict[str, int] = {
    "Critical": 4,
    "High": 3,
    "Medium": 2,
    "Low": 1,
    "Info": 0,
}


@dataclass
class PredictionResult:
    """Result of a single network-flow classification."""

    label: str
    confidence: float
    severity: str
    probabilities: dict[str, float]
    top_3: list[tuple[str, float]]
    predicted_class_idx: int = field(default=0)


class NIDSPredictor:
    """Load the real trained artifacts and classify network flows.

    Artifacts expected in ``models_dir``:

    * ``rf_model.pkl``          — trained RandomForestClassifier
    * ``selected_features.pkl`` — ordered list[str] of feature names (18)
    * ``label_encoder.pkl``     — sklearn LabelEncoder (9 attack classes)
    * ``numeric_imputer.pkl``   — SimpleImputer(median) fit on training X
    * ``scaler.pkl``            — RobustScaler fit on training X
    """

    def __init__(self, models_dir: Optional[Path] = None) -> None:
        if models_dir is None:
            models_dir = Path(__file__).resolve().parents[1] / "models"
        self.models_dir = Path(models_dir)

        self._model = None
        self._imputer = None
        self._scaler = None
        self._label_encoder = None
        self._feature_names: list[str] = []
        self._loaded = False

    # ── Loading ──────────────────────────────────────────────────────────────

    def load(self) -> None:
        """Load all five model artifacts from disk."""
        self._model = joblib.load(self.models_dir / "rf_model.pkl")
        self._feature_names = list(
            joblib.load(self.models_dir / "selected_features.pkl")
        )
        self._label_encoder = joblib.load(self.models_dir / "label_encoder.pkl")
        self._imputer = joblib.load(self.models_dir / "numeric_imputer.pkl")
        self._scaler = joblib.load(self.models_dir / "scaler.pkl")
        self._loaded = True
        logger.info(
            "NIDS Predictor loaded — %d features, %d classes: %s",
            len(self._feature_names),
            len(self._label_encoder.classes_),
            list(self._label_encoder.classes_),
        )

    def _ensure_loaded(self) -> None:
        if not self._loaded:
            self.load()

    # ── Public properties ────────────────────────────────────────────────────

    @property
    def feature_names(self) -> list[str]:
        """Ordered list of the 18 feature names used by the model."""
        self._ensure_loaded()
        return list(self._feature_names)

    @property
    def class_names(self) -> list[str]:
        """List of attack-category strings (9 classes)."""
        self._ensure_loaded()
        return list(self._label_encoder.classes_)

    @property
    def model(self):
        """The underlying RandomForestClassifier."""
        self._ensure_loaded()
        return self._model

    @property
    def model_version(self) -> str:
        """Short version tag derived from rf_model.pkl mtime + size.

        Format: ``rf-<8-char hex>`` — changes every time the model is
        retrained, stable across runs if the file is unchanged.
        """
        model_path = self.models_dir / "rf_model.pkl"
        if not model_path.exists():
            return "unknown"
        stat = model_path.stat()
        tag = hashlib.md5(
            f"{stat.st_mtime}-{stat.st_size}".encode()
        ).hexdigest()[:8]
        return f"rf-{tag}"

    # ── Prediction ───────────────────────────────────────────────────────────

    def predict_single(self, features: dict[str, Any]) -> PredictionResult:
        """Classify a single network flow.

        Args:
            features: Mapping of feature name → raw numeric value.
                      Any feature not present in ``features`` is treated as
                      NaN and filled by the trained imputer (training median).

        Returns:
            :class:`PredictionResult` with label, confidence, severity,
            per-class probabilities, top-3 predictions, and the integer
            class index (used by SHAP explainability).
        """
        self._ensure_loaded()

        # Build feature row in the exact column order from training
        row = {
            feat: float(features.get(feat, np.nan))
            for feat in self._feature_names
        }
        X_raw = pd.DataFrame([row], columns=self._feature_names)

        # Apply imputation + scaling identical to training
        X_imp = pd.DataFrame(
            self._imputer.transform(X_raw),
            columns=self._feature_names,
        )
        X_scaled = pd.DataFrame(
            self._scaler.transform(X_imp),
            columns=self._feature_names,
        )

        # Classify
        y_pred_int = int(self._model.predict(X_scaled)[0])
        y_proba = self._model.predict_proba(X_scaled)[0]

        label: str = self._label_encoder.inverse_transform([y_pred_int])[0]
        confidence = float(np.max(y_proba))
        severity = SEVERITY_MAP.get(label, "Unknown")

        # Per-class probability dict
        prob_dict: dict[str, float] = {
            str(self._label_encoder.inverse_transform([i])[0]): round(float(p), 4)
            for i, p in enumerate(y_proba)
        }

        # Top-3 by probability
        top_indices = np.argsort(y_proba)[::-1][:3]
        top_3 = [
            (
                str(self._label_encoder.inverse_transform([int(i)])[0]),
                round(float(y_proba[i]), 4),
            )
            for i in top_indices
        ]

        return PredictionResult(
            label=label,
            confidence=round(confidence, 4),
            severity=severity,
            probabilities=prob_dict,
            top_3=top_3,
            predicted_class_idx=y_pred_int,
        )

    # ── Helpers ──────────────────────────────────────────────────────────────

    def get_severity(self, label: str) -> str:
        """Return the severity string for an attack-category label."""
        return SEVERITY_MAP.get(label, "Unknown")

    def get_severity_score(self, label: str) -> int:
        """Return a numeric severity score (0–4) for sorting/comparison."""
        return SEVERITY_ORDER.get(self.get_severity(label), 0)
