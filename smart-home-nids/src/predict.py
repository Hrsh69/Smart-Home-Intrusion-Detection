"""Prediction engine for Smart Home NIDS.

This module is the **single bridge** between the original pre-trained models
(rf_model.pkl, device_encoder.pkl, protocol_encoder.pkl) and the dashboard.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Any

import joblib
import numpy as np
import pandas as pd

logger = logging.getLogger("smart_home_nids.predict")

# ── Severity classification ─────────────────────────────────────────────────
SEVERITY_MAP: dict[str, str] = {
    "BENIGN": "Info",
    "Recon": "Low",
    "Port Scan": "Low",
    "Spoofing": "Medium",
    "Brute Force": "High",
    "WebAttack": "High",
    "DoS": "High",
    "DDoS": "Critical",
    "Mirai": "Critical",
    "Malware": "Critical",
    "Botnet C2": "Critical",
}

SEVERITY_ORDER = {"Critical": 4, "High": 3, "Medium": 2, "Low": 1, "Info": 0}

@dataclass
class PredictionResult:
    """Result of a single prediction."""
    label: str
    confidence: float
    severity: str
    probabilities: dict[str, float]
    top_3: list[tuple[str, float]]

class NIDSPredictor:
    """Loads original pre-trained models and provides prediction interfaces."""

    def __init__(self, models_dir: Optional[Path] = None) -> None:
        if models_dir is None:
            models_dir = Path(__file__).resolve().parents[2] / "models"
        self.models_dir = models_dir

        self._model = None
        self._device_encoder = None
        self._protocol_encoder = None
        
        self._features = [
            'device_enc', 'protocol_enc', 'dst_port', 'flow_duration_ms',
            'packet_count', 'byte_rate_bps', 'flag_syn_ratio', 'unique_dst_ips_per_src'
        ]
        
        # User's model outputs [0, 1, 2, 3, 4]
        self.class_mapping = {
            0: "BENIGN",
            1: "DDoS",
            2: "Port Scan",
            3: "Brute Force",
            4: "Botnet C2"
        }
        self._loaded = False

    def load(self) -> None:
        """Load all model artifacts from disk."""
        self._model = joblib.load(self.models_dir / "rf_model.pkl")
        self._device_encoder = joblib.load(self.models_dir / "device_encoder.pkl")
        self._protocol_encoder = joblib.load(self.models_dir / "protocol_encoder.pkl")
        self._loaded = True
        logger.info("NIDS Predictor loaded successfully.")

    def _ensure_loaded(self) -> None:
        if not self._loaded:
            self.load()

    @property
    def feature_names(self) -> list[str]:
        return list(self._features)

    @property
    def class_names(self) -> list[str]:
        return list(self.class_mapping.values())

    @property
    def model(self):
        self._ensure_loaded()
        return self._model

    def predict_single(self, features: dict[str, Any]) -> PredictionResult:
        """Predict a single network flow."""
        self._ensure_loaded()

        # Build feature array
        try:
            device_str = str(features.get('device_type', 'Unknown'))
            device_enc = self._device_encoder.transform([device_str])[0]
        except ValueError:
            device_enc = 0 # Default if unknown

        try:
            proto_str = str(features.get('protocol', 'Unknown'))
            proto_enc = self._protocol_encoder.transform([proto_str])[0]
        except ValueError:
            proto_enc = 0

        x_array = [
            device_enc,
            proto_enc,
            float(features.get('dst_port', 0)),
            float(features.get('flow_duration_ms', 0)),
            float(features.get('packet_count', 0)),
            float(features.get('byte_rate_bps', 0)),
            float(features.get('flag_syn_ratio', 0)),
            float(features.get('unique_dst_ips_per_src', 0))
        ]

        X_df = pd.DataFrame([x_array], columns=self._features)

        # Predict
        y_pred = self._model.predict(X_df)[0]
        y_proba = self._model.predict_proba(X_df)[0]

        label = self.class_mapping.get(y_pred, f"Unknown ({y_pred})")
        confidence = float(np.max(y_proba))
        severity = SEVERITY_MAP.get(label, "Unknown")

        # Build probability dict
        prob_dict = {
            self.class_mapping.get(i, f"Class_{i}"): round(float(p), 4)
            for i, p in enumerate(y_proba)
        }

        # Top 3
        top_indices = np.argsort(y_proba)[::-1][:3]
        top_3 = [
            (self.class_mapping.get(i, f"Class_{i}"), round(float(y_proba[i]), 4))
            for i in top_indices
        ]

        return PredictionResult(
            label=label,
            confidence=round(confidence, 4),
            severity=severity,
            probabilities=prob_dict,
            top_3=top_3,
        )

    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """Predict on a batch of flows (not implemented for mixed string/float raw df yet)."""
        raise NotImplementedError("Batch prediction is not implemented for the raw unencoded dataframe.")

    def get_severity(self, label: str) -> str:
        return SEVERITY_MAP.get(label, "Unknown")

    def get_severity_score(self, label: str) -> int:
        severity = self.get_severity(label)
        return SEVERITY_ORDER.get(severity, 0)
