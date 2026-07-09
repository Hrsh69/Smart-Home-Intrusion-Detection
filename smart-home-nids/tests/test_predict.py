"""Tests for the NIDS prediction engine — real 18-feature / 9-class schema.

All tests use mocked artifacts (no real model file required) so they run
in CI without the 400 MB rf_model.pkl present.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import joblib
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, RobustScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.predict import NIDSPredictor, PredictionResult, SEVERITY_MAP, SEVERITY_ORDER

# ── Canonical constants (must match trained artifacts) ────────────────────────

#: Feature names in the exact order stored in selected_features.pkl.
#: Verified by running: python3 -c "import joblib; print(joblib.load('models/selected_features.pkl'))"
FEATURES_18 = [
    "iat", "avg", "header_length", "header_bytes_per_packet",
    "rst_count", "tot_size", "max", "tot_sum", "flow_duration",
    "urg_count", "rate", "bytes_per_packet", "variance",
    "packets_per_second", "protocol_type", "min", "rst_ratio", "urg_ratio",
]

#: Sorted class names as stored in label_encoder.classes_.
#: Verified by running: python3 -c "import joblib; print(list(joblib.load('models/label_encoder.pkl').classes_))"
CLASS_NAMES_9 = [
    "BENIGN", "BruteForce", "DDoS", "DoS", "Malware",
    "Mirai", "Recon", "Spoofing", "WebAttack",
]

N_FEATURES = len(FEATURES_18)   # 18
N_CLASSES = len(CLASS_NAMES_9)  # 9


# ── Minimal stub model ────────────────────────────────────────────────────────

class _DummyRF:
    """Minimal RandomForest stub — always predicts class 0 (BENIGN)."""

    def predict(self, X):
        return np.zeros(len(X), dtype=int)

    def predict_proba(self, X):
        proba = np.full((len(X), N_CLASSES), 0.1 / (N_CLASSES - 1))
        proba[:, 0] = 0.9
        return proba


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def mock_predictor(tmp_path_factory):
    """Build all 5 artifact files using real sklearn objects, load predictor."""
    tmp = tmp_path_factory.mktemp("models")

    # Fit imputer + scaler on dummy data (same shape as real training)
    rng = np.random.default_rng(42)
    X_dummy = pd.DataFrame(
        rng.random((40, N_FEATURES)), columns=FEATURES_18
    )
    imputer = SimpleImputer(strategy="median")
    X_imp = pd.DataFrame(imputer.fit_transform(X_dummy), columns=FEATURES_18)
    scaler = RobustScaler()
    scaler.fit(X_imp)

    le = LabelEncoder()
    le.fit(CLASS_NAMES_9)

    joblib.dump(_DummyRF(), tmp / "rf_model.pkl")
    joblib.dump(FEATURES_18, tmp / "selected_features.pkl")
    joblib.dump(le, tmp / "label_encoder.pkl")
    joblib.dump(imputer, tmp / "numeric_imputer.pkl")
    joblib.dump(scaler, tmp / "scaler.pkl")

    predictor = NIDSPredictor(models_dir=tmp)
    predictor.load()
    return predictor


# ── Severity mapping tests ────────────────────────────────────────────────────

class TestSeverityMapping:
    def test_benign_is_info(self):
        assert SEVERITY_MAP["BENIGN"] == "Info"

    def test_ddos_is_critical(self):
        assert SEVERITY_MAP["DDoS"] == "Critical"

    def test_mirai_is_critical(self):
        assert SEVERITY_MAP["Mirai"] == "Critical"

    def test_malware_is_critical(self):
        assert SEVERITY_MAP["Malware"] == "Critical"

    def test_bruteforce_is_high(self):
        assert SEVERITY_MAP["BruteForce"] == "High"

    def test_recon_is_low(self):
        assert SEVERITY_MAP["Recon"] == "Low"

    def test_severity_order_monotonic(self):
        assert SEVERITY_ORDER["Critical"] > SEVERITY_ORDER["High"]
        assert SEVERITY_ORDER["High"] > SEVERITY_ORDER["Medium"]
        assert SEVERITY_ORDER["Medium"] > SEVERITY_ORDER["Low"]
        assert SEVERITY_ORDER["Low"] > SEVERITY_ORDER["Info"]

    def test_all_9_classes_have_severity(self):
        for cls in CLASS_NAMES_9:
            assert cls in SEVERITY_MAP, f"No severity mapping for class: {cls}"


# ── Schema tests ──────────────────────────────────────────────────────────────

class TestPredictorSchema:
    def test_feature_count_is_18(self, mock_predictor):
        assert len(mock_predictor.feature_names) == N_FEATURES

    def test_class_count_is_9(self, mock_predictor):
        assert len(mock_predictor.class_names) == N_CLASSES

    def test_all_9_classes_present(self, mock_predictor):
        assert set(mock_predictor.class_names) == set(CLASS_NAMES_9)

    def test_model_version_format(self, mock_predictor):
        version = mock_predictor.model_version
        assert isinstance(version, str)
        assert version.startswith("rf-"), f"Unexpected version format: {version!r}"
        assert len(version) == 11  # "rf-" + 8 hex chars


# ── Prediction correctness tests ──────────────────────────────────────────────

class TestPredictSingle:
    def test_returns_prediction_result(self, mock_predictor):
        features = {feat: 1.0 for feat in FEATURES_18}
        result = mock_predictor.predict_single(features)
        assert isinstance(result, PredictionResult)

    def test_label_is_valid_class(self, mock_predictor):
        features = {feat: 1.0 for feat in FEATURES_18}
        result = mock_predictor.predict_single(features)
        assert result.label in CLASS_NAMES_9

    def test_confidence_in_unit_range(self, mock_predictor):
        features = {feat: 1.0 for feat in FEATURES_18}
        result = mock_predictor.predict_single(features)
        assert 0.0 <= result.confidence <= 1.0

    def test_top3_has_3_entries(self, mock_predictor):
        features = {feat: 1.0 for feat in FEATURES_18}
        result = mock_predictor.predict_single(features)
        assert len(result.top_3) == 3

    def test_top3_probabilities_sum_leq_1(self, mock_predictor):
        features = {feat: 1.0 for feat in FEATURES_18}
        result = mock_predictor.predict_single(features)
        total = sum(p for _, p in result.top_3)
        assert total <= 1.0 + 1e-6

    def test_probabilities_cover_all_9_classes(self, mock_predictor):
        features = {feat: 1.0 for feat in FEATURES_18}
        result = mock_predictor.predict_single(features)
        assert set(result.probabilities.keys()) == set(CLASS_NAMES_9)

    def test_predicted_class_idx_is_valid(self, mock_predictor):
        features = {feat: 1.0 for feat in FEATURES_18}
        result = mock_predictor.predict_single(features)
        assert isinstance(result.predicted_class_idx, int)
        assert 0 <= result.predicted_class_idx < N_CLASSES

    def test_severity_matches_label(self, mock_predictor):
        features = {feat: 1.0 for feat in FEATURES_18}
        result = mock_predictor.predict_single(features)
        expected_severity = SEVERITY_MAP.get(result.label, "Unknown")
        assert result.severity == expected_severity

    def test_handles_empty_feature_dict(self, mock_predictor):
        """All-NaN input: imputer should fill with training medians, not raise."""
        result = mock_predictor.predict_single({})
        assert isinstance(result, PredictionResult)
        assert result.label in CLASS_NAMES_9

    def test_handles_partial_feature_dict(self, mock_predictor):
        """Partial input: missing values filled by imputer."""
        result = mock_predictor.predict_single({"iat": 0.05, "protocol_type": 6.0})
        assert isinstance(result, PredictionResult)
        assert result.label in CLASS_NAMES_9

    def test_handles_nan_values(self, mock_predictor):
        """Explicit NaN values: imputer should handle gracefully."""
        features = {feat: float("nan") for feat in FEATURES_18}
        result = mock_predictor.predict_single(features)
        assert isinstance(result, PredictionResult)

    def test_severity_helper(self, mock_predictor):
        assert mock_predictor.get_severity("BENIGN") == "Info"
        assert mock_predictor.get_severity("DDoS") == "Critical"
        assert mock_predictor.get_severity("UnknownClass") == "Unknown"

    def test_severity_score_helper(self, mock_predictor):
        assert mock_predictor.get_severity_score("DDoS") == 4
        assert mock_predictor.get_severity_score("BENIGN") == 0
