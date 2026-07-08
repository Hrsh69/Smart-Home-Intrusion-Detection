"""Tests for model training and validation."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


class TestTrainingReport:
    """Validate the training report if model has been trained."""

    @pytest.fixture
    def report(self):
        report_path = PROJECT_ROOT / "reports" / "training_report.json"
        if not report_path.exists():
            pytest.skip("Training report not found — model not yet trained.")
        with open(report_path, "r") as f:
            return json.load(f)

    def test_accuracy_above_threshold(self, report):
        """Model accuracy should be above 80%."""
        assert report["accuracy"] > 0.80, f"Accuracy too low: {report['accuracy']:.2%}"

    def test_f1_weighted_above_threshold(self, report):
        """Weighted F1 should be above 80%."""
        assert report["f1_weighted"] > 0.80

    def test_all_classes_present(self, report):
        """All 9 attack categories should be in the report."""
        expected = {"BENIGN", "DDoS", "DoS", "Mirai", "Recon", "Spoofing",
                    "BruteForce", "WebAttack", "Malware"}
        actual = set(report.get("class_names", []))
        assert expected == actual, f"Missing classes: {expected - actual}"

    def test_confusion_matrix_dimensions(self, report):
        """Confusion matrix should be 9×9."""
        cm = report.get("confusion_matrix", [])
        assert len(cm) == 9
        assert all(len(row) == 9 for row in cm)

    def test_feature_importance_count(self, report):
        """Feature importance should have 18 features."""
        feat_imp = report.get("feature_importance", {})
        assert len(feat_imp) == 18

    def test_training_time_reasonable(self, report):
        """Training should complete in under 10 minutes."""
        assert report.get("training_time_seconds", 0) < 600

    def test_per_class_report_complete(self, report):
        """Each class should have precision, recall, f1-score."""
        per_class = report.get("per_class_report", {})
        for cls in report.get("class_names", []):
            assert cls in per_class, f"Missing class: {cls}"
            assert "precision" in per_class[cls]
            assert "recall" in per_class[cls]
            assert "f1-score" in per_class[cls]


class TestModelArtifacts:
    """Validate model artifact files."""

    def test_rf_model_exists(self):
        model_path = PROJECT_ROOT / "models" / "rf_model.pkl"
        if not model_path.exists():
            pytest.skip("Model not yet trained.")
        assert model_path.stat().st_size > 1000  # Should be a non-trivial file

    def test_model_loadable(self):
        model_path = PROJECT_ROOT / "models" / "rf_model.pkl"
        if not model_path.exists():
            pytest.skip("Model not yet trained.")

        import joblib
        model = joblib.load(model_path)
        assert hasattr(model, "predict")
        assert hasattr(model, "predict_proba")

    def test_label_encoder_classes(self):
        le_path = PROJECT_ROOT / "models" / "label_encoder.pkl"
        if not le_path.exists():
            pytest.skip("Label encoder not found.")

        import joblib
        le = joblib.load(le_path)
        assert len(le.classes_) == 9

    def test_selected_features_count(self):
        feat_path = PROJECT_ROOT / "models" / "selected_features.pkl"
        if not feat_path.exists():
            pytest.skip("Selected features not found.")

        import joblib
        features = joblib.load(feat_path)
        assert len(features) == 18


class TestProcessedData:
    """Validate processed dataset files."""

    def test_train_csv_exists(self):
        path = PROJECT_ROOT / "data" / "processed" / "processed_train.csv"
        assert path.exists(), "processed_train.csv not found"

    def test_test_csv_exists(self):
        path = PROJECT_ROOT / "data" / "processed" / "processed_test.csv"
        assert path.exists(), "processed_test.csv not found"

    def test_train_csv_columns(self):
        import pandas as pd
        path = PROJECT_ROOT / "data" / "processed" / "processed_train.csv"
        if not path.exists():
            pytest.skip("Train CSV not found.")

        df = pd.read_csv(path, nrows=5)
        assert "attack_category" in df.columns
        assert len(df.columns) == 19  # 18 features + target
