import pandas as pd
import numpy as np
import pytest
from unittest.mock import MagicMock
from pathlib import Path

from src.predict import NIDSPredictor, PredictionResult, SEVERITY_MAP, SEVERITY_ORDER

class TestSeverityMapping:
    def test_benign_is_info(self):
        assert SEVERITY_MAP.get("BENIGN") == "Info"

    def test_ddos_is_critical(self):
        assert SEVERITY_MAP.get("DDoS") == "Critical"

    def test_severity_order_values(self):
        assert SEVERITY_ORDER["Critical"] > SEVERITY_ORDER["High"]
        assert SEVERITY_ORDER["High"] > SEVERITY_ORDER["Medium"]

class DummyModel:
    def predict(self, X):
        return np.array([0] * len(X))
    def predict_proba(self, X):
        return np.array([[0.9, 0.05, 0.02, 0.02, 0.01]] * len(X))

class DummyEncoder:
    def transform(self, X):
        return [0]

class TestPredictorWithMockedModel:
    @pytest.fixture
    def mock_predictor(self, tmp_path):
        import joblib
        
        mock_model = DummyModel()
        mock_device_enc = DummyEncoder()
        mock_proto_enc = DummyEncoder()

        joblib.dump(mock_model, tmp_path / "rf_model.pkl")
        joblib.dump(mock_device_enc, tmp_path / "device_encoder.pkl")
        joblib.dump(mock_proto_enc, tmp_path / "protocol_encoder.pkl")

        predictor = NIDSPredictor(models_dir=tmp_path)
        predictor.load()
        return predictor

    def test_predict_single(self, mock_predictor):
        features = {
            "device_type": "smart_camera",
            "protocol": "TCP",
            "dst_port": 443,
            "flow_duration_ms": 1000,
            "packet_count": 50,
            "byte_rate_bps": 15000,
            "flag_syn_ratio": 0.1,
            "unique_dst_ips_per_src": 1
        }
        result = mock_predictor.predict_single(features)
        assert isinstance(result, PredictionResult)
        assert result.label == "BENIGN"
        assert result.severity == "Info"
        assert len(result.top_3) == 3

    def test_class_names(self, mock_predictor):
        assert mock_predictor.class_names == ["BENIGN", "DDoS", "Port Scan", "Brute Force", "Botnet C2"]

    def test_feature_names(self, mock_predictor):
        assert len(mock_predictor.feature_names) == 8
