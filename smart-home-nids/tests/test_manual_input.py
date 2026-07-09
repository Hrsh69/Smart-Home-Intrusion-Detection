"""Smoke-test: manual prediction round-trip (predict → DB insert).

Run from smart-home-nids/:
    python tests/test_manual_input.py

Not a pytest test — uses real models and DB.
Kept in tests/ for discoverability but excluded from CI pytest run
(it requires the trained model artifacts in models/).
"""

import sys
from pathlib import Path

# Ensure smart-home-nids/ is on the path regardless of where the script is run from
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.database import NIDSDatabase
from src.predict import NIDSPredictor
from src.alerts import AlertManager

db = NIDSDatabase()
predictor = NIDSPredictor()
alert_mgr = AlertManager(db)

# Realistic BENIGN flow (all 18 real features)
features = {
    "iat": 0.04,
    "avg": 520.0,
    "header_length": 2400.0,
    "header_bytes_per_packet": 48.0,
    "rst_count": 0.0,
    "tot_size": 52000.0,
    "max": 1460.0,
    "tot_sum": 52000.0,
    "flow_duration": 1.5,
    "urg_count": 0.0,
    "rate": 40.0,
    "bytes_per_packet": 520.0,
    "variance": 85000.0,
    "packets_per_second": 40.0,
    "protocol_type": 6.0,
    "min": 40.0,
    "rst_ratio": 0.0,
    "urg_ratio": 0.0,
}

result = predictor.predict_single(features)
print("Label:        ", result.label)
print("Confidence:   ", f"{result.confidence:.1%}")
print("Severity:     ", result.severity)
print("Top-3:        ", result.top_3)

det_id = db.insert_detection(
    prediction=result.label,
    confidence=result.confidence,
    severity=result.severity,
    features=features,
    source_ip="192.168.1.10",
    device_id="test_device_01",
)
print("Inserted to DB with ID:", det_id)
