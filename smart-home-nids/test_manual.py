import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.database import NIDSDatabase
from src.predict import NIDSPredictor
from src.alerts import AlertManager

db = NIDSDatabase()
predictor = NIDSPredictor()
alert_mgr = AlertManager(db)

features = {
    "device_type": "smart_camera",
    "protocol": "TCP",
    "dst_port": 443,
    "flow_duration_ms": 1200.0,
    "packet_count": 50,
    "byte_rate_bps": 15000.0,
    "flag_syn_ratio": 0.1,
    "unique_dst_ips_per_src": 1.0,
}

result = predictor.predict_single(features)
print("Label:", result.label)
print("Confidence:", result.confidence)
print("Severity:", result.severity)
print("Probabilities:", result.probabilities)

det_id = db.insert_detection(
    prediction=result.label,
    confidence=result.confidence,
    severity=result.severity,
    features=features,
    source_ip="192.168.1.10",
    device_id="smart_camera_01"
)
print("Inserted to DB with ID:", det_id)
