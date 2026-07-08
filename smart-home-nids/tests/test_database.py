"""Unit tests for the SQLite database layer."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.database import NIDSDatabase


@pytest.fixture
def db(tmp_path):
    """Create a fresh in-memory-like database for each test."""
    db_path = tmp_path / "test_nids.db"
    return NIDSDatabase(db_path=db_path)


class TestDetections:
    """Test detection CRUD operations."""

    def test_insert_and_retrieve(self, db):
        det_id = db.insert_detection(
            prediction="DDoS",
            confidence=0.95,
            severity="Critical",
            source_ip="192.168.1.100",
        )
        assert det_id > 0

        recent = db.get_recent_detections(limit=10)
        assert len(recent) == 1
        assert recent[0]["prediction"] == "DDoS"
        assert recent[0]["confidence"] == 0.95

    def test_get_detections_count(self, db):
        assert db.get_detections_count() == 0

        db.insert_detection(prediction="BENIGN", confidence=0.99, severity="Info")
        db.insert_detection(prediction="DDoS", confidence=0.85, severity="Critical")

        assert db.get_detections_count() == 2

    def test_search_by_prediction(self, db):
        db.insert_detection(prediction="BENIGN", confidence=0.99, severity="Info")
        db.insert_detection(prediction="DDoS", confidence=0.85, severity="Critical")
        db.insert_detection(prediction="DDoS", confidence=0.75, severity="Critical")

        results = db.search_detections(prediction="DDoS")
        assert len(results) == 2

    def test_search_by_severity(self, db):
        db.insert_detection(prediction="BENIGN", confidence=0.99, severity="Info")
        db.insert_detection(prediction="DDoS", confidence=0.85, severity="Critical")

        results = db.search_detections(severity="Critical")
        assert len(results) == 1

    def test_search_by_ip(self, db):
        db.insert_detection(prediction="DDoS", confidence=0.9, severity="Critical", source_ip="10.0.0.5")
        db.insert_detection(prediction="BENIGN", confidence=0.99, severity="Info", source_ip="192.168.1.1")

        results = db.search_detections(source_ip="10.0")
        assert len(results) == 1

    def test_attack_distribution(self, db):
        db.insert_detection(prediction="BENIGN", confidence=0.99, severity="Info")
        db.insert_detection(prediction="BENIGN", confidence=0.98, severity="Info")
        db.insert_detection(prediction="DDoS", confidence=0.85, severity="Critical")

        dist = db.get_attack_distribution()
        assert dist["BENIGN"] == 2
        assert dist["DDoS"] == 1

    def test_severity_distribution(self, db):
        db.insert_detection(prediction="DDoS", confidence=0.9, severity="Critical")
        db.insert_detection(prediction="DoS", confidence=0.8, severity="High")
        db.insert_detection(prediction="BENIGN", confidence=0.99, severity="Info")

        dist = db.get_severity_distribution()
        assert "Critical" in dist
        assert "High" in dist
        assert "Info" in dist

    def test_features_json_stored(self, db):
        features = {"iat": 0.1, "avg": 200.5}
        db.insert_detection(
            prediction="Recon", confidence=0.7, severity="Low",
            features=features,
        )
        recent = db.get_recent_detections(limit=1)
        import json
        stored = json.loads(recent[0]["features_json"])
        assert stored["iat"] == 0.1


class TestDevices:
    """Test device registry operations."""

    def test_device_auto_created(self, db):
        db.insert_detection(
            prediction="DDoS", confidence=0.9, severity="Critical",
            device_id="cam_01", source_ip="192.168.1.10",
        )

        devices = db.get_all_devices()
        assert len(devices) == 1
        assert devices[0]["id"] == "cam_01"
        assert devices[0]["total_flows"] == 1
        assert devices[0]["malicious_flows"] == 1

    def test_device_risk_score_updates(self, db):
        db.insert_detection(prediction="BENIGN", confidence=0.99, severity="Info", device_id="dev_01")
        db.insert_detection(prediction="BENIGN", confidence=0.98, severity="Info", device_id="dev_01")
        db.insert_detection(prediction="DDoS", confidence=0.85, severity="Critical", device_id="dev_01")

        devices = db.get_all_devices()
        assert len(devices) == 1
        dev = devices[0]
        assert dev["total_flows"] == 3
        assert dev["malicious_flows"] == 1
        assert 30 <= dev["risk_score"] <= 35  # ~33.3%

    def test_update_device_name(self, db):
        db.insert_detection(prediction="BENIGN", confidence=0.99, severity="Info", device_id="dev_01")
        db.update_device_name("dev_01", "Living Room Camera", "Camera")

        devices = db.get_all_devices()
        assert devices[0]["name"] == "Living Room Camera"
        assert devices[0]["type"] == "Camera"

    def test_device_count(self, db):
        assert db.get_device_count() == 0
        db.insert_detection(prediction="BENIGN", confidence=0.99, severity="Info", device_id="dev_01")
        db.insert_detection(prediction="BENIGN", confidence=0.99, severity="Info", device_id="dev_02")
        assert db.get_device_count() == 2


class TestAlerts:
    """Test alert CRUD operations."""

    def test_insert_alert(self, db):
        det_id = db.insert_detection(prediction="DDoS", confidence=0.9, severity="Critical")
        alert_id = db.insert_alert(
            detection_id=det_id,
            alert_type="desktop",
            severity="Critical",
            message="DDoS attack detected!",
        )
        assert alert_id > 0

    def test_unacknowledged_alerts(self, db):
        det_id = db.insert_detection(prediction="DDoS", confidence=0.9, severity="Critical")
        db.insert_alert(det_id, "desktop", "Critical", "Alert 1")
        db.insert_alert(det_id, "email", "High", "Alert 2")

        unack = db.get_unacknowledged_alerts()
        assert len(unack) == 2

    def test_acknowledge_alert(self, db):
        det_id = db.insert_detection(prediction="DDoS", confidence=0.9, severity="Critical")
        alert_id = db.insert_alert(det_id, "desktop", "Critical", "Alert 1")

        db.acknowledge_alert(alert_id)
        unack = db.get_unacknowledged_alerts()
        assert len(unack) == 0

    def test_alert_count(self, db):
        det_id = db.insert_detection(prediction="DDoS", confidence=0.9, severity="Critical")
        db.insert_alert(det_id, "desktop", "Critical", "A")
        db.insert_alert(det_id, "email", "High", "B")

        assert db.get_alert_count() == 2
        assert db.get_alert_count(acknowledged=False) == 2
        assert db.get_alert_count(acknowledged=True) == 0


class TestDashboardStats:
    """Test aggregated dashboard statistics."""

    def test_empty_stats(self, db):
        stats = db.get_dashboard_stats()
        assert stats["total_flows"] == 0
        assert stats["detection_rate"] == 0.0

    def test_stats_after_detections(self, db):
        db.insert_detection(prediction="BENIGN", confidence=0.99, severity="Info")
        db.insert_detection(prediction="DDoS", confidence=0.85, severity="Critical")
        db.insert_detection(prediction="Mirai", confidence=0.80, severity="Critical")

        stats = db.get_dashboard_stats()
        assert stats["total_flows"] == 3
        assert stats["benign_flows"] == 1
        assert stats["threat_flows"] == 2
        assert abs(stats["detection_rate"] - 66.67) < 0.1


class TestCleanup:
    """Test database cleanup operations."""

    def test_clear_all(self, db):
        db.insert_detection(prediction="DDoS", confidence=0.9, severity="Critical", device_id="dev_01")
        det_id = db.get_recent_detections(limit=1)[0]["id"]
        db.insert_alert(det_id, "desktop", "Critical", "Alert")

        db.clear_all()

        assert db.get_detections_count() == 0
        assert db.get_device_count() == 0
        assert db.get_alert_count() == 0


class TestExport:
    """Test CSV export."""

    def test_export_detections(self, db, tmp_path):
        db.insert_detection(prediction="BENIGN", confidence=0.99, severity="Info")
        db.insert_detection(prediction="DDoS", confidence=0.85, severity="Critical")

        export_path = tmp_path / "export.csv"
        count = db.export_detections_csv(str(export_path))

        assert count == 2
        assert export_path.exists()

        import pandas as pd
        df = pd.read_csv(export_path)
        assert len(df) == 2
