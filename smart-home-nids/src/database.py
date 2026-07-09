"""SQLite persistence layer for the Smart Home NIDS.

Provides all CRUD operations for detections, devices, alerts, and statistics.

Usage:
    from src.database import NIDSDatabase
    db = NIDSDatabase()
    db.insert_detection(...)
    recent = db.get_recent_detections(limit=50)
"""

from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

from config.settings import SETTINGS


class NIDSDatabase:
    """Thread-safe SQLite database for NIDS operations."""

    def __init__(self, db_path: Optional[Path] = None) -> None:
        self.db_path = str(db_path or SETTINGS.DB_PATH)
        self._local = threading.local()
        self._init_tables()

    def _get_conn(self) -> sqlite3.Connection:
        """Get a thread-local connection."""
        if not hasattr(self._local, "conn") or self._local.conn is None:
            self._local.conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self._local.conn.row_factory = sqlite3.Row
            self._local.conn.execute("PRAGMA journal_mode=WAL")
            self._local.conn.execute("PRAGMA foreign_keys=ON")
        return self._local.conn

    @contextmanager
    def _cursor(self):
        conn = self._get_conn()
        cursor = conn.cursor()
        try:
            yield cursor
            conn.commit()
        except Exception:
            conn.rollback()
            raise

    def _init_tables(self) -> None:
        """Create tables if they don't exist."""
        with self._cursor() as cur:
            cur.executescript("""
                CREATE TABLE IF NOT EXISTS detections (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp   TEXT    NOT NULL,
                    source_ip   TEXT,
                    dst_ip      TEXT,
                    protocol    TEXT,
                    prediction  TEXT    NOT NULL,
                    confidence  REAL,
                    severity    TEXT,
                    features_json TEXT,
                    device_id   TEXT,
                    model_version TEXT
                );

                CREATE TABLE IF NOT EXISTS devices (
                    id              TEXT PRIMARY KEY,
                    name            TEXT,
                    type            TEXT,
                    ip_address      TEXT,
                    first_seen      TEXT,
                    last_seen       TEXT,
                    total_flows     INTEGER DEFAULT 0,
                    malicious_flows INTEGER DEFAULT 0,
                    risk_score      REAL    DEFAULT 0.0
                );

                CREATE TABLE IF NOT EXISTS alerts (
                    id            INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp     TEXT,
                    detection_id  INTEGER,
                    alert_type    TEXT,
                    severity      TEXT,
                    message       TEXT,
                    acknowledged  INTEGER DEFAULT 0,
                    FOREIGN KEY (detection_id) REFERENCES detections(id)
                );

                CREATE TABLE IF NOT EXISTS statistics (
                    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp           TEXT,
                    period              TEXT,
                    total_flows         INTEGER,
                    benign_flows        INTEGER,
                    malicious_flows     INTEGER,
                    attack_distribution TEXT,
                    top_attacker        TEXT,
                    detection_rate      REAL
                );

                CREATE INDEX IF NOT EXISTS idx_det_timestamp ON detections(timestamp);
                CREATE INDEX IF NOT EXISTS idx_det_prediction ON detections(prediction);
                CREATE INDEX IF NOT EXISTS idx_det_severity ON detections(severity);
                CREATE INDEX IF NOT EXISTS idx_alerts_timestamp ON alerts(timestamp);
                CREATE INDEX IF NOT EXISTS idx_alerts_ack ON alerts(acknowledged);
            """)

            # Migration: add model_version to detections if it doesn't exist
            cur.execute("PRAGMA table_info(detections)")
            columns = [col["name"] for col in cur.fetchall()]
            if "model_version" not in columns:
                cur.execute("ALTER TABLE detections ADD COLUMN model_version TEXT")

    # ── Detections ──────────────────────────────────────────────────────────

    def insert_detection(
        self,
        prediction: str,
        confidence: float,
        severity: str,
        features: Optional[dict] = None,
        source_ip: Optional[str] = None,
        dst_ip: Optional[str] = None,
        protocol: Optional[str] = None,
        device_id: Optional[str] = None,
        timestamp: Optional[str] = None,
        model_version: Optional[str] = None,
    ) -> int:
        """Insert a detection record and return its ID."""
        ts = timestamp or datetime.now().isoformat()
        features_json = json.dumps(features) if features else None

        with self._cursor() as cur:
            cur.execute(
                """INSERT INTO detections
                   (timestamp, source_ip, dst_ip, protocol, prediction,
                    confidence, severity, features_json, device_id, model_version)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (ts, source_ip, dst_ip, protocol, prediction,
                 confidence, severity, features_json, device_id, model_version),
            )
            det_id = cur.lastrowid

        # Update device stats
        if device_id:
            self._upsert_device(device_id, source_ip, prediction, ts)

        return det_id

    def get_recent_detections(self, limit: int = 100) -> list[dict]:
        """Get the most recent detections."""
        with self._cursor() as cur:
            cur.execute(
                "SELECT * FROM detections ORDER BY timestamp DESC LIMIT ?",
                (limit,),
            )
            return [dict(row) for row in cur.fetchall()]

    def get_detections_count(self) -> int:
        """Get total number of detections."""
        with self._cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM detections")
            return cur.fetchone()[0]

    def search_detections(
        self,
        prediction: Optional[str] = None,
        severity: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        source_ip: Optional[str] = None,
        limit: int = 500,
    ) -> list[dict]:
        """Search detections with filters."""
        conditions = []
        params: list[Any] = []

        if prediction:
            conditions.append("prediction = ?")
            params.append(prediction)
        if severity:
            conditions.append("severity = ?")
            params.append(severity)
        if start_date:
            conditions.append("timestamp >= ?")
            params.append(start_date)
        if end_date:
            conditions.append("timestamp <= ?")
            params.append(end_date)
        if source_ip:
            conditions.append("source_ip LIKE ?")
            params.append(f"%{source_ip}%")

        where = " AND ".join(conditions) if conditions else "1=1"
        params.append(limit)

        with self._cursor() as cur:
            cur.execute(
                f"SELECT * FROM detections WHERE {where} ORDER BY timestamp DESC LIMIT ?",
                params,
            )
            return [dict(row) for row in cur.fetchall()]

    def get_attack_distribution(self) -> dict[str, int]:
        """Get count of each attack category."""
        with self._cursor() as cur:
            cur.execute(
                "SELECT prediction, COUNT(*) as cnt FROM detections GROUP BY prediction"
            )
            return {row["prediction"]: row["cnt"] for row in cur.fetchall()}

    def get_severity_distribution(self) -> dict[str, int]:
        """Get count of each severity level."""
        with self._cursor() as cur:
            cur.execute(
                "SELECT severity, COUNT(*) as cnt FROM detections GROUP BY severity"
            )
            return {row["severity"]: row["cnt"] for row in cur.fetchall()}

    def get_timeline_data(self, hours: int = 24) -> list[dict]:
        """Get detection counts per hour for the timeline chart."""
        cutoff = (datetime.now() - timedelta(hours=hours)).isoformat()
        with self._cursor() as cur:
            cur.execute(
                """SELECT
                     strftime('%Y-%m-%d %H:00', timestamp) as hour,
                     prediction,
                     COUNT(*) as cnt
                   FROM detections
                   WHERE timestamp >= ?
                   GROUP BY hour, prediction
                   ORDER BY hour""",
                (cutoff,),
            )
            return [dict(row) for row in cur.fetchall()]

    def get_hourly_totals(self, hours: int = 24) -> list[dict]:
        """Get total detections per hour."""
        cutoff = (datetime.now() - timedelta(hours=hours)).isoformat()
        with self._cursor() as cur:
            cur.execute(
                """SELECT
                     strftime('%Y-%m-%d %H:00', timestamp) as hour,
                     COUNT(*) as total,
                     SUM(CASE WHEN prediction != 'BENIGN' THEN 1 ELSE 0 END) as threats
                   FROM detections
                   WHERE timestamp >= ?
                   GROUP BY hour
                   ORDER BY hour""",
                (cutoff,),
            )
            return [dict(row) for row in cur.fetchall()]

    # ── Devices ─────────────────────────────────────────────────────────────

    def _upsert_device(
        self,
        device_id: str,
        ip_address: Optional[str],
        prediction: str,
        timestamp: str,
    ) -> None:
        """Update or insert device statistics."""
        is_malicious = 1 if prediction != "BENIGN" else 0

        with self._cursor() as cur:
            cur.execute("SELECT * FROM devices WHERE id = ?", (device_id,))
            existing = cur.fetchone()

            if existing:
                new_total = existing["total_flows"] + 1
                new_malicious = existing["malicious_flows"] + is_malicious
                risk_score = round(new_malicious / max(new_total, 1) * 100, 2)

                cur.execute(
                    """UPDATE devices SET
                       last_seen = ?, total_flows = ?, malicious_flows = ?,
                       risk_score = ?, ip_address = COALESCE(?, ip_address)
                       WHERE id = ?""",
                    (timestamp, new_total, new_malicious, risk_score, ip_address, device_id),
                )
            else:
                risk_score = 100.0 if is_malicious else 0.0
                cur.execute(
                    """INSERT INTO devices
                       (id, name, type, ip_address, first_seen, last_seen,
                        total_flows, malicious_flows, risk_score)
                       VALUES (?, ?, ?, ?, ?, ?, 1, ?, ?)""",
                    (device_id, device_id, "Unknown", ip_address,
                     timestamp, timestamp, is_malicious, risk_score),
                )

    def get_all_devices(self) -> list[dict]:
        """Get all registered devices."""
        with self._cursor() as cur:
            cur.execute("SELECT * FROM devices ORDER BY risk_score DESC")
            return [dict(row) for row in cur.fetchall()]

    def get_device_count(self) -> int:
        """Get number of registered devices."""
        with self._cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM devices")
            return cur.fetchone()[0]

    def update_device_name(self, device_id: str, name: str, device_type: str = "") -> None:
        """Update device friendly name and type."""
        with self._cursor() as cur:
            cur.execute(
                "UPDATE devices SET name = ?, type = ? WHERE id = ?",
                (name, device_type, device_id),
            )

    # ── Alerts ──────────────────────────────────────────────────────────────

    def insert_alert(
        self,
        detection_id: int,
        alert_type: str,
        severity: str,
        message: str,
        timestamp: Optional[str] = None,
    ) -> int:
        """Insert an alert record."""
        ts = timestamp or datetime.now().isoformat()
        with self._cursor() as cur:
            cur.execute(
                """INSERT INTO alerts
                   (timestamp, detection_id, alert_type, severity, message, acknowledged)
                   VALUES (?, ?, ?, ?, ?, 0)""",
                (ts, detection_id, alert_type, severity, message),
            )
            return cur.lastrowid

    def get_recent_alerts(self, limit: int = 50) -> list[dict]:
        """Get recent alerts."""
        with self._cursor() as cur:
            cur.execute(
                "SELECT * FROM alerts ORDER BY timestamp DESC LIMIT ?",
                (limit,),
            )
            return [dict(row) for row in cur.fetchall()]

    def get_unacknowledged_alerts(self) -> list[dict]:
        """Get unacknowledged alerts."""
        with self._cursor() as cur:
            cur.execute(
                "SELECT * FROM alerts WHERE acknowledged = 0 ORDER BY timestamp DESC"
            )
            return [dict(row) for row in cur.fetchall()]

    def acknowledge_alert(self, alert_id: int) -> None:
        """Mark an alert as acknowledged."""
        with self._cursor() as cur:
            cur.execute(
                "UPDATE alerts SET acknowledged = 1 WHERE id = ?",
                (alert_id,),
            )

    def get_alert_count(self, acknowledged: Optional[bool] = None) -> int:
        """Get alert count, optionally filtered by acknowledgment status."""
        with self._cursor() as cur:
            if acknowledged is None:
                cur.execute("SELECT COUNT(*) FROM alerts")
            else:
                cur.execute(
                    "SELECT COUNT(*) FROM alerts WHERE acknowledged = ?",
                    (1 if acknowledged else 0,),
                )
            return cur.fetchone()[0]

    # ── Statistics ──────────────────────────────────────────────────────────

    def insert_statistics(
        self,
        total_flows: int,
        benign_flows: int,
        malicious_flows: int,
        attack_distribution: dict,
        top_attacker: str = "",
        detection_rate: float = 0.0,
        period: str = "hourly",
    ) -> None:
        """Insert an aggregated statistics snapshot."""
        with self._cursor() as cur:
            cur.execute(
                """INSERT INTO statistics
                   (timestamp, period, total_flows, benign_flows, malicious_flows,
                    attack_distribution, top_attacker, detection_rate)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (datetime.now().isoformat(), period, total_flows, benign_flows,
                 malicious_flows, json.dumps(attack_distribution), top_attacker,
                 detection_rate),
            )

    def get_dashboard_stats(self) -> dict:
        """Get aggregated stats for the dashboard overview."""
        with self._cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM detections")
            total = cur.fetchone()[0]

            cur.execute("SELECT COUNT(*) FROM detections WHERE prediction = 'BENIGN'")
            benign = cur.fetchone()[0]

            cur.execute("SELECT COUNT(*) FROM detections WHERE prediction != 'BENIGN'")
            threats = cur.fetchone()[0]

            cur.execute("SELECT COUNT(*) FROM devices")
            devices = cur.fetchone()[0]

            cur.execute("SELECT COUNT(*) FROM alerts WHERE acknowledged = 0")
            pending_alerts = cur.fetchone()[0]

            # Detection rate
            detection_rate = round(threats / max(total, 1) * 100, 2)

            return {
                "total_flows": total,
                "benign_flows": benign,
                "threat_flows": threats,
                "device_count": devices,
                "pending_alerts": pending_alerts,
                "detection_rate": detection_rate,
            }

    # ── Export ──────────────────────────────────────────────────────────────

    def export_detections_csv(self, path: str) -> int:
        """Export all detections to CSV. Returns row count."""
        import pandas as pd

        with self._cursor() as cur:
            cur.execute("SELECT * FROM detections ORDER BY timestamp DESC")
            rows = [dict(r) for r in cur.fetchall()]

        if not rows:
            return 0

        df = pd.DataFrame(rows)
        df.to_csv(path, index=False)
        return len(rows)

    # ── Cleanup ─────────────────────────────────────────────────────────────

    def clear_all(self) -> None:
        """Clear all tables. Use with caution."""
        with self._cursor() as cur:
            cur.execute("DELETE FROM alerts")
            cur.execute("DELETE FROM detections")
            cur.execute("DELETE FROM devices")
            cur.execute("DELETE FROM statistics")

    def close(self) -> None:
        """Close the database connection."""
        if hasattr(self._local, "conn") and self._local.conn:
            self._local.conn.close()
            self._local.conn = None
