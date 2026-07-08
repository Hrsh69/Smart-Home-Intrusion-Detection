"""
db.py
-----
Handles all SQLite storage for detected traffic/alerts.

WHY SQLite (not Postgres/MySQL)?
- Zero setup, single file, perfect for a project that needs to "just work"
  when a professor or interviewer runs it locally.
- This is the same reasoning you used for ByteSentry - consistent story.

SCHEMA DESIGN NOTE:
We store EVERY classified flow (not just attacks) so the dashboard can show
both "what's normal" and "what's suspicious" - a real SOC (Security
Operations Center) dashboard needs both for context, not just alerts.
"""

import os
import sqlite3
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGS_DIR = os.path.join(PROJECT_ROOT, "logs")
os.makedirs(LOGS_DIR, exist_ok=True)
DB_PATH = os.path.join(LOGS_DIR, "ids_alerts.db")

LABEL_NAMES = {
    0: "BENIGN", 1: "DDOS", 2: "PORT_SCAN", 3: "BRUTE_FORCE", 4: "BOTNET_C2",
}
SEVERITY = {
    "BENIGN": "none", "DDOS": "critical", "PORT_SCAN": "medium",
    "BRUTE_FORCE": "high", "BOTNET_C2": "critical",
}

def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS detections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            device TEXT NOT NULL,
            protocol TEXT NOT NULL,
            dst_port INTEGER,
            predicted_label TEXT NOT NULL,
            severity TEXT NOT NULL,
            confidence REAL NOT NULL,
            flow_duration_ms REAL,
            packet_count REAL,
            byte_rate_bps REAL
        )
    """)
    conn.commit()
    conn.close()

def log_detection(device, protocol, dst_port, predicted_label, confidence,
                   flow_duration_ms, packet_count, byte_rate_bps):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        INSERT INTO detections
        (timestamp, device, protocol, dst_port, predicted_label, severity,
         confidence, flow_duration_ms, packet_count, byte_rate_bps)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        datetime.now().isoformat(timespec="seconds"),
        device, protocol, int(dst_port), predicted_label,
        SEVERITY.get(predicted_label, "unknown"), float(confidence),
        float(flow_duration_ms), float(packet_count), float(byte_rate_bps),
    ))
    conn.commit()
    conn.close()

def fetch_recent(limit=200):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT * FROM detections ORDER BY id DESC LIMIT ?", (limit,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]

def fetch_all_for_stats():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT * FROM detections").fetchall()
    conn.close()
    return [dict(r) for r in rows]

def clear_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("DELETE FROM detections")
    conn.commit()
    conn.close()

if __name__ == "__main__":
    init_db()
    print(f"Initialized DB at {DB_PATH}")
