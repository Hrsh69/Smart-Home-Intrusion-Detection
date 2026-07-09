"""Smoke-test: run 5 random rows from the test CSV through the full predict pipeline.

Run from smart-home-nids/:
    python tests/test_simulation.py

Not a pytest test — requires processed_test.csv and trained model artifacts.
Kept in tests/ for discoverability but excluded from the CI pytest run.
"""

import sys
from pathlib import Path

# Ensure smart-home-nids/ is on the path regardless of where the script is run from
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.database import NIDSDatabase
from src.predict import NIDSPredictor
from src.alerts import AlertManager

db = NIDSDatabase()
predictor = NIDSPredictor()
alert_mgr = AlertManager(db)

test_path = Path(__file__).resolve().parents[1] / "data" / "processed" / "processed_test.csv"
if not test_path.exists():
    print(f"ERROR: {test_path} not found — run src/preprocessing.py first.")
    sys.exit(1)

test_df = pd.read_csv(test_path)
feature_cols = [c for c in test_df.columns if c != "attack_category"]

sample = test_df.sample(n=5, random_state=42)
print(f"Sampling 5 rows from {test_path.name} ({len(test_df):,} total rows)\n")

for i, (idx, row) in enumerate(sample.iterrows()):
    features = row[feature_cols].to_dict()
    try:
        result = predictor.predict_single(features)
        print(f"[{i+1}] label={result.label:<12} confidence={result.confidence:.1%}  severity={result.severity}")
        det_id = db.insert_detection(
            prediction=result.label,
            confidence=result.confidence,
            severity=result.severity,
            features=features,
            source_ip="1.1.1.1",
            device_id="smoke_test_dev",
        )
        print(f"     DB insert ID: {det_id}")
    except Exception as e:
        print(f"[{i+1}] ERROR: {e}")
