import sys
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.database import NIDSDatabase
from src.predict import NIDSPredictor
from src.alerts import AlertManager

db = NIDSDatabase()
predictor = NIDSPredictor()
alert_mgr = AlertManager(db)

test_path = Path("data/processed/processed_test.csv")
test_df = pd.read_csv(test_path)
feature_cols = [c for c in test_df.columns if c != "attack_category"]

sample = test_df.sample(n=5)
for i, (idx, row) in enumerate(sample.iterrows()):
    features = row[feature_cols].to_dict()
    try:
        result = predictor.predict_single(features)
        print("Success:", result.label)
        det_id = db.insert_detection(
            prediction=result.label,
            confidence=result.confidence,
            severity=result.severity,
            features=features,
            source_ip="1.1.1.1",
            device_id="dev1"
        )
        print("DB Insert:", det_id)
    except Exception as e:
        print("ERROR:", e)
