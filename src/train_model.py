"""
train_model.py
---------------
Trains a Random Forest classifier to detect intrusions in IoT traffic flows.

PIPELINE:
1. Load flow data
2. Encode categorical features (device, protocol) -> numeric
   WHY: sklearn models need numbers, not strings. We use LabelEncoder here
   for simplicity; for production you'd consider one-hot encoding for
   'protocol' since there's no ordinal relationship between TCP/UDP/ICMP.
3. Train/test split (80/20) - STRATIFIED so each attack type is proportionally
   represented in both sets (critical for imbalanced-ish security data)
4. Train RandomForestClassifier
5. Evaluate: accuracy, per-class precision/recall/F1, confusion matrix
   WHY precision/recall matter MORE than accuracy in IDS:
   - False Negative (missed attack) = a real intrusion gets through = BAD
   - False Positive (false alarm) = analyst wastes time = annoying but safer
   So we care a lot about RECALL on attack classes specifically.
6. Save model + encoders for use in the dashboard
"""

import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

LABEL_NAMES = {
    0: "BENIGN",
    1: "DDOS",
    2: "PORT_SCAN",
    3: "BRUTE_FORCE",
    4: "BOTNET_C2",
}

def main():
    df = pd.read_csv(os.path.join(DATA_DIR, "iot_traffic.csv"))

    # --- Encode categoricals ---
    device_encoder = LabelEncoder()
    protocol_encoder = LabelEncoder()
    df["device_enc"] = device_encoder.fit_transform(df["device"])
    df["protocol_enc"] = protocol_encoder.fit_transform(df["protocol"])

    feature_cols = [
        "device_enc", "protocol_enc", "dst_port", "flow_duration_ms",
        "packet_count", "byte_rate_bps", "flag_syn_ratio", "unique_dst_ips_per_src",
    ]
    X = df[feature_cols]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # --- Train ---
    clf = RandomForestClassifier(
        n_estimators=150,
        max_depth=12,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",  # extra safety net if classes were imbalanced
    )
    clf.fit(X_train, y_train)

    # --- Evaluate ---
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Overall Accuracy: {acc:.4f}\n")
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=[LABEL_NAMES[i] for i in sorted(LABEL_NAMES)]))

    print("Confusion Matrix (rows=actual, cols=predicted):")
    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(cm, index=[LABEL_NAMES[i] for i in sorted(LABEL_NAMES)],
                          columns=[LABEL_NAMES[i] for i in sorted(LABEL_NAMES)])
    print(cm_df)

    print("\nFeature Importances (what the model actually learned matters):")
    importances = pd.Series(clf.feature_importances_, index=feature_cols).sort_values(ascending=False)
    print(importances)

    # --- Save artifacts ---
    joblib.dump(clf, os.path.join(MODELS_DIR, "rf_model.pkl"))
    joblib.dump(device_encoder, os.path.join(MODELS_DIR, "device_encoder.pkl"))
    joblib.dump(protocol_encoder, os.path.join(MODELS_DIR, "protocol_encoder.pkl"))
    print("\nSaved model + encoders to models/")

if __name__ == "__main__":
    main()
