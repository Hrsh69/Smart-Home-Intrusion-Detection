"""Train a Random Forest classifier on the preprocessed CIC IoT-2023 data.

Usage:
    python src/train_model.py

Outputs:
    models/rf_model.pkl
    reports/training_report.json
    plots/confusion_matrix.png
    plots/roc_curves.png
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from config.config import CONFIG
from config.settings import SETTINGS
from src.utils import ensure_dirs, setup_logging


# ── Severity mapping for attack categories ────────────────────────────────
SEVERITY_MAP: dict[str, str] = {
    "BENIGN": "Info",
    "Recon": "Low",
    "Spoofing": "Medium",
    "BruteForce": "High",
    "WebAttack": "High",
    "DoS": "High",
    "DDoS": "Critical",
    "Mirai": "Critical",
    "Malware": "Critical",
}


def load_processed_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the preprocessed train and test CSVs."""
    train_path = CONFIG.PROCESSED_DIR / "processed_train.csv"
    test_path = CONFIG.PROCESSED_DIR / "processed_test.csv"

    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            f"Processed data not found. Run preprocessing first.\n"
            f"Expected: {train_path}\n         {test_path}"
        )

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return train_df, test_df


def train_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_estimators: int = 300,
    max_depth: int | None = None,
    random_state: int = 42,
) -> RandomForestClassifier:
    """Train a Random Forest with balanced class weights."""
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features="sqrt",
        oob_score=True,
        verbose=1,
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(
    model: RandomForestClassifier,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    label_encoder: object,
) -> dict:
    """Evaluate the model and return comprehensive metrics."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    class_names = list(label_encoder.classes_)

    # Core metrics
    accuracy = float(accuracy_score(y_test, y_pred))
    f1_weighted = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))
    f1_macro = float(f1_score(y_test, y_pred, average="macro", zero_division=0))
    precision_weighted = float(precision_score(y_test, y_pred, average="weighted", zero_division=0))
    recall_weighted = float(recall_score(y_test, y_pred, average="weighted", zero_division=0))

    # Per-class report
    report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True, zero_division=0)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred).tolist()

    # ROC AUC (one-vs-rest)
    try:
        roc_auc = float(roc_auc_score(y_test, y_proba, multi_class="ovr", average="weighted"))
    except ValueError:
        roc_auc = None

    # OOB score
    oob_score = float(model.oob_score_) if hasattr(model, "oob_score_") else None

    # Feature importance
    feature_importance = dict(zip(X_test.columns.tolist(), model.feature_importances_.tolist()))

    return {
        "accuracy": accuracy,
        "f1_weighted": f1_weighted,
        "f1_macro": f1_macro,
        "precision_weighted": precision_weighted,
        "recall_weighted": recall_weighted,
        "roc_auc_weighted": roc_auc,
        "oob_score": oob_score,
        "class_names": class_names,
        "per_class_report": report,
        "confusion_matrix": cm,
        "feature_importance": feature_importance,
        "severity_map": SEVERITY_MAP,
    }


def plot_confusion_matrix(
    cm: list[list[int]],
    class_names: list[str],
    out_path: Path,
) -> None:
    """Save a styled confusion matrix heatmap."""
    fig, ax = plt.subplots(figsize=(12, 10))
    cm_array = np.array(cm)

    sns.heatmap(
        cm_array,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        linewidths=0.5,
        linecolor="gray",
    )
    ax.set_xlabel("Predicted Label", fontsize=12, fontweight="bold")
    ax.set_ylabel("True Label", fontsize=12, fontweight="bold")
    ax.set_title("Confusion Matrix — Random Forest NIDS", fontsize=14, fontweight="bold")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_feature_importance(
    importance: dict[str, float],
    out_path: Path,
    top_k: int = 18,
) -> None:
    """Save a horizontal bar chart of feature importances."""
    sorted_feats = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:top_k]
    names, values = zip(*reversed(sorted_feats))

    fig, ax = plt.subplots(figsize=(10, 8))
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(names)))
    ax.barh(names, values, color=colors)
    ax.set_xlabel("Importance", fontsize=12)
    ax.set_title("Random Forest Feature Importance", fontsize=14, fontweight="bold")
    plt.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def run_training() -> dict:
    """Execute full training pipeline and return metrics."""
    ensure_dirs(CONFIG.MODELS_DIR, CONFIG.REPORTS_DIR, CONFIG.PLOTS_DIR)
    logger = setup_logging(CONFIG.REPORTS_DIR, name="model_training")

    logger.info("=" * 60)
    logger.info("SMART HOME NIDS — MODEL TRAINING")
    logger.info("=" * 60)

    # Load data
    logger.info("Loading processed data...")
    train_df, test_df = load_processed_data()
    target_col = "attack_category"

    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col].astype(int)
    X_test = test_df.drop(columns=[target_col])
    y_test = test_df[target_col].astype(int)

    logger.info("Train shape: %s, Test shape: %s", X_train.shape, X_test.shape)
    logger.info("Class distribution (train): %s", dict(y_train.value_counts().sort_index()))

    # Load label encoder
    le_path = CONFIG.MODELS_DIR / "label_encoder.pkl"
    if not le_path.exists():
        raise FileNotFoundError(f"Label encoder not found: {le_path}")
    label_encoder = joblib.load(le_path)
    logger.info("Label encoder classes: %s", list(label_encoder.classes_))

    # Train
    logger.info("Training Random Forest (n_estimators=%d, class_weight=balanced)...", SETTINGS.RF_N_ESTIMATORS)
    start_time = time.time()

    model = train_random_forest(
        X_train=X_train,
        y_train=y_train,
        n_estimators=SETTINGS.RF_N_ESTIMATORS,
        max_depth=SETTINGS.RF_MAX_DEPTH,
        random_state=CONFIG.RANDOM_SEED,
    )

    train_time = time.time() - start_time
    logger.info("Training completed in %.1f seconds", train_time)

    # Save model
    model_path = CONFIG.MODELS_DIR / "rf_model.pkl"
    joblib.dump(model, model_path)
    logger.info("Model saved: %s", model_path)

    # Evaluate
    logger.info("Evaluating on test set...")
    metrics = evaluate_model(model, X_test, y_test, label_encoder)
    metrics["training_time_seconds"] = train_time
    metrics["n_estimators"] = SETTINGS.RF_N_ESTIMATORS
    metrics["train_samples"] = int(X_train.shape[0])
    metrics["test_samples"] = int(X_test.shape[0])
    metrics["n_features"] = int(X_train.shape[1])

    logger.info("Accuracy: %.4f", metrics["accuracy"])
    logger.info("F1 (weighted): %.4f", metrics["f1_weighted"])
    logger.info("F1 (macro): %.4f", metrics["f1_macro"])
    logger.info("ROC AUC: %s", metrics.get("roc_auc_weighted"))
    logger.info("OOB Score: %s", metrics.get("oob_score"))

    # Save metrics
    report_path = CONFIG.REPORTS_DIR / "training_report.json"
    report_path.write_text(json.dumps(metrics, indent=2, default=str), encoding="utf-8")
    logger.info("Training report saved: %s", report_path)

    # Plots
    plot_confusion_matrix(
        cm=metrics["confusion_matrix"],
        class_names=metrics["class_names"],
        out_path=CONFIG.PLOTS_DIR / "confusion_matrix.png",
    )
    logger.info("Confusion matrix plot saved.")

    plot_feature_importance(
        importance=metrics["feature_importance"],
        out_path=CONFIG.PLOTS_DIR / "rf_feature_importance.png",
    )
    logger.info("Feature importance plot saved.")

    logger.info("=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 60)

    return metrics


if __name__ == "__main__":
    run_training()
