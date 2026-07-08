from __future__ import annotations

import sys
from pathlib import Path
import os

# Allow running as: python src/preprocessing.py
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / "reports" / ".mplconfig"))
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from config.config import CONFIG
from src.chunk_sampling import build_balanced_sample_dataset
from src.cleaning import clean_data
from src.encoding import fit_categorical_encoders, transform_categorical_columns
from src.feature_engineering import engineer_features
from src.feature_selection import select_features
from src.reporting import (
    build_preprocessing_report_html,
    plot_class_distribution,
    plot_correlation_heatmap,
    plot_feature_importance_bar,
    save_json,
)
from src.scaling import fit_transform_scaler
from src.split_data import stratified_split
from src.utils import ensure_dirs, memory_usage_mb, setup_logging, standardize_column_name


def run_preprocessing() -> None:
    ensure_dirs(CONFIG.PROCESSED_DIR, CONFIG.MODELS_DIR, CONFIG.REPORTS_DIR, CONFIG.PLOTS_DIR)
    logger = setup_logging(CONFIG.REPORTS_DIR)

    # ----------------------------
    # Step 1: Build lightweight balanced sample using chunked reads
    # ----------------------------
    sample_result = build_balanced_sample_dataset(
        raw_dir=CONFIG.RAW_DATA_DIR,
        fallback_dirs=CONFIG.FALLBACK_RAW_DIRS,
        label_col=CONFIG.LABEL_COLUMN,
        allowed_classes=CONFIG.ALLOWED_CLASSES,
        samples_per_class=CONFIG.BALANCED_SAMPLES_PER_CLASS,
        chunk_size=CONFIG.CHUNK_SIZE,
        per_chunk_cap=CONFIG.MAX_ROWS_PER_CHUNK_PER_CLASS,
        random_state=CONFIG.RANDOM_SEED,
        logger=logger,
    )
    df_raw = sample_result.dataframe
    logger.info("Sampled dataset memory usage: %.2f MB", memory_usage_mb(df_raw))

    # ----------------------------
    # Step 2: Clean
    # ----------------------------
    df_clean, cleaning_report = clean_data(df_raw, label_col=CONFIG.LABEL_COLUMN, logger=logger)

    # ----------------------------
    # Step 3: Broad labels are already created during chunked sampling
    # ----------------------------
    df_labeled = df_clean.copy()
    target_col = "attack_category"
    raw_label_col_std = standardize_column_name(CONFIG.LABEL_COLUMN)

    # ----------------------------
    # Step 5: Feature engineering (before selection so selectors can consider engineered signals)
    # ----------------------------
    df_feat = engineer_features(df_labeled, logger=logger)

    # Drop original fine-grained label to prevent leakage into the model.
    if raw_label_col_std in df_feat.columns:
        df_feat = df_feat.drop(columns=[raw_label_col_std], errors="ignore")

    # ----------------------------
    # Step 4: Feature selection
    # ----------------------------
    # Encode target temporarily for selection; selection function expects label_col present.
    df_for_sel = df_feat.copy()
    df_for_sel[target_col] = df_for_sel[target_col].astype(str)

    fs_result = select_features(
        df=df_for_sel,
        label_col=target_col,
        logger=logger,
        n_keep=CONFIG.N_FEATURES_TO_KEEP,
        correlation_threshold=CONFIG.CORRELATION_THRESHOLD,
        rf_estimators=CONFIG.FEATURE_SELECTION_RF_ESTIMATORS,
        top_k_rf=CONFIG.FEATURE_SELECTION_TOP_K_RF,
        top_k_mi=CONFIG.FEATURE_SELECTION_TOP_K_MI,
        sample_rows=CONFIG.FEATURE_SELECTION_SAMPLE_ROWS,
        drop_timestamp_columns=CONFIG.DROP_TIMESTAMP_COLUMNS,
    )

    # Display top 20/15/10 (in logs + persisted json)
    top20_rf = fs_result.rf_importances.head(20).to_dict(orient="records")
    top15_rf = fs_result.rf_importances.head(15).to_dict(orient="records")
    top10_rf = fs_result.rf_importances.head(10).to_dict(orient="records")
    logger.info("Top 20 RF features: %s", [x["feature"] for x in top20_rf])
    logger.info("Top 15 RF features: %s", [x["feature"] for x in top15_rf])
    logger.info("Top 10 RF features: %s", [x["feature"] for x in top10_rf])

    # Save feature importance plots
    plot_feature_importance_bar(
        fs_result.rf_importances,
        value_col="importance",
        out_path=CONFIG.PLOTS_DIR / "feature_importance.png",
        title="Random Forest Feature Importance (Top 20)",
        top_k=20,
    )
    plot_feature_importance_bar(
        fs_result.mi_scores,
        value_col="mi",
        out_path=CONFIG.PLOTS_DIR / "mutual_information.png",
        title="Mutual Information (Top 20)",
        top_k=20,
    )

    # Choose selected features (config override supported)
    selected_features = CONFIG.SELECTED_FEATURES or fs_result.selected_features
    joblib.dump(selected_features, CONFIG.MODELS_DIR / "selected_features.pkl")

    # Correlation heatmap for selected features (numeric only)
    numeric_selected = [c for c in selected_features if c in df_feat.columns and pd.api.types.is_numeric_dtype(df_feat[c])]
    if numeric_selected:
        plot_correlation_heatmap(
            df=df_feat[numeric_selected].fillna(0),
            out_path=CONFIG.PLOTS_DIR / "correlation_heatmap.png",
            title="Correlation Heatmap (Selected Numeric Features)",
        )

    df_model = df_feat[selected_features + [target_col]].copy()
    df_model = df_model.replace([np.inf, -np.inf], np.nan).dropna()

    # ----------------------------
    # Step 9: Split (stratified)
    # ----------------------------
    split = stratified_split(
        df=df_model,
        target_col=target_col,
        test_size=CONFIG.TEST_SIZE,
        random_seed=CONFIG.RANDOM_SEED,
        logger=logger,
    )
    train_df = split.train_df
    test_df = split.test_df

    plot_class_distribution(
        y=train_df[target_col],
        out_path=CONFIG.PLOTS_DIR / "class_distribution.png",
        title="Training Class Distribution (Attack Categories)",
    )

    # ----------------------------
    # Step 6: Categorical encoding + label encoding
    # ----------------------------
    # 6a) Encode categoricals (protocol/device/etc)
    train_enc_art = fit_categorical_encoders(
        train_df.drop(columns=[target_col]),
        models_dir=CONFIG.MODELS_DIR,
        device_column_candidates=CONFIG.DEVICE_COLUMN_CANDIDATES,
        logger=logger,
    )
    train_cat = train_enc_art.encoded_df
    test_cat = transform_categorical_columns(
        test_df.drop(columns=[target_col]),
        models_dir=CONFIG.MODELS_DIR,
        protocol_column=train_enc_art.protocol_column,
        device_column=train_enc_art.device_column,
    )

    # 6b) Fit label encoder on train labels only; apply to both; save for inference
    label_encoder = LabelEncoder()
    y_train_str = train_df[target_col].astype(str).fillna("Unknown")
    y_test_str = test_df[target_col].astype(str).fillna("Unknown")
    y_train_enc = label_encoder.fit_transform(y_train_str)
    y_test_enc = label_encoder.transform(y_test_str)
    joblib.dump(label_encoder, CONFIG.MODELS_DIR / "label_encoder.pkl")

    train_df_enc = train_cat.copy()
    test_df_enc = test_cat.copy()
    train_df_enc[target_col] = y_train_enc
    test_df_enc[target_col] = y_test_enc

    # ----------------------------
    # Step 7: Scaling (fit on train only)
    # ----------------------------
    X_train = train_df_enc.drop(columns=[target_col])
    y_train = train_df_enc[target_col].astype(int)
    X_test = test_df_enc.drop(columns=[target_col])
    y_test = test_df_enc[target_col].astype(int)

    scaling_art = fit_transform_scaler(X_train, models_dir=CONFIG.MODELS_DIR, scaler_name=CONFIG.SCALER, logger=logger)
    X_train_scaled = scaling_art.X_scaled

    # Transform test using saved imputer+scaler
    imputer = joblib.load(CONFIG.MODELS_DIR / "numeric_imputer.pkl")
    scaler = joblib.load(CONFIG.MODELS_DIR / "scaler.pkl")
    X_test_imp = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns.tolist())
    X_test_scaled = pd.DataFrame(scaler.transform(X_test_imp), columns=X_test.columns.tolist())

    # Training set is already approximately balanced by construction.
    X_train_final = X_train_scaled
    y_train_final = y_train

    # ----------------------------
    # Step 10: Save outputs
    # ----------------------------
    processed_train = pd.concat([X_train_final, y_train_final.rename(target_col)], axis=1)
    processed_test = pd.concat([X_test_scaled, y_test.rename(target_col)], axis=1)

    train_out = CONFIG.PROCESSED_DIR / "processed_train.csv"
    test_out = CONFIG.PROCESSED_DIR / "processed_test.csv"
    processed_train.to_csv(train_out, index=False)
    processed_test.to_csv(test_out, index=False)

    summary = {
        "files_loaded": len(sample_result.files_loaded),
        "raw_rows_scanned": sample_result.raw_rows_seen,
        "sampled_rows_before_cleaning": int(df_raw.shape[0]),
        "clean_rows": int(df_clean.shape[0]),
        "balanced_class_counts": sample_result.class_counts,
        "selected_features": selected_features,
        "scaler": CONFIG.SCALER,
        "train_out": str(train_out),
        "test_out": str(test_out),
    }
    cleaning_report_dict = cleaning_report.to_dict()
    feature_selection_summary = {
        "dropped_columns": fs_result.dropped_columns,
        "n_selected_features": len(selected_features),
        "top_20_rf": [r["feature"] for r in top20_rf],
        "top_20_mi": fs_result.mi_scores.head(20)["feature"].tolist(),
    }

    save_json(summary, CONFIG.REPORTS_DIR / "preprocessing_summary.json")
    save_json(cleaning_report_dict, CONFIG.REPORTS_DIR / "cleaning_report.json")
    save_json(feature_selection_summary, CONFIG.REPORTS_DIR / "feature_selection.json")

    build_preprocessing_report_html(
        out_html=CONFIG.REPORTS_DIR / "preprocessing_report.html",
        summary=summary,
        cleaning_report=cleaning_report_dict,
        feature_selection_summary=feature_selection_summary,
        notes=[
            "The dataset is built with chunked CSV reads to avoid loading all raw rows into memory.",
            "Target label used for ML is 'attack_category' (broad classes).",
            "The sampled dataset is approximately balanced per class before splitting.",
            "Scaling is fit on train only; test is transformed using saved artifacts.",
            "RobustScaler is preferred for heavy-tailed network traffic features.",
            (
                "Class imbalance note: BruteForce (2,150), WebAttack (3,530), and Malware (574) "
                "could not reach the target of 25,000 samples each because the entire CIC IoT-2023 "
                "source dataset contains fewer rows for these attack categories. All available rows "
                "were included. Consider SMOTE or class-weighted loss during model training."
            ),
            (
                "protocol_encoder.pkl and device_encoder.pkl were not created because this dataset "
                "version does not contain a separate protocol-string or device-identity column "
                "requiring standalone encoding. The 'protocol_type' feature is already numeric."
            ),
        ],
    )

    logger.info("Saved processed datasets and artifacts successfully.")


if __name__ == "__main__":
    run_preprocessing()

