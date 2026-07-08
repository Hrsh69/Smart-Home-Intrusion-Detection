"""Central configuration for the lightweight preprocessing pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional


ScalerName = Literal["robust", "standard", "minmax"]
BalancingMethod = Literal["none", "under", "over", "smote", "smoteenn"]


@dataclass(frozen=True)
class Config:
    """
    Pipeline configuration.

    Edit values here to control preprocessing behavior without changing code.
    """

    # ----------------------------
    # Paths
    # ----------------------------
    PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
    RAW_DATA_DIR: Path = PROJECT_ROOT / "data" / "raw"
    PROCESSED_DIR: Path = PROJECT_ROOT / "data" / "processed"
    MODELS_DIR: Path = PROJECT_ROOT / "models"
    REPORTS_DIR: Path = PROJECT_ROOT / "reports"
    PLOTS_DIR: Path = PROJECT_ROOT / "plots"

    # Fallback data locations (useful when the user has separate train/test folders)
    # These are only used if RAW_DATA_DIR has no CSV files.
    FALLBACK_RAW_DIRS: tuple[Path, ...] = (
        PROJECT_ROOT.parent / "train",
        PROJECT_ROOT.parent / "test",
        PROJECT_ROOT.parent / "validation",
    )

    # ----------------------------
    # Dataset columns
    # ----------------------------
    LABEL_COLUMN: str = "label"
    # If your dataset contains a device identifier column, list it here. If not present, it's ignored.
    DEVICE_COLUMN_CANDIDATES: tuple[str, ...] = ("device", "device_type", "iot_device", "deviceid")

    # ----------------------------
    # Reproducibility
    # ----------------------------
    RANDOM_SEED: int = 42

    # ----------------------------
    # Cleaning / validation
    # ----------------------------
    CORRELATION_THRESHOLD: float = 0.95
    DROP_TIMESTAMP_COLUMNS: bool = True
    CHUNK_SIZE: int = 100_000
    BALANCED_SAMPLES_PER_CLASS: int = 25_000
    MAX_ROWS_PER_CHUNK_PER_CLASS: int = 5_000
    ALLOWED_CLASSES: tuple[str, ...] = (
        "BENIGN",
        "DDoS",
        "DoS",
        "Mirai",
        "Recon",
        "Spoofing",
        "BruteForce",
        "WebAttack",
        "Malware",
        "Unknown",
    )

    # ----------------------------
    # Feature selection
    # ----------------------------
    N_FEATURES_TO_KEEP: int = 18
    FEATURE_SELECTION_RF_ESTIMATORS: int = 200
    FEATURE_SELECTION_TOP_K_RF: int = 20
    FEATURE_SELECTION_TOP_K_MI: int = 20
    FEATURE_SELECTION_SAMPLE_ROWS: int = 200_000

    # If provided, overrides automatic selection.
    SELECTED_FEATURES: Optional[list[str]] = None

    # ----------------------------
    # Encoding / scaling
    # ----------------------------
    SCALER: ScalerName = "robust"

    # ----------------------------
    # Balancing
    # ----------------------------
    ENABLE_BALANCING: bool = False
    BALANCING_METHOD: BalancingMethod = "none"
    IMBALANCE_RATIO_THRESHOLD: float = 10.0  # max_count / min_count above this is considered severe

    # ----------------------------
    # Split
    # ----------------------------
    TEST_SIZE: float = 0.20


CONFIG = Config()

