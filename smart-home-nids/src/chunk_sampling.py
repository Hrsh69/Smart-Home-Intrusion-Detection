from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from config.attack_mapping import map_attack_label
from src.cleaning import standardize_columns
from src.data_loading import discover_csv_files
from src.utils import replace_inf_with_nan, standardize_column_name


@dataclass(frozen=True)
class SampledDatasetResult:
    dataframe: pd.DataFrame
    files_loaded: list[Path]
    class_counts: dict[str, int]
    raw_rows_seen: int


def _sample_chunk_for_class(
    chunk: pd.DataFrame,
    target_col: str,
    class_name: str,
    remaining: int,
    per_chunk_cap: int,
    random_state: int,
) -> pd.DataFrame:
    class_rows = chunk[chunk[target_col] == class_name]
    if class_rows.empty:
        return class_rows

    sample_n = min(len(class_rows), remaining, per_chunk_cap)
    if sample_n <= 0:
        return class_rows.iloc[0:0]
    if sample_n == len(class_rows):
        return class_rows.copy()
    return class_rows.sample(n=sample_n, random_state=random_state)


def build_balanced_sample_dataset(
    raw_dir: Path,
    fallback_dirs: tuple[Path, ...],
    label_col: str,
    allowed_classes: tuple[str, ...],
    samples_per_class: int,
    chunk_size: int,
    per_chunk_cap: int,
    random_state: int,
    logger: logging.Logger,
) -> SampledDatasetResult:
    """
    Build a lightweight balanced dataframe using chunked CSV reads.

    The full CIC IoT-2023 dataset is too large for many laptops. This function
    streams the raw CSVs in chunks, maps fine-grained labels to broad categories,
    and keeps only a capped random sample per class until the desired quota is met.
    """
    csv_files = discover_csv_files(raw_dir, fallback_dirs)
    if not csv_files:
        raise FileNotFoundError("No CSV files found in data/raw/ or fallback directories.")

    target_counts: Counter[str] = Counter({cls: 0 for cls in allowed_classes})
    sampled_parts: list[pd.DataFrame] = []
    raw_rows_seen = 0
    label_col_std = standardize_column_name(label_col)

    for csv_file in csv_files:
        logger.info("Streaming CSV file: %s", csv_file)
        try:
            reader = pd.read_csv(
                csv_file,
                chunksize=chunk_size,
                low_memory=False,
                on_bad_lines="skip",
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to open %s (%s). Skipping.", csv_file, exc)
            continue

        for chunk in tqdm(reader, desc=f"Sampling {csv_file.name}", unit="chunk"):
            raw_rows_seen += len(chunk)
            chunk = standardize_columns(chunk)
            if label_col_std not in chunk.columns:
                logger.warning("Chunk from %s missing label column '%s'.", csv_file, label_col_std)
                continue

            chunk = replace_inf_with_nan(chunk)
            chunk = chunk.dropna(axis=0, how="all")
            chunk = chunk.dropna(subset=[label_col_std])
            chunk["attack_category"] = chunk[label_col_std].map(map_attack_label)
            chunk = chunk[chunk["attack_category"].isin(allowed_classes)]

            if chunk.empty:
                continue

            for class_name in allowed_classes:
                remaining = samples_per_class - target_counts[class_name]
                if remaining <= 0:
                    continue
                class_sample = _sample_chunk_for_class(
                    chunk=chunk,
                    target_col="attack_category",
                    class_name=class_name,
                    remaining=remaining,
                    per_chunk_cap=per_chunk_cap,
                    random_state=random_state,
                )
                if not class_sample.empty:
                    sampled_parts.append(class_sample)
                    target_counts[class_name] += len(class_sample)

            if all(target_counts[class_name] >= samples_per_class for class_name in allowed_classes):
                logger.info("Collected target sample size for all classes.")
                break

    if not sampled_parts:
        raise RuntimeError("No sampled rows were collected. Check dataset paths and label mapping.")

    sampled_df = pd.concat(sampled_parts, ignore_index=True)
    logger.info("Balanced sampled dataset shape before cleanup: %s", sampled_df.shape)
    return SampledDatasetResult(
        dataframe=sampled_df,
        files_loaded=csv_files,
        class_counts=dict(target_counts),
        raw_rows_seen=raw_rows_seen,
    )

