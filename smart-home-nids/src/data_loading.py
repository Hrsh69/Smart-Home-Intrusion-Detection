from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd
from tqdm import tqdm


@dataclass(frozen=True)
class LoadResult:
    dataframe: pd.DataFrame
    files_loaded: list[Path]
    files_failed: list[Path]


def discover_csv_files(raw_dir: Path, fallback_dirs: Iterable[Path]) -> list[Path]:
    """
    Discover CSV files under the configured raw data directory.

    If `raw_dir` contains no CSVs, fallback dirs are searched (useful when the dataset
    is laid out as ../train/train.csv, ../test/test.csv, etc).
    """
    raw_dir = Path(raw_dir)
    files = sorted(raw_dir.rglob("*.csv"))
    if files:
        return files

    fallback_files: list[Path] = []
    for d in fallback_dirs:
        dd = Path(d)
        if dd.exists():
            fallback_files.extend(sorted(dd.rglob("*.csv")))
    return sorted(set(fallback_files))


def _try_read_csv(path: Path, logger: logging.Logger, **read_kwargs) -> Optional[pd.DataFrame]:
    try:
        return pd.read_csv(path, **read_kwargs)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to read CSV %s (%s). Skipping.", path, exc)
        return None


def load_and_merge_csvs(
    csv_files: list[Path],
    logger: logging.Logger,
    encoding: str = "utf-8",
) -> LoadResult:
    """
    Load multiple CSV files and merge them into a single DataFrame.

    - Uses a progress bar
    - Skips corrupted/unreadable files gracefully
    - Uses low_memory=False to reduce mixed-type surprises
    """
    frames: list[pd.DataFrame] = []
    loaded: list[Path] = []
    failed: list[Path] = []

    if not csv_files:
        raise FileNotFoundError("No CSV files found in data/raw/ (or fallback dirs).")

    for f in tqdm(csv_files, desc="Loading CSVs", unit="file"):
        df = _try_read_csv(
            f,
            logger,
            encoding=encoding,
            low_memory=False,
            on_bad_lines="skip",
        )
        if df is None:
            failed.append(f)
            continue
        frames.append(df)
        loaded.append(f)

    if not frames:
        raise RuntimeError("All CSV files failed to load; no data available.")

    merged = pd.concat(frames, axis=0, ignore_index=True, copy=False)
    logger.info("Loaded %d files (%d failed). Merged shape=%s", len(loaded), len(failed), merged.shape)
    return LoadResult(dataframe=merged, files_loaded=loaded, files_failed=failed)

