from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split


@dataclass(frozen=True)
class SplitResult:
    train_df: pd.DataFrame
    test_df: pd.DataFrame


def stratified_split(
    df: pd.DataFrame,
    target_col: str,
    test_size: float,
    random_seed: int,
    logger: logging.Logger,
) -> SplitResult:
    y = df[target_col]
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_seed,
        stratify=y,
    )
    logger.info("Split done. Train=%s Test=%s", train_df.shape, test_df.shape)
    return SplitResult(train_df=train_df.reset_index(drop=True), test_df=test_df.reset_index(drop=True))

