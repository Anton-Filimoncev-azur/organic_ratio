"""
Cleaning utilities for cohort/target tables.
"""
from __future__ import annotations

import polars as pl


SIZE_COLUMN = "total_installs"


def filter_small_cohorts(df: pl.DataFrame, min_total_installs: int) -> pl.DataFrame:
    """
    Drop cohorts whose total install count is below the threshold.

    Tiny cohorts make `organic_share` a noisy Bernoulli estimate (e.g. 1/1=1
    or 0/2=0), which the model would learn as signal.
    """
    if SIZE_COLUMN not in df.columns:
        raise KeyError(f"Column '{SIZE_COLUMN}' not found in target table")
    if min_total_installs <= 1:
        return df
    return df.filter(pl.col(SIZE_COLUMN) >= min_total_installs)
