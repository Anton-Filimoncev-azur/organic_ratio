"""
Cleaning utilities for cohort / target tables.
"""
from __future__ import annotations

from typing import List, Sequence

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


def select_modeling_columns(
    df: pl.DataFrame,
    *,
    keep_keys: Sequence[str],
    target: str,
    weight: str,
    features: Sequence[str],
) -> pl.DataFrame:
    """
    Keep only: keep_keys + weight + target + features.
    Order is preserved as listed.

    Missing feature columns are warned about and skipped (so a typo in
    parameters.yml does not crash the pipeline).
    """
    available = set(df.columns)

    required: List[str] = []
    for col in (*keep_keys, weight, target):
        if col not in available:
            raise KeyError(f"Required column '{col}' missing from table")
        required.append(col)

    feats_present: List[str] = []
    feats_missing: List[str] = []
    for c in features:
        (feats_present if c in available else feats_missing).append(c)

    if feats_missing:
        print(f"    [warn] {len(feats_missing)} feature columns missing in data, skipped:")
        for c in feats_missing:
            print(f"           - {c}")

    selected = required + feats_present
    return df.select(selected)
