"""
Compute organic_share target at cohort grain (cohort.keys MINUS media_source)
and tag train/test split by install_date.
"""
from __future__ import annotations

from typing import List, Sequence

import polars as pl


ORGANIC_VALUE = "organic"
TARGET_GRAIN_EXCLUDE = {"media_source"}


def derive_target_keys(cohort_keys: Sequence[str]) -> List[str]:
    return [k for k in cohort_keys if k not in TARGET_GRAIN_EXCLUDE]


def _to_date_str(v) -> str:
    return v.isoformat() if hasattr(v, "isoformat") else str(v)


def build_target(
    installs_lf: pl.LazyFrame,
    target_keys: Sequence[str],
    *,
    train_start_date,
    test_start_date,
    test_end_date,
) -> pl.LazyFrame:
    """
    Aggregate installs by target_keys, compute organic_share and add 'split'
    column (train / test / unused) based on install_date thresholds.
    """
    schema = installs_lf.collect_schema()
    if "media_source" not in schema:
        raise KeyError("media_source column not found in installs features")
    missing = [k for k in target_keys if k not in schema]
    if missing:
        raise KeyError(f"Target keys missing in installs: {missing}")

    target_lf = (
        installs_lf.group_by(list(target_keys))
        .agg(
            pl.len().alias("total_installs"),
            (pl.col("media_source") == ORGANIC_VALUE).sum().alias("organic_installs"),
        )
        .with_columns(
            (
                pl.col("organic_installs").cast(pl.Float64)
                / pl.col("total_installs").cast(pl.Float64)
            ).alias("organic_share")
        )
    )

    test_start = pl.lit(_to_date_str(test_start_date)).str.to_date()
    test_end = pl.lit(_to_date_str(test_end_date)).str.to_date()
    train_start = pl.lit(_to_date_str(train_start_date)).str.to_date()

    install_date_d = pl.col("install_date").cast(pl.Date)

    target_lf = target_lf.with_columns(
        pl.when(install_date_d < train_start)
        .then(pl.lit("unused"))
        .when(install_date_d < test_start)
        .then(pl.lit("train"))
        .when(install_date_d < test_end)
        .then(pl.lit("test"))
        .otherwise(pl.lit("unused"))
        .alias("split")
    )

    return target_lf
