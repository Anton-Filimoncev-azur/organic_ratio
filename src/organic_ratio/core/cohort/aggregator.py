"""
Cohort aggregator: collapse user-grain LazyFrame to cohort-grain.

Group keys come from `cfg.cohort.keys`. For each non-key numeric column the
aggregator applies either SUM or MEAN based on a name-pattern policy:

    * SUM      — counts, cumulatives, monetary totals, and per-day sequence
                 columns (sessions_X, iap_X, ads_X, iap_count_X).
    * MEAN     — everything else (rates, ratios, log1p, share, time-deltas).

Two extra cohort-level columns are added:

    cohort_size       — number of users in the cohort.
    n_calendar_days   — distinct install_date values inside the cohort
                        (trivially 1 if `install_date` is among cohort.keys).

Non-numeric columns that are NOT cohort keys are dropped.
"""
from __future__ import annotations

from typing import Iterable, List

import polars as pl


SUM_PATTERNS: tuple[str, ...] = (
    "spend",
    "impressions",
    "clicks",
    "installs_spend",
    "_cum_",
    "_count",
    "purchase_count",
    "rev_d",
    "iap_rev",
    "ads_rev",
)

SEQ_PREFIXES: tuple[str, ...] = ("sessions_", "iap_", "ads_", "iap_count_", "iap_check_size_")

NUMERIC_DTYPES = {
    pl.Float32,
    pl.Float64,
    pl.Int8,
    pl.Int16,
    pl.Int32,
    pl.Int64,
    pl.UInt8,
    pl.UInt16,
    pl.UInt32,
    pl.UInt64,
}


def _is_seq_day_column(col: str) -> bool:
    """sessions_3, iap_12, iap_count_5, ads_0, iap_check_size_4 — ends with int."""
    if "_" not in col:
        return False
    tail = col.rsplit("_", 1)[-1]
    if not tail.isdigit():
        return False
    return any(col.startswith(p) for p in SEQ_PREFIXES)


def _is_sum_column(col: str) -> bool:
    if _is_seq_day_column(col):
        return True
    return any(p in col for p in SUM_PATTERNS)


def build_agg_exprs(schema: dict, group_keys: Iterable[str]) -> List[pl.Expr]:
    keys = set(group_keys)
    exprs: List[pl.Expr] = []
    for col, dtype in schema.items():
        if col in keys:
            continue
        if dtype not in NUMERIC_DTYPES:
            continue
        if _is_sum_column(col):
            exprs.append(pl.col(col).sum().alias(col))
        else:
            exprs.append(pl.col(col).mean().alias(col))
    return exprs


def aggregate_to_cohort(
    user_lf: pl.LazyFrame,
    cohort_keys: List[str],
) -> pl.LazyFrame:
    """
    Group user-grain LazyFrame by cohort_keys and apply numeric aggregation.
    Adds cohort_size and n_calendar_days columns.
    """
    schema = user_lf.collect_schema()

    missing = [k for k in cohort_keys if k not in schema]
    if missing:
        raise KeyError(f"Cohort keys missing in merged data: {missing}")

    agg_exprs = build_agg_exprs(schema, cohort_keys)

    extra_exprs = [pl.len().alias("cohort_size")]
    if "install_date" in schema:
        extra_exprs.append(pl.col("install_date").n_unique().alias("n_calendar_days"))

    return user_lf.group_by(cohort_keys).agg(extra_exprs + agg_exprs)
