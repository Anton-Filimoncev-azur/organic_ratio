"""
Build the MMM panel:

    one row per (platform, country, install_date) with
        organic_installs  : count of organic users (target)
        total_installs    : all users (for ROAS post-hoc)
        spend_<source>    : per-channel paid spend (top-N, rest → other_paid)
        dow_0..dow_6      : dayofweek dummies (control)
        geo               : "<platform>_<country>"  (model dim key)
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np
import polars as pl


OTHER_BUCKET = "other_paid"
ORGANIC_VALUE = "organic"


def pick_top_channels(
    costs_lf: pl.LazyFrame,
    top_n: int,
    date_from=None,
    date_to=None,
) -> List[str]:
    """
    Return media_source names with the largest spend in the date window.
    Channels with total spend == 0 are skipped — they cause identifiability
    issues in MMM (no signal to learn adstock/saturation from).
    """
    lf = costs_lf.filter(pl.col("media_source") != ORGANIC_VALUE)
    if date_from is not None and date_to is not None:
        lf = lf.filter(
            (pl.col("install_date") >= pl.lit(str(date_from)).str.to_date()) &
            (pl.col("install_date") < pl.lit(str(date_to)).str.to_date())
        )

    totals = (
        lf
        .group_by("media_source")
        .agg(pl.col("spend").sum().alias("total_spend"))
        .filter(pl.col("total_spend") > 0)        # drop zero-spend channels
        .sort("total_spend", descending=True)
        .head(top_n)
        .collect()
    )
    return totals["media_source"].to_list()


def aggregate_installs(installs_lf: pl.LazyFrame) -> pl.LazyFrame:
    """Per (platform, country, install_date): organic_installs, total_installs."""
    return (
        installs_lf
        .select(["platform", "country_code", "install_date", "media_source"])
        .group_by(["platform", "country_code", "install_date"])
        .agg(
            pl.len().alias("total_installs"),
            (pl.col("media_source") == ORGANIC_VALUE).sum().alias("organic_installs"),
        )
    )


def aggregate_spend_wide(
    costs_lf: pl.LazyFrame,
    top_channels: List[str],
) -> pl.LazyFrame:
    """
    Aggregate costs by (platform, country, install_date, media_source),
    bucket non-top sources into OTHER_BUCKET, pivot to wide format:
        spend_<source>, spend_other_paid
    """
    bucketed = (
        costs_lf
        .filter(pl.col("media_source") != ORGANIC_VALUE)
        .with_columns(
            pl.when(pl.col("media_source").is_in(top_channels))
            .then(pl.col("media_source"))
            .otherwise(pl.lit(OTHER_BUCKET))
            .alias("channel")
        )
        .group_by(["platform", "country_code", "install_date", "channel"])
        .agg(pl.col("spend").sum().alias("spend"))
        .collect()
    )

    wide = bucketed.pivot(
        values="spend",
        index=["platform", "country_code", "install_date"],
        on="channel",
        aggregate_function="sum",
    )

    # ensure all expected channel columns exist
    expected_cols = top_channels + [OTHER_BUCKET]
    for c in expected_cols:
        if c not in wide.columns:
            wide = wide.with_columns(pl.lit(0.0).alias(c))

    # rename to spend_<channel>
    rename_map = {c: f"spend_{c}" for c in expected_cols}
    wide = wide.rename(rename_map)

    # fill nulls with 0 (no spend that day)
    spend_cols = list(rename_map.values())
    wide = wide.with_columns([pl.col(c).fill_null(0.0).alias(c) for c in spend_cols])

    return wide.lazy()


def add_seasonality(panel: pl.DataFrame) -> pl.DataFrame:
    """Add dayofweek dummy columns dow_0..dow_6 (Monday=0)."""
    panel = panel.with_columns(
        pl.col("install_date").dt.weekday().alias("_dow")
    )
    # polars weekday: 1=Mon..7=Sun in v1+; normalize to 0..6
    panel = panel.with_columns((pl.col("_dow") - 1).alias("_dow"))
    for d in range(7):
        # Float64 dow dummies — avoids strict-dtype mismatch when pandas
        # round-trips through fit/predict in pymc-marketing.
        panel = panel.with_columns((pl.col("_dow") == d).cast(pl.Float64).alias(f"dow_{d}"))
    return panel.drop("_dow")


def filter_countries(panel: pl.DataFrame, min_installs: int) -> pl.DataFrame:
    """Drop countries whose total install count across train window is small."""
    keep = (
        panel
        .group_by("country_code")
        .agg(pl.col("total_installs").sum().alias("country_total"))
        .filter(pl.col("country_total") >= min_installs)
        .select("country_code")
    )
    return panel.join(keep, on="country_code", how="inner")


def keep_top_n_geos(panel: pl.DataFrame, top_n: int) -> pl.DataFrame:
    """Keep only the top-N (platform, country_code) pairs by total install volume."""
    if top_n <= 0:
        return panel
    keep = (
        panel
        .group_by(["platform", "country_code"])
        .agg(pl.col("total_installs").sum().alias("geo_total"))
        .sort("geo_total", descending=True)
        .head(top_n)
        .select(["platform", "country_code"])
    )
    return panel.join(keep, on=["platform", "country_code"], how="inner")


def aggregate_by_cadence(panel: pl.DataFrame, cadence_days: int) -> pl.DataFrame:
    """
    Roll up daily panel to N-day buckets.

    For each (platform, country, bucket_start):
        organic_installs, total_installs    — SUM
        spend_*                              — SUM
        dow_* (если есть)                    — dropped (irrelevant for weekly)
    bucket_start = first day of the window aligned to period anchor.
    """
    if cadence_days <= 1:
        return panel

    anchor = panel["install_date"].min()
    panel = panel.with_columns(
        (
            (pl.col("install_date") - pl.lit(anchor)).dt.total_days()
            // cadence_days
        ).alias("_bucket_idx"),
    )
    panel = panel.with_columns(
        (pl.lit(anchor) + pl.duration(days=pl.col("_bucket_idx") * cadence_days))
        .alias("bucket_start")
    )

    spend_cols = [c for c in panel.columns if c.startswith("spend_")]
    sum_cols = ["organic_installs", "total_installs"] + spend_cols

    agg = (
        panel.group_by(["platform", "country_code", "bucket_start"])
        .agg([pl.col(c).sum().alias(c) for c in sum_cols])
        .sort(["platform", "country_code", "bucket_start"])
        .rename({"bucket_start": "install_date"})  # re-use date column name
    )
    return agg


def build_mmm_panel(
    *,
    installs_path: Path,
    costs_path: Path,
    top_n_channels: int,
    min_country_installs: int,
    top_n_geos: int = 0,
    cadence_days: int = 1,
    date_from,
    date_to,
) -> Tuple[pl.DataFrame, List[str]]:
    """
    Full pipeline:
      1. discover top-N channels by total spend
      2. aggregate installs (organic, total)
      3. aggregate spend wide
      4. join on (platform, country, install_date)
      5. filter date window + min_country_installs
      6. add seasonality dummies + geo key
    Returns (panel, channel_names).
    """
    installs_lf = pl.scan_parquet(installs_path).select(
        ["platform", "country_code", "install_date", "media_source"]
    )
    costs_lf = pl.scan_parquet(costs_path).select(
        ["platform", "country_code", "install_date", "media_source", "spend"]
    )

    top_channels = pick_top_channels(
        costs_lf, top_n=top_n_channels,
        date_from=date_from, date_to=date_to,
    )
    print(f"  top-{top_n_channels} channels (non-zero spend in window): {top_channels}")

    installs_agg = aggregate_installs(installs_lf)
    spend_wide = aggregate_spend_wide(costs_lf, top_channels)

    panel = (
        installs_agg
        .join(spend_wide, on=["platform", "country_code", "install_date"], how="left")
        .collect()
    )

    spend_cols = [f"spend_{c}" for c in top_channels + [OTHER_BUCKET]]
    panel = panel.with_columns([pl.col(c).fill_null(0.0).alias(c) for c in spend_cols])

    # date window
    panel = panel.filter(
        (pl.col("install_date") >= pl.lit(str(date_from)).str.to_date()) &
        (pl.col("install_date") < pl.lit(str(date_to)).str.to_date())
    )

    # country filter (skip if min_country_installs == 0)
    if min_country_installs > 0:
        panel = filter_countries(panel, min_country_installs)

    # top-N geos by total install volume (preferred if set)
    if top_n_geos > 0:
        panel = keep_top_n_geos(panel, top_n_geos)
        n_geos = panel.select(["platform", "country_code"]).unique().height
        print(f"  kept top-{top_n_geos} (platform, country) geos → {n_geos} actual geos")

    # cadence aggregation (daily → N-day buckets)
    if cadence_days > 1:
        print(f"  aggregating to {cadence_days}-day buckets...")
        panel = aggregate_by_cadence(panel, cadence_days)

    # seasonality (dow only meaningful for daily cadence)
    if cadence_days <= 1:
        panel = add_seasonality(panel)

    # geo key
    panel = panel.with_columns(
        (pl.col("platform") + "_" + pl.col("country_code")).alias("geo")
    )

    # sort
    panel = panel.sort(["geo", "install_date"])

    return panel, top_channels
