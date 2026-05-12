"""
Clean train / test target tables:

1. Filter cohorts smaller than `cleaning.min_total_installs`.
2. Keep only modeling columns from `parameters.yml`:
       modeling.keep_keys + modeling.weight + modeling.target + modeling.features

Inputs:   data/train/targets_train.parquet
          data/test/targets_test.parquet
Outputs:  data/train/targets_train_clean.parquet
          data/test/targets_test_clean.parquet

Run from project root:
    python run_clean.py
"""
from dotenv import load_dotenv
load_dotenv()

import sys
from pathlib import Path

import polars as pl

SRC_PATH = Path.cwd() / "src"
print("Adding SRC_PATH:", SRC_PATH)

if not SRC_PATH.exists():
    raise RuntimeError(f"src not found at {SRC_PATH}")

if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from organic_ratio.utils.config import load_config
from organic_ratio.core.cohort.clean import (
    filter_small_cohorts,
    select_modeling_columns,
    SIZE_COLUMN,
)


def _clean_one(
    in_path: Path,
    out_path: Path,
    *,
    min_total_installs: int,
    keep_keys,
    target: str,
    weight: str,
    features,
) -> None:
    df = pl.read_parquet(in_path)
    cols_before = df.width
    rows_before = len(df)

    df = filter_small_cohorts(df, min_total_installs)
    rows_after_filter = len(df)

    df = select_modeling_columns(
        df,
        keep_keys=keep_keys,
        target=target,
        weight=weight,
        features=features,
    )
    cols_after = df.width

    dropped_rows = rows_before - rows_after_filter
    pct_rows = 100.0 * dropped_rows / rows_before if rows_before else 0.0

    print(f"  {in_path.name}")
    print(f"    rows: {rows_before:,} -> {rows_after_filter:,}  "
          f"(dropped {dropped_rows:,}, {pct_rows:.1f}%)")
    print(f"    cols: {cols_before} -> {cols_after}")
    print(f"    {target}: mean={df[target].mean():.3f}, "
          f"std={df[target].std():.3f}, "
          f"median={df[target].median():.3f}")
    print(f"    {SIZE_COLUMN}: min={df[SIZE_COLUMN].min()}, "
          f"median={df[SIZE_COLUMN].median()}, "
          f"max={df[SIZE_COLUMN].max()}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(out_path, compression="zstd")
    print(f"    saved: {out_path}\n")


def main() -> None:
    cfg = load_config()

    min_total_installs = int(cfg.cleaning.min_total_installs)
    keep_keys = list(cfg.modeling.keep_keys)
    target = str(cfg.modeling.target)
    weight = str(cfg.modeling.weight)
    features = list(cfg.modeling.features)

    print(f"Cleaning threshold: min_total_installs = {min_total_installs}")
    print(f"Keep keys: {keep_keys}")
    print(f"Target:    {target}")
    print(f"Weight:    {weight}")
    print(f"Features:  {len(features)} listed in parameters.yml\n")

    out_cfg = cfg.datasets.targets

    train_in = Path(out_cfg.train_dir) / out_cfg.train_filename
    train_out = Path(out_cfg.train_dir) / out_cfg.train_clean_filename

    test_in = Path(out_cfg.test_dir) / out_cfg.test_filename
    test_out = Path(out_cfg.test_dir) / out_cfg.test_clean_filename

    for in_path in (train_in, test_in):
        if not in_path.exists():
            raise FileNotFoundError(f"Input not found: {in_path}")

    print("Cleaning train:")
    _clean_one(
        train_in,
        train_out,
        min_total_installs=min_total_installs,
        keep_keys=keep_keys,
        target=target,
        weight=weight,
        features=features,
    )

    print("Cleaning test:")
    _clean_one(
        test_in,
        test_out,
        min_total_installs=min_total_installs,
        keep_keys=keep_keys,
        target=target,
        weight=weight,
        features=features,
    )


if __name__ == "__main__":
    main()
