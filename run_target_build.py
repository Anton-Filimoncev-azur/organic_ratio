"""
Target build: organic_share at cohort grain (cohort.keys MINUS media_source).

Reads preprocessed installs features (per-user rows with media_source),
groups by target keys, computes total/organic installs and organic_share,
tags rows with train/test/unused based on install_date thresholds from
parameters.yml.

Run from project root:
    python run_target_build.py
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
from organic_ratio.core.cohort.target import build_target, derive_target_keys


def main() -> None:
    cfg = load_config()

    cohort_keys = list(cfg.cohort["keys"])
    target_keys = derive_target_keys(cohort_keys)
    print(f"Cohort keys: {cohort_keys}")
    print(f"Target keys (media_source dropped): {target_keys}")

    installs_cfg = cfg.datasets.installs
    installs_path = Path(installs_cfg.local_feature_dir) / installs_cfg.filename
    if not installs_path.exists():
        raise FileNotFoundError(f"Installs features not found: {installs_path}")
    print(f"Reading installs: {installs_path}")

    installs_lf = pl.scan_parquet(installs_path).select(
        target_keys + ["media_source"]
    )

    target_lf = build_target(
        installs_lf,
        target_keys=target_keys,
        train_start_date=cfg.train_start_date,
        test_start_date=cfg.test_start_date,
        test_end_date=cfg.test_end_date,
    )

    df = target_lf.collect(engine="streaming")
    print(f"Target table shape: {df.shape}")
    print(df.head(10))
    print("\nSplit counts:")
    print(df["split"].value_counts())
    print("\norganic_share summary:")
    print(df["organic_share"].describe())

    out_cfg = cfg.datasets.targets

    # 1. Full table (with `split` column) — source of truth
    out_dir = Path(out_cfg.local_feature_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / out_cfg.filename
    df.write_parquet(out_path, compression="zstd")
    print(f"Saved full:  {out_path}  ({len(df)} rows)")

    # 2. Physical train / test splits
    train_df = df.filter(pl.col("split") == "train").drop("split")
    test_df = df.filter(pl.col("split") == "test").drop("split")

    train_dir = Path(out_cfg.train_dir)
    train_dir.mkdir(parents=True, exist_ok=True)
    train_path = train_dir / out_cfg.train_filename
    train_df.write_parquet(train_path, compression="zstd")
    print(f"Saved train: {train_path}  ({len(train_df)} rows)")

    test_dir = Path(out_cfg.test_dir)
    test_dir.mkdir(parents=True, exist_ok=True)
    test_path = test_dir / out_cfg.test_filename
    test_df.write_parquet(test_path, compression="zstd")
    print(f"Saved test:  {test_path}  ({len(test_df)} rows)")


if __name__ == "__main__":
    main()
