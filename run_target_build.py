"""
Build model-ready train / test tables.

1. Compute target (organic_share + install counts + split) at target grain
   (cohort.keys MINUS media_source) from preprocessed installs.
2. Re-aggregate user-grain features (merge of all per-source parquets) at the
   SAME target grain — single SUM/MEAN aggregation, no media_source.
3. Join target ⨝ features on target keys.
4. Write:
       data/features/targets/targets.parquet     # full table with `split` column
       data/train/targets_train.parquet          # split=='train', no split column
       data/test/targets_test.parquet            # split=='test',  no split column

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
from organic_ratio.core.cohort.merge import merge_datasets
from organic_ratio.core.cohort.aggregator import aggregate_to_cohort
from organic_ratio.core.cohort.sources import build_ordered_feature_scans
from organic_ratio.core.cohort.target import build_target, derive_target_keys


USER_KEYS = ["match_id", "install_date"]


def main() -> None:
    cfg = load_config()

    cohort_keys = list(cfg.cohort["keys"])
    target_keys = derive_target_keys(cohort_keys)
    print(f"Cohort keys:  {cohort_keys}")
    print(f"Target keys:  {target_keys}  (media_source dropped)")

    # ---------- 1. Target ----------
    installs_cfg = cfg.datasets.installs
    installs_path = Path(installs_cfg.local_feature_dir) / installs_cfg.filename
    if not installs_path.exists():
        raise FileNotFoundError(f"Installs features not found: {installs_path}")
    print(f"\nReading installs: {installs_path}")

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

    # ---------- 2. Features at target grain ----------
    print("\nBuilding features at target grain:")
    ordered_scans = build_ordered_feature_scans(cfg)
    user_lf = merge_datasets(
        lfs=ordered_scans,
        on=USER_KEYS,
        how="left",
    )
    features_lf = aggregate_to_cohort(user_lf, target_keys)

    # ---------- 3. Join target + features ----------
    full_lf = target_lf.join(features_lf, on=target_keys, how="left")
    df = full_lf.collect(engine="streaming")
    print(f"\nFull table shape: {df.shape}")
    print(df.head(5))
    print("\nSplit counts:")
    print(df["split"].value_counts())
    print("\norganic_share summary:")
    print(df["organic_share"].describe())

    # ---------- 4. Write outputs ----------
    out_cfg = cfg.datasets.targets

    full_dir = Path(out_cfg.local_feature_dir)
    full_dir.mkdir(parents=True, exist_ok=True)
    full_path = full_dir / out_cfg.filename
    df.write_parquet(full_path, compression="zstd")
    print(f"\nSaved full:  {full_path}  ({len(df)} rows, {df.width} cols)")

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
