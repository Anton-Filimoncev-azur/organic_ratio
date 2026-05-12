"""
Clean train / test target tables: drop cohorts smaller than
`cleaning.min_total_installs` from parameters.yml.

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
from organic_ratio.core.cohort.clean import filter_small_cohorts, SIZE_COLUMN


def _clean_one(in_path: Path, out_path: Path, min_total_installs: int) -> None:
    df = pl.read_parquet(in_path)
    before = len(df)
    df_clean = filter_small_cohorts(df, min_total_installs)
    after = len(df_clean)
    dropped = before - after
    pct = 100.0 * dropped / before if before else 0.0

    print(f"  {in_path.name}")
    print(f"    rows: {before:,} -> {after:,}  (dropped {dropped:,}, {pct:.1f}%)")
    print(f"    organic_share: mean={df_clean['organic_share'].mean():.3f}, "
          f"std={df_clean['organic_share'].std():.3f}, "
          f"median={df_clean['organic_share'].median():.3f}")
    print(f"    {SIZE_COLUMN}: min={df_clean[SIZE_COLUMN].min()}, "
          f"median={df_clean[SIZE_COLUMN].median()}, "
          f"max={df_clean[SIZE_COLUMN].max()}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_clean.write_parquet(out_path, compression="zstd")
    print(f"    saved: {out_path}\n")


def main() -> None:
    cfg = load_config()

    min_total_installs = int(cfg.cleaning.min_total_installs)
    print(f"Cleaning threshold: min_total_installs = {min_total_installs}\n")

    out_cfg = cfg.datasets.targets

    train_in = Path(out_cfg.train_dir) / out_cfg.train_filename
    train_out = Path(out_cfg.train_dir) / out_cfg.train_clean_filename

    test_in = Path(out_cfg.test_dir) / out_cfg.test_filename
    test_out = Path(out_cfg.test_dir) / out_cfg.test_clean_filename

    for in_path in (train_in, test_in):
        if not in_path.exists():
            raise FileNotFoundError(f"Input not found: {in_path}")

    print("Cleaning train:")
    _clean_one(train_in, train_out, min_total_installs)

    print("Cleaning test:")
    _clean_one(test_in, test_out, min_total_installs)


if __name__ == "__main__":
    main()
