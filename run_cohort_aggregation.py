"""
Cohort aggregation:

1. Scan all per-source feature parquets (output of run_preprocessing.py).
2. Merge them on user-grain keys [match_id, install_date], base = installs.
3. Group by cohort.keys (from parameters.yml).
4. Add cohort_size and n_calendar_days; numerics aggregated by SUM/MEAN policy.
5. Write data/features/cohort/cohort_level.parquet.

Run from project root:
    python run_cohort_aggregation.py
"""
from dotenv import load_dotenv
load_dotenv()

import sys
from pathlib import Path
import gc

import polars as pl

SRC_PATH = Path.cwd() / "src"
print("Adding SRC_PATH:", SRC_PATH)

if not SRC_PATH.exists():
    raise RuntimeError(f"src not found at {SRC_PATH}")

if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from organic_ratio.utils.config import load_config
from organic_ratio.core.preprocessing.preprocesser_registry import PREPROCESSORS
from organic_ratio.core.cohort.merge import merge_datasets
from organic_ratio.core.cohort.aggregator import aggregate_to_cohort


USER_KEYS = ["match_id", "install_date"]
BASE_DATASET_NAME = "installs"


def build_ordered_feature_scans(cfg) -> dict:
    lfs: dict = {}
    for name in PREPROCESSORS.keys():
        ds_cfg = cfg.datasets[name]
        feature_path = Path(ds_cfg.local_feature_dir) / ds_cfg.filename
        if not feature_path.exists():
            raise FileNotFoundError(
                f"Feature file not found for {name}: {feature_path}"
            )
        print(f"Scanning features: {name} -> {feature_path}")
        lfs[name] = pl.scan_parquet(feature_path)

    if BASE_DATASET_NAME not in lfs:
        raise KeyError(
            f"Base dataset '{BASE_DATASET_NAME}' is missing in feature registry"
        )

    ordered = {BASE_DATASET_NAME: lfs[BASE_DATASET_NAME]}
    for name, lf in lfs.items():
        if name != BASE_DATASET_NAME:
            ordered[name] = lf
    return ordered


def main() -> None:
    cfg = load_config()
    cohort_keys = list(cfg.cohort["keys"])
    print(f"Cohort keys: {cohort_keys}")

    ordered_scans = build_ordered_feature_scans(cfg)

    user_lf = merge_datasets(
        lfs=ordered_scans,
        on=USER_KEYS,
        how="left",
    )

    cohort_lf = aggregate_to_cohort(user_lf, cohort_keys)

    out_cfg = cfg.datasets.cohort_level
    out_dir = Path(out_cfg.local_feature_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / out_cfg.filename

    df = cohort_lf.collect(engine="streaming")
    print(f"Cohort table shape: {df.shape}")
    print(df.head(5))

    df.write_parquet(out_path, compression="zstd")
    print(f"Saved: {out_path}")

    del df, cohort_lf, user_lf, ordered_scans
    gc.collect()


if __name__ == "__main__":
    main()
