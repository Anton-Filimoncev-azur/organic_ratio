"""
Shared helpers for scanning per-source preprocessed parquets.
Used by run_cohort_aggregation.py and run_target_build.py.
"""
from pathlib import Path

import polars as pl

from organic_ratio.core.preprocessing.preprocesser_registry import PREPROCESSORS


BASE_DATASET_NAME = "installs"


def build_ordered_feature_scans(cfg) -> dict:
    """
    Lazy-scan each per-source feature parquet. Returns a dict ordered so that
    BASE_DATASET_NAME (installs) is the first entry — it will be the merge base.
    """
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
