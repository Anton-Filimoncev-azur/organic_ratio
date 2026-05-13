"""
Shared preprocessing for modeling: median impute, standardize, categorical
indexing. The fitted state is captured in `PrepArtifacts` and reused at
inference time.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd


COUNTRY_COL = "country_code"
PLATFORM_COL = "platform"


@dataclass
class PrepArtifacts:
    feature_names: List[str]
    feature_medians: np.ndarray
    feature_means: np.ndarray
    feature_stds: np.ndarray
    country_categories: List[str]
    platform_categories: List[str]


def fit_prep(df: pd.DataFrame, features: Sequence[str]) -> PrepArtifacts:
    feats = list(features)
    medians = df[feats].median(numeric_only=True).to_numpy()

    df_imp = df[feats].copy()
    for i, c in enumerate(feats):
        df_imp[c] = df_imp[c].fillna(medians[i])

    means = df_imp.mean().to_numpy()
    stds = df_imp.std(ddof=0).to_numpy()
    stds = np.where(stds == 0, 1.0, stds)

    return PrepArtifacts(
        feature_names=feats,
        feature_medians=medians,
        feature_means=means,
        feature_stds=stds,
        country_categories=sorted(df[COUNTRY_COL].astype(str).unique().tolist()),
        platform_categories=sorted(df[PLATFORM_COL].astype(str).unique().tolist()),
    )


def transform_prep(
    df: pd.DataFrame,
    prep: PrepArtifacts,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (X_scaled, country_idx, platform_idx).
    Unknown categories in df fall back to index 0 with a warning.
    """
    X = df[prep.feature_names].copy()
    for i, c in enumerate(prep.feature_names):
        X[c] = X[c].fillna(prep.feature_medians[i])
    X = (X.to_numpy(dtype=np.float64) - prep.feature_means) / prep.feature_stds

    country_idx = pd.Categorical(
        df[COUNTRY_COL].astype(str), categories=prep.country_categories
    ).codes
    n_unknown_country = int((country_idx < 0).sum())
    if n_unknown_country:
        print(f"  [prep] {n_unknown_country} rows with unknown country → idx 0")
        country_idx = np.where(country_idx < 0, 0, country_idx)

    platform_idx = pd.Categorical(
        df[PLATFORM_COL].astype(str), categories=prep.platform_categories
    ).codes
    n_unknown_platform = int((platform_idx < 0).sum())
    if n_unknown_platform:
        print(f"  [prep] {n_unknown_platform} rows with unknown platform → idx 0")
        platform_idx = np.where(platform_idx < 0, 0, platform_idx)

    return X.astype(np.float64), country_idx.astype(np.int32), platform_idx.astype(np.int32)
