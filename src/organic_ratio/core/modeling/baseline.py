"""
Baseline model for cohort-level organic_share.

Approach: weighted Ridge regression on logit(organic_share) with sample
weight = total_installs (so cohorts of 5000 users count proportionally more
than cohorts of 30). Predictions are inverse-logit transformed back to [0, 1].

Features:
  * numeric columns from `modeling.features` — median impute → standardize
  * `platform`  — one-hot
  * `country_code` — weighted target encoding (mean organic_share by country,
                    computed on train, with global mean as fallback)
  * `install_date` — dropped (used for split, not as feature)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


EPS = 1e-3
COUNTRY_COL = "country_code"
PLATFORM_COL = "platform"
DATE_COL = "install_date"


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def inv_logit(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def weighted_country_target_encoding(
    df: pd.DataFrame,
    target: str,
    weight: str,
) -> Tuple[pd.Series, float]:
    """
    Returns (country_te_series_indexed_by_country_code, global_mean).
    Uses install-weighted mean of `target` per country.
    """
    grouped = df.groupby(COUNTRY_COL).apply(
        lambda g: np.average(g[target], weights=g[weight]),
        include_groups=False,
    )
    global_mean = float(
        np.average(df[target].to_numpy(), weights=df[weight].to_numpy())
    )
    return grouped.rename("country_te"), global_mean


@dataclass
class BaselineArtifacts:
    pipeline: Pipeline
    preproc: ColumnTransformer
    country_te: pd.Series
    global_mean: float
    feature_names: List[str]
    coef: np.ndarray


def fit_baseline(
    train: pd.DataFrame,
    *,
    target: str,
    weight: str,
    features: Sequence[str],
    alpha: float = 1.0,
) -> BaselineArtifacts:
    country_te, global_mean = weighted_country_target_encoding(train, target, weight)
    train = train.copy()
    train["country_te"] = train[COUNTRY_COL].map(country_te).fillna(global_mean)

    num_features = list(features) + ["country_te"]
    cat_features = [PLATFORM_COL]

    preproc = ColumnTransformer(
        [
            (
                "num",
                Pipeline(
                    [
                        ("imp", SimpleImputer(strategy="median")),
                        ("sc", StandardScaler()),
                    ]
                ),
                num_features,
            ),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                cat_features,
            ),
        ],
        remainder="drop",
    )

    X = preproc.fit_transform(train)
    y_logit = logit(train[target].to_numpy())
    w = train[weight].to_numpy(dtype=float)

    model = Ridge(alpha=alpha)
    model.fit(X, y_logit, sample_weight=w)

    pipeline = Pipeline([("preproc", preproc), ("model", model)])

    # extract feature names produced by ColumnTransformer
    feat_names: List[str] = []
    feat_names.extend(num_features)
    ohe = preproc.named_transformers_["cat"]
    feat_names.extend(ohe.get_feature_names_out(cat_features).tolist())

    return BaselineArtifacts(
        pipeline=pipeline,
        preproc=preproc,
        country_te=country_te,
        global_mean=global_mean,
        feature_names=feat_names,
        coef=model.coef_.copy(),
    )


def predict_baseline(art: BaselineArtifacts, df: pd.DataFrame) -> np.ndarray:
    df = df.copy()
    df["country_te"] = df[COUNTRY_COL].map(art.country_te).fillna(art.global_mean)
    X = art.preproc.transform(df)
    y_logit = art.pipeline.named_steps["model"].predict(X)
    return inv_logit(y_logit)


def coefficient_importance(art: BaselineArtifacts, top: int = 20) -> pd.DataFrame:
    df = pd.DataFrame({"feature": art.feature_names, "coef": art.coef})
    df["abs_coef"] = df["coef"].abs()
    return df.sort_values("abs_coef", ascending=False).head(top).reset_index(drop=True)
