"""
Halo MMM via pymc-marketing.

Target = organic_installs at (platform × country × install_date)
Channels = top-N paid media_source spends (+ other_paid bucket)
Model learns: how much extra organic each $ of paid spend brings
              (after adstock × saturation transformation).

Per-geo random effects via `dims=("geo",)`.
"""
from __future__ import annotations

from typing import List, Optional

import pandas as pd

from pymc_marketing.mmm import (
    MMM,
    GeometricAdstock,
    LogisticSaturation,
    MichaelisMentenSaturation,
    HillSaturation,
)


SATURATIONS = {
    "logistic": LogisticSaturation,
    "michaelis_menten": MichaelisMentenSaturation,
    "hill": HillSaturation,
}


def make_saturation(kind: str):
    if kind not in SATURATIONS:
        raise ValueError(
            f"Unknown saturation '{kind}'. Options: {list(SATURATIONS)}"
        )
    return SATURATIONS[kind]()


def build_mmm(
    *,
    channel_columns: List[str],
    adstock_l_max: int,
    saturation_kind: str,
    yearly_seasonality: int,
    control_columns: Optional[List[str]] = None,
    date_column: str = "install_date",
) -> MMM:
    """
    Construct a pymc-marketing MMM (single time-series).

    pymc-marketing 0.10's MMM class expects one row per date. For multi-geo
    panels, aggregate first or loop per geo externally.
    """
    mmm = MMM(
        date_column=date_column,
        channel_columns=channel_columns,
        control_columns=control_columns,
        adstock=GeometricAdstock(l_max=adstock_l_max),
        saturation=make_saturation(saturation_kind),
        yearly_seasonality=yearly_seasonality if yearly_seasonality > 0 else None,
    )
    return mmm
