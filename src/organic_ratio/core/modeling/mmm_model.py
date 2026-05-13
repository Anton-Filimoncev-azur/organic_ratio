"""
Halo MMM via pymc-marketing's multidimensional MMM
(`pymc_marketing.mmm.multidimensional.MMM`).

Target = organic_installs at (platform × country × install_date).
Channels = top-N paid media_source spends (+ other_paid bucket).
Per-geo hierarchical priors via dims=("geo",).
"""
from __future__ import annotations

from typing import List, Optional, Tuple

from pymc_marketing.mmm.multidimensional import MMM
from pymc_marketing.mmm import (
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
    dims: Optional[Tuple[str, ...]] = ("geo",),
    yearly_seasonality: int = 0,
    control_columns: Optional[List[str]] = None,
    date_column: str = "install_date",
    target_column: str = "organic_installs",
) -> MMM:
    """
    Multidimensional MMM with optional hierarchical dim (geo).

    dims=("geo",)  → per-geo channel coefficients (partial pooling across geos)
    dims=None      → single time-series (must have one row per date)
    """
    return MMM(
        date_column=date_column,
        channel_columns=channel_columns,
        target_column=target_column,
        control_columns=control_columns,
        adstock=GeometricAdstock(l_max=adstock_l_max),
        saturation=make_saturation(saturation_kind),
        yearly_seasonality=yearly_seasonality if yearly_seasonality > 0 else None,
        dims=dims,
    )
