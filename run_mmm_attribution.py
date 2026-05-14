"""
Per-channel halo attribution from a fitted MMM.

For each (geo, day, channel) compute:
    halo_c = β_c[geo] · saturation(adstock(spend_c[geo, t]; α_c[geo]); λ_c[geo])

using posterior MEAN of α, β, λ (point estimate — fast, no posterior re-sampling).

Aggregates:
    - per-channel total halo across the full panel period
    - per-channel share of total predicted organic
    - per (geo, day, channel) raw contribution

Inputs:
    data/models/mmm/mmm.nc
    data/features/mmm/mmm_panel.parquet
Outputs:
    data/predictions/mmm_attribution.parquet           (per-row)
    data/predictions/mmm_attribution_summary.csv       (per-channel summary)
    data/plots/mmm_halo_per_channel.png
"""
from dotenv import load_dotenv
load_dotenv()

import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import polars as pl

SRC_PATH = Path.cwd() / "src"
print("Adding SRC_PATH:", SRC_PATH)
if not SRC_PATH.exists():
    raise RuntimeError(f"src not found at {SRC_PATH}")
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import arviz as az

from organic_ratio.utils.config import load_config


# ----- MMM transforms (mirrors pymc-marketing defaults) -----

def geometric_adstock_normalized(spend: np.ndarray, alpha: float, l_max: int) -> np.ndarray:
    """
    Causal geometric adstock with normalized weights:
        adstock[t] = Σ_{k=0..l_max} α^k · spend[t-k] / Σ_{k=0..l_max} α^k
    Matches pymc-marketing's GeometricAdstock(l_max=L, normalize=True).
    """
    weights = alpha ** np.arange(l_max + 1)
    weights = weights / weights.sum()
    n = len(spend)
    out = np.zeros(n, dtype=float)
    for t in range(n):
        k_max = min(t + 1, l_max + 1)
        out[t] = np.sum(weights[:k_max] * spend[t:t - k_max:-1] if t >= l_max
                        else weights[:k_max] * spend[t::-1])
    return out


def logistic_saturation(x: np.ndarray, beta: float, lam: float) -> np.ndarray:
    """β · (1 - exp(-λ·x)) / (1 + exp(-λ·x))   — pymc-marketing LogisticSaturation."""
    z = np.exp(-lam * x)
    return beta * (1.0 - z) / (1.0 + z)


# ----- Parameter discovery -----

PARAM_CANDIDATES = {
    "alpha": ["adstock_alpha", "alpha", "adstock_alpha_channel"],
    "beta": ["saturation_beta", "beta_channel", "saturation_beta_channel"],
    "lam":  ["saturation_lam", "lam_channel", "saturation_lam_channel"],
}


def _find_var(post, kind: str) -> str:
    for name in PARAM_CANDIDATES[kind]:
        if name in post.data_vars:
            return name
    raise KeyError(
        f"No {kind} variable in posterior. Looked for {PARAM_CANDIDATES[kind]}. "
        f"Got: {list(post.data_vars)}"
    )


def posterior_means(post) -> Tuple[np.ndarray, np.ndarray, np.ndarray, list, list]:
    """Returns (alpha, beta, lam) arrays of shape (n_geos, n_channels) + names."""
    a_name = _find_var(post, "alpha")
    b_name = _find_var(post, "beta")
    l_name = _find_var(post, "lam")
    print(f"  posterior vars: α={a_name}  β={b_name}  λ={l_name}")

    alpha = post[a_name].mean(dim=("chain", "draw"))
    beta = post[b_name].mean(dim=("chain", "draw"))
    lam = post[l_name].mean(dim=("chain", "draw"))

    # ensure (geo, channel) order
    if alpha.dims == ("channel", "geo"):
        alpha = alpha.transpose("geo", "channel")
        beta = beta.transpose("geo", "channel")
        lam = lam.transpose("geo", "channel")
    elif alpha.dims != ("geo", "channel"):
        raise ValueError(f"Unexpected dims for adstock alpha: {alpha.dims}")

    geos = alpha["geo"].values.tolist()
    channels = alpha["channel"].values.tolist()
    return alpha.values, beta.values, lam.values, geos, channels


def main() -> None:
    cfg = load_config()
    mmm_cfg = cfg.modeling.mmm

    # ----- Load trace -----
    mmm_path = Path("data/models/mmm/mmm.nc")
    if not mmm_path.exists():
        raise FileNotFoundError(f"MMM trace not found: {mmm_path}")
    print(f"Loading: {mmm_path}")
    idata = az.from_netcdf(mmm_path)
    post = idata.posterior

    alpha_geo_ch, beta_geo_ch, lam_geo_ch, geos, channels = posterior_means(post)
    print(f"  geos:     {len(geos)}")
    print(f"  channels: {len(channels)}")

    # ----- Load panel -----
    panel_cfg = cfg.datasets.mmm_panel
    panel_path = Path(panel_cfg.local_feature_dir) / panel_cfg.filename
    if not panel_path.exists():
        raise FileNotFoundError(f"Panel not found: {panel_path}")
    panel = pl.read_parquet(panel_path).to_pandas()
    panel["install_date"] = pd.to_datetime(panel["install_date"])
    panel = panel.sort_values(["geo", "install_date"]).reset_index(drop=True)
    print(f"  panel:    {panel.shape}")

    target_col = str(mmm_cfg.target)
    adstock_l_max = int(mmm_cfg.adstock_l_max)

    # channel posterior names are e.g. "spend_facebook ads" — exact column in panel
    spend_cols = [c for c in panel.columns if c.startswith("spend_")]
    # ensure ordering matches `channels` from posterior
    channel_to_col = {c: c if c in spend_cols else None for c in channels}
    missing = [c for c, v in channel_to_col.items() if v is None]
    if missing:
        print(f"  [warn] channels in posterior but not in panel: {missing}")

    # ----- Compute attribution -----
    print("\nComputing per-channel halo contributions...")
    rows = []
    for gi, geo in enumerate(geos):
        geo_df = panel[panel["geo"] == geo]
        if geo_df.empty:
            continue
        dates = geo_df["install_date"].values

        for ci, ch in enumerate(channels):
            spend_col = channel_to_col.get(ch)
            if spend_col is None:
                continue
            spend = geo_df[spend_col].to_numpy(dtype=float)
            alpha = float(alpha_geo_ch[gi, ci])
            beta = float(beta_geo_ch[gi, ci])
            lam = float(lam_geo_ch[gi, ci])

            adstocked = geometric_adstock_normalized(spend, alpha, adstock_l_max)
            halo = logistic_saturation(adstocked, beta, lam)

            for i, (d, sp, hl) in enumerate(zip(dates, spend, halo)):
                rows.append({
                    "geo": geo,
                    "channel": ch,
                    "install_date": d,
                    "spend": sp,
                    "halo_organic": hl,
                })

    attribution = pd.DataFrame(rows)
    print(f"  attribution table: {attribution.shape}")

    # ----- Save per-row -----
    pred_dir = Path("data/predictions")
    pred_dir.mkdir(parents=True, exist_ok=True)
    out_path = pred_dir / "mmm_attribution.parquet"
    pl.from_pandas(attribution).write_parquet(out_path, compression="zstd")
    print(f"\nSaved: {out_path}")

    # ----- Per-channel summary -----
    total_organic = float(panel[target_col].sum())
    summary = (
        attribution.groupby("channel")
        .agg(
            halo_total=("halo_organic", "sum"),
            spend_total=("spend", "sum"),
        )
        .reset_index()
    )
    summary["share_of_organic"] = summary["halo_total"] / total_organic
    summary["halo_per_1k_spend"] = summary["halo_total"] / (summary["spend_total"] / 1000.0).replace(0, np.nan)
    summary = summary.sort_values("halo_total", ascending=False)

    # Baseline organic (residual) — observed minus total halo
    total_halo = float(summary["halo_total"].sum())
    print(f"\nObserved total organic_installs (panel sum):  {total_organic:>14,.0f}")
    print(f"Sum of channel halo contributions:            {total_halo:>14,.0f}")
    print(f"Implied baseline / non-attributed organic:    {total_organic - total_halo:>14,.0f}")
    print(f"Halo share of total organic:                  {total_halo / total_organic * 100:>13.1f}%")

    print("\nPer-channel attribution summary:")
    print(summary.to_string(index=False, float_format=lambda v: f"{v:,.2f}"))

    csv_path = pred_dir / "mmm_attribution_summary.csv"
    summary.to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path}")

    # ----- Plot -----
    plot_dir = Path("data/plots")
    plot_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, max(4, 0.4 * len(summary))))
    s = summary.sort_values("halo_total")
    ax.barh(s["channel"], s["halo_total"], color="#2ca02c")
    ax.set_xlabel("Halo organic installs (period total)")
    ax.set_title(f"Halo attribution per channel  (total halo = {total_halo:,.0f} "
                 f"= {total_halo/total_organic*100:.1f}% of observed organic)")
    for i, (ch, val, share) in enumerate(zip(s["channel"], s["halo_total"], s["share_of_organic"])):
        ax.text(val, i, f"  {share*100:.1f}%", va="center", fontsize=9)
    fig.tight_layout()
    plot_path = plot_dir / "mmm_halo_per_channel.png"
    fig.savefig(plot_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {plot_path}")


if __name__ == "__main__":
    main()
