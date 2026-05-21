"""
Per-channel halo attribution from a fitted MMM.

Uses the pre-computed `channel_contribution` array from posterior (in scaled
space), calibrates to original scale via per-geo factor derived from training
target, aggregates per channel.

Inputs:
    data/models/mmm/mmm.nc
    data/features/mmm/mmm_panel.parquet (or train panel)
Outputs:
    data/predictions/mmm_attribution.parquet           — per (geo, date, channel)
    data/predictions/mmm_attribution_summary.csv       — per channel summary
    data/plots/mmm_halo_per_channel.png
    data/plots/mmm_halo_over_time.png
"""
from dotenv import load_dotenv
load_dotenv()

import sys
from pathlib import Path

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

import numpy as np
import pandas as pd
import polars as pl

SRC_PATH = Path.cwd() / "src"
print("Adding SRC_PATH:", SRC_PATH)
if not SRC_PATH.exists():
    raise RuntimeError(f"src not found at {SRC_PATH}")
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import arviz as az

from organic_ratio.utils.config import load_config


def main() -> None:
    cfg = load_config()
    target = str(cfg.modeling.mmm.target)

    # ----- Load trace -----
    mmm_path = Path("data/models/mmm/mmm.nc")
    if not mmm_path.exists():
        raise FileNotFoundError(f"MMM not found: {mmm_path}")
    print(f"Loading: {mmm_path}")
    idata = az.from_netcdf(mmm_path)
    post = idata.posterior

    # ----- Posterior mean contributions (scaled space) -----
    ch = post["channel_contribution"].mean(dim=("chain", "draw"))     # (date, geo, channel)
    if "control_contribution" in post.data_vars:
        ctrl = post["control_contribution"].mean(dim=("chain", "draw")).sum("control")  # (date, geo)
    else:
        print("  no control_contribution in posterior — treating controls as 0")
        ctrl = 0.0
    intercept = post["intercept_contribution"].mean(dim=("chain", "draw"))           # (geo,)

    print(f"  channel_contribution dims: {ch.dims}, shape: {ch.shape}")
    print(f"  channels: {ch['channel'].values.tolist()}")
    print(f"  geos:     {ch['geo'].values.tolist()}")

    # ----- Load training panel to get scaling factor -----
    panel_cfg = cfg.datasets.mmm_panel
    panel_path = Path(panel_cfg.local_feature_dir) / panel_cfg.train_filename
    if not panel_path.exists():
        panel_path = Path(panel_cfg.local_feature_dir) / panel_cfg.filename
    panel = pl.read_parquet(panel_path).to_pandas()
    panel["install_date"] = pd.to_datetime(panel["install_date"])
    print(f"  panel: {panel.shape}, dates={panel['install_date'].min().date()} "
          f"→ {panel['install_date'].max().date()}")

    # ----- Per-geo calibration: scaled → original -----
    # pred_scaled[date, geo] = halo + ctrl + intercept   (where halo = sum over channels)
    halo_scaled = ch.sum("channel")                              # (date, geo)
    pred_scaled = halo_scaled + ctrl + intercept                 # (date, geo)

    pred_df = pred_scaled.to_dataframe(name="pred_scaled").reset_index()
    pred_df["date"] = pd.to_datetime(pred_df["date"])
    merged = panel.merge(pred_df, left_on=["install_date", "geo"],
                         right_on=["date", "geo"], how="inner").drop(columns=["date"])

    scale_per_geo = (
        merged.groupby("geo")
        .apply(
            lambda g: g[target].sum() / max(g["pred_scaled"].sum(), 1e-9),
            include_groups=False,
        )
        .rename("scale")
    )
    global_scale = float(scale_per_geo.mean())
    print(f"\nPer-geo scale: mean={global_scale:.2f}, "
          f"std={scale_per_geo.std():.2f}, "
          f"min={scale_per_geo.min():.2f}, max={scale_per_geo.max():.2f}")

    # ----- Apply scale to channel_contribution -----
    # ch shape: (date, geo, channel). Multiply by scale_per_geo (geo,).
    scale_xr = scale_per_geo.reindex(ch["geo"].values).to_xarray().rename({"geo": "geo"})
    ch_original = ch * scale_xr                                  # (date, geo, channel)

    # ----- Flatten to long-form attribution table -----
    attribution = (
        ch_original.to_dataframe(name="halo_organic").reset_index()
        .rename(columns={"date": "install_date"})
    )
    attribution["install_date"] = pd.to_datetime(attribution["install_date"])

    # join spend back for ROAS
    spend_cols = [c for c in panel.columns if c.startswith("spend_")]
    spend_long = panel.melt(
        id_vars=["install_date", "geo"],
        value_vars=spend_cols,
        var_name="channel",
        value_name="spend",
    )
    attribution = attribution.merge(
        spend_long, on=["install_date", "geo", "channel"], how="left",
    )
    print(f"\nAttribution rows: {len(attribution):,}")

    # ----- Per-channel summary -----
    total_actual = float(panel[target].sum())
    per_channel = (
        attribution.groupby("channel")
        .agg(
            halo_total=("halo_organic", "sum"),
            spend_total=("spend", "sum"),
        )
        .reset_index()
    )
    per_channel["share_of_total_organic"] = per_channel["halo_total"] / total_actual
    per_channel["halo_per_1k_spend"] = (
        per_channel["halo_total"] / (per_channel["spend_total"] / 1000.0).replace(0, np.nan)
    )
    per_channel = per_channel.sort_values("halo_total", ascending=False)

    total_halo = float(per_channel["halo_total"].sum())
    baseline_implied = total_actual - total_halo

    print(f"\nObserved total organic_installs (panel sum):  {total_actual:>14,.0f}")
    print(f"Sum of channel halo contributions:            {total_halo:>14,.0f}")
    print(f"Implied baseline / non-attributed organic:    {baseline_implied:>14,.0f}")
    print(f"Halo share of total organic:                  {total_halo / total_actual * 100:>13.1f}%")

    print("\nPer-channel attribution summary:")
    print(per_channel.to_string(index=False, float_format=lambda v: f"{v:,.2f}"))

    # ----- Save outputs -----
    pred_dir = Path("data/predictions")
    pred_dir.mkdir(parents=True, exist_ok=True)
    pl.from_pandas(attribution).write_parquet(
        pred_dir / "mmm_attribution.parquet", compression="zstd",
    )
    print(f"\nSaved: {pred_dir / 'mmm_attribution.parquet'}")

    csv_path = pred_dir / "mmm_attribution_summary.csv"
    per_channel.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")

    # ----- Plot 1: per-channel halo bars -----
    plot_dir = Path("data/plots")
    plot_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, max(4, 0.4 * len(per_channel))))
    s = per_channel.sort_values("halo_total")
    bars = ax.barh(s["channel"], s["halo_total"], color="#2ca02c", alpha=0.85)
    ax.set_xlabel("Halo organic installs (period total)")
    ax.set_title(
        f"Per-channel halo attribution\n"
        f"total halo = {total_halo:,.0f}  ({total_halo/total_actual*100:.1f}% of {total_actual:,.0f} observed)"
    )
    for i, (val, share) in enumerate(zip(s["halo_total"], s["share_of_total_organic"])):
        ax.text(val, i, f"  {share*100:.1f}%", va="center", fontsize=9)
    fig.tight_layout()
    plot_path = plot_dir / "mmm_halo_per_channel.png"
    fig.savefig(plot_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {plot_path}")

    # ----- Plot 2: halo over time, stacked area -----
    halo_over_time = (
        attribution.groupby(["install_date", "channel"])["halo_organic"]
        .sum().unstack(fill_value=0).sort_index()
    )
    # ordering: largest contributors at bottom
    col_order = halo_over_time.sum().sort_values(ascending=False).index.tolist()
    halo_over_time = halo_over_time[col_order]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.stackplot(halo_over_time.index, halo_over_time.T.values,
                 labels=halo_over_time.columns, alpha=0.85)
    ax.set_title("Halo organic contribution over time (stacked by channel)")
    ax.set_ylabel("organic installs / day")
    ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), fontsize=8)
    fig.tight_layout()
    plot_path = plot_dir / "mmm_halo_over_time.png"
    fig.savefig(plot_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {plot_path}")


if __name__ == "__main__":
    main()
