"""
MMM in-sample evaluation using pre-computed contributions from trace.

pymc-marketing's multidim MMM stores `channel_contribution`, `control_contribution`,
`intercept_contribution` in the posterior for the TRAINING period dates. We sum
them per (date, geo) to get prediction in scaled space, then calibrate to
original scale by matching observed training totals per geo.

NOTE: This is in-sample (training period) eval. Out-of-sample test eval needs
a working `mmm.predict()` for multidim — separate todo.

Outputs:
    data/predictions/mmm_insample.parquet
    data/plots/mmm_insample_pred_vs_actual.png
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
from organic_ratio.core.modeling.metrics import (
    report,
    percentage_error,
    pe_summary,
    pe_buckets,
)


def _print_pe_summary(s, label):
    print(f"\n--- PE summary ({label}, n={s['n']:,}) ---")
    for k in ("mean", "median", "q05", "q25", "q75", "q95"):
        print(f"  {k:>8s}: {s[k]*100:+.1f}%")
    for k in ("within_10pct", "within_20pct", "within_50pct"):
        print(f"  {k:>12s}: {s[k]*100:5.1f}%")


def _print_pe_buckets(buckets, label):
    total_w = sum(b["weight"] for b in buckets)
    total_n = sum(b["count"] for b in buckets)
    print(f"\n--- PE buckets ({label}, weighted by total_installs, "
          f"total_w={total_w:,.0f}, cohorts={total_n:,}) ---")
    print(f"  {'bucket':>18s}  {'installs':>12s}   {'pct':>6s}   {'cohorts':>8s}")
    for b in buckets:
        lo, hi = b["lo"], b["hi"]
        if np.isinf(lo):
            l = f"  ≤ {int(hi*100):>+4d}%"
        elif np.isinf(hi):
            l = f"  > {int(lo*100):>+4d}%"
        else:
            l = f"{int(lo*100):>+4d}% .. {int(hi*100):>+4d}%"
        print(f"  {l:>18s}  {b['weight']:>12,.0f}   "
              f"{b['pct']*100:>5.1f}%   {b['count']:>8,}")


def pred_vs_actual_plot(df: pd.DataFrame, target: str, save_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(df["pred"], df[target], alpha=0.4, s=15)
    lim = max(df["pred"].max(), df[target].max())
    ax.plot([0, lim], [0, lim], "k--", alpha=0.5, label="perfect")
    ax.set_xlabel(f"predicted {target}")
    ax.set_ylabel(f"actual {target}")
    ax.set_title(f"MMM in-sample predictions  (n={len(df):,})")
    ax.legend()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)


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

    # ----- Sum contributions to get pred[date, geo] in scaled space -----
    print("\nUsing pre-computed contributions from trace:")
    halo = post["channel_contribution"].mean(dim=("chain", "draw")).sum("channel")
    ctrl = post["control_contribution"].mean(dim=("chain", "draw")).sum("control")
    intercept = post["intercept_contribution"].mean(dim=("chain", "draw"))
    print(f"  halo shape:       {dict(zip(halo.dims, halo.shape))}")
    print(f"  ctrl shape:       {dict(zip(ctrl.dims, ctrl.shape))}")
    print(f"  intercept shape:  {dict(zip(intercept.dims, intercept.shape))}")

    pred_scaled = halo + ctrl + intercept   # broadcast (date, geo)
    pred_df = pred_scaled.to_dataframe(name="pred_scaled").reset_index()
    # date in trace is a coord — typically numeric or datetime
    print(f"  pred_scaled rows: {len(pred_df)}")
    print(f"  date coord sample:    {pred_df['date'].iloc[0]}  "
          f"(dtype: {pred_df['date'].dtype})")

    # ----- Load panel and join -----
    panel_cfg = cfg.datasets.mmm_panel
    panel_path = Path(panel_cfg.local_feature_dir) / panel_cfg.train_filename
    if not panel_path.exists():
        panel_path = Path(panel_cfg.local_feature_dir) / panel_cfg.filename
    panel = pl.read_parquet(panel_path).to_pandas()
    panel["install_date"] = pd.to_datetime(panel["install_date"])
    print(f"\n  panel: {panel.shape}, dates={panel['install_date'].min().date()} "
          f"→ {panel['install_date'].max().date()}")

    # try to align date coord with panel install_date
    try:
        pred_df["date"] = pd.to_datetime(pred_df["date"])
        merged = panel.merge(
            pred_df, left_on=["install_date", "geo"],
            right_on=["date", "geo"], how="inner",
        ).drop(columns=["date"])
    except Exception as e:
        print(f"  [warn] date type mismatch: {e}")
        # fallback: index-based merge (assumes trace dates aligned with panel sorted)
        raise

    print(f"  merged: {merged.shape}")

    # ----- Per-geo calibration: scaled → original -----
    print("\nCalibrating per-geo scale (actual_sum / pred_scaled_sum on training):")
    scale_per_geo = (
        merged.groupby("geo")
        .apply(
            lambda g: g[target].sum() / max(g["pred_scaled"].sum(), 1e-9),
            include_groups=False,
        )
        .rename("scale")
    )
    print(scale_per_geo.describe().to_string())

    merged = merged.merge(scale_per_geo.reset_index(), on="geo", how="left")
    merged["pred"] = merged["pred_scaled"] * merged["scale"].fillna(1.0)
    merged["abs_err"] = (merged[target] - merged["pred"]).abs()

    # ----- Metrics -----
    y_true = merged[target].to_numpy(dtype=float)
    y_p = merged["pred"].to_numpy(dtype=float)
    w = merged["total_installs"].to_numpy(dtype=float)

    print("\n--- In-sample metrics ---")
    report(y_true, y_p, w, label="in-sample")

    pe = percentage_error(y_true, y_p)
    _print_pe_summary(pe_summary(pe), label="in-sample")
    _print_pe_buckets(pe_buckets(pe, w), label="in-sample")

    # ----- Save -----
    keep = ["install_date", "geo", "platform", "country_code",
            target, "total_installs", "pred", "abs_err"]
    keep = [c for c in keep if c in merged.columns]
    out = merged[keep]

    pred_dir = Path("data/predictions")
    pred_dir.mkdir(parents=True, exist_ok=True)
    out_path = pred_dir / "mmm_insample.parquet"
    pl.from_pandas(out).write_parquet(out_path, compression="zstd")
    print(f"\nSaved: {out_path}")

    pred_vs_actual_plot(out, target, Path("data/plots/mmm_insample_pred_vs_actual.png"))
    print(f"Saved plot: data/plots/mmm_insample_pred_vs_actual.png")


if __name__ == "__main__":
    main()
