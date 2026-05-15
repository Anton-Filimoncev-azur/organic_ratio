"""
Out-of-sample MMM eval via `sample_posterior_predictive` on test panel.

Loads the multidim MMM via pymc-marketing's MMM class (not just trace), runs
posterior predictive on test period (with training-tail adstock warmup),
extracts per-(date, geo) predictions and computes test metrics.

Outputs:
    data/predictions/mmm_test.parquet
    data/plots/mmm_test_pred_vs_actual.png
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

from pymc_marketing.mmm.multidimensional import MMM

from organic_ratio.utils.config import load_config
from organic_ratio.core.modeling.metrics import (
    report,
    percentage_error,
    pe_summary,
    pe_buckets,
)
from organic_ratio.core.modeling.pymc_model import report_jax_devices


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
    ax.set_title(f"MMM out-of-sample test predictions  (n={len(df):,})")
    ax.legend()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)


def main() -> None:
    cfg = load_config()
    mmm_cfg = cfg.modeling.mmm
    target = str(mmm_cfg.target)

    if str(mmm_cfg.nuts_sampler) == "numpyro":
        report_jax_devices()

    # ----- Load MMM via pymc-marketing class (preserves transformers) -----
    mmm_path = Path("data/models/mmm/mmm.nc")
    if not mmm_path.exists():
        raise FileNotFoundError(f"MMM not found: {mmm_path}")
    print(f"Loading MMM via pymc-marketing: {mmm_path}")
    mmm = MMM.load(str(mmm_path))
    print(f"  target_column:   {getattr(mmm, 'target_column', None)}")
    print(f"  date_column:     {getattr(mmm, 'date_column', None)}")
    print(f"  channel_columns: {getattr(mmm, 'channel_columns', None)}")
    print(f"  control_columns: {getattr(mmm, 'control_columns', None)}")
    print(f"  dims:            {getattr(mmm, 'dims', None)}")
    print(f"  output_var:      {getattr(mmm, 'output_var', '?')}")

    # ----- Load test panel -----
    panel_cfg = cfg.datasets.mmm_panel
    test_path = Path(panel_cfg.local_feature_dir) / panel_cfg.test_filename
    if not test_path.exists():
        raise FileNotFoundError(f"Test panel not found: {test_path}")
    test = pl.read_parquet(test_path).to_pandas()
    test["install_date"] = pd.to_datetime(test["install_date"])
    test = test.sort_values(["geo", "install_date"]).reset_index(drop=True)
    print(f"\nTest panel: {test.shape}, "
          f"{test['install_date'].min().date()} → {test['install_date'].max().date()}, "
          f"geos={test['geo'].nunique()}")

    # ----- Assemble X with the exact columns MMM expects -----
    channel_cols = list(mmm.channel_columns)
    control_cols = list(mmm.control_columns or [])
    date_col = mmm.date_column
    geo_col = "geo"

    keep_cols = [date_col, geo_col] + channel_cols + control_cols
    missing = [c for c in keep_cols if c not in test.columns]
    if missing:
        raise KeyError(f"Test panel missing columns: {missing}")
    X_test = test[keep_cols].copy()
    print(f"X_test: {X_test.shape}, cols={list(X_test.columns)}")

    # ----- Posterior predictive (include training tail for adstock warmup) -----
    print("\nRunning mmm.sample_posterior_predictive(...) with original_scale=True...")
    pp_idata = mmm.sample_posterior_predictive(
        X_test,
        extend_idata=False,
        include_last_observations=True,
        original_scale=True,
        progressbar=True,
    )

    # ----- Extract output variable -----
    output_var = getattr(mmm, "output_var", "y")
    # pp_idata can be either an arviz InferenceData OR a bare xarray Dataset
    if hasattr(pp_idata, "groups"):
        print(f"\nposterior_predictive groups: {list(pp_idata.groups())}")
        pp_group = pp_idata.posterior_predictive
    else:
        print(f"\nReturned object type: {type(pp_idata).__name__}")
        pp_group = pp_idata
    print(f"posterior_predictive vars: {list(pp_group.data_vars)}")

    if output_var not in pp_group.data_vars:
        # try the first variable as fallback
        output_var = list(pp_group.data_vars)[0]
        print(f"  [warn] output_var fallback to: {output_var}")

    pp_da = pp_group[output_var]
    print(f"  {output_var} dims:  {pp_da.dims}")
    print(f"  {output_var} shape: {pp_da.shape}")

    # Collapse all sampling dims (chain/draw or flat 'sample') → posterior mean
    reduce_dims = [d for d in pp_da.dims if d in ("chain", "draw", "sample")]
    pred_mean = pp_da.mean(dim=reduce_dims)
    print(f"  after collapsing {reduce_dims}: dims={pred_mean.dims}, shape={pred_mean.shape}")

    # original_scale=True in sample_posterior_predictive should put pred in
    # original units. No extra calibration needed.

    # Convert to long-form dataframe
    pred_df = pred_mean.to_dataframe(name="pred").reset_index()
    print(f"  pred_df: {pred_df.shape}, cols={list(pred_df.columns)}")

    # If pred has "date" dim (could be from training+test concat), filter to test dates
    if "date" in pred_df.columns:
        pred_df["date"] = pd.to_datetime(pred_df["date"])
        test_dates = pd.to_datetime(test["install_date"].unique())
        pred_df = pred_df[pred_df["date"].isin(test_dates)]
        merge_left = "date"
    else:
        merge_left = None

    # Merge with test by (date, geo)
    if merge_left is not None and "geo" in pred_df.columns:
        merged = test.merge(
            pred_df, left_on=["install_date", "geo"],
            right_on=[merge_left, "geo"], how="inner",
        )
        if merge_left != "install_date":
            merged = merged.drop(columns=[merge_left])
    else:
        raise RuntimeError(
            f"Cannot align posterior_predictive output with test panel. "
            f"pred_df columns: {list(pred_df.columns)}"
        )

    print(f"\nMerged: {merged.shape}")
    if len(merged) == 0:
        raise RuntimeError("Empty merge — date/geo mismatch between pred and test")

    merged["abs_err"] = (merged[target] - merged["pred"]).abs()

    # ----- Metrics -----
    y_true = merged[target].to_numpy(dtype=float)
    y_p = merged["pred"].to_numpy(dtype=float)
    w = merged["total_installs"].to_numpy(dtype=float)

    print("\n--- Test (out-of-sample) metrics ---")
    report(y_true, y_p, w, label="test")

    pe = percentage_error(y_true, y_p)
    _print_pe_summary(pe_summary(pe), label="test")
    _print_pe_buckets(pe_buckets(pe, w), label="test")

    # ----- Save -----
    keep = ["install_date", "geo", "platform", "country_code",
            target, "total_installs", "pred", "abs_err"]
    keep = [c for c in keep if c in merged.columns]
    out = merged[keep]

    pred_dir = Path("data/predictions")
    pred_dir.mkdir(parents=True, exist_ok=True)
    out_path = pred_dir / "mmm_test.parquet"
    pl.from_pandas(out).write_parquet(out_path, compression="zstd")
    print(f"\nSaved: {out_path}")

    pred_vs_actual_plot(out, target, Path("data/plots/mmm_test_pred_vs_actual.png"))
    print(f"Saved plot: data/plots/mmm_test_pred_vs_actual.png")


if __name__ == "__main__":
    main()
