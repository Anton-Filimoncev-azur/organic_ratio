"""
Out-of-sample MMM evaluation.

Loads:
    data/models/mmm/mmm.nc                  — fitted MMM (trained on train panel)
    data/features/mmm/mmm_panel_test.parquet
Computes posterior-predictive on test, prints test metrics, writes per-row
predictions.

Outputs:
    data/predictions/mmm_test.parquet
    data/plots/mmm_test_pred_vs_actual.png

Run from project root:
    python run_mmm_eval.py
"""
from dotenv import load_dotenv
load_dotenv()

import sys
from pathlib import Path

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
    total = sum(b["count"] for b in buckets)
    print(f"\n--- PE buckets ({label}, n={total:,}) ---")
    for b in buckets:
        lo, hi = b["lo"], b["hi"]
        if np.isinf(lo):
            l = f"  ≤ {int(hi*100):>+4d}%"
        elif np.isinf(hi):
            l = f"  > {int(lo*100):>+4d}%"
        else:
            l = f"{int(lo*100):>+4d}% .. {int(hi*100):>+4d}%"
        print(f"  {l:>18s}  {b['count']:>8,}   {b['pct']*100:>5.1f}%")


def pred_vs_actual_plot(df: pd.DataFrame, save_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(df["pred"], df["organic_installs"], alpha=0.4, s=15)
    lim = max(df["pred"].max(), df["organic_installs"].max())
    ax.plot([0, lim], [0, lim], "k--", alpha=0.5, label="perfect")
    ax.set_xlabel("predicted organic_installs")
    ax.set_ylabel("actual organic_installs")
    ax.set_title(f"Test predictions  (n={len(df):,})")
    ax.legend()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)


def main() -> None:
    cfg = load_config()

    # ----- Load model -----
    mmm_path = Path("data/models/mmm/mmm.nc")
    if not mmm_path.exists():
        raise FileNotFoundError(f"MMM not found: {mmm_path}")
    print(f"Loading MMM: {mmm_path}")
    if str(cfg.modeling.mmm.nuts_sampler) == "numpyro":
        report_jax_devices()
    mmm = MMM.load(str(mmm_path))

    # ----- Load test panel -----
    panel_cfg = cfg.datasets.mmm_panel
    test_path = Path(panel_cfg.local_feature_dir) / panel_cfg.test_filename
    if not test_path.exists():
        raise FileNotFoundError(f"Test panel not found: {test_path}")
    test = pl.read_parquet(test_path).to_pandas()
    test["install_date"] = pd.to_datetime(test["install_date"])
    print(f"Test panel: {test.shape}, "
          f"dates={test['install_date'].min().date()} → {test['install_date'].max().date()}, "
          f"geos={test['geo'].nunique()}")

    target = str(cfg.modeling.mmm.target)
    channel_cols = [c for c in test.columns if c.startswith("spend_")]
    control_cols = [c for c in test.columns if c.startswith("dow_")]
    geo_dim = bool(cfg.modeling.mmm.geo_dim)

    keep = ["install_date"] + (["geo"] if geo_dim else []) + channel_cols + control_cols
    X_test = test[keep].reset_index(drop=True)

    # ----- Predict -----
    print("\nPredicting on test (this uses MMM internal posterior predictive)...")
    # include_last_observations=True so adstock carryover from end-of-train is applied
    y_pred = mmm.predict(X_test, include_last_observations=True)
    y_pred = np.asarray(y_pred)

    id_cols = ["install_date"] + (["geo", "platform", "country_code"]
                                    if "platform" in test.columns else [])
    id_cols = [c for c in id_cols if c in test.columns]
    pred_df = test[id_cols + [target, "total_installs"]].reset_index(drop=True).copy()
    pred_df["pred"] = y_pred
    pred_df["abs_err"] = (pred_df[target] - pred_df["pred"]).abs()

    # ----- Metrics -----
    y_true = pred_df[target].to_numpy(dtype=float)
    y_p = pred_df["pred"].to_numpy(dtype=float)
    w = pred_df["total_installs"].to_numpy(dtype=float)

    print("\n--- Test metrics ---")
    report(y_true, y_p, w, label="test")

    pe = percentage_error(y_true, y_p)
    _print_pe_summary(pe_summary(pe), label="test")
    _print_pe_buckets(pe_buckets(pe), label="test")

    # ----- Save predictions + plot -----
    pred_dir = Path("data/predictions")
    pred_dir.mkdir(parents=True, exist_ok=True)
    out_path = pred_dir / "mmm_test.parquet"
    pl.from_pandas(pred_df).write_parquet(out_path, compression="zstd")
    print(f"\nSaved: {out_path}")

    pred_vs_actual_plot(pred_df, Path("data/plots/mmm_test_pred_vs_actual.png"))
    print(f"Saved plot: data/plots/mmm_test_pred_vs_actual.png")


if __name__ == "__main__":
    main()
