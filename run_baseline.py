"""
Baseline runner: weighted Ridge on logit(organic_share).

Reads:
    data/train/targets_train_clean.parquet
    data/test/targets_test_clean.parquet
Writes:
    data/predictions/baseline_train.parquet
    data/predictions/baseline_test.parquet
    data/plots/baseline_calibration.png
    data/plots/baseline_coefficients.png

Run from project root:
    python run_baseline.py
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
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt

from organic_ratio.utils.config import load_config
from organic_ratio.core.modeling.baseline import (
    fit_baseline,
    predict_baseline,
    coefficient_importance,
)
from organic_ratio.core.modeling.metrics import report, percentage_error, pe_summary


def calibration_plot(y_true, y_pred, weight, save_path: Path, bins: int = 20) -> None:
    """
    Weighted reliability curve: bin predictions by quantile, compare mean
    predicted vs mean actual within each bin.
    """
    df = pd.DataFrame({"y": y_true, "p": y_pred, "w": weight})
    df["bin"] = pd.qcut(df["p"], q=bins, duplicates="drop")
    grp = df.groupby("bin", observed=True).apply(
        lambda g: pd.Series(
            {
                "pred_mean": np.average(g["p"], weights=g["w"]),
                "true_mean": np.average(g["y"], weights=g["w"]),
                "total_w": g["w"].sum(),
            }
        ),
        include_groups=False,
    )

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="perfect")
    ax.scatter(grp["pred_mean"], grp["true_mean"],
               s=np.sqrt(grp["total_w"]) * 0.5, alpha=0.7)
    ax.set_xlabel("predicted organic_share (bin mean)")
    ax.set_ylabel("actual organic_share (bin mean, weighted)")
    ax.set_title("Calibration — test")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)


def pe_distribution_plot(pe: np.ndarray, save_path: Path, clip: float = 2.0) -> None:
    """
    Histogram of percentage error (clipped to ±clip for visibility).
    """
    pe = pe[~np.isnan(pe)]
    pe_clipped = np.clip(pe, -clip, clip)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.hist(pe_clipped * 100, bins=80, edgecolor="black")
    ax.axvline(0, color="k", ls="--", alpha=0.5)
    ax.axvline(np.median(pe) * 100, color="r", ls="--",
               label=f"median = {np.median(pe) * 100:+.1f}%")
    ax.axvline(np.mean(pe) * 100, color="orange", ls="--",
               label=f"mean = {np.mean(pe) * 100:+.1f}%")
    ax.set_xlabel("PE (%)")
    ax.set_ylabel("count")
    ax.set_title(f"Percentage error — test  (n={len(pe):,}, clipped to ±{int(clip * 100)}%)")
    ax.legend()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)


def print_pe_summary(summary: dict, label: str = "") -> None:
    print(f"\n--- PE distribution ({label}, n={summary['n']:,} non-zero targets) ---")
    print(f"  mean    : {summary['mean'] * 100:+.1f}%")
    print(f"  median  : {summary['median'] * 100:+.1f}%")
    print(f"  Q05     : {summary['q05'] * 100:+.1f}%")
    print(f"  Q25     : {summary['q25'] * 100:+.1f}%")
    print(f"  Q75     : {summary['q75'] * 100:+.1f}%")
    print(f"  Q95     : {summary['q95'] * 100:+.1f}%")
    print(f"  within ±10%: {summary['within_10pct'] * 100:5.1f}%")
    print(f"  within ±20%: {summary['within_20pct'] * 100:5.1f}%")
    print(f"  within ±50%: {summary['within_50pct'] * 100:5.1f}%")


def coefficient_plot(coef_df: pd.DataFrame, save_path: Path) -> None:
    df = coef_df.sort_values("coef")
    fig, ax = plt.subplots(figsize=(8, max(4, 0.3 * len(df))))
    colors = ["#d62728" if c < 0 else "#2ca02c" for c in df["coef"]]
    ax.barh(df["feature"], df["coef"], color=colors)
    ax.axvline(0, color="k", lw=0.5)
    ax.set_title("Baseline Ridge coefficients (standardized features)")
    ax.set_xlabel("coefficient")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)


def main() -> None:
    cfg = load_config()

    target = str(cfg.modeling.target)
    weight = str(cfg.modeling.weight)
    features = list(cfg.modeling.features)

    out_cfg = cfg.datasets.targets
    train_path = Path(out_cfg.train_dir) / out_cfg.train_clean_filename
    test_path = Path(out_cfg.test_dir) / out_cfg.test_clean_filename
    print(f"Loading train: {train_path}")
    print(f"Loading test:  {test_path}")

    train = pl.read_parquet(train_path).to_pandas()
    test = pl.read_parquet(test_path).to_pandas()
    print(f"train: {train.shape}, test: {test.shape}")

    # Drop install_date from feature input — it's an identifier, not a feature
    print(f"\nFitting Ridge on {len(features)} numeric features + country_te + platform OHE")
    art = fit_baseline(train, target=target, weight=weight, features=features, alpha=1.0)

    pred_train = predict_baseline(art, train)
    pred_test = predict_baseline(art, test)

    print("\n--- Metrics ---")
    report(train[target].to_numpy(), pred_train, train[weight].to_numpy(dtype=float), label="train")
    report(test[target].to_numpy(), pred_test, test[weight].to_numpy(dtype=float), label="test")

    pe_train = percentage_error(train[target].to_numpy(), pred_train)
    pe_test = percentage_error(test[target].to_numpy(), pred_test)
    print_pe_summary(pe_summary(pe_train), label="train")
    print_pe_summary(pe_summary(pe_test), label="test")

    print("\n--- Top-20 coefficients (|coef|) ---")
    coef_top = coefficient_importance(art, top=20)
    print(coef_top.to_string(index=False))

    # ----- Save predictions -----
    pred_dir = Path("data/predictions")
    pred_dir.mkdir(parents=True, exist_ok=True)

    for df, pred, name in [
        (train, pred_train, "baseline_train.parquet"),
        (test, pred_test, "baseline_test.parquet"),
    ]:
        out = df[["platform", "country_code", "install_date", target, weight]].copy()
        out["pred"] = pred
        out["abs_err"] = np.abs(out[target] - out["pred"])
        out_path = pred_dir / name
        pl.from_pandas(out).write_parquet(out_path, compression="zstd")
        print(f"Saved predictions: {out_path}")

    # ----- Plots -----
    plot_dir = Path("data/plots")
    calibration_plot(
        test[target].to_numpy(),
        pred_test,
        test[weight].to_numpy(dtype=float),
        plot_dir / "baseline_calibration.png",
    )
    coefficient_plot(coef_top, plot_dir / "baseline_coefficients.png")
    pe_distribution_plot(pe_test, plot_dir / "baseline_pe_distribution.png")
    print(
        f"Saved plots:        {plot_dir}/baseline_calibration.png, "
        f"baseline_coefficients.png, baseline_pe_distribution.png"
    )

    # save PE per-row for further inspection
    pred_dir = Path("data/predictions")
    pe_df = test[["platform", "country_code", "install_date", target, weight]].copy()
    pe_df["pred"] = pred_test
    pe_df["pe"] = pe_test
    pl.from_pandas(pe_df).write_parquet(pred_dir / "baseline_pe_test.parquet", compression="zstd")
    print(f"Saved PE table:     {pred_dir}/baseline_pe_test.parquet")


if __name__ == "__main__":
    main()
