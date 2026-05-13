"""
Train hierarchical Bayesian model for cohort organic_share (PyMC).

Reads cleaned train / test parquets, fits the model on train, evaluates on
both splits with posterior-mean point predictions, saves trace + prep.

Outputs:
    data/models/pymc/trace.nc           — ArviZ InferenceData
    data/models/pymc/prep.pkl           — preprocessing artifacts
    data/predictions/pymc_{train,test}.parquet
    data/plots/pymc_{calibration,beta,trace_summary}.png

Run from project root:
    python run_train.py
"""
from dotenv import load_dotenv
load_dotenv()

import os

# CPU-only fallback: uncomment to force multiple host CPU devices so
# `chain_method: parallel` works on a CPU-only machine. Not needed on GPU
# (use `chain_method: vectorized` in parameters.yml instead).
# os.environ.setdefault(
#     "XLA_FLAGS",
#     "--xla_force_host_platform_device_count=4",
# )

import sys
import pickle
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
import arviz as az

from organic_ratio.utils.config import load_config
from organic_ratio.core.modeling.preprocess import fit_prep, transform_prep
from organic_ratio.core.modeling.pymc_model import (
    build_model,
    sample,
    posterior_mean_params,
    predict_mean,
    beta_summary,
    sampler_diagnostics,
    report_jax_devices,
)
from organic_ratio.core.modeling.metrics import (
    report,
    percentage_error,
    pe_summary,
    pe_buckets,
)


def _print_pe_buckets(buckets, label):
    total = sum(b["count"] for b in buckets)
    print(f"\n--- PE buckets ({label}, n={total:,}) ---")
    print(f"  {'bucket':>18s}  {'count':>8s}   {'pct':>6s}")
    for b in buckets:
        lo, hi = b["lo"], b["hi"]
        if np.isinf(lo):
            label_str = f"  ≤ {int(hi*100):>+4d}%"
        elif np.isinf(hi):
            label_str = f"  > {int(lo*100):>+4d}%"
        else:
            label_str = f"{int(lo*100):>+4d}% .. {int(hi*100):>+4d}%"
        print(f"  {label_str:>18s}  {b['count']:>8,}   {b['pct']*100:>5.1f}%")


def _print_pe_summary(summary, label):
    print(f"\n--- PE summary ({label}, n={summary['n']:,}) ---")
    for k in ("mean", "median", "q05", "q25", "q75", "q95"):
        print(f"  {k:>8s}: {summary[k]*100:+.1f}%")
    for k in ("within_10pct", "within_20pct", "within_50pct"):
        print(f"  {k:>12s}: {summary[k]*100:5.1f}%")


def beta_plot(beta_df: pd.DataFrame, save_path: Path) -> None:
    df = beta_df.sort_values("mean")
    fig, ax = plt.subplots(figsize=(8, max(4, 0.3 * len(df))))
    errs = np.vstack([df["mean"] - df["hdi_3%"], df["hdi_97%"] - df["mean"]])
    colors = ["#d62728" if m < 0 else "#2ca02c" for m in df["mean"]]
    ax.errorbar(df["mean"], df["feature"], xerr=errs, fmt="o",
                ecolor="gray", elinewidth=1, capsize=2,
                markerfacecolor=colors, markeredgecolor="black")
    ax.axvline(0, color="k", lw=0.5)
    ax.set_title("PyMC β coefficients (posterior mean ± 94% HDI)")
    ax.set_xlabel("β (standardized features)")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)


def calibration_plot(y_true, y_pred, weight, save_path: Path, bins: int = 20) -> None:
    df = pd.DataFrame({"y": y_true, "p": y_pred, "w": weight})
    df["bin"] = pd.qcut(df["p"], q=bins, duplicates="drop")
    grp = df.groupby("bin", observed=True).apply(
        lambda g: pd.Series({
            "pred_mean": np.average(g["p"], weights=g["w"]),
            "true_mean": np.average(g["y"], weights=g["w"]),
            "total_w": g["w"].sum(),
        }),
        include_groups=False,
    )
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="perfect")
    ax.scatter(grp["pred_mean"], grp["true_mean"],
               s=np.sqrt(grp["total_w"]) * 0.5, alpha=0.7)
    ax.set_xlabel("predicted organic_share (bin mean)")
    ax.set_ylabel("actual organic_share (bin mean, weighted)")
    ax.set_title("PyMC calibration — test")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.legend()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)


def main() -> None:
    cfg = load_config()

    target = str(cfg.modeling.target)
    weight = str(cfg.modeling.weight)
    features = list(cfg.modeling.features)

    pymc_cfg = cfg.modeling.pymc
    draws = int(pymc_cfg.draws)
    tune = int(pymc_cfg.tune)
    chains = int(pymc_cfg.chains)
    target_accept = float(pymc_cfg.target_accept)
    nuts_sampler = str(pymc_cfg.nuts_sampler)
    chain_method = str(pymc_cfg.get("chain_method", "parallel"))
    beta_prior_sigma = float(pymc_cfg.get("beta_prior_sigma", 1.0))
    random_seed = int(pymc_cfg.random_seed)

    if nuts_sampler == "numpyro":
        report_jax_devices()

    out_cfg = cfg.datasets.targets
    train_path = Path(out_cfg.train_dir) / out_cfg.train_clean_filename
    test_path = Path(out_cfg.test_dir) / out_cfg.test_clean_filename

    print(f"Loading train: {train_path}")
    print(f"Loading test:  {test_path}")
    train = pl.read_parquet(train_path).to_pandas()
    test = pl.read_parquet(test_path).to_pandas()
    print(f"train: {train.shape}, test: {test.shape}")

    # Reconstruct integer success/trial counts for Binomial likelihood
    for df in (train, test):
        df["organic_installs"] = (df[target] * df[weight]).round().astype(int)

    # ---------- Prep ----------
    prep = fit_prep(train, features)
    print(f"\nPrep fitted: {len(prep.feature_names)} features, "
          f"{len(prep.country_categories)} countries, "
          f"{len(prep.platform_categories)} platforms")

    X_train, c_train, p_train = transform_prep(train, prep)
    X_test, c_test, p_test = transform_prep(test, prep)

    # ---------- Model ----------
    model = build_model(
        X=X_train,
        country_idx=c_train,
        platform_idx=p_train,
        organic_installs=train["organic_installs"].to_numpy(),
        total_installs=train[weight].to_numpy(),
        n_countries=len(prep.country_categories),
        n_platforms=len(prep.platform_categories),
        feature_names=prep.feature_names,
        beta_prior_sigma=beta_prior_sigma,
    )
    print(f"\nModel built on {X_train.shape[0]} obs × {X_train.shape[1]} features")

    # ---------- Sampling ----------
    print(f"Sampling: draws={draws}, tune={tune}, chains={chains}, "
          f"target_accept={target_accept}, sampler={nuts_sampler}"
          + (f", chain_method={chain_method}" if nuts_sampler == "numpyro" else ""))
    trace = sample(
        model,
        draws=draws, tune=tune, chains=chains,
        target_accept=target_accept,
        nuts_sampler=nuts_sampler,
        chain_method=chain_method,
        random_seed=random_seed,
    )

    # ---------- Save trace + prep ----------
    model_dir = Path("data/models/pymc")
    model_dir.mkdir(parents=True, exist_ok=True)
    trace_path = model_dir / "trace.nc"
    trace.to_netcdf(trace_path)
    print(f"\nSaved trace: {trace_path}")

    prep_path = model_dir / "prep.pkl"
    with open(prep_path, "wb") as f:
        pickle.dump(prep, f)
    print(f"Saved prep:  {prep_path}")

    # ---------- Diagnostics ----------
    print("\n--- Sampler diagnostics ---")
    diag = sampler_diagnostics(trace)
    print(diag.to_string())

    # ---------- Predictions (posterior mean) ----------
    params = posterior_mean_params(trace)
    pred_train = predict_mean(params, X_train, c_train, p_train)
    pred_test = predict_mean(params, X_test, c_test, p_test)

    print("\n--- Metrics ---")
    report(train[target].to_numpy(), pred_train, train[weight].to_numpy(dtype=float), label="train")
    report(test[target].to_numpy(), pred_test, test[weight].to_numpy(dtype=float), label="test")

    pe_train = percentage_error(train[target].to_numpy(), pred_train)
    pe_test = percentage_error(test[target].to_numpy(), pred_test)
    _print_pe_summary(pe_summary(pe_test), label="test")
    _print_pe_buckets(pe_buckets(pe_train), label="train")
    _print_pe_buckets(pe_buckets(pe_test), label="test")

    # ---------- β summary ----------
    print("\n--- Top-20 |β| (posterior mean ± 94% HDI) ---")
    beta_df = beta_summary(trace, prep.feature_names, top=20)
    print(beta_df.to_string(index=False))

    # ---------- Save predictions + plots ----------
    pred_dir = Path("data/predictions")
    pred_dir.mkdir(parents=True, exist_ok=True)

    for df, pred, name in [
        (train, pred_train, "pymc_train.parquet"),
        (test, pred_test, "pymc_test.parquet"),
    ]:
        out = df[["platform", "country_code", "install_date", target, weight]].copy()
        out["pred"] = pred
        out["abs_err"] = np.abs(out[target] - out["pred"])
        pl.from_pandas(out).write_parquet(pred_dir / name, compression="zstd")
        print(f"Saved predictions: {pred_dir/name}")

    plot_dir = Path("data/plots")
    calibration_plot(test[target].to_numpy(), pred_test,
                     test[weight].to_numpy(dtype=float),
                     plot_dir / "pymc_calibration.png")
    beta_plot(beta_df, plot_dir / "pymc_beta.png")
    print(f"Saved plots: {plot_dir}/pymc_calibration.png, pymc_beta.png")


if __name__ == "__main__":
    main()
