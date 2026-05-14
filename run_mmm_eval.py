"""
Out-of-sample MMM evaluation — manual prediction from posterior means.

mmm.predict() in multidim MMM may aggregate across geos. We bypass it by
reproducing the model formula in numpy from the saved posterior:

    pred[g, t] = intercept[g]
               + Σ_c  β_c[g] · saturation(adstock(spend_c[g, :]; α_c[g]); λ_c[g])[t]
               + Σ_d  γ_d · dow_d[t]

Then compare to actual organic_installs on the test panel.

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
import arviz as az

from organic_ratio.utils.config import load_config
from organic_ratio.core.modeling.metrics import (
    report,
    percentage_error,
    pe_summary,
    pe_buckets,
)


# ----- MMM transforms (mirror pymc-marketing defaults) -----

def geometric_adstock_normalized(spend: np.ndarray, alpha: float, l_max: int) -> np.ndarray:
    weights = alpha ** np.arange(l_max + 1)
    weights = weights / weights.sum()
    n = len(spend)
    out = np.zeros(n, dtype=float)
    for t in range(n):
        k_max = min(t + 1, l_max + 1)
        # spend[t], spend[t-1], ..., spend[t-k_max+1] weighted
        out[t] = np.sum(weights[:k_max] * spend[t - np.arange(k_max)])
    return out


def logistic_saturation(x: np.ndarray, beta: float, lam: float) -> np.ndarray:
    z = np.exp(-lam * x)
    return beta * (1.0 - z) / (1.0 + z)


# ----- Parameter discovery -----

PARAM_CANDIDATES = {
    "alpha":     ["adstock_alpha", "alpha", "adstock_alpha_channel"],
    "beta":      ["saturation_beta", "beta_channel", "saturation_beta_channel"],
    "lam":       ["saturation_lam", "lam_channel", "saturation_lam_channel"],
    "intercept": ["intercept", "baseline", "intercept_geo"],
    "gamma":     ["gamma_control", "gamma", "control_contribution"],
}


def _find_var(post, kind: str, required: bool = True):
    for name in PARAM_CANDIDATES[kind]:
        if name in post.data_vars:
            return name
    if required:
        raise KeyError(
            f"No {kind} variable. Looked for {PARAM_CANDIDATES[kind]}. "
            f"Got: {list(post.data_vars)}"
        )
    return None


def extract_params(post):
    a_name = _find_var(post, "alpha")
    b_name = _find_var(post, "beta")
    l_name = _find_var(post, "lam")
    i_name = _find_var(post, "intercept", required=False)
    g_name = _find_var(post, "gamma", required=False)
    print(f"  posterior vars: α={a_name}  β={b_name}  λ={l_name}  "
          f"intercept={i_name}  gamma={g_name}")

    def _mean(name):
        return post[name].mean(dim=("chain", "draw")) if name else None

    alpha = _mean(a_name); beta = _mean(b_name); lam = _mean(l_name)
    # ensure (geo, channel) order
    for arr in (alpha, beta, lam):
        if arr.dims == ("channel", "geo"):
            arr_t = arr.transpose("geo", "channel")
            alpha = alpha.transpose("geo", "channel") if alpha.dims == ("channel", "geo") else alpha
            beta = beta.transpose("geo", "channel") if beta.dims == ("channel", "geo") else beta
            lam = lam.transpose("geo", "channel") if lam.dims == ("channel", "geo") else lam
            break

    intercept = _mean(i_name)
    gamma = _mean(g_name)

    return {
        "alpha": alpha,
        "beta": beta,
        "lam": lam,
        "intercept": intercept,
        "gamma": gamma,
        "geos": alpha["geo"].values.tolist(),
        "channels": alpha["channel"].values.tolist(),
    }


# ----- Manual predict -----

def predict_panel(panel: pd.DataFrame, params: dict, adstock_l_max: int,
                  target_col: str) -> pd.DataFrame:
    """
    Apply the MMM model formula row-by-row using posterior means.
    Returns a copy of panel with 'pred' column.
    """
    panel = panel.sort_values(["geo", "install_date"]).reset_index(drop=True).copy()

    geos = params["geos"]
    channels = params["channels"]
    alpha = params["alpha"].values   # (geo, channel)
    beta = params["beta"].values
    lam = params["lam"].values
    intercept = params["intercept"].values if params["intercept"] is not None else None
    gamma = params["gamma"]

    dow_cols = [c for c in panel.columns if c.startswith("dow_")]
    if gamma is not None and "control" in gamma.dims:
        control_names = gamma["control"].values.tolist()
        gamma_vals = {}
        for c in control_names:
            v = gamma.sel(control=c)
            # collapse any non-control dim (e.g., geo) by mean
            extra_dims = [d for d in v.dims if d != "control"]
            if extra_dims:
                v = v.mean(dim=extra_dims)
            gamma_vals[c] = float(v.item() if v.ndim == 0 else v.values)
    else:
        gamma_vals = {}

    pred = np.zeros(len(panel), dtype=float)

    for gi, geo in enumerate(geos):
        geo_mask = panel["geo"].to_numpy() == geo
        idx = np.where(geo_mask)[0]
        if len(idx) == 0:
            continue
        geo_df = panel.iloc[idx]

        # baseline
        base = float(intercept[gi]) if intercept is not None else 0.0
        pred[idx] = base

        # channel halo
        for ci, ch in enumerate(channels):
            if ch not in geo_df.columns:
                continue
            spend = geo_df[ch].to_numpy(dtype=float)
            adstocked = geometric_adstock_normalized(spend, float(alpha[gi, ci]), adstock_l_max)
            halo = logistic_saturation(adstocked, float(beta[gi, ci]), float(lam[gi, ci]))
            pred[idx] += halo

    # seasonality controls
    for dow_col in dow_cols:
        if dow_col in gamma_vals:
            pred += gamma_vals[dow_col] * panel[dow_col].to_numpy(dtype=float)

    panel["pred"] = pred
    return panel


# ----- Helpers -----

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
    ax.set_title(f"MMM test predictions  (n={len(df):,})")
    ax.legend()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)


def main() -> None:
    cfg = load_config()
    mmm_cfg = cfg.modeling.mmm

    target = str(mmm_cfg.target)
    adstock_l_max = int(mmm_cfg.adstock_l_max)

    # ----- Load trace -----
    mmm_path = Path("data/models/mmm/mmm.nc")
    if not mmm_path.exists():
        raise FileNotFoundError(f"MMM not found: {mmm_path}")
    print(f"Loading: {mmm_path}")
    idata = az.from_netcdf(mmm_path)
    params = extract_params(idata.posterior)
    print(f"  geos:     {len(params['geos'])}")
    print(f"  channels: {len(params['channels'])}")

    # ----- Load FULL panel (need train tail for adstock warmup on test) -----
    panel_cfg = cfg.datasets.mmm_panel
    full_path = Path(panel_cfg.local_feature_dir) / panel_cfg.filename
    if not full_path.exists():
        raise FileNotFoundError(f"Full panel not found: {full_path}")
    panel = pl.read_parquet(full_path).to_pandas()
    panel["install_date"] = pd.to_datetime(panel["install_date"])
    print(f"  full panel: {panel.shape}")

    # ----- Predict on full panel, then slice test -----
    print("\nComputing predictions (manual model-formula)...")
    panel_pred = predict_panel(panel, params, adstock_l_max, target)

    test_start = pd.to_datetime(str(cfg.test_start_date))
    test_end = pd.to_datetime(str(cfg.test_end_date))
    test = panel_pred[
        (panel_pred["install_date"] >= test_start) &
        (panel_pred["install_date"] < test_end)
    ].copy()
    test["abs_err"] = (test[target] - test["pred"]).abs()
    print(f"  test rows: {len(test)}  "
          f"({test['install_date'].min().date()} → {test['install_date'].max().date()})")

    # ----- Metrics -----
    y_true = test[target].to_numpy(dtype=float)
    y_p = test["pred"].to_numpy(dtype=float)
    w = test["total_installs"].to_numpy(dtype=float)

    print("\n--- Test metrics ---")
    report(y_true, y_p, w, label="test")

    pe = percentage_error(y_true, y_p)
    _print_pe_summary(pe_summary(pe), label="test")
    _print_pe_buckets(pe_buckets(pe, w), label="test")

    # ----- Save -----
    keep_cols = ["install_date", "geo", "platform", "country_code",
                 target, "total_installs", "pred", "abs_err"]
    keep_cols = [c for c in keep_cols if c in test.columns]
    out = test[keep_cols]

    pred_dir = Path("data/predictions")
    pred_dir.mkdir(parents=True, exist_ok=True)
    out_path = pred_dir / "mmm_test.parquet"
    pl.from_pandas(out).write_parquet(out_path, compression="zstd")
    print(f"\nSaved: {out_path}")

    pred_vs_actual_plot(out, target, Path("data/plots/mmm_test_pred_vs_actual.png"))
    print(f"Saved plot: data/plots/mmm_test_pred_vs_actual.png")


if __name__ == "__main__":
    main()
