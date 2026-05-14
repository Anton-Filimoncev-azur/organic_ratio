"""
Analyze the fitted MMM trace without invoking pymc-marketing's heavy
posterior-predictive / plot methods.

Inputs:  data/models/mmm/mmm.nc
Outputs:
    data/plots/mmm_channel_coefs.png
    data/plots/mmm_adstock_saturation.png
    data/plots/mmm_geo_baseline.png
    data/predictions/mmm_summary.csv

Run from project root:
    python run_mmm_analyze.py
"""
from dotenv import load_dotenv
load_dotenv()

import sys
from pathlib import Path

import numpy as np
import pandas as pd

SRC_PATH = Path.cwd() / "src"
if not SRC_PATH.exists():
    raise RuntimeError(f"src not found at {SRC_PATH}")
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import arviz as az

from organic_ratio.utils.config import load_config


def _summary(post, var: str) -> pd.DataFrame:
    """ArviZ-like summary for one variable across all its dims."""
    return az.summary(post, var_names=[var], kind="stats", round_to=4)


def plot_channel_coefs(post, save: Path) -> None:
    """Posterior mean ± 94% HDI for channel saturation coefficients (beta)."""
    # Common channel coefficient names in pymc-marketing: 'beta_channel' or
    # 'saturation_beta'. Try several until one exists.
    candidates = ["beta_channel", "saturation_beta_alpha", "channel_contribution",
                  "saturation_beta", "saturation_lam"]
    found = [v for v in candidates if v in post.data_vars]
    if not found:
        print(f"[warn] no channel coef found; available: {list(post.data_vars)}")
        return

    fig, axes = plt.subplots(len(found), 1, figsize=(10, 3 * len(found)))
    if len(found) == 1:
        axes = [axes]
    for ax, var in zip(axes, found):
        sub = post[var].mean(dim=("chain", "draw"))
        hdi = az.hdi(post, var_names=[var], hdi_prob=0.94)[var]
        if "channel" in sub.dims:
            channels = sub["channel"].values.tolist()
            mean = sub.values
            if mean.ndim > 1:
                # collapse geo dim by mean → channel-level summary
                mean = mean.mean(axis=tuple(i for i, d in enumerate(sub.dims) if d != "channel"))
                lo = hdi.values[..., 0]
                hi = hdi.values[..., 1]
                if lo.ndim > 1:
                    lo = lo.mean(axis=tuple(i for i, d in enumerate(sub.dims) if d != "channel"))
                    hi = hi.mean(axis=tuple(i for i, d in enumerate(sub.dims) if d != "channel"))
            else:
                lo = hdi.sel(hdi="lower").values
                hi = hdi.sel(hdi="higher").values
            order = np.argsort(mean)
            ax.barh(np.array(channels)[order], mean[order],
                    xerr=[mean[order] - lo[order], hi[order] - mean[order]],
                    color=["#2ca02c" if m > 0 else "#d62728" for m in mean[order]],
                    ecolor="gray", capsize=3)
            ax.axvline(0, color="k", lw=0.5)
            ax.set_title(f"{var} — posterior mean ± 94% HDI (geo-averaged)")
        else:
            ax.text(0.5, 0.5, f"{var}: dims={sub.dims}", ha="center")
    save.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(save, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save}")


def diagnostics_table(post, save: Path) -> pd.DataFrame:
    summary = az.summary(post, kind="all", round_to=4)
    save.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(save)
    print(f"Saved: {save}  ({len(summary)} parameters)")
    return summary


def diagnostics_overview(summary: pd.DataFrame) -> None:
    print("\n--- Convergence overview ---")
    bad_rhat = summary[summary["r_hat"] > 1.01]
    bad_ess = summary[summary["ess_bulk"] < 200]
    print(f"  total params:       {len(summary)}")
    print(f"  R̂ > 1.01:           {len(bad_rhat)} ({100*len(bad_rhat)/len(summary):.1f}%)")
    print(f"  ESS_bulk < 200:     {len(bad_ess)} ({100*len(bad_ess)/len(summary):.1f}%)")
    if len(bad_rhat):
        print(f"  max R̂:              {summary['r_hat'].max():.3f}")
    if len(bad_ess):
        print(f"  min ESS_bulk:       {summary['ess_bulk'].min():.0f}")

    print("\n--- Top-level params (no dim) ---")
    scalar = summary[summary.index.str.fullmatch(r"[A-Za-z_]+")]
    if len(scalar):
        print(scalar.to_string())


def main() -> None:
    cfg = load_config()

    mmm_path = Path("data/models/mmm/mmm.nc")
    if not mmm_path.exists():
        raise FileNotFoundError(f"MMM trace not found: {mmm_path}")
    print(f"Loading: {mmm_path}")

    idata = az.from_netcdf(mmm_path)
    post = idata.posterior
    print(f"Posterior vars: {list(post.data_vars.keys())}")
    print(f"Coords:         {dict(post.coords)}")

    summary = diagnostics_table(post, Path("data/predictions/mmm_summary.csv"))
    diagnostics_overview(summary)

    plot_channel_coefs(post, Path("data/plots/mmm_channel_coefs.png"))


if __name__ == "__main__":
    main()
