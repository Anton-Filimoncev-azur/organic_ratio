"""
Fit halo MMM on organic_installs panel.

Inputs:  data/features/mmm/mmm_panel.parquet
Outputs:
    data/models/mmm/mmm.nc            — fitted MMM (pymc-marketing serialized)
    data/predictions/mmm_train.parquet
    data/plots/mmm_contribution.png
    data/plots/mmm_saturation.png
    data/plots/mmm_channel_share.png
"""
from dotenv import load_dotenv
load_dotenv()

import os
# Uncomment for CPU parallel chains:
# os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=4")

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

from organic_ratio.utils.config import load_config
from organic_ratio.core.modeling.mmm_model import build_mmm
from organic_ratio.core.modeling.pymc_model import report_jax_devices


def _channel_columns(panel: pd.DataFrame) -> list[str]:
    return [c for c in panel.columns if c.startswith("spend_")]


def _control_columns(panel: pd.DataFrame) -> list[str]:
    return [c for c in panel.columns if c.startswith("dow_")]


def main() -> None:
    cfg = load_config()
    mmm_cfg = cfg.modeling.mmm

    if str(mmm_cfg.nuts_sampler) == "numpyro":
        report_jax_devices()

    # ----- Load panel -----
    panel_cfg = cfg.datasets.mmm_panel
    panel_path = Path(panel_cfg.local_feature_dir) / panel_cfg.filename
    if not panel_path.exists():
        raise FileNotFoundError(
            f"MMM panel not found: {panel_path}. Run run_mmm_data.py first."
        )
    print(f"Loading panel: {panel_path}")
    panel = pl.read_parquet(panel_path).to_pandas()
    panel["install_date"] = pd.to_datetime(panel["install_date"])
    print(f"Panel: {panel.shape}, geos={panel['geo'].nunique()}, "
          f"dates={panel['install_date'].min().date()} → {panel['install_date'].max().date()}")

    target = str(mmm_cfg.target)
    channel_cols = _channel_columns(panel)
    control_cols = _control_columns(panel)
    print(f"Target:       {target}")
    print(f"Channels ({len(channel_cols)}): {channel_cols}")
    print(f"Controls ({len(control_cols)}): {control_cols}")

    # Multidim MMM: long-format panel with date + geo + channels + controls.
    geo_dim = bool(mmm_cfg.geo_dim)
    dims = ("geo",) if geo_dim else None

    keep = ["install_date"] + (["geo"] if geo_dim else []) + channel_cols + control_cols
    X = panel[keep].reset_index(drop=True)
    y = panel[target].astype(float).reset_index(drop=True)

    if geo_dim:
        print(f"Per-geo MMM: {panel['geo'].nunique()} geos × "
              f"{panel['install_date'].nunique()} dates")

    # ----- Build model -----
    mmm = build_mmm(
        channel_columns=channel_cols,
        adstock_l_max=int(mmm_cfg.adstock_l_max),
        saturation_kind=str(mmm_cfg.saturation),
        dims=dims,
        yearly_seasonality=int(mmm_cfg.yearly_seasonality),
        control_columns=control_cols if control_cols else None,
        date_column="install_date",
        target_column=target,
    )

    # ----- Fit -----
    print(f"\nFitting MMM: draws={mmm_cfg.draws}, tune={mmm_cfg.tune}, "
          f"chains={mmm_cfg.chains}, target_accept={mmm_cfg.target_accept}, "
          f"sampler={mmm_cfg.nuts_sampler}, chain_method={mmm_cfg.chain_method}")

    sampler_kwargs = {}
    if str(mmm_cfg.nuts_sampler) == "numpyro":
        sampler_kwargs["chain_method"] = str(mmm_cfg.chain_method)

    mmm.fit(
        X=X,
        y=y,
        draws=int(mmm_cfg.draws),
        tune=int(mmm_cfg.tune),
        chains=int(mmm_cfg.chains),
        target_accept=float(mmm_cfg.target_accept),
        nuts_sampler=str(mmm_cfg.nuts_sampler),
        nuts_sampler_kwargs=sampler_kwargs or None,
        random_seed=int(mmm_cfg.random_seed),
        progressbar=True,
    )

    # ----- Save trace IMMEDIATELY (the most important artifact) -----
    model_dir = Path("data/models/mmm")
    model_dir.mkdir(parents=True, exist_ok=True)
    mmm_path = model_dir / "mmm.nc"
    print(f"\nSaving MMM trace → {mmm_path}")
    mmm.save(str(mmm_path))
    print(f"Saved.")

    compute_predict = bool(mmm_cfg.get("compute_predict", False))
    save_plots = bool(mmm_cfg.get("save_plots", False))

    # ----- (optional) In-sample predictions — can be slow for large per-geo models
    if compute_predict:
        print("\nComputing posterior predictive (in-sample)...")
        try:
            y_pred = mmm.predict(X)
            id_cols = ["install_date"] + (["geo", "platform", "country_code"] if geo_dim else [])
            id_cols = [c for c in id_cols if c in panel.columns]
            pred_df = panel[id_cols + [target]].reset_index(drop=True).copy()
            pred_df["pred"] = np.asarray(y_pred)
            pred_df["abs_err"] = (pred_df[target] - pred_df["pred"]).abs()

            pred_dir = Path("data/predictions")
            pred_dir.mkdir(parents=True, exist_ok=True)
            pl.from_pandas(pred_df).write_parquet(
                pred_dir / "mmm_train.parquet", compression="zstd"
            )
            print(f"Saved predictions: {pred_dir / 'mmm_train.parquet'}")

            err = pred_df[target] - pred_df["pred"]
            rmse = float(np.sqrt((err ** 2).mean()))
            mae = float(err.abs().mean())
            ybar = pred_df[target].mean()
            r2 = 1.0 - (err ** 2).sum() / ((pred_df[target] - ybar) ** 2).sum()
            print(f"In-sample: RMSE={rmse:.2f}, MAE={mae:.2f}, R²={r2:.4f}")
        except Exception as e:
            print(f"[warn] predict failed: {e}")
    else:
        print("\nSkipping predict (set mmm.compute_predict: true to enable).")

    # ----- (optional) Plots
    if save_plots:
        plot_dir = Path("data/plots")
        plot_dir.mkdir(parents=True, exist_ok=True)
        for name, fn in [
            ("contribution", "plot_channel_contributions_grid"),
            ("saturation", "plot_direct_contribution_curves"),
            ("channel_share", "plot_waterfall_components_decomposition"),
        ]:
            try:
                fig = getattr(mmm, fn)()
                fig.savefig(plot_dir / f"mmm_{name}.png", dpi=120, bbox_inches="tight")
                plt.close(fig)
                print(f"Saved: {plot_dir / f'mmm_{name}.png'}")
            except Exception as e:
                print(f"[warn] {name} plot failed: {e}")
    else:
        print("Skipping plots (set mmm.save_plots: true to enable).")


if __name__ == "__main__":
    main()
