"""
Hierarchical Bayesian model for cohort-level organic_share.

    organic_installs[i] ~ Binomial(total_installs[i], p[i])
    logit(p[i]) = α + X[i] · β + u_country[c[i]] + u_platform[plat[i]]

    α          ~ Normal(0, 1.5)
    β[f]       ~ Normal(0, 0.5)
    u_country  ~ Normal(0, σ_country),  σ_country  ~ HalfNormal(1.0)
    u_platform ~ Normal(0, σ_platform), σ_platform ~ HalfNormal(1.0)

Binomial likelihood naturally weights each cohort by its install count —
no explicit sample_weight needed.
"""
from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd

import pymc as pm
import arviz as az


def build_model(
    X: np.ndarray,
    country_idx: np.ndarray,
    platform_idx: np.ndarray,
    organic_installs: np.ndarray,
    total_installs: np.ndarray,
    n_countries: int,
    n_platforms: int,
    feature_names: List[str],
) -> pm.Model:
    coords = {
        "feature": feature_names,
        "country": np.arange(n_countries),
        "platform": np.arange(n_platforms),
        "obs": np.arange(len(organic_installs)),
    }

    with pm.Model(coords=coords) as model:
        alpha = pm.Normal("alpha", mu=0.0, sigma=1.5)
        beta = pm.Normal("beta", mu=0.0, sigma=0.5, dims="feature")

        sigma_country = pm.HalfNormal("sigma_country", sigma=1.0)
        u_country = pm.Normal("u_country", mu=0.0, sigma=sigma_country, dims="country")

        sigma_platform = pm.HalfNormal("sigma_platform", sigma=1.0)
        u_platform = pm.Normal("u_platform", mu=0.0, sigma=sigma_platform, dims="platform")

        eta = (
            alpha
            + pm.math.dot(X, beta)
            + u_country[country_idx]
            + u_platform[platform_idx]
        )
        p = pm.math.invlogit(eta)

        pm.Binomial(
            "organic",
            n=total_installs,
            p=p,
            observed=organic_installs,
            dims="obs",
        )

    return model


def sample(
    model: pm.Model,
    *,
    draws: int,
    tune: int,
    chains: int,
    target_accept: float,
    nuts_sampler: str,
    random_seed: int,
    chain_method: str = "parallel",
) -> az.InferenceData:
    """
    Run NUTS via the selected backend.

    nuts_sampler:
      * "pymc"     — default PyTensor/C backend (slow, no extra deps)
      * "numpyro"  — JAX-backed NUTS (fast; needs `numpyro`, `jax`).
                     chain_method = parallel | sequential | vectorized
      * "nutpie"   — Rust-backed NUTS (fast; needs `nutpie`)
    """
    sampler_kwargs = {}
    if nuts_sampler == "numpyro":
        sampler_kwargs["chain_method"] = chain_method

    with model:
        trace = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            target_accept=target_accept,
            nuts_sampler=nuts_sampler,
            nuts_sampler_kwargs=sampler_kwargs or None,
            random_seed=random_seed,
            progressbar=True,
        )
    return trace


def report_jax_devices() -> None:
    """Print JAX devices visible to numpyro (GPU/CPU). No-op if JAX missing."""
    try:
        import jax
    except ImportError:
        print("[jax] not installed — skipping device report")
        return
    print(f"[jax] default_backend: {jax.default_backend()}")
    print(f"[jax] devices: {jax.devices()}")


def posterior_mean_params(trace: az.InferenceData) -> dict:
    """Mean of posterior for inference without re-sampling."""
    post = trace.posterior
    return {
        "alpha": float(post["alpha"].mean(dim=("chain", "draw")).item()),
        "beta": post["beta"].mean(dim=("chain", "draw")).values,
        "u_country": post["u_country"].mean(dim=("chain", "draw")).values,
        "u_platform": post["u_platform"].mean(dim=("chain", "draw")).values,
    }


def predict_mean(
    params: dict,
    X: np.ndarray,
    country_idx: np.ndarray,
    platform_idx: np.ndarray,
) -> np.ndarray:
    eta = (
        params["alpha"]
        + X @ params["beta"]
        + params["u_country"][country_idx]
        + params["u_platform"][platform_idx]
    )
    return 1.0 / (1.0 + np.exp(-eta))


def beta_summary(trace: az.InferenceData, feature_names: List[str], top: int = 20) -> pd.DataFrame:
    post = trace.posterior["beta"]
    mean = post.mean(dim=("chain", "draw")).values
    sd = post.std(dim=("chain", "draw")).values
    hdi = az.hdi(post, hdi_prob=0.94)["beta"].values
    df = pd.DataFrame({
        "feature": feature_names,
        "mean": mean,
        "sd": sd,
        "hdi_3%": hdi[:, 0],
        "hdi_97%": hdi[:, 1],
    })
    df["abs_mean"] = df["mean"].abs()
    return df.sort_values("abs_mean", ascending=False).head(top).reset_index(drop=True)


def sampler_diagnostics(trace: az.InferenceData) -> pd.DataFrame:
    return az.summary(
        trace,
        var_names=["alpha", "sigma_country", "sigma_platform"],
        kind="all",
    )
