"""
Weighted regression metrics for cohort-level targets.
"""
from __future__ import annotations

import numpy as np


def weighted_rmse(y_true, y_pred, w) -> float:
    err = np.asarray(y_true) - np.asarray(y_pred)
    return float(np.sqrt(np.average(err ** 2, weights=w)))


def weighted_mae(y_true, y_pred, w) -> float:
    err = np.asarray(y_true) - np.asarray(y_pred)
    return float(np.average(np.abs(err), weights=w))


def weighted_r2(y_true, y_pred, w) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    w = np.asarray(w, dtype=float)
    ybar = np.average(y_true, weights=w)
    ss_res = float(np.sum(w * (y_true - y_pred) ** 2))
    ss_tot = float(np.sum(w * (y_true - ybar) ** 2))
    if ss_tot == 0:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def weighted_wmape(y_true, y_pred, w) -> float:
    """
    Weighted MAPE-style metric, in [0, ∞):
        WMAPE = Σ wᵢ·|yᵢ − ŷᵢ| / Σ wᵢ·|yᵢ|
    For organic_share ∈ [0, 1] reads as: total weighted absolute error per
    unit of actual organic mass. Robust to per-row y=0 cohorts.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    w = np.asarray(w, dtype=float)
    num = float(np.sum(w * np.abs(y_true - y_pred)))
    den = float(np.sum(w * np.abs(y_true)))
    if den == 0:
        return float("nan")
    return num / den


def percentage_error(y_true, y_pred, eps: float = 1e-6) -> np.ndarray:
    """
    Per-row PE = (ŷ - y) / y. y=0 rows return NaN.
    Sign convention: positive PE → over-prediction.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    out = np.full_like(y_true, np.nan, dtype=float)
    mask = np.abs(y_true) > eps
    out[mask] = (y_pred[mask] - y_true[mask]) / y_true[mask]
    return out


DEFAULT_PE_BUCKETS = [-1.00, -0.50, -0.25, -0.10, 0.0, 0.10, 0.25, 0.50, 1.00]


def pe_buckets(pe: np.ndarray, edges=DEFAULT_PE_BUCKETS) -> list[dict]:
    """
    Histogram of PE binned with explicit edges (default: ±10/25/50/100%).

    Includes two open tails:
        (-inf, edges[0])   and   (edges[-1], +inf)

    Returns list of dicts: [{lo, hi, count, pct}, ...] (NaN PEs dropped).
    """
    pe = np.asarray(pe, dtype=float)
    pe = pe[~np.isnan(pe)]
    n = pe.size

    full_edges = [-np.inf, *edges, np.inf]
    rows = []
    for lo, hi in zip(full_edges[:-1], full_edges[1:]):
        # left-open right-closed, except the very last is right-open to +inf
        if np.isinf(hi):
            mask = pe > lo
        else:
            mask = (pe > lo) & (pe <= hi)
        count = int(mask.sum())
        rows.append({
            "lo": lo,
            "hi": hi,
            "count": count,
            "pct": float(count / n) if n else 0.0,
        })
    return rows


def pe_summary(pe: np.ndarray) -> dict:
    """Compact summary of a percentage-error array (NaNs dropped)."""
    pe = np.asarray(pe, dtype=float)
    pe = pe[~np.isnan(pe)]
    if pe.size == 0:
        return {"n": 0}
    return {
        "n": int(pe.size),
        "mean": float(np.mean(pe)),
        "median": float(np.median(pe)),
        "q05": float(np.quantile(pe, 0.05)),
        "q25": float(np.quantile(pe, 0.25)),
        "q75": float(np.quantile(pe, 0.75)),
        "q95": float(np.quantile(pe, 0.95)),
        "within_10pct": float(np.mean(np.abs(pe) <= 0.10)),
        "within_20pct": float(np.mean(np.abs(pe) <= 0.20)),
        "within_50pct": float(np.mean(np.abs(pe) <= 0.50)),
    }


def report(y_true, y_pred, w, label: str = "") -> dict:
    out = {
        "rmse_w": weighted_rmse(y_true, y_pred, w),
        "mae_w": weighted_mae(y_true, y_pred, w),
        "r2_w": weighted_r2(y_true, y_pred, w),
        "wmape": weighted_wmape(y_true, y_pred, w),
        "rmse_u": float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2))),
        "mae_u": float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred)))),
    }
    if label:
        print(
            f"{label:>6}: "
            f"RMSE_w={out['rmse_w']:.4f}  MAE_w={out['mae_w']:.4f}  "
            f"R²_w={out['r2_w']:.4f}  WMAPE={out['wmape']*100:.2f}%  "
            f"| RMSE_u={out['rmse_u']:.4f}  MAE_u={out['mae_u']:.4f}"
        )
    return out
