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


def report(y_true, y_pred, w, label: str = "") -> dict:
    out = {
        "rmse_w": weighted_rmse(y_true, y_pred, w),
        "mae_w": weighted_mae(y_true, y_pred, w),
        "r2_w": weighted_r2(y_true, y_pred, w),
        "rmse_u": float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2))),
        "mae_u": float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred)))),
    }
    if label:
        print(
            f"{label:>6}: "
            f"RMSE_w={out['rmse_w']:.4f}  MAE_w={out['mae_w']:.4f}  R²_w={out['r2_w']:.4f}  "
            f"| RMSE_u={out['rmse_u']:.4f}  MAE_u={out['mae_u']:.4f}"
        )
    return out
