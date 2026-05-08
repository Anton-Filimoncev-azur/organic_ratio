"""
Sessions preprocessor.

1. Uses sessions_X (X=0..learn_days) for static features:
   sessions_d1, session_velocity_slope, last_active_day, zero_session_days.

2. Extends sessions_X sequence channel up to max_horizon. Missing days are
   filled with zeros.
"""

import numpy as np
import pandas as pd

from organic_ratio.utils.config import load_config

LEARN_WINDOW_END_DAY = 7
SESSIONS_D1_END_DAY = 1
NO_ACTIVITY_LAST_DAY_VALUE = 0
DAY_INDEX_SHIFT = 1


def _existing_session_day_columns(df: pd.DataFrame, max_day_inclusive: int) -> list[str]:
    return [
        f"sessions_{day}"
        for day in range(max_day_inclusive + 1)
        if f"sessions_{day}" in df.columns
    ]


def build_sessions_features(df: pd.DataFrame) -> pd.DataFrame:
    cfg = load_config()
    max_horizon = int(cfg.project.max_horizon)

    df["install_date"] = pd.to_datetime(df["install_date"], utc=False)

    learn_day_columns = _existing_session_day_columns(df, LEARN_WINDOW_END_DAY)
    if learn_day_columns:
        sessions_matrix = df[learn_day_columns].fillna(0.0).to_numpy(dtype=float)

        d1_columns = _existing_session_day_columns(df, SESSIONS_D1_END_DAY)
        df["sessions_d1"] = (
            df[d1_columns].fillna(0.0).sum(axis=1) if d1_columns else 0.0
        )

        x = np.arange(sessions_matrix.shape[1], dtype=float)
        x_centered = x - x.mean()
        denominator = np.square(x_centered).sum()
        if denominator > 0:
            y_mean = sessions_matrix.mean(axis=1, keepdims=True)
            numerator = ((sessions_matrix - y_mean) * x_centered).sum(axis=1)
            df["session_velocity_slope"] = numerator / denominator
        else:
            df["session_velocity_slope"] = 0.0

        active_mask = sessions_matrix > 0
        has_activity = active_mask.any(axis=1)
        last_active_idx = active_mask.shape[1] - 1 - np.argmax(
            active_mask[:, ::-1], axis=1
        )
        last_active_day = np.where(
            has_activity,
            last_active_idx + DAY_INDEX_SHIFT,
            NO_ACTIVITY_LAST_DAY_VALUE,
        )
        df["last_active_day"] = last_active_day.astype(int)
    else:
        df["sessions_d1"] = 0.0
        df["session_velocity_slope"] = 0.0
        df["last_active_day"] = NO_ACTIVITY_LAST_DAY_VALUE

    if "active_days_7d" in df.columns:
        df["zero_session_days"] = LEARN_WINDOW_END_DAY - df["active_days_7d"]
    elif learn_day_columns:
        active_days_7d = (
            df[learn_day_columns]
            .fillna(0.0)
            .to_numpy(dtype=float)[:, :LEARN_WINDOW_END_DAY]
            > 0
        ).sum(axis=1)
        df["zero_session_days"] = LEARN_WINDOW_END_DAY - active_days_7d
    else:
        df["zero_session_days"] = float(LEARN_WINDOW_END_DAY)
    df["zero_session_days"] = df["zero_session_days"].clip(lower=0)

    real_session_days = [
        int(c.split("_")[1])
        for c in df.columns
        if c.startswith("sessions_") and c.split("_", 1)[1].isdigit()
    ]
    if real_session_days:
        max_real_day = max(real_session_days)
        print(
            f"[sessions preproc] raw sessions_X spans days 0..{max_real_day}; "
            f"target max_horizon-1 = {max_horizon - 1}"
        )
        if max_real_day + 1 < max_horizon:
            print(
                f"[sessions preproc] WARN: sessions_{max_real_day + 1}..sessions_{max_horizon - 1} "
                f"will be filled with zeros (missing in raw)."
            )
        for day in range(max_horizon):
            col = f"sessions_{day}"
            if col not in df.columns:
                df[col] = 0.0
            else:
                df[col] = df[col].fillna(0.0).astype(float)
    else:
        print(
            f"[sessions preproc] WARN: no sessions_X columns in raw, "
            f"creating sessions_0..sessions_{max_horizon - 1} = 0.0"
        )
        for day in range(max_horizon):
            df[f"sessions_{day}"] = 0.0

    try:
        df = df.drop_duplicates(subset=["match_id", "install_date"])
    except KeyError:
        df = df.drop_duplicates()

    return df
