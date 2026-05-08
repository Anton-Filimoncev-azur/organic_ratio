import pandas as pd
from typing import Iterable


def cast_datetime(
    df: pd.DataFrame,
    columns: Iterable[str],
    utc: bool = False,
) -> pd.DataFrame:
    """
    Cast given columns to datetime. Missing columns are skipped.
    """
    df = df.copy()

    for col in columns:
        if col not in df.columns:
            continue

        df[col] = pd.to_datetime(
            df[col],
            errors="coerce",
            utc=utc,
        )

    return df
