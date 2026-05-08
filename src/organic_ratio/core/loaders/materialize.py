from pathlib import Path
import pandas as pd


def materialize_df(
    df: pd.DataFrame,
    path,
    filename,
    *,
    overwrite: bool = True,
    index: bool = False,
):
    """
    Save DataFrame locally according to data-registry config.
    """
    out_dir = Path(path)
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / filename

    if out_path.exists() and not overwrite:
        return out_path

    df.to_parquet(out_path, index=index)

    return out_path
