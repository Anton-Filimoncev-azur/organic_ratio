from typing import Dict, Iterable

import polars as pl


def merge_datasets(
    lfs: Dict[str, pl.LazyFrame],
    on: Iterable[str],
    how: str = "left",
) -> pl.LazyFrame:
    """
    Universal merge of several Polars LazyFrames.

    First dataset = base, all others are joined onto it.
    """
    if not lfs:
        raise ValueError("No datasets to merge")

    it = iter(lfs.items())
    base_name, base_lf = next(it)

    result = base_lf
    for name, lf in it:
        result = result.join(lf, on=list(on), how=how)

    return result
