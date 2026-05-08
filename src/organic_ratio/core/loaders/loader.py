import os
import pandas as pd


def load_from_s3(ds_cfg) -> pd.DataFrame:
    """
    Universal S3 parquet loader. Reads ds_cfg.s3_path with credentials
    from S3_ACCESS_KEY / S3_SECRET_KEY env vars.
    """
    return pd.read_parquet(
        ds_cfg.s3_path,
        storage_options={
            "key": os.environ["S3_ACCESS_KEY"],
            "secret": os.environ["S3_SECRET_KEY"],
        },
    )
