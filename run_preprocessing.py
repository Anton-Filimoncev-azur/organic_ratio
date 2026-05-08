"""
Per-source preprocessing: read raw parquet, run dataset-specific feature
builder from the registry, save user-grain feature parquet.

Run from project root:
    python run_preprocessing.py
"""
from dotenv import load_dotenv
load_dotenv()

import sys
from pathlib import Path
import gc

SRC_PATH = Path.cwd() / "src"
print("Adding SRC_PATH:", SRC_PATH)

if not SRC_PATH.exists():
    raise RuntimeError(f"src not found at {SRC_PATH}")

if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

import pandas as pd

from organic_ratio.utils.config import load_config
from organic_ratio.core.preprocessing.preprocesser_registry import PREPROCESSORS
from organic_ratio.core.preprocessing.common.dtypes import cast_datetime
from organic_ratio.core.loaders.materialize import materialize_df


def main():
    cfg = load_config()

    results = {}

    for name, ds_cfg in cfg.datasets.items():
        if name not in PREPROCESSORS:
            continue

        print(f"\nProcessing dataset: {name}")

        raw_path = Path(ds_cfg.local_raw_dir) / ds_cfg.filename
        df = pd.read_parquet(raw_path)

        df = cast_datetime(df, columns=["install_time", "install_date"])

        features_df = PREPROCESSORS[name](df)

        out_path = materialize_df(
            df=features_df,
            path=ds_cfg.local_feature_dir,
            filename=ds_cfg.filename,
        )
        results[name] = out_path
        print(f"Saved features for {name}: {out_path}")

        del df, features_df
        gc.collect()

    return results


if __name__ == "__main__":
    main()
