"""
Ingestion: pull each dataset from S3 to local raw partition.

Run from project root:
    python run.py
"""
from dotenv import load_dotenv
load_dotenv()

import sys
from pathlib import Path

SRC_PATH = Path.cwd() / "src"
print("Adding SRC_PATH:", SRC_PATH)

if not SRC_PATH.exists():
    raise RuntimeError(f"src not found at {SRC_PATH}")

if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from organic_ratio.utils.config import load_config
from organic_ratio.core.loaders.loader import load_from_s3
from organic_ratio.core.loaders.materialize import materialize_df


def run(ds_cfg):
    print(f"Running ingestion for {ds_cfg.filename}")
    df = load_from_s3(ds_cfg)
    print(f"Loaded {ds_cfg.filename}:", df.shape)

    out_path = materialize_df(df, ds_cfg.local_raw_dir, ds_cfg.filename)
    print("Saved to:", out_path)

    return out_path


if __name__ == "__main__":
    cfg = load_config()
    print(cfg)

    for data_name, ds_cfg in cfg.datasets.items():
        if "s3_path" not in ds_cfg:
            continue
        run(ds_cfg)
