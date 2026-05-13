"""
Build MMM panel:
    per (platform × country × install_date) →
        organic_installs (target), total_installs,
        spend_<top10>, spend_other_paid,
        dow_0..dow_6, geo

Output: data/features/mmm/mmm_panel.parquet
"""
from dotenv import load_dotenv
load_dotenv()

import sys
from pathlib import Path

import polars as pl

SRC_PATH = Path.cwd() / "src"
if not SRC_PATH.exists():
    raise RuntimeError(f"src not found at {SRC_PATH}")
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from organic_ratio.utils.config import load_config
from organic_ratio.core.modeling.mmm_data import build_mmm_panel


def main() -> None:
    cfg = load_config()
    mmm_cfg = cfg.modeling.mmm

    installs_cfg = cfg.datasets.installs
    costs_cfg = cfg.datasets.costs

    # installs feature parquet keeps cohort keys (platform, country, source).
    # costs feature parquet drops them (user-level after join); use RAW costs.
    installs_path = Path(installs_cfg.local_feature_dir) / installs_cfg.filename
    costs_path = Path(costs_cfg.local_raw_dir) / costs_cfg.filename
    print(f"Reading installs (features): {installs_path}")
    print(f"Reading costs (raw):         {costs_path}")

    panel, top_channels = build_mmm_panel(
        installs_path=installs_path,
        costs_path=costs_path,
        top_n_channels=int(mmm_cfg.top_n_channels),
        min_country_installs=int(mmm_cfg.min_country_installs),
        date_from=cfg.train_start_date,
        date_to=cfg.test_end_date,
    )

    print(f"\nPanel shape: {panel.shape}")
    print(f"Date range:  {panel['install_date'].min()}  →  {panel['install_date'].max()}")
    print(f"Geos:        {panel['geo'].n_unique()}  unique (platform × country)")
    print(f"Channels:    {len(top_channels)} + other_paid")

    print("\norganic_installs summary:")
    print(panel["organic_installs"].describe())

    print("\nspend per channel (sum across panel):")
    spend_cols = [c for c in panel.columns if c.startswith("spend_")]
    for c in spend_cols:
        print(f"  {c:30s}  total = {panel[c].sum():>14,.0f}")

    out_cfg = cfg.datasets.mmm_panel
    out_dir = Path(out_cfg.local_feature_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / out_cfg.filename
    panel.write_parquet(out_path, compression="zstd")
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
