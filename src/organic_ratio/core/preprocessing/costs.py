import pandas as pd
import numpy as np
from organic_ratio.utils.config import load_config


def build_costs_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    User-grain UA cost features. Joins UA-cost rows (cohort key in costs raw)
    onto installs by [install_date, platform, media_source, country_code,
    campaign, af_adset, af_ad].
    """
    cfg = load_config()

    df.rename(columns={"installs": "installs_spend"}, inplace=True)
    df["install_date"] = pd.to_datetime(df["install_date"], utc=False)

    cols_installs = [
        "match_id",
        "install_date",
        "platform",
        "media_source",
        "country_code",
        "campaign",
        "af_adset",
        "af_ad",
    ]

    df_installs = pd.read_parquet(
        f'{cfg.datasets["installs"]["local_feature_dir"]}/{cfg.datasets["installs"]["filename"]}',
        columns=cols_installs,
    )

    df["cpi"] = df["spend"] / df["installs_spend"].replace(0, np.nan)
    df["ctr"] = df["clicks"] / df["impressions"].replace(0, np.nan)
    df["cvr"] = df["installs_spend"] / df["clicks"].replace(0, np.nan)
    df["cpm"] = df["spend"] * 1000 / df["impressions"].replace(0, np.nan)

    merged = df_installs.merge(
        df,
        on=["install_date", "platform", "media_source", "country_code", "campaign", "af_adset", "af_ad"],
        how="left",
    )

    merged = merged.drop(columns=["platform", "media_source", "country_code", "campaign", "af_adset", "af_ad"])

    try:
        merged = merged.drop_duplicates(subset=["match_id", "install_date"])
    except KeyError:
        merged = merged.drop_duplicates()

    return merged
