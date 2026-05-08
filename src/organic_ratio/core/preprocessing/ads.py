import numpy as np
import pandas as pd

from organic_ratio.utils.config import load_config


def build_ads_features(df: pd.DataFrame) -> pd.DataFrame:
    cfg = load_config()
    HORIZONT = int(cfg.project.max_horizon)
    LEARN_DAYS = int(cfg.project.learn_days_min)
    PREDICT_TARGET = cfg.project.predict_target
    MAX_HORIZON = max(cfg.project.horizons)

    df["install_date"] = pd.to_datetime(df["install_date"], utc=False)

    df_long = (
        df
        .explode(["purchase_date_list", "purchase_ads_list"])
        .rename(columns={
            "purchase_date_list": "purchase_date",
            "purchase_ads_list": "ads"
        })
    )

    df_long["purchase_date"] = pd.to_datetime(df_long["purchase_date"], utc=False)

    df_long["day"] = (
            df_long["purchase_date"].dt.normalize()
            - df_long["install_date"].dt.normalize()
    ).dt.days

    df_long = df_long[df_long["day"].between(0, 360)]

    ads_days_max = (
        MAX_HORIZON
        if PREDICT_TARGET == "ads"
        else LEARN_DAYS
    )
    full_days_ads = range(0, ads_days_max + 1)
    full_days_ads_aux = range(0, LEARN_DAYS + 1)

    df_grouped = (
        df_long
        .groupby(["match_id", "install_date", "day"], as_index=False)["ads"]
        .sum()
    )
    df_grouped_count = (
        df_long
        .groupby(["match_id", "install_date", "day"], as_index=False)["ads"]
        .size()
        .rename(columns={"size": "ads_count"})
    )

    df_agg = (
        df_grouped
        .pivot_table(
            index=["match_id", "install_date"],
            columns="day",
            values="ads",
            aggfunc="sum",
            fill_value=0,
        )
        .reindex(columns=full_days_ads, fill_value=0)
        .infer_objects(copy=False)
    )
    df_agg_count = (
        df_grouped_count
        .pivot_table(
            index=["match_id", "install_date"],
            columns="day",
            values="ads_count",
            aggfunc="sum",
            fill_value=0,
        )
        .reindex(columns=full_days_ads_aux, fill_value=0)
        .infer_objects(copy=False)
    )

    df_agg.columns = [f"ads_{d}" for d in df_agg.columns]
    df_agg_count.columns = [f"ads_count_{d}" for d in df_agg_count.columns]

    df_final = (
        df.set_index(["match_id", "install_date"])
        .join(df_agg, how="left")
        .join(df_agg_count, how="left")
        .reset_index()
    )

    drop_after_explode = [
        "next_install_date",
        "purchase_date_list",
        "purchase_ads_list",
    ]
    existing_drop_after_explode = [c for c in drop_after_explode if c in df_final.columns]
    if existing_drop_after_explode:
        df_final = df_final.drop(columns=existing_drop_after_explode)

    day_cols_learn = [f"ads_{d}" for d in range(LEARN_DAYS + 1)]

    df_final["ads_cum_learn"] = df_final[day_cols_learn].sum(axis=1)
    df_final["log1p_ads_cum_learn"] = np.log1p(df_final["ads_cum_learn"])

    df_final["ads_days_with_purchase_learn"] = (df_final[day_cols_learn] > 0).sum(axis=1)

    df_learn = df_long[df_long["day"].between(0, LEARN_DAYS)]

    ads_purchase_count_learn = (
        df_learn
        .groupby(["match_id", "install_date"])["ads"]
        .size()
    )

    ads_purchase_count_map = ads_purchase_count_learn.rename("ads_purchase_count_learn")
    df_final = df_final.merge(
        ads_purchase_count_map.reset_index(),
        on=["match_id", "install_date"],
        how="left",
    )
    df_final["ads_purchase_count_learn"] = df_final["ads_purchase_count_learn"].fillna(0).astype(int)

    df_final["ads_avg_check_learn"] = (
        df_final["ads_cum_learn"] /
        df_final["ads_purchase_count_learn"].replace(0, np.nan)
    ).fillna(0)

    day_columns = {f"ads_{day}" for day in range(HORIZONT + 1)}
    d1_columns = [col for col in ["ads_0", "ads_1"] if col in day_columns]
    d3_columns = [col for col in ["ads_0", "ads_1", "ads_2", "ads_3"] if col in day_columns]

    df_final["ads_rev_d1"] = (
        df_final[d1_columns].sum(axis=1) if d1_columns else 0.0
    )
    df_final["ads_rev_d3"] = (
        df_final[d3_columns].sum(axis=1) if d3_columns else 0.0
    )

    try:
        df_final = df_final.drop_duplicates(subset=["match_id", "install_date"])
    except KeyError:
        df_final = df_final.drop_duplicates()

    return df_final
