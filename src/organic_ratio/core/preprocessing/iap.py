import numpy as np
import pandas as pd

from organic_ratio.utils.config import load_config

EPSILON = 1e-8
MIN_PURCHASES_FOR_REPEAT = 2
THIRD_PURCHASE_RANK = 2


def build_iap_features(df: pd.DataFrame) -> pd.DataFrame:  # noqa: PLR0915
    cfg = load_config()
    HORIZONT = int(cfg.project.max_horizon)
    LEARN_DAYS = int(cfg.project.learn_days_min)
    PREDICT_TARGET = cfg.project.predict_target
    MAX_HORIZON = max(cfg.project.horizons)

    df["install_date"] = pd.to_datetime(df["install_date"], utc=False)

    df_long = (
        df
        .explode(["purchase_time_list", "purchase_iap_list"])
        .rename(columns={
            "purchase_time_list": "purchase_time",
            "purchase_iap_list": "iap"
        })
    )

    df_long["purchase_time"] = pd.to_datetime(df_long["purchase_time"], utc=False)

    df_long["day"] = (
            df_long["purchase_time"].dt.normalize()
            - df_long["install_date"].dt.normalize()
    ).dt.days

    df_long = df_long[df_long["day"].between(0, 360)]

    seq_days_max = MAX_HORIZON if PREDICT_TARGET == "iap" else LEARN_DAYS
    full_days_seq = range(0, seq_days_max + 1)

    df_grouped = (
        df_long
        .groupby(["match_id", "install_date", "day"], as_index=False)["iap"]
        .sum()
    )
    df_grouped_count = (
        df_long
        .groupby(["match_id", "install_date", "day"], as_index=False)["iap"]
        .size()
        .rename(columns={"size": "iap_count"})
    )

    df_agg = (
        df_grouped
        .pivot_table(
            index=["match_id", "install_date"],
            columns="day",
            values="iap",
            aggfunc="sum",
            fill_value=0,
        )
        .reindex(columns=full_days_seq, fill_value=0)
        .infer_objects(copy=False)
    )
    df_agg_count = (
        df_grouped_count
        .pivot_table(
            index=["match_id", "install_date"],
            columns="day",
            values="iap_count",
            aggfunc="sum",
            fill_value=0,
        )
        .reindex(columns=full_days_seq, fill_value=0)
        .infer_objects(copy=False)
    )

    df_agg_check = df_agg.div(df_agg_count + EPSILON)

    df_agg.columns = [f"iap_{d}" for d in df_agg.columns]
    df_agg_count.columns = [f"iap_count_{d}" for d in df_agg_count.columns]
    df_agg_check.columns = [f"iap_check_size_{d}" for d in df_agg_check.columns]

    df_final = (
        df.set_index(["match_id", "install_date"])
        .join(df_agg, how="left")
        .join(df_agg_count, how="left")
        .join(df_agg_check, how="left")
        .reset_index()
    )

    drop_after_explode = [
        "country_code_list",
        "purchase_iap_content_id_list",
        "purchase_time_list",
        "next_install_date",
        "purchase_usd_raw_list",
        "purchase_iap_list",
    ]
    existing_drop_after_explode = [c for c in drop_after_explode if c in df_final.columns]
    if existing_drop_after_explode:
        df_final = df_final.drop(columns=existing_drop_after_explode)

    first_iap = (
        df_long
        .groupby(["match_id", "install_date"])["purchase_time"]
        .min()
    )

    first_iap_map = first_iap.rename("first_iap_time")
    df_final = df_final.merge(
        first_iap_map.reset_index(),
        on=["match_id", "install_date"],
        how="left",
    )
    df_final["time_to_first_iap_minutes"] = (
        (df_final["first_iap_time"] - df_final["install_date"])
        .dt.total_seconds()
        .div(60)
        .fillna(0)
    )

    df_learn = df_long[df_long["day"].between(0, LEARN_DAYS)]

    if not df_learn.empty:
        df_learn = df_learn.sort_values("purchase_time")

        df_learn_ranked = df_learn.copy()
        df_learn_ranked["purchase_rank"] = (
            df_learn_ranked
            .groupby(["match_id", "install_date"])
            .cumcount()
        )

        second_iap_map = (
            df_learn_ranked[df_learn_ranked["purchase_rank"] == 1][
                ["match_id", "install_date", "purchase_time"]
            ]
            .rename(columns={"purchase_time": "second_iap_time"})
        )
        df_final = df_final.merge(
            second_iap_map,
            on=["match_id", "install_date"],
            how="left",
        )

        time_between_first_and_second_iap_minutes = (
            (df_final["second_iap_time"] - df_final["first_iap_time"])
            .dt.total_seconds()
            .div(60)
        )

        df_final["time_first_and_second_iap_minutes"] = (
            time_between_first_and_second_iap_minutes.fillna(0)
        )
    else:
        df_final["time_first_and_second_iap_minutes"] = 0

    day_cols_learn = [f"iap_{d}" for d in range(LEARN_DAYS + 1)]

    df_final["iap_cum_learn"] = df_final[day_cols_learn].sum(axis=1)
    df_final["log1p_iap_cum_learn"] = np.log1p(df_final["iap_cum_learn"])

    df_final["iap_days_with_purchase_learn"] = (df_final[day_cols_learn] > 0).sum(axis=1)

    iap_purchase_count_learn = (
        df_learn
        .groupby(["match_id", "install_date"])["iap"]
        .size()
    )
    iap_purchase_count_map = iap_purchase_count_learn.rename("iap_purchase_count_learn")
    df_final = df_final.merge(
        iap_purchase_count_map.reset_index(),
        on=["match_id", "install_date"],
        how="left",
    )
    df_final["iap_purchase_count_learn"] = df_final["iap_purchase_count_learn"].fillna(0).astype(int)

    df_final["iap_avg_check_learn"] = (
        df_final["iap_cum_learn"] /
        df_final["iap_purchase_count_learn"].replace(0, np.nan)
    ).fillna(0)

    day_columns = {f"iap_{day}" for day in range(HORIZONT + 1)}
    d1_columns = [col for col in ["iap_0", "iap_1"] if col in day_columns]
    d3_columns = [col for col in ["iap_0", "iap_1", "iap_2", "iap_3"] if col in day_columns]

    df_final["rev_d1_iap"] = (
        df_final[d1_columns].sum(axis=1) if d1_columns else 0.0
    )
    df_final["rev_d3_iap"] = (
        df_final[d3_columns].sum(axis=1) if d3_columns else 0.0
    )

    user_d1_share = df_final["rev_d1_iap"] / (df_final["iap_cum_learn"] + EPSILON)
    d1_share_by_install = (
        df_final.assign(d1_share_user_iap=user_d1_share)
        .groupby("install_date", as_index=False)["d1_share_user_iap"]
        .mean()
        .rename(columns={"d1_share_user_iap": "d1_share_iap"})
    )
    df_final = df_final.merge(
        d1_share_by_install,
        on="install_date",
        how="left",
    )

    if not df_learn.empty:
        df_d1_3 = df_learn[df_learn["day"].between(0, 3)]
        df_d4_7 = df_learn[df_learn["day"].between(4, 7)]

        d1_3_agg = (
            df_d1_3
            .groupby(["match_id", "install_date"])["iap"]
            .agg(["sum", "size"])
            .reset_index()
            .rename(
                columns={
                    "sum": "iap_rev_d1_3",
                    "size": "iap_purchase_count_d1_3",
                }
            )
        )
        d4_7_agg = (
            df_d4_7
            .groupby(["match_id", "install_date"])["iap"]
            .agg(["sum", "size"])
            .reset_index()
            .rename(
                columns={
                    "sum": "iap_rev_d4_7",
                    "size": "iap_purchase_count_d4_7",
                }
            )
        )

        df_final = df_final.merge(d1_3_agg, on=["match_id", "install_date"], how="left")
        df_final = df_final.merge(d4_7_agg, on=["match_id", "install_date"], how="left")
    else:
        df_final["iap_rev_d1_3"] = 0.0
        df_final["iap_purchase_count_d1_3"] = 0
        df_final["iap_rev_d4_7"] = 0.0
        df_final["iap_purchase_count_d4_7"] = 0

    df_final["iap_rev_d1_3"] = df_final["iap_rev_d1_3"].fillna(0.0)
    df_final["iap_purchase_count_d1_3"] = df_final["iap_purchase_count_d1_3"].fillna(0).astype(int)
    df_final["iap_rev_d4_7"] = df_final["iap_rev_d4_7"].fillna(0.0)
    df_final["iap_purchase_count_d4_7"] = df_final["iap_purchase_count_d4_7"].fillna(0).astype(int)

    df_final["avg_check_d1_3_iap"] = (
        df_final["iap_rev_d1_3"] /
        (df_final["iap_purchase_count_d1_3"] + EPSILON)
    )
    df_final["avg_check_d4_7_iap"] = (
        df_final["iap_rev_d4_7"] /
        (df_final["iap_purchase_count_d4_7"] + EPSILON)
    )
    df_final["acceleration_iap"] = (
        df_final["avg_check_d4_7_iap"] /
        (df_final["avg_check_d1_3_iap"] + EPSILON)
    )

    df_final["has_second_iap"] = (
        df_final["iap_purchase_count_learn"] >= MIN_PURCHASES_FOR_REPEAT
    ).astype(int)

    if not df_learn.empty:
        df_learn_sorted = df_learn.sort_values("purchase_time").copy()
        df_learn_sorted["purchase_rank"] = (
            df_learn_sorted
            .groupby(["match_id", "install_date"])
            .cumcount()
        )

        second_iap_for_2_3 = (
            df_learn_sorted[df_learn_sorted["purchase_rank"] == 1][
                ["match_id", "install_date", "purchase_time"]
            ]
            .rename(columns={"purchase_time": "second_iap_time_for_2_3"})
        )
        third_iap_map = (
            df_learn_sorted[df_learn_sorted["purchase_rank"] == THIRD_PURCHASE_RANK][
                ["match_id", "install_date", "purchase_time"]
            ]
            .rename(columns={"purchase_time": "third_iap_time"})
        )

        df_final = df_final.merge(
            second_iap_for_2_3,
            on=["match_id", "install_date"],
            how="left",
        )
        df_final = df_final.merge(
            third_iap_map,
            on=["match_id", "install_date"],
            how="left",
        )
        df_final["time_between_iap_2_3"] = (
            (df_final["third_iap_time"] - df_final["second_iap_time_for_2_3"])
            .dt.total_seconds()
            .div(60)
            .fillna(0.0)
        )
    else:
        df_final["time_between_iap_2_3"] = 0.0

    drop_cols = [
        "first_iap_time",
        "second_iap_time",
        "second_iap_time_for_2_3",
        "third_iap_time",
        "iap_rev_d1_3",
        "iap_purchase_count_d1_3",
        "iap_rev_d4_7",
        "iap_purchase_count_d4_7",
    ]
    existing_drop_cols = [col for col in drop_cols if col in df_final.columns]
    if existing_drop_cols:
        df_final = df_final.drop(columns=existing_drop_cols)

    try:
        df_final = df_final.drop_duplicates(subset=["match_id", "install_date"])
    except KeyError:
        df_final = df_final.drop_duplicates()

    return df_final
