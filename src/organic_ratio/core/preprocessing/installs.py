import pandas as pd


def build_installs_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    user-grain features from install events. Carries cohort keys
    (country_code, media_source, platform, install_date) downstream.
    """
    df["install_date"] = pd.to_datetime(df["install_date"], utc=False)

    try:
        df = df.drop_duplicates(subset=["match_id", "install_date"])
    except KeyError:
        df = df.drop_duplicates()

    return df
