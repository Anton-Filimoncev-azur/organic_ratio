import pandas as pd


def build_personal_features(df: pd.DataFrame) -> pd.DataFrame:
    df["install_date"] = pd.to_datetime(df["install_date"], utc=False)

    try:
        df = df.drop_duplicates(subset=["match_id", "install_date"])
    except KeyError:
        df = df.drop_duplicates()

    return df
