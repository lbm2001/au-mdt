"""Load and preprocess ENTSO-E day-ahead price CSV files."""

from pathlib import Path

import pandas as pd

_DATA_DIR = Path(__file__).parent.parent / "data" / "entsoe"

# MTU column is like "01/01/2015 00:00:00 - 01/01/2015 01:00:00"
_MTU_COL = "MTU (CET/CEST)"
_PRICE_COL = "Day-ahead Price (EUR/MWh)"


def _parse_file(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, thousands=None)
    # Extract start timestamp from MTU range string
    # Some DST-transition rows have " (CET)" or " (CEST)" appended — strip it.
    start_str = (
        df[_MTU_COL]
        .str.split(" - ").str[0]
        .str.replace(r"\s*\(CES?T\)$", "", regex=True)
    )
    df["timestamp"] = pd.to_datetime(start_str, format="%d/%m/%Y %H:%M:%S")
    df["price_eur_mwh"] = pd.to_numeric(df[_PRICE_COL], errors="coerce")
    return df[["timestamp", "price_eur_mwh"]].dropna()


def load_prices(data_dir: Path | None = None) -> pd.DataFrame:
    """
    Load all ENTSO-E CSV files and return a single DataFrame.

    Columns
    -------
    timestamp       : UTC-naive CET/CEST datetime (start of interval)
    price_eur_mwh   : day-ahead price in EUR/MWh
    price_eur_kwh   : day-ahead price in EUR/kWh
    dow             : day-of-week (0=Monday … 6=Sunday)
    is_weekend      : True for Saturday/Sunday
    hour            : hour of day (0–23)
    minute          : minute within hour (0 or 15/30/45 for 15-min data)
    month           : month (1–12)
    season          : 'spring' | 'summer' | 'autumn' | 'winter'
    """
    root = data_dir or _DATA_DIR
    files = sorted(root.glob("GUI_ENERGY_PRICES_*.csv"))
    if not files:
        raise FileNotFoundError(f"No ENTSO-E CSV files found in {root}")

    parts = [_parse_file(f) for f in files]
    df = pd.concat(parts, ignore_index=True).drop_duplicates("timestamp").sort_values("timestamp")

    df["price_eur_kwh"] = df["price_eur_mwh"] / 1000.0
    df["dow"] = df["timestamp"].dt.dayofweek
    df["is_weekend"] = df["dow"] >= 5
    df["hour"] = df["timestamp"].dt.hour
    df["minute"] = df["timestamp"].dt.minute
    df["month"] = df["timestamp"].dt.month
    df["season"] = df["month"].map(_month_to_season)

    return df.reset_index(drop=True)


def _month_to_season(month: int) -> str:
    if month in (12, 1, 2):
        return "winter"
    if month in (3, 4, 5):
        return "spring"
    if month in (6, 7, 8):
        return "summer"
    return "autumn"
