"""Load and preprocess ENTSO-E day-ahead price data straight from the live API."""

import os
from pathlib import Path

import pandas as pd

_DEFAULT_START   = "2015-01-01"  # ENTSO-E DK1 day-ahead history starts here
_DEFAULT_COUNTRY = "DK_1"        # West Denmark bidding zone (entsoe-py area code)
# On-disk cache: <repo root>/entsoe-data/<country_code>_<startYear>-<endYear>.parquet
# (the year span reflects the data actually stored, e.g. DK_1_2015-2026.parquet)
_DEFAULT_CACHE_DIR = Path(__file__).resolve().parents[2] / "entsoe-data"


def load_prices(
    api_key: str | None = None,
    _log=None,
    *,
    start: "str | pd.Timestamp" = _DEFAULT_START,
    end: "str | pd.Timestamp | None" = None,
    country_code: str = _DEFAULT_COUNTRY,
    cache: bool = True,
    cache_dir: "str | Path | None" = None,
) -> pd.DataFrame:
    """
    Fetch the full ENTSO-E day-ahead price history from the API and return a
    preprocessed DataFrame.

    The range [start, end) is pulled in one-year chunks and concatenated; end
    defaults to tomorrow (so the latest day-ahead prices are included).  Requires
    an API key, resolved in order:
      1. The ``api_key`` argument
      2. The ``ENTSOE_API_KEY`` environment variable
      3. ``[entsoe] api_key`` in .streamlit/secrets.toml

    _log: optional callable(message: str) called after each yearly chunk.

    Caching
    -------
    If ``cache`` is True (default), the preprocessed DataFrame is stored as
    ``<cache_dir>/<country_code>_<startYear>-<endYear>.parquet`` — the year span
    reflecting the data actually stored (``cache_dir`` defaults to
    ``<repo root>/entsoe-data``).  On a subsequent call, any matching cache file
    is loaded directly and the API is not contacted at all — so no API key is
    required once the cache is populated.  Delete the file to force a refresh.

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
    cache_root = Path(cache_dir or _DEFAULT_CACHE_DIR)
    if cache:
        existing = _find_cache(cache_root, country_code)
        if existing is not None:
            if _log is not None:
                _log(f"Cache: loading {country_code} prices from {existing}…")
            df = pd.read_parquet(existing)
            if _log is not None:
                y0, y1 = df["timestamp"].dt.year.min(), df["timestamp"].dt.year.max()
                _log(f"Cache: loaded {len(df):,} samples ({y0}–{y1})")
            return df

    key = _resolve_api_key(api_key)
    if not key:
        raise RuntimeError(
            "No ENTSO-E API key configured. Set ENTSOE_API_KEY, pass api_key=…, "
            "or add [entsoe] api_key to .streamlit/secrets.toml."
        )

    if end is None:
        end = pd.Timestamp.now().normalize() + pd.Timedelta(days=1)

    if _log is not None:
        _log(f"API: fetching {country_code} day-ahead prices "
             f"{pd.Timestamp(start).date()} → {pd.Timestamp(end).date()}…")

    fetcher = EntsoeFetcher(key, country_code=country_code)
    df = fetcher.fetch_range(start, end, _log=_log)

    if _log is not None:
        y0 = df["timestamp"].dt.year.min()
        y1 = df["timestamp"].dt.year.max()
        _log(f"API: loaded {len(df):,} samples ({y0}–{y1})")

    if cache:
        cache_root.mkdir(parents=True, exist_ok=True)
        cache_path = _dated_cache_path(cache_root, country_code, df)
        df.to_parquet(cache_path, index=False)
        if _log is not None:
            _log(f"Cache: saved {len(df):,} samples → {cache_path}")

    return df


def _dated_cache_path(cache_dir: Path, country_code: str, df: pd.DataFrame) -> Path:
    """``<cache_dir>/<country_code>_<startYear>-<endYear>.parquet`` for the data in ``df``."""
    y0 = int(df["timestamp"].dt.year.min())
    y1 = int(df["timestamp"].dt.year.max())
    return cache_dir / f"{country_code}_{y0}-{y1}.parquet"


def _find_cache(cache_dir: Path, country_code: str) -> Path | None:
    """Return an existing on-disk price cache for the country, or None.

    Matches the dated form ``<country_code>_<y0>-<y1>.parquet`` (widest end-year
    wins) and the legacy un-dated ``<country_code>.parquet``.
    """
    dated = sorted(cache_dir.glob(f"{country_code}_*.parquet"))
    if dated:
        return dated[-1]
    legacy = cache_dir / f"{country_code}.parquet"
    return legacy if legacy.exists() else None


def _resolve_api_key(explicit: str | None) -> str | None:
    """Return the first non-empty API key from: argument → env var → streamlit secrets."""
    if explicit:
        return explicit
    env = os.environ.get("ENTSOE_API_KEY", "")
    if env:
        return env
    try:
        import streamlit as st
        key = st.secrets.get("entsoe", {}).get("api_key", "")
        return key or None
    except Exception:
        return None


# ── API loader ────────────────────────────────────────────────────────────────

class EntsoeFetcher:
    """
    Fetch day-ahead prices directly from the ENTSO-E Transparency Platform API
    via entsoe-py and return the same preprocessed DataFrame as load_prices().

    Parameters
    ----------
    api_key     : ENTSO-E API key (obtain at transparency.entsoe.eu → My Account)
    country_code: ENTSO-E bidding-zone code, default "DK_1" (West Denmark)

    Usage
    -----
        fetcher = EntsoeFetcher(api_key="YOUR_KEY")
        df = fetcher.fetch("2023-01-01", "2024-01-01")
        # df has the same columns as load_prices()
    """

    def __init__(self, api_key: str, country_code: str = "DK_1") -> None:
        self.api_key      = api_key
        self.country_code = country_code

    def fetch(
        self,
        start: str | pd.Timestamp,
        end:   str | pd.Timestamp,
        tz:    str = "Europe/Copenhagen",
    ) -> pd.DataFrame:
        """
        Fetch day-ahead prices for [start, end) and return a preprocessed DataFrame.

        Parameters
        ----------
        start : start date/datetime (inclusive), e.g. "2023-01-01"
        end   : end date/datetime (exclusive),   e.g. "2024-01-01"
        tz    : timezone for the timestamps (default: Europe/Copenhagen = CET/CEST)

        Returns
        -------
        DataFrame with the same columns as load_prices().
        """
        from entsoe import EntsoePandasClient

        client = EntsoePandasClient(api_key=self.api_key)

        ts_start = pd.Timestamp(start, tz=tz)
        ts_end   = pd.Timestamp(end,   tz=tz)

        series = client.query_day_ahead_prices(
            self.country_code, start=ts_start, end=ts_end,
        )

        raw = (
            series
            .rename("price_eur_mwh")
            .reset_index()
            .rename(columns={"index": "timestamp"})
        )
        # Strip timezone info to match the CSV-based loader
        raw["timestamp"] = raw["timestamp"].dt.tz_localize(None)
        raw = raw.dropna(subset=["price_eur_mwh"])

        return _add_features(raw)

    def fetch_range(
        self,
        start: str | pd.Timestamp,
        end:   str | pd.Timestamp,
        tz:    str = "Europe/Copenhagen",
        _log=None,
        retries: int = 4,
        backoff: float = 2.0,
    ) -> pd.DataFrame:
        """
        Fetch [start, end) in one-year chunks and return the combined DataFrame.

        Each chunk is retried with exponential backoff because the ENTSO-E API
        intermittently returns 503/504.  A genuinely empty period (e.g. a future
        chunk with no published prices) is skipped; a chunk that still fails after
        all retries raises, so we never silently return a dataset with missing
        years.

        _log: optional callable(message: str) — progress and retry notices.
        """
        # Note: HTTP error messages embed the API key in the URL, so only the
        # exception *type* is ever logged/raised — never the message.
        try:
            from entsoe.exceptions import NoMatchingDataError
        except Exception:                       # pragma: no cover
            NoMatchingDataError = ()            # nothing to special-case → all retryable

        import time

        start = pd.Timestamp(start)
        end   = pd.Timestamp(end)
        frames: list[pd.DataFrame] = []
        cur = start
        while cur < end:
            nxt = min(pd.Timestamp(year=cur.year + 1, month=1, day=1), end)
            for attempt in range(retries):
                try:
                    part = self.fetch(cur, nxt, tz=tz)
                    frames.append(part)
                    if _log is not None:
                        _log(f"API: {cur.date()}–{nxt.date()}  {len(part):,} rows")
                    break
                except NoMatchingDataError:
                    if _log is not None:
                        _log(f"API: {cur.date()}–{nxt.date()} no data, skipped")
                    break
                except Exception as exc:
                    if attempt < retries - 1:
                        wait = backoff * (2 ** attempt)
                        if _log is not None:
                            _log(f"API: {cur.date()}–{nxt.date()} {type(exc).__name__}, "
                                 f"retry {attempt + 1}/{retries - 1} in {wait:.0f}s…")
                        time.sleep(wait)
                    else:
                        raise RuntimeError(
                            f"ENTSO-E API failed for {cur.date()}–{nxt.date()} after "
                            f"{retries} attempts ({type(exc).__name__}). The service is "
                            f"likely temporarily unavailable (503/504) — try again shortly."
                        ) from None
            cur = nxt

        if not frames:
            raise RuntimeError(
                f"ENTSO-E API returned no data for {start.date()}–{end.date()} "
                f"({self.country_code})."
            )
        return (
            pd.concat(frames, ignore_index=True)
            .drop_duplicates("timestamp")
            .sort_values("timestamp")
            .reset_index(drop=True)
        )

    def fetch_and_append(
        self,
        existing: pd.DataFrame,
        end: str | pd.Timestamp,
        tz: str = "Europe/Copenhagen",
    ) -> pd.DataFrame:
        """
        Fetch data from the day after the last timestamp in `existing` up to `end`
        and return the combined, deduplicated DataFrame.

        Useful for topping up the CSV-based dataset with recent data.
        """
        last = existing["timestamp"].max()
        start = last + pd.Timedelta(hours=1)
        new   = self.fetch(start, end, tz=tz)
        combined = (
            pd.concat([existing, new], ignore_index=True)
            .drop_duplicates("timestamp")
            .sort_values("timestamp")
            .reset_index(drop=True)
        )
        return combined


# ── Shared feature engineering ────────────────────────────────────────────────

def _add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived columns to a raw (timestamp, price_eur_mwh) DataFrame."""
    df = df.copy()
    df["price_eur_kwh"] = df["price_eur_mwh"] / 1000.0
    df["dow"]           = df["timestamp"].dt.dayofweek
    df["is_weekend"]    = df["dow"] >= 5
    df["hour"]          = df["timestamp"].dt.hour
    df["minute"]        = df["timestamp"].dt.minute
    df["month"]         = df["timestamp"].dt.month
    df["season"]        = df["month"].map(_month_to_season)
    return df.reset_index(drop=True)


def _month_to_season(month: int) -> str:
    if month in (12, 1, 2):
        return "winter"
    if month in (3, 4, 5):
        return "spring"
    if month in (6, 7, 8):
        return "summer"
    return "autumn"
