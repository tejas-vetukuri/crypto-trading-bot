import pandas as pd
import requests
import streamlit as st


BINANCE_KLINES_URL = "https://api.binance.com/api/v3/klines"


@st.cache_data(ttl=30)
def fetch_klines(symbol: str, interval: str, limit: int = 200) -> pd.DataFrame:
    """
    Fetch recent Binance spot klines and return a clean DataFrame.
    Cached for 30 seconds to avoid repeated API hits on every rerun.
    """
    params = {
        "symbol": symbol,
        "interval": interval,
        "limit": limit,
    }

    response = requests.get(BINANCE_KLINES_URL, params=params, timeout=10)
    response.raise_for_status()
    data = response.json()

    cols = [
        "open_time",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "close_time",
        "quote_asset_volume",
        "number_of_trades",
        "taker_buy_base_asset_volume",
        "taker_buy_quote_asset_volume",
        "ignore",
    ]

    df = pd.DataFrame(data, columns=cols)

    numeric_cols = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "quote_asset_volume",
        "taker_buy_base_asset_volume",
        "taker_buy_quote_asset_volume",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["number_of_trades"] = pd.to_numeric(df["number_of_trades"], errors="coerce")
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)

    return df