from __future__ import annotations

import pandas as pd

from data.bybit_derivatives import BybitDerivativesClient


def merge_bybit_oi_lsr_features(
    price_df: pd.DataFrame,
    symbol: str = "BTCUSDT",
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    df = price_df.copy()
    client = BybitDerivativesClient()

    oi = client.get_open_interest_history(
        symbol=symbol,
        category="linear",
        interval_time="1h",
        start_date=start_date,
        end_date=end_date,
    )

    lsr = client.get_long_short_ratio_history(
        symbol=symbol,
        category="linear",
        period="1h",
        start_date=start_date,
        end_date=end_date,
    )

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.floor("h")

    if not oi.empty:
        df = df.merge(oi, on="timestamp", how="left")
    else:
        df["by_oi"] = pd.NA

    if not lsr.empty:
        df = df.merge(lsr, on="timestamp", how="left")
    else:
        df["by_lsr"] = pd.NA

    # reasonable for slow-changing market structure features
    if "by_oi" in df.columns:
        df["by_oi"] = df["by_oi"].ffill()
    if "by_lsr" in df.columns:
        df["by_lsr"] = df["by_lsr"].ffill()

    return df.sort_values("timestamp").reset_index(drop=True)