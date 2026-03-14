# data/binance.py

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Optional

import pandas as pd
import requests


class BinanceDataClient:
    def __init__(self, market: str = "spot", timeout: int = 30):
        """
        market:
            - 'spot'    -> https://api.binance.com/api/v3/klines
            - 'futures' -> https://fapi.binance.com/fapi/v1/klines
        """
        self.market = market
        self.timeout = timeout

        if market == "spot":
            self.base_url = "https://api.binance.com"
            self.endpoint = "/api/v3/klines"
        elif market == "futures":
            self.base_url = "https://fapi.binance.com"
            self.endpoint = "/fapi/v1/klines"
        else:
            raise ValueError("market must be 'spot' or 'futures'")

    @staticmethod
    def _to_millis(dt_str: Optional[str]) -> Optional[int]:
        if dt_str is None:
            return None

        # Accepts YYYY-MM-DD or ISO-like strings
        dt = pd.to_datetime(dt_str, utc=True)
        return int(dt.timestamp() * 1000)

    def get_candles(
        self,
        symbol: str,
        resolution: str,
        start_date: str,
        end_date: str | None = None,
        limit: int = 1000,
    ) -> pd.DataFrame:
        """
        Returns a DataFrame with:
        timestamp, open, high, low, close, volume
        """
        start_ms = self._to_millis(start_date)
        end_ms = self._to_millis(end_date) if end_date else int(time.time() * 1000)

        all_rows = []
        current_start = start_ms

        while current_start is not None and current_start < end_ms:
            params = {
                "symbol": symbol,
                "interval": resolution,
                "startTime": current_start,
                "endTime": end_ms,
                "limit": limit,
            }

            url = self.base_url + self.endpoint
            response = requests.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()
            rows = response.json()

            if not rows:
                break

            all_rows.extend(rows)

            last_open_time = rows[-1][0]
            next_start = int(last_open_time) + 1

            if next_start <= current_start:
                break

            current_start = next_start

            if len(rows) < limit:
                break

        if not all_rows:
            return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

        df = pd.DataFrame(
            all_rows,
            columns=[
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
            ],
        )

        df = df[["open_time", "open", "high", "low", "close", "volume"]].copy()
        df.rename(columns={"open_time": "timestamp"}, inplace=True)

        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df = (
            df.dropna(subset=["timestamp", "open", "high", "low", "close"])
              .drop_duplicates(subset=["timestamp"])
              .sort_values("timestamp")
              .reset_index(drop=True)
        )

        return df