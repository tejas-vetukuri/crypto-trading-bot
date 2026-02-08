import requests
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional
from pathlib import Path


class DeltaDataClient:
    BASE_URL = "https://api.delta.exchange"

    RESOLUTION_MINUTES = {
        "1m": 1,
        "5m": 5,
        "15m": 15,
        "30m": 30,
        "1h": 60,
        "2h": 120,
        "4h": 240,
        "6h": 360,
        "12h": 720,
        "1d": 1440,
        "1w": 10080,
    }

    MAX_CANDLES_PER_REQUEST = 1000   # Delta hard cap
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    DATA_DIR = PROJECT_ROOT / "data" / "raw"

    def __init__(self):
        self.DATA_DIR.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------
    # Internal single-request fetch
    # --------------------------------------------------
    def _fetch_candles_chunk(
        self,
        symbol: str,
        resolution: str,
        start_time: datetime,
        end_time: datetime,
        tz: str,
    ) -> pd.DataFrame:

        params = {
            "symbol": symbol,
            "resolution": resolution,
            "start": int(start_time.timestamp()),
            "end": int(end_time.timestamp()),
        }

        url = f"{self.BASE_URL}/v2/history/candles"
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()

        data = response.json()
        if "result" not in data:
            raise RuntimeError("Invalid response from Delta Exchange")

        df = pd.DataFrame(data["result"]).rename(columns={"time": "timestamp"})

        df["timestamp"] = (
            pd.to_datetime(df["timestamp"], unit="s")
            .dt.tz_localize("UTC")
            .dt.tz_convert(tz)
        )

        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        return df.sort_values("timestamp")

    # --------------------------------------------------
    # Public cached + paginated fetch
    # --------------------------------------------------
    def get_candles(
        self,
        symbol: str,
        resolution: str = "5m",
        limit: int = 5000,
        tz: str = "UTC",
        force_refresh: bool = False,
    ) -> pd.DataFrame:

        if resolution not in self.RESOLUTION_MINUTES:
            raise ValueError(f"Unsupported resolution: {resolution}")

        csv_path = self.DATA_DIR / f"{symbol}_{resolution}.csv"

        # -----------------------------
        # Load cached data if available
        # -----------------------------
        if csv_path.exists() and not force_refresh:
            df = pd.read_csv(csv_path, parse_dates=["timestamp"])
            df["timestamp"] = df["timestamp"].dt.tz_convert(tz)

            if len(df) >= limit:
                return df.tail(limit).reset_index(drop=True)

            candles_needed = limit - len(df)
            end_time = df["timestamp"].min().to_pydatetime()
            all_dfs = [df]
        else:
            candles_needed = limit
            end_time = datetime.utcnow()
            all_dfs = []

        minutes_per_candle = self.RESOLUTION_MINUTES[resolution]

        # -----------------------------
        # Pagination loop
        # -----------------------------
        while candles_needed > 0:
            chunk_size = min(self.MAX_CANDLES_PER_REQUEST, candles_needed)

            start_time = end_time - timedelta(
                minutes=chunk_size * minutes_per_candle
            )

            chunk = self._fetch_candles_chunk(
                symbol, resolution, start_time, end_time, tz
            )

            if chunk.empty:
                break

            all_dfs.append(chunk)
            candles_needed -= len(chunk)
            end_time = chunk["timestamp"].min().to_pydatetime()

        # -----------------------------
        # Merge + deduplicate
        # -----------------------------
        final_df = (
            pd.concat(all_dfs)
            .drop_duplicates("timestamp")
            .sort_values("timestamp")
            .reset_index(drop=True)
        )

        final_df.to_csv(csv_path, index=False)

        return final_df.tail(limit).reset_index(drop=True)
