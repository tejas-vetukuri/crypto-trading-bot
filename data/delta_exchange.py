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

    MAX_CANDLES_PER_REQUEST = 1000  # Delta hard cap
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

        if df.empty:
            return df

        df["timestamp"] = (
            pd.to_datetime(df["timestamp"], unit="s", utc=True)
            .dt.tz_convert(tz)
        )

        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        return df.sort_values("timestamp")

    # --------------------------------------------------
    # Public time-based + paginated fetch
    # --------------------------------------------------
    def get_candles(
        self,
        symbol: str,
        resolution: str = "5m",
        start_date: str = None,  # "YYYY-MM-DD"
        end_date: Optional[str] = None,
        tz: str = "UTC",
        force_refresh: bool = False,
    ) -> pd.DataFrame:

        if resolution not in self.RESOLUTION_MINUTES:
            raise ValueError(f"Unsupported resolution: {resolution}")

        if start_date is None:
            raise ValueError("start_date must be provided (YYYY-MM-DD)")

        start_time = pd.to_datetime(start_date, utc=True)
        end_time = (
            pd.to_datetime(end_date, utc=True)
            if end_date
            else pd.Timestamp.utcnow()
        )

        csv_path = self.DATA_DIR / f"{symbol}_{resolution}_{start_date}_to_{end_date or 'now'}.csv"

        # -----------------------------
        # Load cached data
        # -----------------------------
        if csv_path.exists() and not force_refresh:
            df = pd.read_csv(csv_path, parse_dates=["timestamp"])
            df["timestamp"] = df["timestamp"].dt.tz_convert(tz)
            return df

        minutes_per_candle = self.RESOLUTION_MINUTES[resolution]

        current_start = start_time
        all_dfs = []

        # -----------------------------
        # Safe Pagination Loop
        # -----------------------------
        while current_start < end_time:

            current_end = min(
                current_start + timedelta(
                    minutes=self.MAX_CANDLES_PER_REQUEST * minutes_per_candle
                ),
                end_time,
            )

            chunk = self._fetch_candles_chunk(
                symbol, resolution, current_start, current_end, tz
            )

            if chunk.empty:
                break

            all_dfs.append(chunk)

            # Move window safely using last returned candle
            last_timestamp = chunk["timestamp"].max()

            # Convert back to UTC for next request
            current_start = last_timestamp.tz_convert("UTC").to_pydatetime()

            # Step forward one candle to prevent infinite loop
            current_start += timedelta(minutes=minutes_per_candle)

        if not all_dfs:
            raise RuntimeError("No data returned from API for given time range.")

        # -----------------------------
        # Merge + Deduplicate
        # -----------------------------
        final_df = (
            pd.concat(all_dfs)
            .drop_duplicates("timestamp")
            .sort_values("timestamp")
            .reset_index(drop=True)
        )

        final_df.to_csv(csv_path, index=False)

        return final_df
