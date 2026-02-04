import requests
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional


class DeltaDataClient:
    """
    Synchronous Delta Exchange market data client.
    Safe for backtesting, research, and live trading.
    """

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

    def get_candles(
        self,
        symbol: str,
        resolution: str = "5m",
        limit: int = 200,
        end_time: Optional[datetime] = None,
        tz: str = "UTC",
    ) -> pd.DataFrame:
        """
        Fetch historical OHLCV candles.

        Parameters
        ----------
        symbol : str
            Market symbol (e.g. BTCUSD)
        resolution : str
            Candle timeframe (e.g. 1m, 5m, 1h)
        limit : int
            Number of candles to fetch
        end_time : datetime, optional
            Fetch candles up to this time (defaults to now)
        tz : str
            Timezone for returned timestamps

        Returns
        -------
        pd.DataFrame
            Columns: timestamp, open, high, low, close, volume
        """

        if resolution not in self.RESOLUTION_MINUTES:
            raise ValueError(f"Unsupported resolution: {resolution}")

        minutes_per_candle = self.RESOLUTION_MINUTES[resolution]

        if end_time is None:
            end_time = datetime.utcnow()

        # Ensure only CLOSED candles are used
        end_time = end_time.replace(second=0, microsecond=0)
        end_time -= timedelta(minutes=minutes_per_candle)

        start_time = end_time - timedelta(
            minutes=(limit - 1) * minutes_per_candle
        )

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

        # Timestamp handling
        df["timestamp"] = (
            pd.to_datetime(df["timestamp"], unit="s")
            .dt.tz_localize("UTC")
            .dt.tz_convert(tz)
        )

        # Numeric safety
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.sort_values("timestamp").reset_index(drop=True)

        return df

    def get_ticker(self, symbol: str) -> dict:
        """
        Fetch latest market ticker data.
        """
        url = f"{self.BASE_URL}/v2/tickers/{symbol}"
        response = requests.get(url, timeout=10)
        response.raise_for_status()

        data = response.json()
        if "result" not in data:
            raise RuntimeError("Invalid ticker response")

        return data["result"]
