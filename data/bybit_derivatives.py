from __future__ import annotations

import time
from typing import Optional

import pandas as pd
import requests


class BybitDerivativesClient:
    def __init__(self, timeout: int = 30):
        self.base_url = "https://api.bybit.com"
        self.timeout = timeout

    @staticmethod
    def _to_millis(dt_str: Optional[str]) -> Optional[int]:
        if dt_str is None:
            return None
        dt = pd.to_datetime(dt_str, utc=True)
        return int(dt.timestamp() * 1000)

    def _get(self, endpoint: str, params: dict) -> dict:
        url = self.base_url + endpoint
        r = requests.get(url, params=params, timeout=self.timeout)
        r.raise_for_status()
        payload = r.json()

        ret_code = payload.get("retCode", 0)
        if ret_code != 0:
            raise RuntimeError(
                f"Bybit API error for {endpoint}: retCode={ret_code}, retMsg={payload.get('retMsg')}"
            )
        return payload

    def get_open_interest_history(
        self,
        symbol: str = "BTCUSDT",
        category: str = "linear",
        interval_time: str = "1h",
        start_date: str | None = None,
        end_date: str | None = None,
        limit: int = 200,
    ) -> pd.DataFrame:
        """
        Bybit V5 open interest history.
        """
        endpoint = "/v5/market/open-interest"

        start_ms = self._to_millis(start_date)
        end_ms = self._to_millis(end_date) if end_date else int(time.time() * 1000)

        all_rows = []
        cursor = None

        while True:
            params = {
                "category": category,
                "symbol": symbol,
                "intervalTime": interval_time,
                "limit": limit,
            }
            if start_ms is not None:
                params["startTime"] = start_ms
            if end_ms is not None:
                params["endTime"] = end_ms
            if cursor:
                params["cursor"] = cursor

            payload = self._get(endpoint, params)
            result = payload.get("result", {})
            rows = result.get("list", [])

            if not rows:
                break

            all_rows.extend(rows)

            next_cursor = result.get("nextPageCursor")
            if not next_cursor or next_cursor == cursor:
                break
            cursor = next_cursor

        if not all_rows:
            return pd.DataFrame(columns=["timestamp", "by_oi"])

        df = pd.DataFrame(all_rows).copy()

        # Expected fields: timestamp, openInterest
        df.rename(columns={
            "openInterest": "by_oi",
        }, inplace=True)

        df["timestamp"] = pd.to_datetime(df["timestamp"].astype("int64"), unit="ms", utc=True)
        df["timestamp"] = df["timestamp"].dt.floor("h")
        df["by_oi"] = pd.to_numeric(df["by_oi"], errors="coerce")

        df = (
            df[["timestamp", "by_oi"]]
            .dropna(subset=["timestamp", "by_oi"])
            .drop_duplicates(subset=["timestamp"])
            .sort_values("timestamp")
            .reset_index(drop=True)
        )
        return df

    def get_long_short_ratio_history(
        self,
        symbol: str = "BTCUSDT",
        category: str = "linear",
        period: str = "1h",
        start_date: str | None = None,
        end_date: str | None = None,
        limit: int = 200,
    ) -> pd.DataFrame:
        """
        Bybit V5 long/short ratio history.
        """
        endpoint = "/v5/market/account-ratio"

        start_ms = self._to_millis(start_date)
        end_ms = self._to_millis(end_date) if end_date else int(time.time() * 1000)

        all_rows = []
        cursor = None

        while True:
            params = {
                "category": category,
                "symbol": symbol,
                "period": period,
                "limit": limit,
            }
            if start_ms is not None:
                params["startTime"] = start_ms
            if end_ms is not None:
                params["endTime"] = end_ms
            if cursor:
                params["cursor"] = cursor

            payload = self._get(endpoint, params)
            result = payload.get("result", {})
            rows = result.get("list", [])

            if not rows:
                break

            all_rows.extend(rows)

            next_cursor = result.get("nextPageCursor")
            if not next_cursor or next_cursor == cursor:
                break
            cursor = next_cursor

        if not all_rows:
            return pd.DataFrame(columns=["timestamp", "by_lsr"])

        df = pd.DataFrame(all_rows).copy()

        # Common field name in docs: buyRatio / sellRatio. Build long/short ratio from that.
        if "buyRatio" in df.columns and "sellRatio" in df.columns:
            df["buyRatio"] = pd.to_numeric(df["buyRatio"], errors="coerce")
            df["sellRatio"] = pd.to_numeric(df["sellRatio"], errors="coerce")
            df["by_lsr"] = df["buyRatio"] / df["sellRatio"].replace(0, pd.NA)
        elif "longShortRatio" in df.columns:
            df["by_lsr"] = pd.to_numeric(df["longShortRatio"], errors="coerce")
        else:
            raise RuntimeError(f"Unexpected Bybit long/short ratio columns: {list(df.columns)}")

        ts_col = "timestamp" if "timestamp" in df.columns else "time"
        df["timestamp"] = pd.to_datetime(df[ts_col].astype("int64"), unit="ms", utc=True)
        df["timestamp"] = df["timestamp"].dt.floor("h")

        df = (
            df[["timestamp", "by_lsr"]]
            .dropna(subset=["timestamp", "by_lsr"])
            .drop_duplicates(subset=["timestamp"])
            .sort_values("timestamp")
            .reset_index(drop=True)
        )
        return df