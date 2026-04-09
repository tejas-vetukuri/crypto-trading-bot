#tests/test_data_loading.py
import pandas as pd
import pytest

from data.binance import BinanceDataClient
from data.bybit_derivatives import BybitDerivativesClient


class MockResponse:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise Exception(f"HTTP {self.status_code}")

    def json(self):
        return self._payload


def test_binance_get_candles_returns_clean_dataframe(monkeypatch):
    """
    Verifies that BinanceDataClient.get_candles():
    - returns the expected columns
    - converts timestamp to datetime
    - converts OHLCV columns to numeric
    - sorts by timestamp
    - removes duplicate timestamps
    """
    mock_rows = [
        [
            1704067200000, "42000", "42100", "41900", "42050", "1000",
            1704070799999, "0", "0", "0", "0", "0"
        ],
        [
            1704067200000, "42000", "42100", "41900", "42050", "1000",  # duplicate
            1704070799999, "0", "0", "0", "0", "0"
        ],
        [
            1704070800000, "42050", "42200", "42000", "42150", "1200",
            1704074399999, "0", "0", "0", "0", "0"
        ],
    ]

    def mock_get(url, params=None, timeout=None):
        return MockResponse(mock_rows)

    monkeypatch.setattr("requests.get", mock_get)

    client = BinanceDataClient(market="spot")
    df = client.get_candles(
        symbol="BTCUSDT",
        resolution="1h",
        start_date="2024-01-01",
        end_date="2024-01-02",
    )

    expected_columns = ["timestamp", "open", "high", "low", "close", "volume"]
    assert list(df.columns) == expected_columns
    assert len(df) == 2
    assert pd.api.types.is_datetime64tz_dtype(df["timestamp"])
    assert pd.api.types.is_numeric_dtype(df["open"])
    assert pd.api.types.is_numeric_dtype(df["high"])
    assert pd.api.types.is_numeric_dtype(df["low"])
    assert pd.api.types.is_numeric_dtype(df["close"])
    assert pd.api.types.is_numeric_dtype(df["volume"])
    assert df["timestamp"].is_monotonic_increasing


def test_bybit_open_interest_history_returns_clean_dataframe(monkeypatch):
    """
    Verifies that BybitDerivativesClient.get_open_interest_history():
    - extracts timestamp and by_oi correctly
    - converts timestamp to floored hourly datetime
    - converts by_oi to numeric
    - sorts by timestamp
    """
    mock_payload = {
        "retCode": 0,
        "retMsg": "OK",
        "result": {
            "list": [
                {"timestamp": "1704070800000", "openInterest": "12345.67"},
                {"timestamp": "1704067200000", "openInterest": "12000.50"},
            ],
            "nextPageCursor": "",
        },
    }

    def mock_get(url, params=None, timeout=None):
        return MockResponse(mock_payload)

    monkeypatch.setattr("requests.get", mock_get)

    client = BybitDerivativesClient()
    df = client.get_open_interest_history(
        symbol="BTCUSDT",
        interval_time="1h",
        start_date="2024-01-01",
        end_date="2024-01-02",
    )

    expected_columns = ["timestamp", "by_oi"]
    assert list(df.columns) == expected_columns
    assert len(df) == 2
    assert pd.api.types.is_datetime64tz_dtype(df["timestamp"])
    assert pd.api.types.is_numeric_dtype(df["by_oi"])
    assert df["timestamp"].is_monotonic_increasing


def test_bybit_long_short_ratio_history_builds_ratio_from_buy_sell(monkeypatch):
    """
    Verifies that BybitDerivativesClient.get_long_short_ratio_history():
    - builds by_lsr from buyRatio / sellRatio
    - returns the expected columns
    - sorts by timestamp
    """
    mock_payload = {
        "retCode": 0,
        "retMsg": "OK",
        "result": {
            "list": [
                {"timestamp": "1704070800000", "buyRatio": "1.2", "sellRatio": "0.8"},
                {"timestamp": "1704067200000", "buyRatio": "1.0", "sellRatio": "0.5"},
            ],
            "nextPageCursor": "",
        },
    }

    def mock_get(url, params=None, timeout=None):
        return MockResponse(mock_payload)

    monkeypatch.setattr("requests.get", mock_get)

    client = BybitDerivativesClient()
    df = client.get_long_short_ratio_history(
        symbol="BTCUSDT",
        period="1h",
        start_date="2024-01-01",
        end_date="2024-01-02",
    )

    expected_columns = ["timestamp", "by_lsr"]
    assert list(df.columns) == expected_columns
    assert len(df) == 2
    assert pd.api.types.is_datetime64tz_dtype(df["timestamp"])
    assert pd.api.types.is_numeric_dtype(df["by_lsr"])
    assert df["timestamp"].is_monotonic_increasing

    # After sorting, first row corresponds to 1.0 / 0.5 = 2.0
    assert df.iloc[0]["by_lsr"] == pytest.approx(2.0)
    # Second row corresponds to 1.2 / 0.8 = 1.5
    assert df.iloc[1]["by_lsr"] == pytest.approx(1.5)