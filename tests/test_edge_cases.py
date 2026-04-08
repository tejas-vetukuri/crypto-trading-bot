import pytest
import pandas as pd
import numpy as np

from data.feature_engineering import feature_engineering_xgb
from data.binance import BinanceDataClient
from models.lstm.sequence_builder import make_windows


# -------------------------------------------------
# 1. Feature Engineering: Missing Required Columns
# -------------------------------------------------

def test_feature_engineering_xgb_raises_on_missing_columns():
    """
    Ensures feature_engineering_xgb fails safely when required OHLCV
    columns are missing.
    """
    df = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=5, freq="h", tz="UTC"),
        "open": [100, 101, 102, 103, 104],
        "high": [101, 102, 103, 104, 105],
        "low": [99, 100, 101, 102, 103],
        # 'close' intentionally missing
        "volume": [1000, 1100, 1200, 1300, 1400],
    })

    with pytest.raises(ValueError, match="Missing required columns"):
        feature_engineering_xgb(df)


# -------------------------------------------------
# 2. Data Loading: Empty API Response
# -------------------------------------------------

def test_binance_get_candles_handles_empty_response(monkeypatch):
    """
    Ensures BinanceDataClient returns an empty DataFrame
    with correct columns when API returns no data.
    """

    class MockResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return []  # simulate empty API response

    def mock_get(url, params=None, timeout=None):
        return MockResponse()

    monkeypatch.setattr("requests.get", mock_get)

    client = BinanceDataClient(market="spot")

    df = client.get_candles(
        symbol="BTCUSDT",
        resolution="1h",
        start_date="2024-01-01",
        end_date="2024-01-02",
    )

    assert isinstance(df, pd.DataFrame)
    assert df.empty

    expected_cols = ["timestamp", "open", "high", "low", "close", "volume"]
    assert list(df.columns) == expected_cols


# -------------------------------------------------
# 3. LSTM: Input Shorter Than Window Size
# -------------------------------------------------

def test_make_windows_returns_empty_when_data_too_short():
    """
    Ensures make_windows handles insufficient sequence length
    without crashing and returns empty arrays.
    """
    timestamps = pd.date_range("2024-01-01", periods=3, freq="h", tz="UTC")

    df = pd.DataFrame({
        "timestamp": timestamps,
        "open": [100, 101, 102],
        "high": [101, 102, 103],
        "low": [99, 100, 101],
        "close": [100, 101, 102],
        "volume": [1000, 1100, 1200],
        "log_ret_1": [0.0, 0.01, 0.01],
        "body": [0.1, 0.1, 0.1],
        "range": [2, 2, 2],
        "upper_wick": [0.5, 0.5, 0.5],
        "lower_wick": [0.5, 0.5, 0.5],
        "clv": [0.0, 0.1, 0.1],
        "vol_10": [0.01, 0.01, 0.01],
        "vol_30": [0.01, 0.01, 0.01],
    })

    feature_cols = [
        "open", "high", "low", "close", "volume",
        "log_ret_1", "body", "range", "upper_wick", "lower_wick", "clv",
        "vol_10", "vol_30",
    ]

    X, y = make_windows(df, x_window_size=5, feature_cols=feature_cols)

    assert isinstance(X, np.ndarray)
    assert isinstance(y, np.ndarray)

    assert X.shape[0] == 0
    assert y.shape[0] == 0