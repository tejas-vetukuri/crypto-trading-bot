import numpy as np
import pandas as pd
import pytest

from data.feature_engineering import feature_engineering_xgb, feature_engineering_lstm
from data.feature_engineering_bybit import feature_engineering_bybit


def make_sample_ohlcv(n=80):
    timestamps = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")

    base = np.linspace(100, 180, n)
    df = pd.DataFrame({
        "timestamp": timestamps,
        "open": base + 0.5,
        "high": base + 2.0,
        "low": base - 2.0,
        "close": base + 1.0,
        "volume": np.linspace(1000, 2000, n),
    })
    return df


def test_feature_engineering_xgb_creates_expected_columns_and_clean_output():
    """
    Verifies that feature_engineering_xgb():
    - creates the key engineered features
    - creates the target column
    - drops rows with insufficient rolling history
    - returns a non-empty clean dataframe
    """
    df = make_sample_ohlcv(80)
    out = feature_engineering_xgb(df)

    expected_columns = {
        "timestamp", "open", "high", "low", "close", "volume",
        "ema_20", "ema_50", "rsi", "atr",
        "log_ret_1", "ret_1", "ret_3", "ret_5", "ret_10",
        "body", "range", "upper_wick", "lower_wick", "body_pct", "range_pct", "clv",
        "ema_spread", "ema20_dist", "ema50_dist", "ema20_slope_3", "ema50_slope_3",
        "atr_pct", "rsi_delta", "rsi_ma_10", "rsi_dist",
        "volatility_5", "vol_10", "vol_30", "vol_ratio",
        "vol_chg_1", "vol_chg_5", "vol_z20",
        "actual_trend",
    }

    assert not out.empty
    assert expected_columns.issubset(set(out.columns))
    assert out["actual_trend"].isin(["up", "down"]).all()
    assert out.replace([np.inf, -np.inf], np.nan).notna().all().all()
    assert out["timestamp"].is_monotonic_increasing


def test_feature_engineering_lstm_creates_expected_columns_and_no_infinities():
    """
    Verifies that feature_engineering_lstm():
    - creates the expected LSTM preprocessing features
    - keeps the dataframe structure intact
    - does not leave infinities in the output
    """
    df = make_sample_ohlcv(50)
    out = feature_engineering_lstm(df)

    expected_columns = {
        "timestamp", "open", "high", "low", "close", "volume",
        "log_ret_1", "body", "range", "upper_wick", "lower_wick", "clv",
        "vol_10", "vol_30",
    }

    assert expected_columns.issubset(set(out.columns))
    assert len(out) == len(df)
    assert out.replace([np.inf, -np.inf], np.nan).equals(out)


def test_feature_engineering_bybit_creates_stationary_features():
    """
    Verifies that feature_engineering_bybit():
    - creates oi_change, oi_z, and lsr_z when by_oi and by_lsr are present
    - returns numeric outputs
    """
    timestamps = pd.date_range("2024-01-01", periods=100, freq="h", tz="UTC")
    df = pd.DataFrame({
        "timestamp": timestamps,
        "by_oi": np.linspace(10000, 12000, 100),
        "by_lsr": np.linspace(0.8, 1.2, 100),
    })

    out = feature_engineering_bybit(df)

    expected_columns = {"oi_change", "oi_z", "lsr_z"}
    assert expected_columns.issubset(set(out.columns))
    assert pd.api.types.is_numeric_dtype(out["oi_change"])
    assert pd.api.types.is_numeric_dtype(out["oi_z"])
    assert pd.api.types.is_numeric_dtype(out["lsr_z"])