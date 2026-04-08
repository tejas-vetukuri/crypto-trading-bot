import numpy as np
import pandas as pd

from models.lstm.sequence_builder import make_windows


def make_lstm_input_df(n=8):
    timestamps = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")

    df = pd.DataFrame({
        "timestamp": timestamps,
        "open":        [100, 101, 102, 103, 104, 105, 106, 107],
        "high":        [101, 102, 103, 104, 105, 106, 107, 108],
        "low":         [ 99, 100, 101, 102, 103, 104, 105, 106],
        "close":       [100, 102, 101, 104, 103, 106, 105, 108],
        "volume":      [1000, 1010, 1020, 1030, 1040, 1050, 1060, 1070],
        "log_ret_1":   [0.0, 0.01, -0.01, 0.02, -0.01, 0.02, -0.01, 0.03],
        "body":        [0.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0],
        "range":       [2.0] * 8,
        "upper_wick":  [0.5] * 8,
        "lower_wick":  [0.5] * 8,
        "clv":         [0.0, 0.5, -0.5, 0.3, -0.3, 0.4, -0.4, 0.6],
        "vol_10":      [0.1] * 8,
        "vol_30":      [0.2] * 8,
    })
    return df


def test_make_windows_returns_correct_shape():
    """
    Verifies that make_windows():
    - returns X with shape (N, window_size, n_features)
    - returns y with shape (N,)
    - preserves float32 / int32 output types
    """
    df = make_lstm_input_df(n=8)

    feature_cols = [
        "open", "high", "low", "close", "volume",
        "log_ret_1", "body", "range", "upper_wick", "lower_wick", "clv",
        "vol_10", "vol_30",
    ]

    X, y = make_windows(df, x_window_size=3, feature_cols=feature_cols)

    # N = len(df) - window_size = 8 - 3 = 5
    assert X.shape == (5, 3, len(feature_cols))
    assert y.shape == (5,)
    assert X.dtype == np.float32
    assert y.dtype == np.int32


def test_make_windows_labels_align_with_next_candle_direction():
    """
    Verifies that labels are constructed correctly:
    y = 1 if next_close > previous_close else 0
    """
    df = make_lstm_input_df(n=8)

    feature_cols = [
        "open", "high", "low", "close", "volume",
        "log_ret_1", "body", "range", "upper_wick", "lower_wick", "clv",
        "vol_10", "vol_30",
    ]

    X, y = make_windows(df, x_window_size=3, feature_cols=feature_cols)

    # Close values:
    # [100, 102, 101, 104, 103, 106, 105, 108]
    #
    # With window_size=3, labels start from i=3:
    # i=3: prev_close=101, next_close=104 -> 1
    # i=4: prev_close=104, next_close=103 -> 0
    # i=5: prev_close=103, next_close=106 -> 1
    # i=6: prev_close=106, next_close=105 -> 0
    # i=7: prev_close=105, next_close=108 -> 1

    expected_y = np.array([1, 0, 1, 0, 1], dtype=np.int32)
    assert np.array_equal(y, expected_y)