#models/lstm/sequence_builder.py

import pandas as pd
import numpy as np

def make_windows(
    df: pd.DataFrame,
    x_window_size: int = 100,
    feature_cols=None,
):
    """
    Builds raw (UNSCALED) windows + next-candle direction labels.

    Scaling is intentionally NOT done here.
    We'll do train-only StandardScaler after splitting.
    """
    if feature_cols is None:
        feature_cols = [
            "open", "high", "low", "close", "volume",
            "log_ret_1", "body", "range", "upper_wick", "lower_wick", "clv",
            "vol_10", "vol_30",
        ]

    data = df[feature_cols].astype(float).reset_index(drop=True)

    X_list, y_list = [], []
    for i in range(x_window_size, len(data)):
        # Raw window (no min-max)
        x_win = data.iloc[i - x_window_size:i].values  # (x_window_size, n_features)

        prev_close = float(df.iloc[i - 1]["close"])
        next_close = float(df.iloc[i]["close"])
        y = 1 if next_close > prev_close else 0

        X_list.append(x_win)
        y_list.append(y)

    X = np.asarray(X_list, dtype=np.float32)  # (N, x_window_size, F)
    y = np.asarray(y_list, dtype=np.int32)    # (N,)
    return X, y