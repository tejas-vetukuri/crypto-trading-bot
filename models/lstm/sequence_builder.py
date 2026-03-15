import numpy as np
import pandas as pd


def make_windows(
    df: pd.DataFrame,
    x_window_size: int,
    feature_cols: list[str],
    target_col: str | None = None,
):
    """
    Build rolling windows for LSTM.

    Training mode:
      - if target_col is provided, returns (X, y)

    Inference mode:
      - if target_col is None, returns (X, None)

    For each sample i:
      X[i] = rows [i - x_window_size, ..., i - 1]
      y[i] = df[target_col].iloc[i]   (training only)

    Returns:
      X: (N, T, F)
      y: (N,) or None
    """
    required_cols = set(feature_cols)
    if target_col is not None:
        required_cols.add(target_col)

    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns for windowing: {missing}")

    feature_array = df.loc[:, feature_cols].to_numpy(dtype=np.float32)

    n_rows, n_features = feature_array.shape
    n_samples = n_rows - x_window_size

    if n_samples <= 0:
        raise ValueError(
            f"Not enough rows to build windows: len(df)={n_rows}, x_window_size={x_window_size}"
        )

    if target_col is None:
        X = np.empty((n_samples, x_window_size, n_features), dtype=np.float32)

        out_i = 0
        for i in range(x_window_size, n_rows):
            window = feature_array[i - x_window_size:i]

            if np.isnan(window).any():
                continue

            X[out_i] = window
            out_i += 1

        return X[:out_i], None

    target_array = df[target_col].to_numpy(dtype=np.int32)

    X = np.empty((n_samples, x_window_size, n_features), dtype=np.float32)
    y = np.empty(n_samples, dtype=np.int32)

    out_i = 0
    for i in range(x_window_size, n_rows):
        window = feature_array[i - x_window_size:i]

        if np.isnan(window).any():
            continue

        X[out_i] = window
        y[out_i] = target_array[i]
        out_i += 1

    return X[:out_i], y[:out_i]