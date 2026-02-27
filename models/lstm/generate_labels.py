import numpy as np
from scipy.signal import savgol_filter


# ==========================================================
# 1️⃣ Generate smoothed trend labels
# ==========================================================
def generate_regime_labels(close_prices, window=25, polyorder=3):
    """
    Create up/down trend labels based on smoothed price curve.
    1 = uptrend, 0 = downtrend
    """
    # Smooth close prices
    smooth_prices = savgol_filter(close_prices, window_length=window, polyorder=polyorder)
    # Calculate slope: positive slope = uptrend
    slope = np.diff(smooth_prices, prepend=smooth_prices[0])
    labels = (slope > 0).astype(int)
    return labels

def generate_regime_labels_causal(close_prices, window: int = 25):
    """
    Past-only trend labels using rolling linear-regression slope over the last `window` closes.
    1 = uptrend (positive slope), 0 = downtrend (non-positive slope)
    """
    close = np.asarray(close_prices, dtype=float)
    n = len(close)
    labels = np.zeros(n, dtype=int)

    if window < 2:
        raise ValueError("window must be >= 2")
    if n == 0:
        return labels

    x = np.arange(window, dtype=float)
    x_mean = x.mean()
    denom = ((x - x_mean) ** 2).sum()

    for i in range(window - 1, n):
        y = close[i - window + 1 : i + 1]
        y_mean = y.mean()
        slope = ((x - x_mean) * (y - y_mean)).sum() / denom
        labels[i] = 1 if slope > 0 else 0

    labels[: window - 1] = labels[window - 1]
    return labels
