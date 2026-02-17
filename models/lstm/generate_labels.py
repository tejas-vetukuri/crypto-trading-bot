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
