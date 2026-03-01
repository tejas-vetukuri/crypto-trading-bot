import numpy as np
import pandas as pd
from data.technical_indicators import TechnicalIndicators

def feature_engineering_xgb(df: pd.DataFrame) -> pd.DataFrame:
    """
    Feature engineering for XGBoost trend classifier.
    Uses ONLY past information.
    Label = next-candle direction.
    """
    required = {"timestamp", "open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.copy()
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Indicators (must be causal implementations)
    df["ema_20"] = TechnicalIndicators.calculate_ema(df, 20)
    df["ema_50"] = TechnicalIndicators.calculate_ema(df, 50)
    df["rsi"] = TechnicalIndicators.calculate_rsi(df)
    df["atr"] = TechnicalIndicators.calculate_atr(df)

    # Additional causal features
    df["momentum_3"] = df["close"] - df["close"].shift(3)
    df["volatility_5"] = df["close"].rolling(5).std()

    # Next-candle label (NO leakage)
    df["actual_trend"] = np.where(
        df["close"].shift(-1) > df["close"],
        "up",
        "down"
    )

    feature_cols = ["ema_20", "ema_50", "rsi", "momentum_3", "volatility_5"]
    df = df.dropna(subset=feature_cols + ["actual_trend"]).reset_index(drop=True)

    return df


def feature_engineering_lstm(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = df[c].astype(float)

    eps = 1e-12

    # Returns
    df["log_ret_1"] = np.log((df["close"] + eps) / (df["close"].shift(1) + eps))

    # Candle geometry (scale-free)
    o = df["open"]
    h = df["high"]
    l = df["low"]
    c = df["close"]

    df["body"] = (c - o) / (o + eps)
    df["range"] = (h - l) / (o + eps)

    max_oc = np.maximum(o, c)
    min_oc = np.minimum(o, c)

    df["upper_wick"] = (h - max_oc) / (o + eps)
    df["lower_wick"] = (min_oc - l) / (o + eps)

    df["clv"] = (2.0 * c - h - l) / ((h - l) + eps)

    # Volatility
    df["vol_10"] = df["log_ret_1"].rolling(10).std()
    df["vol_30"] = df["log_ret_1"].rolling(30).std()

    df = df.replace([np.inf, -np.inf], np.nan)
    return df