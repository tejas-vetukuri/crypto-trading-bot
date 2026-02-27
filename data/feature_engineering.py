import numpy as np
import pandas as pd
from data.technical_indicators import TechnicalIndicators

def feature_engineering(df, horizon=6):
    """
    Feature engineering for LSTM + forward-looking target

    Args:
        df (pd.DataFrame): OHLCV dataframe
        horizon (int): Number of candles ahead to predict

    Returns:
        pd.DataFrame: dataframe with technical features, price dynamics, and future target
    """

    # ==============================
    # Technical Indicators
    # ==============================
    df["ema_20"] = TechnicalIndicators.calculate_ema(df, 20)
    df["ema_50"] = TechnicalIndicators.calculate_ema(df, 50)
    df["rsi"] = TechnicalIndicators.calculate_rsi(df)
    df["atr"] = TechnicalIndicators.calculate_atr(df)

    df["momentum_3"] = df["close"] - df["close"].shift(3)
    df["volatility_5"] = df["close"].rolling(5).std()

    # ==============================
    # Price Dynamics
    # ==============================

    # Log returns
    df["returns"] = np.log(df["close"] / df["close"].shift(1))

    # Candle structure
    df["candle_body"] = df["close"] - df["open"]
    df["upper_wick"] = df["high"] - np.maximum(df["open"], df["close"])
    df["lower_wick"] = np.minimum(df["open"], df["close"]) - df["low"]

    # Total range
    df["range"] = df["high"] - df["low"]

    # Body strength relative to range (avoid division by zero)
    df["body_ratio"] = df["candle_body"] / (df["range"] + 1e-9)

    # ==============================
    # Forward-Looking Target (NEW)
    # ==============================

    # Calculate future return over 'horizon' candles
    df["future_return"] = (df["close"].shift(-horizon) - df["close"]) / df["close"]

    # Convert to categorical target: 1 = price up, 0 = price down
    df["future_trend"] = np.where(df["future_return"] > 0, 1, 0)

    # Drop NaNs caused by rolling/shifting
    df.dropna(inplace=True)

    return df


def add_return_wick_vol_features(df: pd.DataFrame) -> pd.DataFrame:
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