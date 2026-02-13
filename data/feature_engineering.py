import numpy as np
from data.technical_indicators import TechnicalIndicators

def feature_engineering(df):
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
    # Price Dynamics (NEW)
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
    # Target
    # ==============================
    df["actual_trend"] = np.where(df["close"] > df["close"].shift(1), "up", "down")

    # Drop NaNs caused by shifting/rolling
    df.dropna(inplace=True)

    return df
