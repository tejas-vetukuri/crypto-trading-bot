import numpy as np
from technical_indicators import TechnicalIndicators

def feature_engineering(df):
    df["ema_20"] = TechnicalIndicators.calculate_ema(df, 20)
    df["ema_50"] = TechnicalIndicators.calculate_ema(df, 50)
    df["rsi"] = TechnicalIndicators.calculate_rsi(df)
    df["atr"] = TechnicalIndicators.calculate_atr(df)
    df["momentum_3"] = df["close"] - df["close"].shift(3)
    df["volatility_5"] = df["close"].rolling(5).std()
    df["actual_trend"] = np.where(df["close"] > df["close"].shift(1), "up", "down")
    df.dropna(inplace=True)     # Handle missing values