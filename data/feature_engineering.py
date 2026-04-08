#data/feature_engineering.py

import numpy as np
import pandas as pd
from data.technical_indicators import TechnicalIndicators


def feature_engineering_xgb(df: pd.DataFrame) -> pd.DataFrame:
    required = {"timestamp", "open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.copy()
    df = df.sort_values("timestamp").reset_index(drop=True)

    eps = 1e-12

    # -----------------------------
    # 1) Returns (core)
    # -----------------------------
    df["log_ret_1"] = np.log(df["close"] + eps).diff()
    df["ret_1"] = df["close"].pct_change(1)
    df["ret_3"] = df["close"].pct_change(3)
    df["ret_5"] = df["close"].pct_change(5)
    df["ret_10"] = df["close"].pct_change(10)

    # -----------------------------
    # 2) Candle structure (very useful)
    # -----------------------------
    df["body"] = df["close"] - df["open"]
    df["range"] = df["high"] - df["low"]
    df["upper_wick"] = df["high"] - df[["open", "close"]].max(axis=1)
    df["lower_wick"] = df[["open", "close"]].min(axis=1) - df["low"]

    df["body_pct"] = df["body"] / (df["close"] + eps)
    df["range_pct"] = df["range"] / (df["close"] + eps)

    # close location value (-1..1-ish)
    df["clv"] = ((df["close"] - df["low"]) - (df["high"] - df["close"])) / (df["high"] - df["low"] + eps)

    # -----------------------------
    # 3) Indicators (trend / momentum)
    # -----------------------------
    df["ema_20"] = TechnicalIndicators.calculate_ema(df, 20)
    df["ema_50"] = TechnicalIndicators.calculate_ema(df, 50)
    df["rsi"] = TechnicalIndicators.calculate_rsi(df)
    df["atr"] = TechnicalIndicators.calculate_atr(df)

    # Derived indicator features (often better than raw)
    df["ema_spread"] = (df["ema_20"] - df["ema_50"]) / (df["close"] + eps)
    df["ema20_dist"] = (df["close"] - df["ema_20"]) / (df["close"] + eps)
    df["ema50_dist"] = (df["close"] - df["ema_50"]) / (df["close"] + eps)
    df["ema20_slope_3"] = df["ema_20"].diff(3)
    df["ema50_slope_3"] = df["ema_50"].diff(3)

    df["atr_pct"] = df["atr"] / (df["close"] + eps)

    df["rsi_delta"] = df["rsi"].diff(1)
    df["rsi_ma_10"] = df["rsi"].rolling(10).mean()
    df["rsi_dist"] = df["rsi"] - df["rsi_ma_10"]

    # -----------------------------
    # 4) Volatility regime
    # -----------------------------
    df["volatility_5"] = df["close"].rolling(5).std()
    df["vol_10"] = df["log_ret_1"].rolling(10).std()
    df["vol_30"] = df["log_ret_1"].rolling(30).std()
    df["vol_ratio"] = df["vol_10"] / (df["vol_30"] + eps)

    # -----------------------------
    # 5) Volume regime
    # -----------------------------
    df["vol_chg_1"] = df["volume"].pct_change(1)
    df["vol_chg_5"] = df["volume"].pct_change(5)
    vmean = df["volume"].rolling(20).mean()
    vstd = df["volume"].rolling(20).std()
    df["vol_z20"] = (df["volume"] - vmean) / (vstd + eps)

    # -----------------------------
    # 6) Target: next-candle direction (binary)
    # -----------------------------
    df["actual_trend"] = np.where(df["close"].shift(-1) > df["close"], "up", "down")

    # IMPORTANT: include every feature you train on
    feature_cols = [
        "ema_20", "ema_50", "rsi", "atr",
        "log_ret_1", "ret_1", "ret_3", "ret_5", "ret_10",
        "body", "range", "upper_wick", "lower_wick", "body_pct", "range_pct", "clv",
        "ema_spread", "ema20_dist", "ema50_dist", "ema20_slope_3", "ema50_slope_3",
        "atr_pct", "rsi_delta", "rsi_ma_10", "rsi_dist",
        "volatility_5", "vol_10", "vol_30", "vol_ratio",
        "vol_chg_1", "vol_chg_5", "vol_z20",
    ]

    #Replace inf with Nans and drop Nans
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
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