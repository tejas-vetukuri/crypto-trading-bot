from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from joblib import load

from data.feature_engineering import feature_engineering_xgb, feature_engineering_lstm

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _normalize_input_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "timestamp" not in df.columns:
        for candidate in ["open_time", "datetime", "date", "time"]:
            if candidate in df.columns:
                df = df.rename(columns={candidate: "timestamp"})
                break

    if "timestamp" not in df.columns:
        raise ValueError(f"Missing timestamp column. Got columns: {list(df.columns)}")

    return df


def _resolve_path(path_like) -> Path:
    p = Path(path_like)
    if p.is_absolute():
        return p
    return (PROJECT_ROOT / p).resolve()


def build_latest_xgb_features(df: pd.DataFrame, xgb_artifacts: dict):
    df = _normalize_input_df(df)

    feat_df = feature_engineering_xgb(df.copy())
    feat_df = feat_df.dropna().reset_index(drop=True)

    feature_cols = xgb_artifacts["features"]

    if feat_df.empty:
        raise ValueError("No usable XGB rows after feature engineering.")

    missing_cols = [col for col in feature_cols if col not in feat_df.columns]
    if missing_cols:
        raise ValueError(f"Missing XGB feature columns: {missing_cols}")

    latest_row = feat_df.iloc[[-1]]
    X_latest = latest_row[feature_cols].values

    return latest_row, X_latest


def build_latest_lstm_window(df: pd.DataFrame, lstm_artifacts: dict):
    df = _normalize_input_df(df)

    feat_df = feature_engineering_lstm(df.copy())
    feat_df = feat_df.dropna().reset_index(drop=True)

    feature_cols = list(lstm_artifacts["feature_cols"])
    lookback = int(lstm_artifacts["x_window_size"])

    scaler_path = _resolve_path(lstm_artifacts["scaler_path"])
    print("Resolved scaler path:", scaler_path)

    scaler = load(scaler_path)

    if len(feat_df) < lookback:
        raise ValueError(
            f"Not enough rows for LSTM window. Need {lookback}, got {len(feat_df)}."
        )

    missing_cols = [col for col in feature_cols if col not in feat_df.columns]
    if missing_cols:
        raise ValueError(f"Missing LSTM feature columns: {missing_cols}")

    X_all = feat_df[feature_cols].values.astype(np.float32)
    n_features = X_all.shape[1]

    X_scaled = scaler.transform(X_all).astype(np.float32)
    X_latest = X_scaled[-lookback:].reshape(1, lookback, n_features)

    latest_row = feat_df.iloc[[-1]]
    return latest_row, X_latest