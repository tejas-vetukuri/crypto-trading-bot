from __future__ import annotations

import numpy as np
import pandas as pd

from pathlib import Path
from joblib import dump, load

from sklearn.preprocessing import StandardScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

from data.binance import BinanceDataClient
from data.feature_engineering import feature_engineering_lstm
from models.lstm.confidence_threshold import eval_with_ignore_zone


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ALT_DIR = PROJECT_ROOT / "models" / "alternate"
SAVED_DIR = ALT_DIR / "saved"

START_DATES_BY_INTERVAL = {
    "5m": "2025-01-01",
    "15m": "2024-01-01",
    "1h": "2017-09-01",
    "4h": "2017-09-01",
}


def resolve_project_path(path_str: str | Path) -> Path:
    p = Path(path_str)
    if p.is_absolute():
        return p
    return PROJECT_ROOT / p


def get_default_start_date(resolution: str) -> str:
    if resolution not in START_DATES_BY_INTERVAL:
        raise ValueError(
            f"Unsupported resolution '{resolution}'. "
            f"Expected one of {list(START_DATES_BY_INTERVAL.keys())}"
        )
    return START_DATES_BY_INTERVAL[resolution]


def build_lstm_returns_save_paths(symbol: str, resolution: str) -> dict:
    out_dir = SAVED_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    return {
        "model_path": out_dir / "lstm_returns_model.keras",
        "scaler_path": out_dir / "lstm_returns_scaler.joblib",
        "artifacts_path": out_dir / "lstm_returns_artifacts.joblib",
        "test_probs_path": out_dir / "lstm_returns_test_probs.csv",
        "metrics_path": out_dir / "lstm_returns_metrics.csv",
    }


def create_vol_adjusted_binary_target(
    df: pd.DataFrame,
    horizon: int = 12,
    vol_window: int = 24,
) -> pd.DataFrame:
    """
    target_score = forward_log_return / sigma_h

    where:
      forward_log_return = log(close[t+h] / close[t])
      sigma_t = rolling std of 1-step log returns
      sigma_h = sigma_t * sqrt(horizon)

    Binary labels:
      up   if target_score > 0
      down otherwise
    """
    df = df.copy()

    df["log_ret_1_raw"] = np.log(df["close"] / df["close"].shift(1))
    df["sigma_t"] = df["log_ret_1_raw"].rolling(vol_window).std()
    df["sigma_h"] = df["sigma_t"] * np.sqrt(horizon)

    df["future_close"] = df["close"].shift(-horizon)
    df["forward_log_return"] = np.log(df["future_close"] / df["close"])

    eps = 1e-12
    df["target_score"] = df["forward_log_return"] / (df["sigma_h"] + eps)

    df["actual_trend"] = np.where(df["target_score"] > 0, "up", "down")
    df["target"] = (df["actual_trend"] == "up").astype(np.int32)

    return df


def _add_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    df = df.copy()

    prev_close = df["close"].shift(1)
    tr1 = df["high"] - df["low"]
    tr2 = (df["high"] - prev_close).abs()
    tr3 = (df["low"] - prev_close).abs()

    df["tr"] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df["atr"] = df["tr"].rolling(period).mean()
    return df


def make_windows_for_target(
    df: pd.DataFrame,
    x_window_size: int = 100,
    feature_cols=None,
    target_col: str = "target",
):
    if feature_cols is None:
        feature_cols = [
            "open", "high", "low", "close", "volume",
            "log_ret_1", "body", "range", "upper_wick", "lower_wick", "clv",
            "vol_10", "vol_30",
        ]

    data = df[feature_cols].astype(float).reset_index(drop=True)
    target = df[target_col].astype(int).reset_index(drop=True)

    X_list, y_list = [], []

    for i in range(x_window_size, len(data)):
        x_win = data.iloc[i - x_window_size:i].values
        y = int(target.iloc[i])

        X_list.append(x_win)
        y_list.append(y)

    X = np.asarray(X_list, dtype=np.float32)
    y = np.asarray(y_list, dtype=np.int32)
    return X, y


def _predict_from_probs_with_ignore_zone(
    probs: np.ndarray,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    probs = np.asarray(probs, dtype=float)
    used_mask = (probs >= threshold) | (probs <= (1.0 - threshold))
    pred_used = (probs[used_mask] >= 0.5).astype(int)
    return used_mask, pred_used


def build_lstm_returns_trade_metrics(
    probs_df: pd.DataFrame,
    threshold: float,
    rr: float = 1.25,
    fee_bps: float = 2.0,
    trade_penalty_bps: float = 2.0,
    sl_atr_mult: float = 1.0,
    max_horizon: int = 3,
    min_atr_pct: float = 0.001,
) -> dict:
    """
    Simple trade simulation from saved LSTM-return predictions.

    Expected columns in probs_df:
      timestamp, close, high, low, atr, p, y_true, actual_trend, row_idx
    """
    df = probs_df.copy().reset_index(drop=True)

    required_cols = {"close", "high", "low", "atr", "p", "y_true"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns for trade simulation: {missing}")

    probs = df["p"].astype(float).values
    used_mask, pred_used = _predict_from_probs_with_ignore_zone(probs, threshold)

    setups = int(used_mask.sum())
    total_rows = int(len(df))

    trade_rows = []
    equity_curve = [0.0]

    fee_r = fee_bps / 10000.0
    trade_penalty_r = trade_penalty_bps / 10000.0

    used_indices = np.where(used_mask)[0]

    for idx, pred in zip(used_indices, pred_used):
        entry = float(df.loc[idx, "close"])
        atr = float(df.loc[idx, "atr"]) if pd.notna(df.loc[idx, "atr"]) else np.nan

        if not np.isfinite(atr):
            continue

        atr_pct = atr / entry if entry > 0 else np.nan
        if not np.isfinite(atr_pct) or atr_pct < min_atr_pct:
            continue

        risk_dist = max(atr * sl_atr_mult, entry * min_atr_pct)
        if risk_dist <= 0:
            continue

        direction = "long" if int(pred) == 1 else "short"

        if direction == "long":
            stop_price = entry - risk_dist
            take_price = entry + rr * risk_dist
        else:
            stop_price = entry + risk_dist
            take_price = entry - rr * risk_dist

        exit_type = "horizon"
        gross_r = 0.0

        future_slice = df.iloc[idx + 1: idx + 1 + int(max_horizon)].copy()

        if len(future_slice) == 0:
            continue

        exited = False

        for _, row in future_slice.iterrows():
            high = float(row["high"])
            low = float(row["low"])

            if direction == "long":
                hit_sl = low <= stop_price
                hit_tp = high >= take_price

                if hit_sl and hit_tp:
                    exit_type = "sl_exits"
                    gross_r = -1.0
                    exited = True
                    break
                if hit_sl:
                    exit_type = "sl_exits"
                    gross_r = -1.0
                    exited = True
                    break
                if hit_tp:
                    exit_type = "tp_exits"
                    gross_r = float(rr)
                    exited = True
                    break
            else:
                hit_sl = high >= stop_price
                hit_tp = low <= take_price

                if hit_sl and hit_tp:
                    exit_type = "sl_exits"
                    gross_r = -1.0
                    exited = True
                    break
                if hit_sl:
                    exit_type = "sl_exits"
                    gross_r = -1.0
                    exited = True
                    break
                if hit_tp:
                    exit_type = "tp_exits"
                    gross_r = float(rr)
                    exited = True
                    break

        if not exited:
            final_close = float(future_slice.iloc[-1]["close"])
            if direction == "long":
                gross_r = (final_close - entry) / risk_dist
            else:
                gross_r = (entry - final_close) / risk_dist
            exit_type = "horizon_exits"

        net_r = gross_r - fee_r - trade_penalty_r

        actual = int(df.loc[idx, "y_true"])
        directional_correct = int(pred) == actual
        won = gross_r > 0

        trade_rows.append({
            "idx": idx,
            "direction": direction,
            "gross_r": gross_r,
            "net_r": net_r,
            "won": bool(won),
            "directional_correct": bool(directional_correct),
            "exit_type": exit_type,
        })

        equity_curve.append(equity_curve[-1] + net_r)

    trades_df = pd.DataFrame(trade_rows)

    taken = int(len(trades_df))
    skipped = int(total_rows - taken)
    take_rate = (taken / total_rows) if total_rows else 0.0

    if taken == 0:
        return {
            "setups": setups,
            "taken": 0,
            "skipped": total_rows,
            "take_rate": take_rate,
            "directional_accuracy": 0.0,
            "win_rate": 0.0,
            "avg_gross_r_per_trade": 0.0,
            "avg_net_r_per_trade": 0.0,
            "total_return": 0.0,
            "max_drawdown": 0.0,
            "tp_exits": 0,
            "sl_exits": 0,
            "horizon_exits": 0,
            "other_exits": 0,
            "trades_df": trades_df,
        }

    equity = pd.Series(equity_curve[1:], dtype=float)
    rolling_peak = equity.cummax()
    drawdown = rolling_peak - equity
    max_drawdown = float(drawdown.max()) if len(drawdown) else 0.0

    return {
        "setups": setups,
        "taken": taken,
        "skipped": skipped,
        "take_rate": take_rate,
        "directional_accuracy": float(trades_df["directional_correct"].mean()),
        "win_rate": float(trades_df["won"].mean()),
        "avg_gross_r_per_trade": float(trades_df["gross_r"].mean()),
        "avg_net_r_per_trade": float(trades_df["net_r"].mean()),
        "total_return": float(trades_df["net_r"].sum()),
        "max_drawdown": max_drawdown,
        "tp_exits": int((trades_df["exit_type"] == "tp_exits").sum()),
        "sl_exits": int((trades_df["exit_type"] == "sl_exits").sum()),
        "horizon_exits": int((trades_df["exit_type"] == "horizon_exits").sum()),
        "other_exits": int((~trades_df["exit_type"].isin(["tp_exits", "sl_exits", "horizon_exits"])).sum()),
        "trades_df": trades_df,
    }


def train_lstm_returns_model(
    symbol: str = "BTCUSDT",
    resolution: str = "1h",
    start_date: str = "2017-09-01",
    end_date: str | None = None,
    train_ratio: float = 0.80,
    x_window_size: int = 100,
    epochs: int = 10,
    batch_size: int = 64,
    validation_split: float = 0.05,
    learning_rate: float = 0.001,
    lstm_units: int = 100,
    dropout_rate: float = 0.20,
    early_stopping_patience: int = 3,
    horizon: int = 12,
    vol_window: int = 24,
    model_path: str | Path = "models/alternate/saved/lstm_returns_model.keras",
    scaler_path: str | Path = "models/alternate/saved/lstm_returns_scaler.joblib",
    artifacts_path: str | Path = "models/alternate/saved/lstm_returns_artifacts.joblib",
    test_probs_path: str | Path = "models/alternate/saved/lstm_returns_test_probs.csv",
    metrics_path: str | Path = "models/alternate/saved/lstm_returns_metrics.csv",
    thresholds: tuple[float, ...] = (0.50, 0.53, 0.55, 0.57),
):
    if not (0.0 < train_ratio < 1.0):
        raise ValueError(f"train_ratio must be in (0,1). Got {train_ratio}")

    client = BinanceDataClient(market="spot")
    df = client.get_candles(
        symbol=symbol,
        resolution=resolution,
        start_date=start_date,
        end_date=end_date,
    )

    required = {"timestamp", "open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.sort_values("timestamp").reset_index(drop=True)
    df = df.dropna(subset=list(required)).reset_index(drop=True)
    df = _add_atr(df, period=14)

    df = create_vol_adjusted_binary_target(
        df=df,
        horizon=horizon,
        vol_window=vol_window,
    )

    df = df.dropna(
        subset=[
            "future_close",
            "forward_log_return",
            "sigma_t",
            "sigma_h",
            "target_score",
            "actual_trend",
            "target",
            "atr",
        ]
    ).reset_index(drop=True)

    df["row_idx"] = np.arange(len(df))

    df = feature_engineering_lstm(df)

    feature_cols = [
        "open", "high", "low", "close", "volume",
        "log_ret_1", "body", "range", "upper_wick", "lower_wick", "clv",
        "vol_10", "vol_30",
    ]

    df = df.dropna(subset=feature_cols + ["target"]).reset_index(drop=True)

    X, y = make_windows_for_target(
        df=df,
        x_window_size=int(x_window_size),
        feature_cols=feature_cols,
        target_col="target",
    )

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int32)

    if X.ndim != 3:
        raise ValueError(f"Expected X shape (N, T, F), got {X.shape}")

    meta_df = df.iloc[x_window_size:].reset_index(drop=True)

    if len(meta_df) != len(y):
        raise ValueError(f"meta_df length {len(meta_df)} does not match y length {len(y)}")

    n = len(X)
    train_end = int(n * train_ratio)
    if train_end <= 0 or train_end >= n:
        raise ValueError(f"Invalid split after windowing. n={n}, train_end={train_end}")

    X_train, X_test = X[:train_end], X[train_end:]
    y_train, y_test = y[:train_end], y[train_end:]
    meta_train = meta_df.iloc[:train_end].copy()
    meta_test = meta_df.iloc[train_end:].copy()

    n_features = X_train.shape[-1]
    scaler = StandardScaler()

    X_train_2d = X_train.reshape(-1, n_features)
    scaler.fit(X_train_2d)

    scaler_path_p = resolve_project_path(scaler_path)
    scaler_path_p.parent.mkdir(parents=True, exist_ok=True)
    dump(scaler, str(scaler_path_p))

    X_train_s = scaler.transform(X_train_2d).reshape(X_train.shape).astype(np.float32)
    X_test_s = scaler.transform(X_test.reshape(-1, n_features)).reshape(X_test.shape).astype(np.float32)

    model = Sequential([
        Input(shape=(x_window_size, n_features)),
        LSTM(int(lstm_units), return_sequences=False),
        Dropout(float(dropout_rate)),
        Dense(1, activation="sigmoid"),
    ])

    model.compile(
        optimizer=Adam(learning_rate=float(learning_rate)),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    history = model.fit(
        X_train_s,
        y_train,
        epochs=int(epochs),
        batch_size=int(batch_size),
        validation_split=float(validation_split),
        shuffle=False,
        callbacks=[EarlyStopping(patience=int(early_stopping_patience), restore_best_weights=True)],
        verbose=1,
    )

    model_path_p = resolve_project_path(model_path)
    model_path_p.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(model_path_p))

    p_test = model.predict(X_test_s, batch_size=int(batch_size), verbose=0).reshape(-1)

    probs_df = pd.DataFrame({
        "timestamp": meta_test["timestamp"].values,
        "row_idx": meta_test["row_idx"].values,
        "close": meta_test["close"].values,
        "high": meta_test["high"].values,
        "low": meta_test["low"].values,
        "atr": meta_test["atr"].values,
        "y_true": y_test.astype(int),
        "actual_trend": np.where(y_test == 1, "up", "down"),
        "p": p_test,
        "p_up": p_test,
        "p_down": 1.0 - p_test,
        "probability": p_test,
        "future_close": meta_test["future_close"].values,
        "forward_log_return": meta_test["forward_log_return"].values,
        "sigma_t": meta_test["sigma_t"].values,
        "sigma_h": meta_test["sigma_h"].values,
        "target_score": meta_test["target_score"].values,
    })

    test_probs_path_p = resolve_project_path(test_probs_path)
    test_probs_path_p.parent.mkdir(parents=True, exist_ok=True)
    probs_df.to_csv(test_probs_path_p, index=False)

    metrics = [eval_with_ignore_zone(y_test, p_test, threshold=t) for t in thresholds]
    metrics_df = pd.DataFrame(metrics)

    metrics_path_p = resolve_project_path(metrics_path)
    metrics_path_p.parent.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(metrics_path_p, index=False)

    artifacts_path_p = resolve_project_path(artifacts_path)
    artifacts_path_p.parent.mkdir(parents=True, exist_ok=True)

    artifacts = {
        "model_path": str(model_path_p),
        "scaler_path": str(scaler_path_p),
        "artifacts_path": str(artifacts_path_p),
        "test_probs_path": str(test_probs_path_p),
        "metrics_path": str(metrics_path_p),
        "symbol": symbol,
        "resolution": resolution,
        "start_date": start_date,
        "end_date": end_date,
        "train_ratio": float(train_ratio),
        "x_window_size": int(x_window_size),
        "feature_cols": feature_cols,
        "thresholds_eval": thresholds,
        "target_type": "volatility_adjusted_return",
        "target_definition": "up if forward_log_return / sigma_h > 0 else down",
        "class_mapping": {"down": 0, "up": 1},
        "horizon": int(horizon),
        "vol_window": int(vol_window),
        "train_samples": int(len(X_train)),
        "test_samples": int(len(X_test)),
        "test_start_timestamp": str(meta_test["timestamp"].iloc[0]) if len(meta_test) else None,
        "test_end_timestamp": str(meta_test["timestamp"].iloc[-1]) if len(meta_test) else None,
    }

    dump(artifacts, str(artifacts_path_p))

    return model, history, probs_df, metrics_df, scaler


def load_lstm_returns_artifacts(artifacts_path: str | Path) -> dict:
    return load(str(resolve_project_path(artifacts_path)))