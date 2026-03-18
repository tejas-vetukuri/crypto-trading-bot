from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

from joblib import dump

from data.binance import BinanceDataClient
from data.feature_engineering import feature_engineering_lstm
from models.lstm.sequence_builder import make_windows
from models.lstm.confidence_threshold import eval_with_ignore_zone


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LSTM_DIR = PROJECT_ROOT / "models" / "lstm"
SAVED_DIR = LSTM_DIR / "saved"

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
            f"Expected one of: {list(START_DATES_BY_INTERVAL.keys())}"
        )
    return START_DATES_BY_INTERVAL[resolution]


def symbol_tag(symbol: str) -> str:
    symbol = symbol.upper()
    if symbol.endswith("USDT"):
        return symbol[:-4]
    return symbol


def combo_tag(symbol: str, resolution: str) -> str:
    return f"{symbol_tag(symbol)}_{resolution}"


def build_lstm_save_paths(symbol: str, resolution: str) -> dict[str, Path]:
    tag = combo_tag(symbol, resolution)
    SAVED_DIR.mkdir(parents=True, exist_ok=True)

    return {
        "model_path": SAVED_DIR / f"lstm_{tag}.keras",
        "scaler_path": SAVED_DIR / f"lstm_scaler_{tag}.joblib",
        "artifacts_path": SAVED_DIR / f"lstm_artifacts_{tag}.joblib",
        "test_probs_path": SAVED_DIR / f"lstm_test_probs_{tag}.csv",
        "metrics_path": SAVED_DIR / f"lstm_metrics_{tag}.csv",
    }


def train_lstm_model(
    symbol: str = "BTCUSDT",
    resolution: str = "1h",
    start_date: str | None = None,
    end_date: str | None = None,
    x_window_size: int = 100,
    epochs: int = 10,
    batch_size: int = 64,
    model_path: str | Path | None = None,
    scaler_path: str | Path | None = None,
    artifacts_path: str | Path | None = None,
    test_probs_path: str | Path | None = None,
    metrics_path: str | Path | None = None,
    thresholds: tuple[float, ...] = (0.50, 0.52, 0.55, 0.60),
):
    """
    Trains next-candle direction LSTM and saves combination-specific outputs.

    Save naming example for BTCUSDT, 5m:
      - models/lstm/saved/lstm_BTC_5m.keras
      - models/lstm/saved/lstm_scaler_BTC_5m.joblib
      - models/lstm/saved/lstm_artifacts_BTC_5m.joblib
      - models/lstm/saved/lstm_test_probs_BTC_5m.csv
      - models/lstm/saved/lstm_metrics_BTC_5m.csv

    Returns:
      model, history, out_df, metrics_df, scaler
    """
    symbol = symbol.upper()
    if start_date is None:
        start_date = get_default_start_date(resolution)

    default_paths = build_lstm_save_paths(symbol, resolution)

    model_path_p = resolve_project_path(model_path) if model_path else default_paths["model_path"]
    scaler_path_p = resolve_project_path(scaler_path) if scaler_path else default_paths["scaler_path"]
    artifacts_path_p = resolve_project_path(artifacts_path) if artifacts_path else default_paths["artifacts_path"]
    test_probs_path_p = resolve_project_path(test_probs_path) if test_probs_path else default_paths["test_probs_path"]
    metrics_path_p = resolve_project_path(metrics_path) if metrics_path else default_paths["metrics_path"]

    for p in [model_path_p, scaler_path_p, artifacts_path_p, test_probs_path_p, metrics_path_p]:
        p.parent.mkdir(parents=True, exist_ok=True)

    print("\n================ LSTM TRAIN CONFIG ================")
    print(f"Symbol:        {symbol}")
    print(f"Resolution:    {resolution}")
    print(f"Start date:    {start_date}")
    print(f"End date:      {end_date}")
    print(f"Window size:   {x_window_size}")
    print(f"Epochs:        {epochs}")
    print(f"Batch size:    {batch_size}")
    print(f"Combo tag:     {combo_tag(symbol, resolution)}")
    print("===================================================\n")

    # 1) Fetch candles
    client = BinanceDataClient(market="spot")
    df = client.get_candles(
        symbol=symbol,
        resolution=resolution,
        start_date=start_date,
        end_date=end_date,
    )

    required = {"open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.dropna(subset=list(required)).reset_index(drop=True)
    if df.empty:
        raise ValueError("No rows returned after raw candle cleaning.")

    # 2) Feature engineering
    df = feature_engineering_lstm(df)

    feature_cols = [
        "open", "high", "low", "close", "volume",
        "log_ret_1", "body", "range", "upper_wick", "lower_wick", "clv",
        "vol_10", "vol_30",
    ]
    df = df.dropna(subset=feature_cols).reset_index(drop=True)

    if df.empty:
        raise ValueError("No rows available after feature engineering and NA drop.")

    # 3) Windowing + labels
    X, y = make_windows(df, x_window_size=x_window_size, feature_cols=feature_cols)
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int32)

    if X.ndim != 3:
        raise ValueError(f"Expected X with shape (N, T, F), got {X.shape}")
    if X.shape[1] != x_window_size:
        raise ValueError(f"Expected T={x_window_size}, got {X.shape[1]}")
    if X.shape[2] != len(feature_cols):
        raise ValueError(f"Expected F={len(feature_cols)}, got {X.shape[2]}")

    # 4) Chronological split
    n = len(X)
    train_end = int(n * 0.95)
    if train_end <= 0 or train_end >= n:
        raise ValueError(f"Not enough samples after windowing. n={n}")

    X_train, X_test = X[:train_end], X[train_end:]
    y_train, y_test = y[:train_end], y[train_end:]

    # 5) Train-only StandardScaler
    n_features = X_train.shape[-1]
    scaler = StandardScaler()

    X_train_2d = X_train.reshape(-1, n_features)
    scaler.fit(X_train_2d)

    dump(scaler, str(scaler_path_p))
    print(f"✅ Saved scaler: {scaler_path_p}")

    X_train_s = scaler.transform(X_train_2d).reshape(X_train.shape).astype(np.float32)
    X_test_s = scaler.transform(X_test.reshape(-1, n_features)).reshape(X_test.shape).astype(np.float32)

    # 6) Model
    model = Sequential([
        Input(shape=(x_window_size, n_features)),
        LSTM(100, return_sequences=False),
        Dropout(0.2),
        Dense(1, activation="sigmoid"),
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    history = model.fit(
        X_train_s,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.05,
        shuffle=False,
        callbacks=[EarlyStopping(patience=3, restore_best_weights=True)],
        verbose=1,
    )

    model.save(str(model_path_p))
    print(f"✅ Saved model: {model_path_p}")

    # 7) Test probs + metrics
    p_test = model.predict(X_test_s, batch_size=batch_size, verbose=0).reshape(-1)

    out_df = pd.DataFrame({
        "symbol": symbol,
        "resolution": resolution,
        "y_true": y_test.astype(int),
        "p": p_test.astype(float),
    })
    out_df.to_csv(str(test_probs_path_p), index=False)
    print(f"✅ Saved test probs: {test_probs_path_p}")

    metrics = [eval_with_ignore_zone(y_test, p_test, threshold=t) for t in thresholds]
    metrics_df = pd.DataFrame(metrics)
    metrics_df.insert(0, "symbol", symbol)
    metrics_df.insert(1, "resolution", resolution)
    metrics_df.to_csv(str(metrics_path_p), index=False)
    print(f"✅ Saved metrics: {metrics_path_p}")

    # 8) Save artifacts
    lstm_artifacts = {
        "model_path": str(model_path_p.relative_to(PROJECT_ROOT)),
        "scaler_path": str(scaler_path_p.relative_to(PROJECT_ROOT)),
        "test_probs_path": str(test_probs_path_p.relative_to(PROJECT_ROOT)),
        "metrics_path": str(metrics_path_p.relative_to(PROJECT_ROOT)),
        "x_window_size": int(x_window_size),
        "feature_cols": feature_cols,
        "symbol": symbol,
        "symbol_tag": symbol_tag(symbol),
        "resolution": resolution,
        "combo_tag": combo_tag(symbol, resolution),
        "start_date": start_date,
        "end_date": end_date,
        "thresholds_eval": tuple(float(t) for t in thresholds),
        "ignore_zone_threshold_for_sideways": 0.53,
        "market": "spot",
        "train_split_ratio": 0.95,
    }

    dump(lstm_artifacts, str(artifacts_path_p))
    print(f"✅ Saved artifacts: {artifacts_path_p}")

    print("\n📊 Label balance:")
    print(f"Train UP rate: {float(y_train.mean()):.4f}")
    print(f"Test  UP rate: {float(y_test.mean()):.4f}")

    return model, history, out_df, metrics_df, scaler