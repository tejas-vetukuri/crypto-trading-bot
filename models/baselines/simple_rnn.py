#models/baselines/simple_rnn.py
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

from joblib import dump

from data.binance import BinanceDataClient
from data.feature_engineering import feature_engineering_lstm
from models.lstm.sequence_builder import make_windows
from models.lstm.confidence_threshold import eval_with_ignore_zone


PROJECT_ROOT = Path(__file__).resolve().parents[2]
BASELINES_DIR = PROJECT_ROOT / "models" / "baselines"
SAVED_DIR = BASELINES_DIR / "saved"

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


def build_simple_rnn_save_paths(symbol: str, resolution: str) -> dict[str, Path]:
    tag = combo_tag(symbol, resolution)
    SAVED_DIR.mkdir(parents=True, exist_ok=True)

    return {
        "model_path": SAVED_DIR / f"simple_rnn_{tag}.keras",
        "scaler_path": SAVED_DIR / f"simple_rnn_scaler_{tag}.joblib",
        "artifacts_path": SAVED_DIR / f"simple_rnn_artifacts_{tag}.joblib",
        "test_probs_path": SAVED_DIR / f"simple_rnn_test_probs_{tag}.csv",
        "metrics_path": SAVED_DIR / f"simple_rnn_metrics_{tag}.csv",
    }


def train_simple_rnn_model(
    symbol: str = "BTCUSDT",
    resolution: str = "1h",
    start_date: str | None = None,
    end_date: str | None = None,
    x_window_size: int = 100,
    epochs: int = 10,
    batch_size: int = 64,
    train_ratio: float = 0.80,
    validation_split: float = 0.05,
    learning_rate: float = 0.001,
    rnn_units: int = 100,
    dropout_rate: float = 0.20,
    early_stopping_patience: int = 3,
    model_path: str | Path | None = None,
    scaler_path: str | Path | None = None,
    artifacts_path: str | Path | None = None,
    test_probs_path: str | Path | None = None,
    metrics_path: str | Path | None = None,
    thresholds: tuple[float, ...] = (0.50, 0.52, 0.55, 0.60),
):
    """
    Trains next-candle direction Simple RNN baseline and saves combination-specific outputs.

    Split logic:
      - chronological train/test split using train_ratio
      - validation_split is taken only from the training block inside model.fit()

    Default effective split:
      - 80% train block
      - 5% of that 80% used as validation
      - 20% held-out test

    Save naming example for BTCUSDT, 5m:
      - models/baselines/saved/simple_rnn_BTC_5m.keras
      - models/baselines/saved/simple_rnn_scaler_BTC_5m.joblib
      - models/baselines/saved/simple_rnn_artifacts_BTC_5m.joblib
      - models/baselines/saved/simple_rnn_test_probs_BTC_5m.csv
      - models/baselines/saved/simple_rnn_metrics_BTC_5m.csv

    Returns:
      model, history, out_df, metrics_df, scaler
    """
    symbol = symbol.upper()
    if start_date is None:
        start_date = get_default_start_date(resolution)

    if not (0.0 < train_ratio < 1.0):
        raise ValueError(f"train_ratio must be in (0, 1). Got {train_ratio}")

    if not (0.0 < validation_split < 1.0):
        raise ValueError(f"validation_split must be in (0, 1). Got {validation_split}")

    if validation_split >= train_ratio:
        raise ValueError(
            f"validation_split ({validation_split}) should be smaller than train_ratio ({train_ratio})."
        )

    if learning_rate <= 0:
        raise ValueError(f"learning_rate must be > 0. Got {learning_rate}")

    if rnn_units <= 0:
        raise ValueError(f"rnn_units must be > 0. Got {rnn_units}")

    if not (0.0 <= dropout_rate < 1.0):
        raise ValueError(f"dropout_rate must be in [0, 1). Got {dropout_rate}")

    if early_stopping_patience < 1:
        raise ValueError(
            f"early_stopping_patience must be >= 1. Got {early_stopping_patience}"
        )

    default_paths = build_simple_rnn_save_paths(symbol, resolution)

    model_path_p = resolve_project_path(model_path) if model_path else default_paths["model_path"]
    scaler_path_p = resolve_project_path(scaler_path) if scaler_path else default_paths["scaler_path"]
    artifacts_path_p = resolve_project_path(artifacts_path) if artifacts_path else default_paths["artifacts_path"]
    test_probs_path_p = resolve_project_path(test_probs_path) if test_probs_path else default_paths["test_probs_path"]
    metrics_path_p = resolve_project_path(metrics_path) if metrics_path else default_paths["metrics_path"]

    for p in [model_path_p, scaler_path_p, artifacts_path_p, test_probs_path_p, metrics_path_p]:
        p.parent.mkdir(parents=True, exist_ok=True)

    print("\n================ SIMPLE RNN TRAIN CONFIG ================")
    print(f"Symbol:                  {symbol}")
    print(f"Resolution:              {resolution}")
    print(f"Start date:              {start_date}")
    print(f"End date:                {end_date}")
    print(f"Window size:             {x_window_size}")
    print(f"Epochs:                  {epochs}")
    print(f"Batch size:              {batch_size}")
    print(f"Train ratio:             {train_ratio}")
    print(f"Validation split:        {validation_split} (within train block)")
    print(f"Learning rate:           {learning_rate}")
    print(f"RNN units:               {rnn_units}")
    print(f"Dropout rate:            {dropout_rate}")
    print(f"Early stopping patience: {early_stopping_patience}")
    print(f"Combo tag:               {combo_tag(symbol, resolution)}")
    print("=========================================================\n")

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

    df = feature_engineering_lstm(df)

    feature_cols = [
        "open", "high", "low", "close", "volume",
        "log_ret_1", "body", "range", "upper_wick", "lower_wick", "clv",
        "vol_10", "vol_30",
    ]
    df = df.dropna(subset=feature_cols).reset_index(drop=True)

    if df.empty:
        raise ValueError("No rows available after feature engineering and NA drop.")

    X, y = make_windows(df, x_window_size=x_window_size, feature_cols=feature_cols)
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int32)

    if X.ndim != 3:
        raise ValueError(f"Expected X with shape (N, T, F), got {X.shape}")
    if X.shape[1] != x_window_size:
        raise ValueError(f"Expected T={x_window_size}, got {X.shape[1]}")
    if X.shape[2] != len(feature_cols):
        raise ValueError(f"Expected F={len(feature_cols)}, got {X.shape[2]}")

    n = len(X)
    train_end = int(n * train_ratio)

    if train_end <= 0 or train_end >= n:
        raise ValueError(f"Not enough samples after windowing. n={n}")

    X_train_full, X_test = X[:train_end], X[train_end:]
    y_train_full, y_test = y[:train_end], y[train_end:]

    if len(X_train_full) < 2 or len(X_test) < 1:
        raise ValueError(
            f"Insufficient samples after split. "
            f"train={len(X_train_full)}, test={len(X_test)}"
        )

    n_features = X_train_full.shape[-1]
    scaler = StandardScaler()

    X_train_2d = X_train_full.reshape(-1, n_features)
    scaler.fit(X_train_2d)

    dump(scaler, str(scaler_path_p))
    print(f"✅ Saved scaler: {scaler_path_p}")

    X_train_s_full = (
        scaler.transform(X_train_full.reshape(-1, n_features))
        .reshape(X_train_full.shape)
        .astype(np.float32)
    )
    X_test_s = (
        scaler.transform(X_test.reshape(-1, n_features))
        .reshape(X_test.shape)
        .astype(np.float32)
    )

    model = Sequential([
        Input(shape=(x_window_size, n_features)),
        SimpleRNN(rnn_units, return_sequences=False),
        Dropout(dropout_rate),
        Dense(1, activation="sigmoid"),
    ])

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    history = model.fit(
        X_train_s_full,
        y_train_full,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        shuffle=False,
        callbacks=[
            EarlyStopping(
                patience=early_stopping_patience,
                restore_best_weights=True,
            )
        ],
        verbose=1,
    )

    model.save(str(model_path_p))
    print(f"✅ Saved model: {model_path_p}")

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

    simple_rnn_artifacts = {
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
        "train_ratio": float(train_ratio),
        "validation_split_within_train": float(validation_split),
        "learning_rate": float(learning_rate),
        "rnn_units": int(rnn_units),
        "dropout_rate": float(dropout_rate),
        "early_stopping_patience": int(early_stopping_patience),
        "model_type": "simple_rnn_baseline",
    }

    dump(simple_rnn_artifacts, str(artifacts_path_p))
    print(f"✅ Saved artifacts: {artifacts_path_p}")

    print("\n📊 Label balance:")
    print(f"Train UP rate: {float(y_train_full.mean()):.4f}")
    print(f"Test  UP rate: {float(y_test.mean()):.4f}")

    return model, history, out_df, metrics_df, scaler