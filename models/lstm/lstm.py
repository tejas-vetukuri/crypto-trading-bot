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


def train_lstm_model(
    symbol: str = "BTCUSDT",
    resolution: str = "1h",
    start_date: str = "2017-09-01",
    end_date: str | None = None,
    train_ratio: float = 0.80,
    x_window_size: int = 100,
    epochs: int = 10,
    batch_size: int = 64,

    # target settings aligned with XGB
    horizon: int = 12,
    vol_window: int = 24,

    model_path: str = "models/lstm/lstm_vol_adj_target.keras",
    scaler_path: str = "models/lstm/lstm_scaler.joblib",
    artifacts_path: str = "models/lstm/lstm_artifacts.joblib",
    preds_csv_path: str = "models/lstm/lstm_vol_adj_target_test_probs.csv",
    metrics_csv_path: str = "models/lstm/lstm_vol_adj_target_metrics.csv",

    thresholds: tuple[float, ...] = (0.5, 0.55, 0.6),
):
    """
    LSTM binary classifier using the same volatility-adjusted forward-return
    target family as the XGBoost setup.

    Returns:
      model, history, out_df, metrics_df, scaler
    """

    if not (0.0 < train_ratio < 1.0):
        raise ValueError(f"train_ratio must be in (0,1). Got {train_ratio}")

    # -----------------------------
    # 1) Fetch candles
    # -----------------------------
    client = BinanceDataClient(market="spot")
    df = client.get_candles(
        symbol=symbol,
        resolution=resolution,
        start_date=start_date,
        end_date=end_date
    )

    required = {"timestamp", "open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.sort_values("timestamp").reset_index(drop=True)
    df = df.dropna(subset=list(required)).reset_index(drop=True)

    # -----------------------------
    # 2) Create target FIRST
    # -----------------------------
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
        ]
    ).reset_index(drop=True)

    # -----------------------------
    # 3) Feature engineering
    # -----------------------------
    df = feature_engineering_lstm(df)

    feature_cols = [
        "open", "high", "low", "close", "volume",
        "log_ret_1", "body", "range", "upper_wick", "lower_wick", "clv",
        "vol_10", "vol_30",
    ]

    df = df.dropna(subset=feature_cols + ["target"]).reset_index(drop=True)

    print("\nDF shape before make_windows:", df.shape)
    print("Feature cols present:", all(c in df.columns for c in feature_cols))
    print("Selected feature shape:", df.loc[:, feature_cols].shape)

    # -----------------------------
    # 4) Windowing + label
    # -----------------------------
    X, y = make_windows(
        df,
        x_window_size=x_window_size,
        feature_cols=feature_cols,
        target_col="target",
    )

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int32)

    if X.ndim != 3:
        raise ValueError(f"Expected X with shape (N, T, F), got {X.shape}")
    if X.shape[1] != x_window_size:
        raise ValueError(f"Expected T={x_window_size}, got {X.shape[1]}")
    if X.shape[2] != len(feature_cols):
        raise ValueError(f"Expected F={len(feature_cols)}, got {X.shape[2]}")

    # align metadata rows with y
    meta_df = df.iloc[x_window_size:].reset_index(drop=True)

    if len(meta_df) != len(y):
        raise ValueError(f"meta_df length {len(meta_df)} does not match y length {len(y)}")

    # -----------------------------
    # 5) Chronological split (aligned to XGB)
    # -----------------------------
    n = len(X)
    train_end = int(n * train_ratio)
    if train_end <= 0 or train_end >= n:
        raise ValueError(f"Invalid split after windowing. n={n}, train_end={train_end}")

    X_train, X_test = X[:train_end], X[train_end:]
    y_train, y_test = y[:train_end], y[train_end:]
    meta_train = meta_df.iloc[:train_end].copy()
    meta_test = meta_df.iloc[train_end:].copy()

    print("\n================ LSTM SPLIT SUMMARY ================")
    print(f"Total samples after windowing: {n}")
    print(f"Train ratio used:              {train_ratio:.2f}")
    print(f"Train samples:                 {len(X_train)}")
    print(f"Test samples:                  {len(X_test)}")
    if len(meta_train) > 0:
        print(f"Train start timestamp:         {meta_train['timestamp'].iloc[0]}")
        print(f"Train end timestamp:           {meta_train['timestamp'].iloc[-1]}")
    if len(meta_test) > 0:
        print(f"Test start timestamp:          {meta_test['timestamp'].iloc[0]}")
        print(f"Test end timestamp:            {meta_test['timestamp'].iloc[-1]}")

    print("\nTraining label distribution:")
    print(pd.Series(y_train).map({0: "down", 1: "up"}).value_counts())

    print("\nTraining label ratios:")
    print(pd.Series(y_train).map({0: "down", 1: "up"}).value_counts(normalize=True).sort_index())

    print("\nTest label distribution:")
    print(pd.Series(y_test).map({0: "down", 1: "up"}).value_counts())

    print("\nTest label ratios:")
    print(pd.Series(y_test).map({0: "down", 1: "up"}).value_counts(normalize=True).sort_index())

    # -----------------------------
    # 6) Train-only StandardScaler
    # -----------------------------
    n_features = X_train.shape[-1]
    scaler = StandardScaler()

    X_train_2d = X_train.reshape(-1, n_features)
    scaler.fit(X_train_2d)

    scaler_path_p = Path(scaler_path)
    scaler_path_p.parent.mkdir(parents=True, exist_ok=True)
    dump(scaler, str(scaler_path_p))
    print(f"✅ Saved: {scaler_path_p}")

    X_train_s = scaler.transform(X_train_2d).reshape(X_train.shape).astype(np.float32)
    X_test_s = scaler.transform(X_test.reshape(-1, n_features)).reshape(X_test.shape).astype(np.float32)

    # -----------------------------
    # 7) Model
    # -----------------------------
    model = Sequential([
        Input(shape=(x_window_size, n_features)),
        LSTM(100, return_sequences=False),
        Dropout(0.2),
        Dense(1, activation="sigmoid"),
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    history = model.fit(
        X_train_s, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.05,
        shuffle=False,
        callbacks=[EarlyStopping(patience=3, restore_best_weights=True)],
        verbose=1
    )

    model_path_p = Path(model_path)
    model_path_p.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(model_path_p))
    print(f"✅ Saved: {model_path_p}")

    # -----------------------------
    # 8) Test probs + metrics
    # -----------------------------
    p_test = model.predict(X_test_s, batch_size=batch_size).reshape(-1)

    out_df = pd.DataFrame({
        "timestamp": meta_test["timestamp"].values,
        "y_true": y_test.astype(int),  # 0=down, 1=up
        "actual_trend": np.where(y_test == 1, "up", "down"),
        "p_down": 1.0 - p_test,
        "p_up": p_test,
        "probability": p_test,
        "future_close": meta_test["future_close"].values,
        "forward_log_return": meta_test["forward_log_return"].values,
        "sigma_t": meta_test["sigma_t"].values,
        "sigma_h": meta_test["sigma_h"].values,
        "target_score": meta_test["target_score"].values,
    })

    preds_csv_path_p = Path(preds_csv_path)
    preds_csv_path_p.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(preds_csv_path_p, index=False)
    print(f"✅ Saved: {preds_csv_path_p}")

    metrics = [eval_with_ignore_zone(y_test, p_test, threshold=t) for t in thresholds]
    metrics_df = pd.DataFrame(metrics)

    metrics_csv_path_p = Path(metrics_csv_path)
    metrics_csv_path_p.parent.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(metrics_csv_path_p, index=False)
    print(f"✅ Saved: {metrics_csv_path_p}")

    # -----------------------------
    # 9) Save artifacts
    # -----------------------------
    lstm_artifacts = {
        "model_path": str(model_path_p),
        "scaler_path": str(scaler_path_p),
        "preds_csv_path": str(preds_csv_path_p),
        "metrics_csv_path": str(metrics_csv_path_p),
        "x_window_size": int(x_window_size),
        "feature_cols": feature_cols,
        "symbol": symbol,
        "resolution": resolution,
        "start_date": start_date,
        "end_date": end_date,
        "train_ratio": float(train_ratio),
        "thresholds_eval": thresholds,
        "ignore_zone_threshold_for_sideways": 0.52,
        "target_type": "volatility_adjusted_return",
        "target_definition": (
            "binary target: up if forward_log_return / sigma_h > 0, "
            "down otherwise"
        ),
        "class_mapping": {
            "down": 0,
            "up": 1,
        },
        "horizon": int(horizon),
        "vol_window": int(vol_window),
        "train_samples": int(len(X_train)),
        "test_samples": int(len(X_test)),
        "test_start_timestamp": str(meta_test["timestamp"].iloc[0]) if len(meta_test) > 0 else None,
        "test_end_timestamp": str(meta_test["timestamp"].iloc[-1]) if len(meta_test) > 0 else None,
    }

    artifacts_path_p = Path(artifacts_path)
    artifacts_path_p.parent.mkdir(parents=True, exist_ok=True)
    dump(lstm_artifacts, str(artifacts_path_p))
    print(f"✅ Saved: {artifacts_path_p}")

    return model, history, out_df, metrics_df, scaler
