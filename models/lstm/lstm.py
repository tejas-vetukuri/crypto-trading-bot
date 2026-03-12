# models/lstm/lstm.py

import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

from joblib import dump

from data.delta_exchange import DeltaDataClient
from data.feature_engineering import feature_engineering_lstm
from models.lstm.confidence_threshold import eval_with_ignore_zone


def build_tp_horizon_windows(
    df: pd.DataFrame,
    x_window_size: int,
    feature_cols: list[str],
    horizon: int = 12,
    tp_pct: float = 0.02,
    sl_pct: float = 0.01,
    skip_ambiguous: bool = True,
):
    """
    Build LSTM windows with a barrier-based label.

    Label definition:
      y = 1 -> upper TP barrier hit first within horizon
      y = 0 -> lower SL barrier hit first within horizon

    We skip samples where:
      - neither barrier is hit within horizon
      - both barriers are hit in the same candle (ambiguous intrabar order), if skip_ambiguous=True

    Entry is assumed at the close of the last candle in the input window.
    Future path starts from the next candle.
    """

    X_list = []
    y_list = []
    meta_rows = []

    n = len(df)
    if n < x_window_size + horizon:
        raise ValueError(
            f"Not enough rows for x_window_size={x_window_size} and horizon={horizon}. Got n={n}"
        )

    for end_idx in range(x_window_size, n - horizon + 1):
        # Window uses rows [end_idx - x_window_size, ..., end_idx - 1]
        window = df.iloc[end_idx - x_window_size:end_idx][feature_cols].values.astype(np.float32)

        entry_idx = end_idx - 1
        entry_price = float(df.iloc[entry_idx]["close"])

        upper_barrier = entry_price * (1.0 + tp_pct)
        lower_barrier = entry_price * (1.0 - sl_pct)

        future_slice = df.iloc[end_idx:end_idx + horizon]

        label = None
        hit_step = None

        for step, row in enumerate(future_slice.itertuples(index=False), start=1):
            hit_upper = float(row.high) >= upper_barrier
            hit_lower = float(row.low) <= lower_barrier

            if hit_upper and hit_lower:
                # Same-candle double touch -> intrabar order unknown
                if skip_ambiguous:
                    label = None
                    hit_step = step
                    break
                else:
                    # Conservative fallback: skip anyway
                    label = None
                    hit_step = step
                    break

            if hit_upper:
                label = 1
                hit_step = step
                break

            if hit_lower:
                label = 0
                hit_step = step
                break

        # Skip unresolved samples (no barrier hit within horizon)
        if label is None:
            continue

        X_list.append(window)
        y_list.append(label)
        meta_rows.append(
            {
                "entry_idx": entry_idx,
                "entry_close": entry_price,
                "upper_barrier": upper_barrier,
                "lower_barrier": lower_barrier,
                "hit_step": hit_step,
                "label": label,
            }
        )

    X = np.asarray(X_list, dtype=np.float32)
    y = np.asarray(y_list, dtype=np.int32)
    meta_df = pd.DataFrame(meta_rows)

    return X, y, meta_df


def train_lstm_model(
    symbol: str = "BTCUSD",
    resolution: str = "1h",
    start_date: str = "2019-06-01",
    end_date: str | None = None,
    x_window_size: int = 100,
    epochs: int = 10,
    batch_size: int = 64,

    # New target params
    horizon: int = 12,      # recommended to match RL max_horizon
    tp_pct: float = 0.02,   # example: +2%
    sl_pct: float = 0.01,   # example: -1%
    skip_ambiguous: bool = True,

    # Save paths
    model_path: str = "models/lstm/lstm_tp_horizon_stdscale.keras",
    scaler_path: str = "models/lstm/lstm_tp_horizon_scaler.joblib",
    artifacts_path: str = "models/lstm/lstm_tp_horizon_artifacts.joblib",

    thresholds: tuple[float, ...] = (0.5, 0.55, 0.6),
):
    """
    LSTM with barrier-based target:
      - feature_engineering_lstm(df)
      - train-only StandardScaler
      - label = which barrier is hit first within horizon
      - chronological 95/5 split

    Target:
      y = 1 -> TP/upper barrier hit first within horizon
      y = 0 -> SL/lower barrier hit first within horizon

    This is much more RL-friendly than plain t+1 direction because it learns
    path-dependent trade outcome over the same holding horizon.
    """

    # -----------------------------
    # 1) Fetch candles
    # -----------------------------
    client = DeltaDataClient()
    df = client.get_candles(
        symbol=symbol,
        resolution=resolution,
        start_date=start_date,
        end_date=end_date
    )

    required = {"open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.dropna(subset=list(required)).reset_index(drop=True)

    # -----------------------------
    # 2) Feature engineering
    # -----------------------------
    df = feature_engineering_lstm(df)

    feature_cols = [
        "open", "high", "low", "close", "volume",
        "log_ret_1", "body", "range", "upper_wick", "lower_wick", "clv",
        "vol_10", "vol_30",
    ]
    df = df.dropna(subset=feature_cols).reset_index(drop=True)

    # -----------------------------
    # 3) Windowing + new target
    # -----------------------------
    X, y, meta_df = build_tp_horizon_windows(
        df=df,
        x_window_size=x_window_size,
        feature_cols=feature_cols,
        horizon=horizon,
        tp_pct=tp_pct,
        sl_pct=sl_pct,
        skip_ambiguous=skip_ambiguous,
    )

    if len(X) == 0:
        raise ValueError(
            "No valid samples after TP-horizon labeling. "
            "Try increasing dataset size, increasing horizon, or adjusting tp/sl."
        )

    if X.ndim != 3:
        raise ValueError(f"Expected X with shape (N, T, F), got {X.shape}")
    if X.shape[1] != x_window_size:
        raise ValueError(f"Expected T={x_window_size}, got {X.shape[1]}")
    if X.shape[2] != len(feature_cols):
        raise ValueError(f"Expected F={len(feature_cols)}, got {X.shape[2]}")

    # -----------------------------
    # 4) Chronological split (95/5)
    # -----------------------------
    n = len(X)
    train_end = int(n * 0.95)
    if train_end <= 0 or train_end >= n:
        raise ValueError(f"Not enough labeled samples after windowing. n={n}")

    X_train, X_test = X[:train_end], X[train_end:]
    y_train, y_test = y[:train_end], y[train_end:]
    meta_train, meta_test = meta_df.iloc[:train_end].reset_index(drop=True), meta_df.iloc[train_end:].reset_index(drop=True)

    # -----------------------------
    # 5) Train-only StandardScaler
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
    # 6) Model
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
        X_train_s,
        y_train,
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
    # 7) Test probs + metrics
    # -----------------------------
    p_test = model.predict(X_test_s, batch_size=batch_size).reshape(-1)

    out_df = meta_test.copy()
    out_df["y_true"] = y_test.astype(int)
    out_df["p"] = p_test.astype(float)
    out_df.to_csv("lstm_tp_horizon_test_probs.csv", index=False)

    metrics = [eval_with_ignore_zone(y_test, p_test, threshold=t) for t in thresholds]
    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv("lstm_tp_horizon_metrics.csv", index=False)

    # -----------------------------
    # 8) Save artifacts for RL
    # -----------------------------
    lstm_artifacts = {
        "model_path": str(model_path_p),
        "scaler_path": str(scaler_path_p),
        "x_window_size": int(x_window_size),
        "feature_cols": feature_cols,
        "symbol": symbol,
        "resolution": resolution,
        "start_date": start_date,
        "end_date": end_date,
        "thresholds_eval": thresholds,
        "ignore_zone_threshold_for_sideways": 0.52,

        # New target metadata
        "target_type": "tp_before_sl_within_horizon",
        "horizon": int(horizon),
        "tp_pct": float(tp_pct),
        "sl_pct": float(sl_pct),
        "skip_ambiguous": bool(skip_ambiguous),
    }

    artifacts_path_p = Path(artifacts_path)
    artifacts_path_p.parent.mkdir(parents=True, exist_ok=True)
    dump(lstm_artifacts, str(artifacts_path_p))
    print(f"✅ Saved: {artifacts_path_p}")

    print("✅ Saved: lstm_tp_horizon_test_probs.csv")
    print("✅ Saved: lstm_tp_horizon_metrics.csv")

    print("\n📊 Label balance:")
    print(f"Train positive rate: {float(y_train.mean()):.4f}")
    print(f"Test  positive rate: {float(y_test.mean()):.4f}")
    print(f"Train samples: {len(y_train)}")
    print(f"Test  samples: {len(y_test)}")

    return model, history, out_df, metrics_df, scaler