import numpy as np
import pandas as pd

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

from data.delta_exchange import DeltaDataClient


def make_windows_next_candle_direction(
    df: pd.DataFrame,
    x_window_size: int = 100,
):
    """
    Same windowing + per-window min-max normalization as gyusu-style,
    BUT target is next-candle direction:

      - X window: last x_window_size candles (ending at i-1)
      - y label: 1 if close[i] > close[i-1] else 0   (next candle direction)
      - normalization: min-max per X window (feature-wise)
    """
    cols = ["open", "high", "low", "close", "volume"]
    data = df[cols].astype(float).reset_index(drop=True)

    X_list, y_list = [], []

    # i is the "next candle" index whose movement we want to predict
    # X uses candles [i-x_window_size, ..., i-1]
    # y compares close[i] vs close[i-1]
    for i in range(x_window_size, len(data)):
        x_win = data.iloc[i - x_window_size:i].copy()

        prev_close = float(data.iloc[i - 1]["close"])
        next_close = float(data.iloc[i]["close"])
        y = 1 if next_close > prev_close else 0

        # per-window min-max normalization (feature-wise)
        x_min = x_win.min(axis=0)
        x_max = x_win.max(axis=0)
        denom = (x_max - x_min).replace(0, 1.0)
        x_norm = (x_win - x_min) / denom

        X_list.append(x_norm.values)
        y_list.append(y)

    X = np.asarray(X_list, dtype=np.float32)   # (N, x_window_size, 5)
    y = np.asarray(y_list, dtype=np.int32)     # (N,)
    return X, y


def eval_with_ignore_zone(y_true: np.ndarray, p: np.ndarray, threshold: float):
    """
    Confidence gating:
      predict 1 if p >= threshold
      predict 0 if p < 1 - threshold
      else ignored

    Returns accuracy/precision/recall/f1 computed ONLY on non-ignored samples.
    """
    y_true = y_true.reshape(-1)
    p = p.reshape(-1)

    TP = FP = TN = FN = ignored = 0

    for yt, prob in zip(y_true, p):
        if prob >= threshold:
            pred = 1
        elif prob < 1.0 - threshold:
            pred = 0
        else:
            ignored += 1
            continue

        if yt == 1 and pred == 1:
            TP += 1
        elif yt == 0 and pred == 1:
            FP += 1
        elif yt == 0 and pred == 0:
            TN += 1
        elif yt == 1 and pred == 0:
            FN += 1

    total = TP + FP + TN + FN
    eps = 1e-8
    acc = (TP + TN) / (total + eps)
    prec = TP / (TP + FP + eps)
    rec = TP / (TP + FN + eps)
    f1 = 2 * prec * rec / (prec + rec + eps)

    coverage = total / (total + ignored + eps)

    return {
        "threshold": threshold,
        "coverage": float(coverage),
        "ignored": int(ignored),
        "used_samples": int(total),
        "TP": int(TP), "FP": int(FP), "TN": int(TN), "FN": int(FN),
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
    }


def train_lstm_model_next_direction_gyusu_norm(
    symbol: str = "BTCUSD",
    resolution: str = "1m",
    start_date: str = "2024-01-01",
    end_date: str = None,
    x_window_size: int = 100,
    epochs: int = 10,
    batch_size: int = 64,
    model_path: str = "lstm_next_direction_gyusu_norm.h5",
):
    """
    Keeps the rest of the gyusu-style pipeline the same:
    - Uses ONLY OHLCV
    - Per-window min-max normalization (feature-wise)
    - Same model: LSTM(100) + Dropout(0.2) + Dense(1,sigmoid)
    - Same class-weighted BCE logic
    - Same ignore-zone metrics at thresholds [0.5, 0.55, 0.6]

    ONLY change:
    - Target is next-candle direction (close[i] > close[i-1])
    """

    # -----------------------------
    # Fetch historical data
    # -----------------------------
    client = DeltaDataClient()
    df = client.get_candles(
        symbol=symbol,
        resolution=resolution,
        start_date=start_date,
        end_date=end_date
    )

    # Ensure required columns exist
    required = {"open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.dropna(subset=["open", "high", "low", "close", "volume"]).reset_index(drop=True)

    # -----------------------------
    # Windowing + labels (UPDATED TARGET)
    # -----------------------------
    X, y = make_windows_next_candle_direction(
        df,
        x_window_size=x_window_size,
    )

    # -----------------------------
    # Chronological split (95/5 like their config)
    # -----------------------------
    n = len(X)
    train_end = int(n * 0.95)

    X_train, X_test = X[:train_end], X[train_end:]
    y_train, y_test = y[:train_end], y[train_end:]

    # -----------------------------
    # Class weights (imbalance handling)
    # -----------------------------
    pos = float(y_train.sum())
    neg = float(len(y_train) - pos)
    class_weight = {0: 1.0, 1: (neg / (pos + 1e-8))}

    # -----------------------------
    # Model: SAME as before
    # -----------------------------
    model = Sequential([
        LSTM(100, input_shape=(x_window_size, 5), return_sequences=False),
        Dropout(0.2),
        Dense(1, activation="sigmoid"),
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.05,
        class_weight=class_weight,
        shuffle=False,
        callbacks=[EarlyStopping(patience=3, restore_best_weights=True)],
        verbose=1
    )

    model.save(model_path)

    # -----------------------------
    # Test predictions + ignore-zone metrics
    # -----------------------------
    p_test = model.predict(X_test, batch_size=batch_size).reshape(-1)

    metrics = []
    for t in [0.5, 0.55, 0.6]:
        m = eval_with_ignore_zone(y_test, p_test, threshold=t)
        metrics.append(m)
        print(m)

    out = pd.DataFrame({"y_true": y_test, "p": p_test})
    out.to_csv("lstm_next_direction_gyusu_norm_test_probs.csv", index=False)

    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv("lstm_next_direction_gyusu_norm_metrics.csv", index=False)

    print("✅ Saved:", model_path)
    print("✅ Saved: lstm_next_direction_gyusu_norm_test_probs.csv")
    print("✅ Saved: lstm_next_direction_gyusu_norm_metrics.csv")

    # Quick sanity check: class balance
    print("\n📊 Label balance:")
    print(f"Train UP rate: {y_train.mean():.4f}")
    print(f"Test  UP rate: {y_test.mean():.4f}")

    return model, history, out, metrics_df