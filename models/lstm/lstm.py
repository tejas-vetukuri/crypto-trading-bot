import numpy as np
import pandas as pd

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

from data.delta_exchange import DeltaDataClient


def add_return_wick_vol_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds:
      - log return (1-step)
      - candle body/range + wicks (scale-free)
      - rolling volatility of log returns

    Uses ONLY current/past info (no leakage), but will create NaNs at the start.
    """
    df = df.copy()

    # Ensure float
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = df[c].astype(float)

    eps = 1e-12

    # 1) Returns
    df["log_ret_1"] = np.log((df["close"] + eps) / (df["close"].shift(1) + eps))

    # 2) Candle geometry (all normalized by open to be scale-free)
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

    # Close location value (optional but very useful for wicks/range context)
    df["clv"] = (2.0 * c - h - l) / ((h - l) + eps)  # in [-1, 1] (roughly)

    # 3) Volatility (rolling std of returns)
    df["vol_10"] = df["log_ret_1"].rolling(10).std()
    df["vol_30"] = df["log_ret_1"].rolling(30).std()

    # Replace inf and keep NaNs (we'll drop them later)
    df = df.replace([np.inf, -np.inf], np.nan)

    return df


def make_windows_next_candle_direction_with_features(
    df: pd.DataFrame,
    x_window_size: int = 100,
    feature_cols=None,
):
    """
    Same as before:
      - X window: last x_window_size candles (ending at i-1)
      - y label: 1 if close[i] > close[i-1] else 0
      - per-window min-max normalization (feature-wise)

    BUT now you can pass expanded feature_cols (returns+wicks+volatility).
    """
    if feature_cols is None:
        feature_cols = [
            "open", "high", "low", "close", "volume",
            "log_ret_1", "body", "range", "upper_wick", "lower_wick", "clv",
            "vol_10", "vol_30",
        ]

    data = df[feature_cols].astype(float).reset_index(drop=True)

    X_list, y_list = [], []

    # i is the "next candle" index whose movement we predict
    for i in range(x_window_size, len(data)):
        x_win = data.iloc[i - x_window_size:i].copy()

        prev_close = float(df.iloc[i - 1]["close"])
        next_close = float(df.iloc[i]["close"])
        y = 1 if next_close > prev_close else 0

        # per-window min-max normalization (feature-wise)
        x_min = x_win.min(axis=0)
        x_max = x_win.max(axis=0)
        denom = (x_max - x_min).replace(0, 1.0)
        x_norm = (x_win - x_min) / denom

        X_list.append(x_norm.values)
        y_list.append(y)

    X = np.asarray(X_list, dtype=np.float32)
    y = np.asarray(y_list, dtype=np.int32)
    return X, y


def eval_with_ignore_zone(y_true: np.ndarray, p: np.ndarray, threshold: float):
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
    resolution: str = "1h",
    start_date: str = "2019-06-01",
    end_date: str = None,
    x_window_size: int = 100,
    epochs: int = 10,
    batch_size: int = 64,
    model_path: str = "lstm_next_direction_gyusu_norm.h5",
):
    """
    Same model + same per-window min-max normalization + same 95/5 split,
    but adds:
      - returns
      - wick/body/range
      - rolling volatility

    Target remains next-candle direction.
    """

    # Fetch data
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

    # ✅ Add engineered features
    df = add_return_wick_vol_features(df)

    # Drop rows where features not available (rolling/std/shift)
    feature_cols = [
        "open", "high", "low", "close", "volume",
        "log_ret_1", "body", "range", "upper_wick", "lower_wick", "clv",
        "vol_10", "vol_30",
    ]
    df = df.dropna(subset=feature_cols).reset_index(drop=True)

    # Windowing + labels
    X, y = make_windows_next_candle_direction_with_features(
        df,
        x_window_size=x_window_size,
        feature_cols=feature_cols
    )

    # 95/5 split
    n = len(X)
    train_end = int(n * 0.95)

    X_train, X_test = X[:train_end], X[train_end:]
    y_train, y_test = y[:train_end], y[train_end:]

    # Class weights
    pos = float(y_train.sum())
    neg = float(len(y_train) - pos)
    class_weight = {0: 1.0, 1: (neg / (pos + 1e-8))}

    # Model (unchanged except input dim now = len(feature_cols))
    model = Sequential([
        LSTM(100, input_shape=(x_window_size, len(feature_cols)), return_sequences=False),
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

    # Test predictions + ignore-zone metrics
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

    # Sanity: label balance
    print("\n📊 Label balance:")
    print(f"Train UP rate: {y_train.mean():.4f}")
    print(f"Test  UP rate: {y_test.mean():.4f}")

    return model, history, out, metrics_df