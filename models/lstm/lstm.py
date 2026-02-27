import numpy as np
import pandas as pd

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

from sklearn.preprocessing import StandardScaler

from data.delta_exchange import DeltaDataClient
from data.feature_engineering import add_return_wick_vol_features


def make_windows_next_candle_direction_with_features(
    df: pd.DataFrame,
    x_window_size: int = 100,
    feature_cols=None,
):
    """
    Builds raw (UNSCALED) windows + next-candle direction labels.

    Scaling is intentionally NOT done here.
    We'll do train-only StandardScaler after splitting.
    """
    if feature_cols is None:
        feature_cols = [
            "open", "high", "low", "close", "volume",
            "log_ret_1", "body", "range", "upper_wick", "lower_wick", "clv",
            "vol_10", "vol_30",
        ]

    data = df[feature_cols].astype(float).reset_index(drop=True)

    X_list, y_list = [], []
    for i in range(x_window_size, len(data)):
        # Raw window (no min-max)
        x_win = data.iloc[i - x_window_size:i].values  # (x_window_size, n_features)

        prev_close = float(df.iloc[i - 1]["close"])
        next_close = float(df.iloc[i]["close"])
        y = 1 if next_close > prev_close else 0

        X_list.append(x_win)
        y_list.append(y)

    X = np.asarray(X_list, dtype=np.float32)  # (N, x_window_size, F)
    y = np.asarray(y_list, dtype=np.int32)    # (N,)
    return X, y


def standard_scale_train_only(X_train: np.ndarray, X_test: np.ndarray):
    """
    Fit StandardScaler on TRAIN ONLY, feature-wise across all timesteps.

    We reshape (N, T, F) -> (N*T, F) to fit/transform.
    """
    n_features = X_train.shape[-1]
    scaler = StandardScaler()

    X_train_2d = X_train.reshape(-1, n_features)
    scaler.fit(X_train_2d)

    def transform(X):
        X2d = X.reshape(-1, n_features)
        X2d = scaler.transform(X2d)
        return X2d.reshape(X.shape).astype(np.float32)

    return transform(X_train), transform(X_test), scaler


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


def train_lstm_model_next_direction_stdscale(
    symbol: str = "BTCUSD",
    resolution: str = "1h",
    start_date: str = "2019-06-01",
    end_date: str = None,
    x_window_size: int = 100,
    epochs: int = 10,
    batch_size: int = 64,
    model_path: str = "lstm_next_direction_stdscale.keras",
):
    """
    Same target + same model architecture as before,
    BUT replaces per-window min-max with:
      ✅ Train-only StandardScaler applied feature-wise across all timesteps.

    This preserves absolute scale/regime info better (often improves separability).
    """
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

    # Feature engineering (returns+wicks+vol)
    df = add_return_wick_vol_features(df)

    feature_cols = [
        "open", "high", "low", "close", "volume",
        "log_ret_1", "body", "range", "upper_wick", "lower_wick", "clv",
        "vol_10", "vol_30",
    ]
    df = df.dropna(subset=feature_cols).reset_index(drop=True)

    # Build RAW windows + labels
    X, y = make_windows_next_candle_direction_with_features(
        df,
        x_window_size=x_window_size,
        feature_cols=feature_cols
    )

    # 95/5 chrono split
    n = len(X)
    train_end = int(n * 0.95)

    X_train, X_test = X[:train_end], X[train_end:]
    y_train, y_test = y[:train_end], y[train_end:]

    # ✅ Train-only scaling
    X_train_s, X_test_s, scaler = standard_scale_train_only(X_train, X_test)

    # Model (architecture unchanged; use Input to remove Keras warning)
    model = Sequential([
        Input(shape=(x_window_size, len(feature_cols))),
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

    model.save(model_path)

    # Test predictions + ignore-zone metrics
    p_test = model.predict(X_test_s, batch_size=batch_size).reshape(-1)

    metrics = []
    for t in [0.5, 0.55, 0.6]:
        m = eval_with_ignore_zone(y_test, p_test, threshold=t)
        metrics.append(m)
        print(m)

    out = pd.DataFrame({"y_true": y_test, "p": p_test})
    out.to_csv("lstm_next_direction_stdscale_test_probs.csv", index=False)

    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv("lstm_next_direction_stdscale_metrics.csv", index=False)

    print("✅ Saved:", model_path)
    print("✅ Saved: lstm_next_direction_stdscale_test_probs.csv")
    print("✅ Saved: lstm_next_direction_stdscale_metrics.csv")

    print("\n📊 Label balance:")
    print(f"Train UP rate: {y_train.mean():.4f}")
    print(f"Test  UP rate: {y_test.mean():.4f}")

    return model, history, out, metrics_df, scaler