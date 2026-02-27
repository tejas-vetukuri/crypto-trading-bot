import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

from data.delta_exchange import DeltaDataClient
from data.feature_engineering import feature_engineering_lstm
from models.lstm.sequence_builder import make_windows
from models.lstm.confidence_threshold import eval_with_ignore_zone


def train_lstm_model(
    symbol: str = "BTCUSD",
    resolution: str = "1h",
    start_date: str = "2019-06-01",
    end_date: str | None = None,
    x_window_size: int = 100,
    epochs: int = 10,
    batch_size: int = 64,
    model_path: str = "lstm_next_direction_stdscale.keras",
    thresholds: tuple[float, ...] = (0.5, 0.55, 0.6),
):
    """
    Next-candle direction LSTM with:
      - feature_engineering_lstm(df)
      - RAW windows via make_windows(df, x_window_size, feature_cols)
      - train-only StandardScaler (fit on train windows across all timesteps)
      - LSTM(100) + Dropout(0.2) + Dense(sigmoid)
      - chronological 95/5 split
      - optional ignore-zone metrics at given thresholds

    Returns:
      model, history, out_df (y_true + prob), metrics_df, scaler
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
    # 3) Windowing + label
    # -----------------------------
    X, y = make_windows(df, x_window_size=x_window_size, feature_cols=feature_cols)
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int32)

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
        raise ValueError(f"Not enough samples after windowing. n={n}")

    X_train, X_test = X[:train_end], X[train_end:]
    y_train, y_test = y[:train_end], y[train_end:]

    # -----------------------------
    # 5) Train-only StandardScaler
    # -----------------------------
    n_features = X_train.shape[-1]
    scaler = StandardScaler()

    X_train_2d = X_train.reshape(-1, n_features)
    scaler.fit(X_train_2d)

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
        X_train_s, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.05,
        shuffle=False,
        callbacks=[EarlyStopping(patience=3, restore_best_weights=True)],
        verbose=1
    )

    model.save(model_path)

    # -----------------------------
    # 7) Test probs + metrics
    # -----------------------------
    p_test = model.predict(X_test_s, batch_size=batch_size).reshape(-1)

    out_df = pd.DataFrame({"y_true": y_test.astype(int), "p": p_test.astype(float)})
    out_df.to_csv("lstm_next_direction_stdscale_test_probs.csv", index=False)

    metrics = [eval_with_ignore_zone(y_test, p_test, threshold=t) for t in thresholds]
    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv("lstm_next_direction_stdscale_metrics.csv", index=False)

    print("✅ Saved:", model_path)
    print("✅ Saved: lstm_next_direction_stdscale_test_probs.csv")
    print("✅ Saved: lstm_next_direction_stdscale_metrics.csv")
    print("\n📊 Label balance:")
    print(f"Train UP rate: {float(y_train.mean()):.4f}")
    print(f"Test  UP rate: {float(y_test.mean()):.4f}")

    return model, history, out_df, metrics_df, scaler