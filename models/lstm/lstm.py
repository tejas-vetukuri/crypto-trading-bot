import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping

from data.delta_exchange import DeltaDataClient
from data.feature_engineering import feature_engineering
from models.lstm.sequence_builder import make_sequences


def train_lstm_model(
    symbol: str = "BTCUSD",
    resolution: str = "5m",
    limit: int = 5000,
    sequence_length: int = 20,
    epochs: int = 10,
    batch_size: int = 64,
    model_path: str = "lstm_model.h5"
):
    """
    Train an LSTM model for next-candle direction prediction.
    """

    # -----------------------------
    # Fetch historical data
    # -----------------------------
    client = DeltaDataClient()
    df = client.get_candles(
        symbol=symbol,
        resolution=resolution,
        limit=limit
    )

    # -----------------------------
    # Feature engineering
    # -----------------------------
    df = feature_engineering(df)

    features = [
        "ema_20",
        "ema_50",
        "rsi",
        "atr",
        "momentum_3",
        "volatility_5",
    ]

    X = df[features].values
    y = df["actual_trend"].values  # "up" / "down"

    # -----------------------------
    # Encode labels
    # -----------------------------
    le = LabelEncoder()
    y = le.fit_transform(y)  # down=0, up=1

    # -----------------------------
    # Chronological split
    # -----------------------------
    n = len(X)
    train_end = int(n * 0.7)
    val_end = int(n * 0.85)

    X_train, X_val, X_test = X[:train_end], X[train_end:val_end], X[val_end:]
    y_train, y_val, y_test = y[:train_end], y[train_end:val_end], y[val_end:]

    # -----------------------------
    # Scale features
    # -----------------------------
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # -----------------------------
    # Build sequences
    # -----------------------------
    X_train_seq, y_train_seq = make_sequences(X_train, y_train, sequence_length)
    X_val_seq, y_val_seq = make_sequences(X_val, y_val, sequence_length)
    X_test_seq, y_test_seq = make_sequences(X_test, y_test, sequence_length)

    y_train_cat = to_categorical(y_train_seq, num_classes=2)
    y_val_cat = to_categorical(y_val_seq, num_classes=2)
    y_test_cat = to_categorical(y_test_seq, num_classes=2)

    # -----------------------------
    # LSTM model
    # -----------------------------
    model = Sequential([
        LSTM(
            64,
            input_shape=(sequence_length, X_train_seq.shape[2]),
            return_sequences=False
        ),
        Dropout(0.3),
        Dense(32, activation="relu"),
        Dropout(0.2),
        Dense(2, activation="softmax")
    ])

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    # -----------------------------
    # Train
    # -----------------------------
    history = model.fit(
        X_train_seq,
        y_train_cat,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val_seq, y_val_cat),
        callbacks=[EarlyStopping(patience=3, restore_best_weights=True)],
        verbose=1
    )

    # -----------------------------
    # Save model
    # -----------------------------
    model.save(model_path)

    # -----------------------------
    # Test predictions
    # -----------------------------
    y_pred_probs = model.predict(X_test_seq)
    y_pred = np.argmax(y_pred_probs, axis=1)

    close_prices = df.iloc[
        val_end + sequence_length :
    ]["close"].values

    results_df = pd.DataFrame({
        "actual": y_test_seq,
        "pred": y_pred,
        "close": close_prices
    })

    results_df.to_csv("lstm_test_results.csv", index=False)

    print("✅ LSTM model trained and saved")
    print("✅ Predictions saved to lstm_test_results.csv")

    return model, history, results_df
