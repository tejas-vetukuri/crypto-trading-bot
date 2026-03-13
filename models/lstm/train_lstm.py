from models.lstm.lstm import train_lstm_model


if __name__ == "__main__":
    SIDE = "short"   # change to "short" when needed

    train_lstm_model(
        symbol="BTCUSD",
        resolution="1h",
        start_date="2019-06-01",
        end_date=None,
        x_window_size=100,
        epochs=10,
        batch_size=64,
        train_ratio=0.80,
        horizon=12,
        tp_pct=0.02,
        sl_pct=0.01,
        side=SIDE,
        skip_ambiguous=True,
        model_path=f"models/lstm/lstm_tp_horizon_stdscale_{SIDE}.keras",
        scaler_path=f"models/lstm/lstm_tp_horizon_scaler_{SIDE}.joblib",
        artifacts_path=f"models/lstm/lstm_tp_horizon_artifacts_{SIDE}.joblib",
        thresholds=(0.20, 0.25, 0.30, 0.35, 0.40, 0.50),
    )