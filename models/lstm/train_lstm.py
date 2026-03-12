# scripts/train_lstm.py

from models.lstm.lstm import train_lstm_model


if __name__ == "__main__":
    model, history, out_df, metrics_df, scaler = train_lstm_model(
        symbol="BTCUSD",
        resolution="1h",
        start_date="2019-06-01",
        end_date=None,

        x_window_size=100,
        epochs=10,
        batch_size=64,

        # TP-horizon target settings
        horizon=12,
        tp_pct=0.02,
        sl_pct=0.01,
        skip_ambiguous=True,

        # save paths
        model_path="models/lstm/lstm_tp_horizon_stdscale.keras",
        scaler_path="models/lstm/lstm_tp_horizon_scaler.joblib",
        artifacts_path="models/lstm/lstm_tp_horizon_artifacts.joblib",

        # eval thresholds
        thresholds=(0.50, 0.55, 0.60),
    )

    print("\n✅ Training complete.")
    print(f"Train/Test output rows saved: {len(out_df)}")
    print("\nMetrics preview:")
    print(metrics_df)