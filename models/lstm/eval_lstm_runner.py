from models.lstm.eval_lstm import evaluate_lstm_tp_horizon


if __name__ == "__main__":
    # change to "short" for short-side eval
    SIDE = "short"

    evaluate_lstm_tp_horizon(
        artifacts_path=f"models/lstm/lstm_tp_horizon_artifacts_{SIDE}.joblib",
        thresholds=(0.20, 0.25, 0.30, 0.35, 0.40, 0.50),
        batch_size=64,
        out_probs_path=f"models/lstm/lstm_tp_horizon_eval_test_probs_{SIDE}.csv",
        out_thresholds_path=f"models/lstm/lstm_tp_horizon_eval_threshold_sweep_{SIDE}.csv",
        out_ignore_zone_path=f"models/lstm/lstm_tp_horizon_eval_ignore_zone_metrics_{SIDE}.csv",
    )