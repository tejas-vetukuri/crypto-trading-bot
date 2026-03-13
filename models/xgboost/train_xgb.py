from models.xgboost.xgb import train_xgb_tp_horizon_model


if __name__ == "__main__":
    # change to "short" for short-side training
    SIDE = "short"

    train_xgb_tp_horizon_model(
        symbol="BTCUSD",
        resolution="1h",
        start_date="2019-06-01",
        end_date=None,
        train_ratio=0.80,
        horizon=12,
        tp_pct=0.02,
        sl_pct=0.01,
        side=SIDE,
        skip_ambiguous=True,
        artifacts_path=f"models/xgboost/xgb_tp_horizon_artifacts_{SIDE}.joblib",
        preds_csv_path=f"models/xgboost/xgb_tp_horizon_predictions_{SIDE}.csv",
        thresholds=(0.20, 0.25, 0.30, 0.35, 0.40, 0.50),
    )