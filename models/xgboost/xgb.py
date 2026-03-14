# xgboost.py

import numpy as np
import pandas as pd

from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

from joblib import dump, load

from data.binance import BinanceDataClient
from data.feature_engineering import feature_engineering_xgb


def train_xgb_model(
    symbol: str = "BTCUSDT",
    resolution: str = "1h",
    start_date: str = "2017-09-01",
    end_date: str | None = None,
    train_ratio: float = 0.80,

    # Decision boundary (shift from 0.5 to fix skew)
    decision_boundary: float = 0.46,

    # Ignore zone around boundary
    margin_threshold: float = 0.07,

    # Path to the tuned artifacts that contains best_params
    tuned_artifacts_path: str = "xgb_tuned_artifacts.joblib",

    artifacts_path: str = "xgb_trend_artifacts.joblib",
    preds_csv_path: str = "xgb_predictions.csv",
):
    """
    XGBoost next-direction classifier with:
      - DeltaDataClient fetch
      - Feature engineering
      - Chronological split
      - Binary labels: down=0, up=1 (stable)
      - Decision boundary shift + optional ignore zone
      - Loads tuned hyperparameters from tuned_artifacts_path (best_params)
      - Artifact saving

    Returns:
      model, artifacts, output_df
    """

    # -----------------------------
    # 1) Fetch candles
    # -----------------------------
    client = BinanceDataClient(market="spot")
    df = client.get_candles(
        symbol=symbol,
        resolution=resolution,
        start_date=start_date,
        end_date=end_date,
    )

    df = df.sort_values("timestamp").reset_index(drop=True)

    # -----------------------------
    # 2) Chronological split
    # -----------------------------
    n = len(df)
    train_end = int(n * train_ratio)
    if train_end <= 0 or train_end >= n:
        raise ValueError(f"Invalid split. n={n}, train_end={train_end}")

    train_df = df.iloc[:train_end].copy()
    test_df = df.iloc[train_end:].copy()

    # -----------------------------
    # 3) Feature engineering (separate to avoid leakage)
    # -----------------------------
    train_df = feature_engineering_xgb(train_df)
    test_df = feature_engineering_xgb(test_df)

    features = [
        "ema_20", "ema_50", "rsi", "atr",
        "log_ret_1", "ret_1", "ret_3", "ret_5", "ret_10",
        "body", "range", "upper_wick", "lower_wick", "body_pct", "range_pct", "clv",
        "ema_spread", "ema20_dist", "ema50_dist", "ema20_slope_3", "ema50_slope_3",
        "atr_pct", "rsi_delta", "rsi_ma_10", "rsi_dist",
        "volatility_5", "vol_10", "vol_30", "vol_ratio",
        "vol_chg_1", "vol_chg_5", "vol_z20",
    ]

    X_train = train_df[features]
    y_train = train_df["actual_trend"].astype(str)

    X_test = test_df[features]
    y_test = test_df["actual_trend"].astype(str)

    # -----------------------------
    # 4) Encode labels (stable binary)
    # -----------------------------
    # Stable mapping for XGB: up=1, down=0
    y_train_bin = (y_train.values == "up").astype(int)
    y_test_bin = (y_test.values == "up").astype(int)

    # Keep LabelEncoder just for artifacts / compatibility
    le = LabelEncoder()
    le.fit(y_train)
    if "up" not in le.classes_ or "down" not in le.classes_:
        raise ValueError(f"Expected classes to include 'up' and 'down', got {list(le.classes_)}")

    # -----------------------------
    # 5) Load tuned hyperparameters (best_params)
    # -----------------------------
    best_params = None
    try:
        tuned_artifacts = load(tuned_artifacts_path)
        if isinstance(tuned_artifacts, dict) and "best_params" in tuned_artifacts:
            best_params = tuned_artifacts["best_params"]
            print(f"✅ Loaded tuned best_params from {tuned_artifacts_path}")
        else:
            print(f"⚠️ No 'best_params' found in {tuned_artifacts_path}. Using defaults.")
    except Exception as e:
        print(f"⚠️ Could not load tuned artifacts from {tuned_artifacts_path}: {e}")
        print("⚠️ Using default XGB params.")

    # -----------------------------
    # 6) Train model (binary)
    # -----------------------------
    default_params = dict(
        objective="binary:logistic",
        eval_metric="logloss",
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        tree_method="hist",
    )

    # Merge tuned params over defaults (tuned wins)
    if isinstance(best_params, dict):
        model_params = {**default_params, **best_params}
    else:
        model_params = default_params

    # Safety: force the essentials we rely on
    model_params["objective"] = "binary:logistic"
    model_params["eval_metric"] = "logloss"

    model = XGBClassifier(**model_params)
    model.fit(X_train, y_train_bin)

    # -----------------------------
    # 7) Predict + Decision Boundary (+ optional ignore zone)
    # -----------------------------
    if not (0.0 < decision_boundary < 1.0):
        raise ValueError(f"decision_boundary must be in (0, 1). Got {decision_boundary}")

    if margin_threshold < 0.0:
        raise ValueError(f"margin_threshold must be >= 0. Got {margin_threshold}")

    probs = model.predict_proba(X_test)  # shape (N, 2)
    p_up = probs[:, 1]                   # because we trained with y_train_bin where up=1

    margin = np.abs(p_up - decision_boundary)

    final_preds = np.full(len(p_up), 2, dtype=int)  # sideways default
    confident = margin >= margin_threshold
    final_preds[confident] = (p_up[confident] > decision_boundary).astype(int)  # 1=up, 0=down

    probability_for_pred = np.where(final_preds == 2, decision_boundary, p_up)

    # -----------------------------
    # 8) Save predictions
    # -----------------------------
    output_df = pd.DataFrame({
        "timestamp": test_df["timestamp"].values,
        "prediction": final_preds,             # 0/1 or 2(sideways)
        "actual_trend": y_test.values,         # "down"/"up"
        "p_up": p_up,
        "decision_boundary": float(decision_boundary),
        "margin": margin,
        "used": confident.astype(int),
        "probability": probability_for_pred,
    })

    output_df.to_csv(preds_csv_path, index=False)
    print(f"✅ Predictions saved to {preds_csv_path}")

    # -----------------------------
    # 9) Up/Down-only evaluation (USED trades)
    # -----------------------------
    mask = final_preds != 2
    filtered_preds = final_preds[mask]   # 0/1
    filtered_true = y_test_bin[mask]     # 0/1

    if len(filtered_preds) > 0:
        print("\n📊 Classification Report (Up & Down only):")
        print(classification_report(filtered_true, filtered_preds, target_names=["down", "up"]))

        print("\n🔢 Confusion Matrix:")
        print(confusion_matrix(filtered_true, filtered_preds))
    else:
        print("\n⚠️ No predictions passed the margin threshold.")

    sideways_count = int((final_preds == 2).sum())
    sideways_pct = (sideways_count / len(final_preds)) * 100
    print(f"\n➡️ Sideways count: {sideways_count} ({sideways_pct:.2f}%)")
    print(f"➡️ Decision boundary: {decision_boundary}")
    print(f"➡️ Margin threshold: {margin_threshold}")

    print("\n📊 Label balance:")
    print(f"Train UP rate: {float(y_train_bin.mean()):.4f}")
    print(f"Test  UP rate: {float(y_test_bin.mean()):.4f}")

    # -----------------------------
    # 10) Save artifacts (includes best_params used)
    # -----------------------------
    artifacts = {
        "model": model,
        "label_encoder": le,
        "features": features,
        "decision_boundary": float(decision_boundary),
        "margin_threshold": float(margin_threshold),
        "symbol": symbol,
        "resolution": resolution,
        "start_date": start_date,
        "end_date": end_date,
        "ignore_zone": f"sideways if |p_up-{decision_boundary}| < {margin_threshold}",
        "best_params_used": model_params,
        "tuned_artifacts_path": tuned_artifacts_path,
    }

    dump(artifacts, artifacts_path)
    print(f"✅ Model saved to {artifacts_path}")

    return model, artifacts, output_df


if __name__ == "__main__":
    a = load("xgb_tuned_artifacts.joblib")
    print(a.keys())
    print(a["best_params"])

    train_xgb_model()