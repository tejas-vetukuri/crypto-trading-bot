# xgboost.py

import numpy as np
import pandas as pd

from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

from joblib import dump

from data.delta_exchange import DeltaDataClient
from data.feature_engineering import feature_engineering_xgb


def train_xgb_model(
    symbol: str = "BTCUSD",
    resolution: str = "1h",
    start_date: str = "2019-06-01",
    end_date: str | None = None,
    train_ratio: float = 0.80,

    # NEW: shift the UP/DOWN decision boundary away from 0.5
    # predict UP if p_up > decision_boundary else DOWN (when not sideways)
    decision_boundary: float = 0.46,

    # OPTIONAL: keep ignore zone, but now centered around decision_boundary
    # keep only if |p_up - decision_boundary| >= margin_threshold
    margin_threshold: float = 0.05,

    artifacts_path: str = "xgb_trend_artifacts.joblib",
    preds_csv_path: str = "xgb_predictions.csv",
):
    """
    XGBoost next-direction classifier with:

      - DeltaDataClient fetch
      - Feature engineering
      - Chronological split
      - Train-only LabelEncoder
      - Decision boundary shift (default 0.50)
      - Optional margin-based ignore zone around the boundary
      - Artifact saving

    Returns:
      model, artifacts, output_df
    """

    # -----------------------------
    # 1) Fetch candles
    # -----------------------------
    client = DeltaDataClient()
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
    # 4) Encode labels (train only)
    # -----------------------------
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc = le.transform(y_test)

    if "up" not in le.classes_ or "down" not in le.classes_:
        raise ValueError(f"Expected classes to include 'up' and 'down', got {list(le.classes_)}")

    up_class_idx = int(np.where(le.classes_ == "up")[0][0])

    # -----------------------------
    # 5) Train model (binary)
    # -----------------------------
    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
    )

    model.fit(X_train, y_train_enc)

    # -----------------------------
    # 6) Predict + Decision Boundary (+ optional ignore zone)
    # -----------------------------
    if not (0.0 < decision_boundary < 1.0):
        raise ValueError(f"decision_boundary must be in (0, 1). Got {decision_boundary}")

    if margin_threshold < 0.0:
        raise ValueError(f"margin_threshold must be >= 0. Got {margin_threshold}")

    probs = model.predict_proba(X_test)  # shape (N, 2)
    p_up = probs[:, up_class_idx]        # P(up)

    # confidence distance from boundary
    margin = np.abs(p_up - decision_boundary)

    # default: sideways (2)
    final_preds = np.full(len(p_up), 2, dtype=int)

    confident = margin >= margin_threshold

    # If confident: predict up if p_up > decision_boundary else down
    final_preds[confident] = (p_up[confident] > decision_boundary).astype(int)

    probability_for_pred = np.where(final_preds == 2, decision_boundary, p_up)

    # -----------------------------
    # 7) Save predictions
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
    # 8) Save artifacts
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
    }

    dump(artifacts, artifacts_path)
    print(f"✅ Model saved to {artifacts_path}")

    # -----------------------------
    # 9) Up/Down-only evaluation
    # -----------------------------
    mask = final_preds != 2
    filtered_preds = final_preds[mask]
    filtered_true = y_test_enc[mask]

    if len(filtered_preds) > 0:
        print("\n📊 Classification Report (Up & Down only):")
        print(classification_report(filtered_true, filtered_preds, target_names=le.classes_))

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
    train_up_rate = float((y_train.values == "up").mean())
    test_up_rate = float((y_test.values == "up").mean())
    print(f"Train UP rate: {train_up_rate:.4f}")
    print(f"Test  UP rate: {test_up_rate:.4f}")

    return model, artifacts, output_df


if __name__ == "__main__":
    train_xgb_model()