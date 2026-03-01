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
    resolution: str = "1m",
    start_date: str = "2023-06-01",
    end_date: str | None = None,
    train_ratio: float = 0.80,
    threshold: float = 0.50,
    artifacts_path: str = "xgb_trend_artifacts.joblib",
    preds_csv_path: str = "xgb_predictions.csv",
):
    """
    XGBoost next-direction classifier with:

      - DeltaDataClient fetch
      - Feature engineering
      - Chronological split (95/5 default)
      - Train-only LabelEncoder
      - Ignore-zone threshold
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

    features = ["ema_20", "ema_50", "rsi", "atr", "momentum_3", "volatility_5"]

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

    # -----------------------------
    # 5) Train model
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
    # 6) Predict + Ignore Zone
    # -----------------------------
    probs = model.predict_proba(X_test)
    max_probs = np.max(probs, axis=1)
    predicted_classes = np.argmax(probs, axis=1)

    final_preds = np.where(max_probs >= threshold, predicted_classes, 2)

    # -----------------------------
    # 7) Save predictions
    # -----------------------------
    output_df = pd.DataFrame({
        "timestamp": test_df["timestamp"].values,
        "prediction": final_preds,
        "actual_trend": y_test.values,
        "probability": max_probs
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
        "threshold": threshold,
        "symbol": symbol,
        "resolution": resolution,
        "start_date": start_date,
        "end_date": end_date,
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
        print("\n⚠️ No predictions passed the threshold.")

    sideways_count = int((final_preds == 2).sum())
    sideways_pct = (sideways_count / len(final_preds)) * 100
    print(f"\n➡️ Sideways count: {sideways_count} ({sideways_pct:.2f}%)")

    print("\n📊 Label balance:")
    print(f"Train UP rate: {(y_train_enc.mean()):.4f}")
    print(f"Test  UP rate: {(y_test_enc.mean()):.4f}")

    return model, artifacts, output_df


if __name__ == "__main__":
    train_xgb_model()