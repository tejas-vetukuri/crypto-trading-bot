# models/lstm/eval_lstm.py

import numpy as np
import pandas as pd
from pathlib import Path

from joblib import load
from tensorflow.keras.models import load_model

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
)

from data.delta_exchange import DeltaDataClient
from data.feature_engineering import feature_engineering_lstm
from models.lstm.confidence_threshold import eval_with_ignore_zone
from models.lstm.lstm import build_tp_horizon_windows


def print_probability_summary(p: np.ndarray):
    s = pd.Series(p)
    print("\n📈 Probability summary")
    print(s.describe())

    bins = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 1.0]
    bucket_counts = pd.cut(s, bins=bins, include_lowest=True).value_counts().sort_index()
    print("\n📦 Probability buckets")
    print(bucket_counts)


def evaluate_thresholds(
    y_true: np.ndarray,
    p_test: np.ndarray,
    thresholds: list[float] | tuple[float, ...],
):
    rows = []

    print("\n================ THRESHOLD SWEEP ================")
    for t in thresholds:
        y_pred = (p_test >= t).astype(int)

        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        pred_pos = int((y_pred == 1).sum())
        pred_neg = int((y_pred == 0).sum())

        row = {
            "threshold": float(t),
            "accuracy": float(acc),
            "precision": float(prec),
            "recall": float(rec),
            "f1": float(f1),
            "pred_pos": pred_pos,
            "pred_neg": pred_neg,
        }
        rows.append(row)

        print(f"\nThreshold = {t:.2f}")
        print(f"Predicted positives: {pred_pos}")
        print(f"Predicted negatives: {pred_neg}")
        print(f"Accuracy:            {acc:.4f}")
        print(f"Precision:           {prec:.4f}")
        print(f"Recall:              {rec:.4f}")
        print(f"F1:                  {f1:.4f}")
        print("Confusion matrix:")
        print(confusion_matrix(y_true, y_pred))

    return pd.DataFrame(rows)


def evaluate_lstm_tp_horizon(
    symbol: str | None = None,
    resolution: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    artifacts_path: str = "models/lstm/lstm_tp_horizon_artifacts_long.joblib",
    thresholds: tuple[float, ...] = (0.20, 0.25, 0.30, 0.35, 0.40, 0.50),
    batch_size: int = 64,
    out_probs_path: str = "models/lstm/lstm_tp_horizon_eval_test_probs_long.csv",
    out_thresholds_path: str = "models/lstm/lstm_tp_horizon_eval_threshold_sweep_long.csv",
    out_ignore_zone_path: str = "models/lstm/lstm_tp_horizon_eval_ignore_zone_metrics_long.csv",
):
    artifacts_path_p = Path(artifacts_path)
    if not artifacts_path_p.exists():
        raise FileNotFoundError(f"Artifacts not found: {artifacts_path_p}")

    artifacts = load(str(artifacts_path_p))

    model_path = artifacts["model_path"]
    scaler_path = artifacts["scaler_path"]
    x_window_size = int(artifacts["x_window_size"])
    feature_cols = list(artifacts["feature_cols"])

    target_type = artifacts.get("target_type", "unknown")
    side = artifacts.get("side", "long")
    horizon = int(artifacts["horizon"])
    tp_pct = float(artifacts["tp_pct"])
    sl_pct = float(artifacts["sl_pct"])
    skip_ambiguous = bool(artifacts.get("skip_ambiguous", True))
    train_ratio = float(artifacts.get("train_ratio", 0.80))

    symbol = symbol or artifacts["symbol"]
    resolution = resolution or artifacts["resolution"]
    start_date = start_date or artifacts["start_date"]
    end_date = end_date if end_date is not None else artifacts["end_date"]

    print("\n================ LSTM TP-HORIZON EVALUATION ================")
    print(f"Target type:      {target_type}")
    print(f"Side:             {side}")
    print(f"Symbol:           {symbol}")
    print(f"Resolution:       {resolution}")
    print(f"Start date:       {start_date}")
    print(f"End date:         {end_date}")
    print(f"Window size:      {x_window_size}")
    print(f"Horizon:          {horizon}")
    print(f"TP %:             {tp_pct:.4f}")
    print(f"SL %:             {sl_pct:.4f}")
    print(f"Train ratio:      {train_ratio:.2f}")
    print(f"Skip ambiguous:   {skip_ambiguous}")
    print(f"Model path:       {model_path}")
    print(f"Scaler path:      {scaler_path}")

    model = load_model(model_path)
    scaler = load(scaler_path)

    client = DeltaDataClient()
    df = client.get_candles(
        symbol=symbol,
        resolution=resolution,
        start_date=start_date,
        end_date=end_date,
    )

    required = {"open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.dropna(subset=list(required)).reset_index(drop=True)
    df = feature_engineering_lstm(df)
    df = df.dropna(subset=feature_cols).reset_index(drop=True)

    X, y, meta_df = build_tp_horizon_windows(
        df=df,
        x_window_size=x_window_size,
        feature_cols=feature_cols,
        horizon=horizon,
        tp_pct=tp_pct,
        sl_pct=sl_pct,
        side=side,
        skip_ambiguous=skip_ambiguous,
    )

    if len(X) == 0:
        raise ValueError("No labeled samples generated for evaluation.")

    n = len(X)
    train_end = int(n * train_ratio)
    if train_end <= 0 or train_end >= n:
        raise ValueError(f"Not enough samples after labeling. n={n}, train_end={train_end}")

    X_test = X[train_end:]
    y_test = y[train_end:]
    meta_test = meta_df.iloc[train_end:].reset_index(drop=True)

    print("\n📊 Test set")
    print(f"Samples:          {len(y_test)}")
    print(f"Positive rate:    {float(y_test.mean()):.4f}")
    print(f"Negative rate:    {1.0 - float(y_test.mean()):.4f}")

    n_features = X_test.shape[-1]
    X_test_s = scaler.transform(X_test.reshape(-1, n_features)).reshape(X_test.shape).astype(np.float32)

    p_test = model.predict(X_test_s, batch_size=batch_size, verbose=0).reshape(-1)

    print_probability_summary(p_test)

    y_pred_05 = (p_test >= 0.50).astype(int)

    acc = accuracy_score(y_test, y_pred_05)
    try:
        auc = roc_auc_score(y_test, p_test)
    except Exception:
        auc = float("nan")

    print("\n---------------- Basic Metrics @0.50 ----------------")
    print(f"Accuracy:      {acc:.4f}")
    print(f"ROC-AUC:       {auc:.4f}")
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred_05))

    print("\nClassification report @0.50")
    print(classification_report(y_test, y_pred_05, digits=4, zero_division=0))

    threshold_df = evaluate_thresholds(y_test, p_test, thresholds)

    ignore_zone_rows = []
    print("\n================ IGNORE-ZONE METRICS ================")
    for t in thresholds:
        row = eval_with_ignore_zone(y_test, p_test, threshold=t)
        ignore_zone_rows.append(row)

        print(f"\nThreshold = {t}")
        for k, v in row.items():
            print(f"{k}: {v}")

    ignore_zone_df = pd.DataFrame(ignore_zone_rows)

    out_df = meta_test.copy()
    out_df["y_true"] = y_test.astype(int)
    out_df["p"] = p_test.astype(float)
    out_df["pred_0_50"] = y_pred_05.astype(int)

    for t in thresholds:
        col = f"pred_{str(t).replace('.', '_')}"
        out_df[col] = (p_test >= t).astype(int)

    Path(out_probs_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_thresholds_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_ignore_zone_path).parent.mkdir(parents=True, exist_ok=True)

    out_df.to_csv(out_probs_path, index=False)
    threshold_df.to_csv(out_thresholds_path, index=False)
    ignore_zone_df.to_csv(out_ignore_zone_path, index=False)

    print(f"\n✅ Saved: {out_probs_path}")
    print(f"✅ Saved: {out_thresholds_path}")
    print(f"✅ Saved: {out_ignore_zone_path}")

    return out_df, threshold_df, ignore_zone_df


if __name__ == "__main__":
    evaluate_lstm_tp_horizon()