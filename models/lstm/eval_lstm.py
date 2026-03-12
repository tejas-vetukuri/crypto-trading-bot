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
)

from data.delta_exchange import DeltaDataClient
from data.feature_engineering import feature_engineering_lstm
from models.lstm.confidence_threshold import eval_with_ignore_zone


def build_tp_horizon_windows(
    df: pd.DataFrame,
    x_window_size: int,
    feature_cols: list[str],
    horizon: int = 12,
    tp_pct: float = 0.02,
    sl_pct: float = 0.01,
    skip_ambiguous: bool = True,
):
    """
    Build LSTM windows with barrier-based labels.

    Label definition:
      y = 1 -> upper TP barrier hit first within horizon
      y = 0 -> lower SL barrier hit first within horizon

    Samples are skipped when:
      - neither barrier is hit within horizon
      - both barriers are hit in the same candle and order is unknown
    """

    X_list = []
    y_list = []
    meta_rows = []

    n = len(df)
    if n < x_window_size + horizon:
        raise ValueError(
            f"Not enough rows for x_window_size={x_window_size} and horizon={horizon}. Got n={n}"
        )

    for end_idx in range(x_window_size, n - horizon + 1):
        window = df.iloc[end_idx - x_window_size:end_idx][feature_cols].values.astype(np.float32)

        entry_idx = end_idx - 1
        entry_price = float(df.iloc[entry_idx]["close"])

        upper_barrier = entry_price * (1.0 + tp_pct)
        lower_barrier = entry_price * (1.0 - sl_pct)

        future_slice = df.iloc[end_idx:end_idx + horizon]

        label = None
        hit_step = None
        outcome = "unresolved"

        for step, row in enumerate(future_slice.itertuples(index=False), start=1):
            hit_upper = float(row.high) >= upper_barrier
            hit_lower = float(row.low) <= lower_barrier

            if hit_upper and hit_lower:
                if skip_ambiguous:
                    label = None
                    hit_step = step
                    outcome = "ambiguous_same_bar"
                    break
                label = None
                hit_step = step
                outcome = "ambiguous_same_bar"
                break

            if hit_upper:
                label = 1
                hit_step = step
                outcome = "tp_first"
                break

            if hit_lower:
                label = 0
                hit_step = step
                outcome = "sl_first"
                break

        if label is None:
            continue

        X_list.append(window)
        y_list.append(label)
        meta_rows.append(
            {
                "entry_idx": entry_idx,
                "entry_close": entry_price,
                "upper_barrier": upper_barrier,
                "lower_barrier": lower_barrier,
                "hit_step": hit_step,
                "label": label,
                "outcome": outcome,
            }
        )

    X = np.asarray(X_list, dtype=np.float32)
    y = np.asarray(y_list, dtype=np.int32)
    meta_df = pd.DataFrame(meta_rows)

    return X, y, meta_df


def evaluate_lstm_tp_horizon(
    symbol: str | None = None,
    resolution: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    artifacts_path: str = "models/lstm/lstm_tp_horizon_artifacts.joblib",
    thresholds: tuple[float, ...] = (0.50, 0.55, 0.60),
    batch_size: int = 64,
):
    """
    Evaluate saved TP-horizon LSTM model using the artifact config.

    Uses the same:
      - feature engineering
      - barrier labeling
      - chronological 95/5 split
      - scaler
    as training.
    """

    # -----------------------------
    # 1) Load artifacts
    # -----------------------------
    artifacts_path_p = Path(artifacts_path)
    if not artifacts_path_p.exists():
        raise FileNotFoundError(f"Artifacts not found: {artifacts_path_p}")

    artifacts = load(str(artifacts_path_p))

    model_path = artifacts["model_path"]
    scaler_path = artifacts["scaler_path"]
    x_window_size = int(artifacts["x_window_size"])
    feature_cols = list(artifacts["feature_cols"])

    target_type = artifacts.get("target_type", "unknown")
    horizon = int(artifacts["horizon"])
    tp_pct = float(artifacts["tp_pct"])
    sl_pct = float(artifacts["sl_pct"])
    skip_ambiguous = bool(artifacts.get("skip_ambiguous", True))

    symbol = symbol or artifacts["symbol"]
    resolution = resolution or artifacts["resolution"]
    start_date = start_date or artifacts["start_date"]
    end_date = end_date if end_date is not None else artifacts["end_date"]

    print("\n================ LSTM TP-HORIZON EVALUATION ================")
    print(f"Target type:      {target_type}")
    print(f"Symbol:           {symbol}")
    print(f"Resolution:       {resolution}")
    print(f"Start date:       {start_date}")
    print(f"End date:         {end_date}")
    print(f"Window size:      {x_window_size}")
    print(f"Horizon:          {horizon}")
    print(f"TP %:             {tp_pct:.4f}")
    print(f"SL %:             {sl_pct:.4f}")
    print(f"Skip ambiguous:   {skip_ambiguous}")
    print(f"Model path:       {model_path}")
    print(f"Scaler path:      {scaler_path}")

    # -----------------------------
    # 2) Load model + scaler
    # -----------------------------
    model = load_model(model_path)
    scaler = load(scaler_path)

    # -----------------------------
    # 3) Fetch candles
    # -----------------------------
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

    # -----------------------------
    # 4) Feature engineering
    # -----------------------------
    df = feature_engineering_lstm(df)
    df = df.dropna(subset=feature_cols).reset_index(drop=True)

    # -----------------------------
    # 5) Build windows + labels
    # -----------------------------
    X, y, meta_df = build_tp_horizon_windows(
        df=df,
        x_window_size=x_window_size,
        feature_cols=feature_cols,
        horizon=horizon,
        tp_pct=tp_pct,
        sl_pct=sl_pct,
        skip_ambiguous=skip_ambiguous,
    )

    if len(X) == 0:
        raise ValueError("No labeled samples generated for evaluation.")

    # -----------------------------
    # 6) Chronological 95/5 split
    # -----------------------------
    n = len(X)
    train_end = int(n * 0.95)
    if train_end <= 0 or train_end >= n:
        raise ValueError(f"Not enough samples after labeling. n={n}")

    X_test = X[train_end:]
    y_test = y[train_end:]
    meta_test = meta_df.iloc[train_end:].reset_index(drop=True)

    # -----------------------------
    # 7) Scale test windows
    # -----------------------------
    n_features = X_test.shape[-1]
    X_test_s = scaler.transform(X_test.reshape(-1, n_features)).reshape(X_test.shape).astype(np.float32)

    # -----------------------------
    # 8) Predict
    # -----------------------------
    p_test = model.predict(X_test_s, batch_size=batch_size, verbose=0).reshape(-1)
    y_pred_05 = (p_test >= 0.5).astype(int)

    # -----------------------------
    # 9) Standard metrics
    # -----------------------------
    acc = accuracy_score(y_test, y_pred_05)
    cm = confusion_matrix(y_test, y_pred_05)

    try:
        auc = roc_auc_score(y_test, p_test)
    except Exception:
        auc = float("nan")

    print("\n---------------- Basic Metrics ----------------")
    print(f"Test samples:             {len(y_test)}")
    print(f"Positive rate (test):     {float(y_test.mean()):.4f}")
    print(f"Accuracy @0.50:           {acc:.4f}")
    print(f"ROC-AUC:                  {auc:.4f}")

    print("\n---------------- Confusion Matrix @0.50 ----------------")
    print(cm)

    print("\n---------------- Classification Report @0.50 ----------------")
    print(classification_report(y_test, y_pred_05, digits=4))

    # -----------------------------
    # 10) Ignore-zone metrics
    # -----------------------------
    ignore_zone_rows = []
    print("\n---------------- Ignore-Zone Metrics ----------------")
    for t in thresholds:
        row = eval_with_ignore_zone(y_test, p_test, threshold=t)
        ignore_zone_rows.append(row)
        print(f"\nThreshold = {t}")
        for k, v in row.items():
            print(f"{k}: {v}")

    metrics_df = pd.DataFrame(ignore_zone_rows)

    # -----------------------------
    # 11) Save outputs
    # -----------------------------
    out_df = meta_test.copy()
    out_df["y_true"] = y_test.astype(int)
    out_df["p"] = p_test.astype(float)
    out_df["pred_0_50"] = y_pred_05.astype(int)

    out_probs_path = "lstm_tp_horizon_eval_test_probs.csv"
    out_metrics_path = "lstm_tp_horizon_eval_metrics.csv"

    out_df.to_csv(out_probs_path, index=False)
    metrics_df.to_csv(out_metrics_path, index=False)

    print(f"\n✅ Saved: {out_probs_path}")
    print(f"✅ Saved: {out_metrics_path}")

    return out_df, metrics_df


if __name__ == "__main__":
    evaluate_lstm_tp_horizon()