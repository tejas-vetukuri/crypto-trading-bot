# models/xgboost/xgb_tp_horizon.py

import numpy as np
import pandas as pd
from pathlib import Path

from xgboost import XGBClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
)
from joblib import dump

from data.delta_exchange import DeltaDataClient
from data.feature_engineering import feature_engineering_xgb


def build_tp_horizon_labels(
    df: pd.DataFrame,
    horizon: int = 12,
    tp_pct: float = 0.02,
    sl_pct: float = 0.01,
    side: str = "long",
    skip_ambiguous: bool = True,
):
    if side not in {"long", "short"}:
        raise ValueError(f"side must be 'long' or 'short', got {side}")

    rows = []

    n = len(df)
    if n < horizon + 1:
        raise ValueError(f"Not enough rows for horizon={horizon}. Got n={n}")

    for i in range(n - horizon):
        entry_row = df.iloc[i]
        entry_price = float(entry_row["close"])
        timestamp = entry_row["timestamp"] if "timestamp" in df.columns else i

        if side == "long":
            tp_barrier = entry_price * (1.0 + tp_pct)
            sl_barrier = entry_price * (1.0 - sl_pct)
        else:
            tp_barrier = entry_price * (1.0 - tp_pct)
            sl_barrier = entry_price * (1.0 + sl_pct)

        future_slice = df.iloc[i + 1:i + 1 + horizon]

        label = None
        hit_step = None
        outcome = "unresolved"

        for step, row in enumerate(future_slice.itertuples(index=False), start=1):
            high = float(row.high)
            low = float(row.low)

            if side == "long":
                hit_tp = high >= tp_barrier
                hit_sl = low <= sl_barrier
            else:
                hit_tp = low <= tp_barrier
                hit_sl = high >= sl_barrier

            if hit_tp and hit_sl:
                hit_step = step
                outcome = "ambiguous_same_bar"
                label = None
                break

            if hit_tp:
                label = 1
                hit_step = step
                outcome = "tp_first"
                break

            if hit_sl:
                label = 0
                hit_step = step
                outcome = "sl_first"
                break

        if label is None:
            if skip_ambiguous or outcome != "ambiguous_same_bar":
                continue
            continue

        rows.append(
            {
                "row_idx": i,
                "timestamp": timestamp,
                "entry_close": entry_price,
                "tp_barrier": tp_barrier,
                "sl_barrier": sl_barrier,
                "hit_step": hit_step,
                "label": label,
                "outcome": outcome,
                "side": side,
            }
        )

    return pd.DataFrame(rows)


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


def train_xgb_tp_horizon_model(
    symbol: str = "BTCUSD",
    resolution: str = "1h",
    start_date: str = "2019-06-01",
    end_date: str | None = None,
    train_ratio: float = 0.80,

    horizon: int = 12,
    tp_pct: float = 0.02,
    sl_pct: float = 0.01,
    side: str = "long",
    skip_ambiguous: bool = True,

    scale_pos_weight: float | None = None,

    artifacts_path: str | None = None,
    preds_csv_path: str | None = None,

    thresholds: tuple[float, ...] = (0.20, 0.25, 0.30, 0.35, 0.40, 0.50),
):
    if side not in {"long", "short"}:
        raise ValueError(f"side must be 'long' or 'short', got {side}")

    side_suffix = side

    if artifacts_path is None:
        artifacts_path = f"models/xgboost/xgb_tp_horizon_artifacts_{side_suffix}.joblib"
    if preds_csv_path is None:
        preds_csv_path = f"models/xgboost/xgb_tp_horizon_predictions_{side_suffix}.csv"

    threshold_csv_path = f"models/xgboost/xgb_tp_horizon_threshold_sweep_{side_suffix}.csv"

    client = DeltaDataClient()
    df = client.get_candles(
        symbol=symbol,
        resolution=resolution,
        start_date=start_date,
        end_date=end_date,
    )

    df = df.sort_values("timestamp").reset_index(drop=True)

    required = {"open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = feature_engineering_xgb(df)
    df = df.dropna().reset_index(drop=True)

    label_df = build_tp_horizon_labels(
        df=df,
        horizon=horizon,
        tp_pct=tp_pct,
        sl_pct=sl_pct,
        side=side,
        skip_ambiguous=skip_ambiguous,
    )

    if len(label_df) == 0:
        raise ValueError("No valid labeled samples generated.")

    feature_df = df.iloc[label_df["row_idx"].values].copy().reset_index(drop=True)
    feature_df["label"] = label_df["label"].values
    feature_df["entry_close"] = label_df["entry_close"].values
    feature_df["tp_barrier"] = label_df["tp_barrier"].values
    feature_df["sl_barrier"] = label_df["sl_barrier"].values
    feature_df["hit_step"] = label_df["hit_step"].values
    feature_df["outcome"] = label_df["outcome"].values
    feature_df["side"] = label_df["side"].values

    features = [
        "ema_20", "ema_50", "rsi", "atr",
        "log_ret_1", "ret_1", "ret_3", "ret_5", "ret_10",
        "body", "range", "upper_wick", "lower_wick", "body_pct", "range_pct", "clv",
        "ema_spread", "ema20_dist", "ema50_dist", "ema20_slope_3", "ema50_slope_3",
        "atr_pct", "rsi_delta", "rsi_ma_10", "rsi_dist",
        "volatility_5", "vol_10", "vol_30", "vol_ratio",
        "vol_chg_1", "vol_chg_5", "vol_z20",
    ]

    missing_features = [c for c in features if c not in feature_df.columns]
    if missing_features:
        raise ValueError(f"Missing engineered features: {missing_features}")

    X = feature_df[features].copy()
    y = feature_df["label"].astype(int).values

    n = len(X)
    train_end = int(n * train_ratio)
    if train_end <= 0 or train_end >= n:
        raise ValueError(f"Invalid split. n={n}, train_end={train_end}")

    X_train = X.iloc[:train_end].copy()
    X_test = X.iloc[train_end:].copy()
    y_train = y[:train_end]
    y_test = y[train_end:]
    meta_test = feature_df.iloc[train_end:].reset_index(drop=True)

    print("\n📊 TRAIN label stats")
    print(f"Samples:        {len(y_train)}")
    print(f"Positive rate:  {float(y_train.mean()):.4f}")
    print(f"Negative rate:  {1.0 - float(y_train.mean()):.4f}")
    print(f"Pos count:      {int((y_train == 1).sum())}")
    print(f"Neg count:      {int((y_train == 0).sum())}")

    print("\n📊 TEST label stats")
    print(f"Samples:        {len(y_test)}")
    print(f"Positive rate:  {float(y_test.mean()):.4f}")
    print(f"Negative rate:  {1.0 - float(y_test.mean()):.4f}")
    print(f"Pos count:      {int((y_test == 1).sum())}")
    print(f"Neg count:      {int((y_test == 0).sum())}")

    if scale_pos_weight is None:
        pos = max(int((y_train == 1).sum()), 1)
        neg = max(int((y_train == 0).sum()), 1)
        scale_pos_weight = neg / pos

    print(f"\n⚖️ scale_pos_weight: {scale_pos_weight:.6f}")

    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        tree_method="hist",
        scale_pos_weight=scale_pos_weight,
    )

    model.fit(X_train, y_train)

    p_test = model.predict_proba(X_test)[:, 1]

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

    output_df = pd.DataFrame({
        "timestamp": meta_test["timestamp"].values,
        "y_true": y_test.astype(int),
        "p_tp_first": p_test.astype(float),
        "entry_close": meta_test["entry_close"].values,
        "tp_barrier": meta_test["tp_barrier"].values,
        "sl_barrier": meta_test["sl_barrier"].values,
        "hit_step": meta_test["hit_step"].values,
        "outcome": meta_test["outcome"].values,
        "side": meta_test["side"].values,
    })

    output_df["pred_0_50"] = y_pred_05.astype(int)

    for t in thresholds:
        col = f"pred_{str(t).replace('.', '_')}"
        output_df[col] = (p_test >= t).astype(int)

    Path(preds_csv_path).parent.mkdir(parents=True, exist_ok=True)
    Path(threshold_csv_path).parent.mkdir(parents=True, exist_ok=True)

    output_df.to_csv(preds_csv_path, index=False)
    threshold_df.to_csv(threshold_csv_path, index=False)

    print(f"\n✅ Predictions saved to {preds_csv_path}")
    print(f"✅ Threshold sweep saved to {threshold_csv_path}")

    artifacts = {
        "model": model,
        "features": features,
        "symbol": symbol,
        "resolution": resolution,
        "start_date": start_date,
        "end_date": end_date,
        "train_ratio": float(train_ratio),
        "target_type": "tp_before_sl_within_horizon",
        "side": side,
        "horizon": int(horizon),
        "tp_pct": float(tp_pct),
        "sl_pct": float(sl_pct),
        "skip_ambiguous": bool(skip_ambiguous),
        "scale_pos_weight": float(scale_pos_weight),
        "thresholds_eval": thresholds,
    }

    artifacts_path_p = Path(artifacts_path)
    artifacts_path_p.parent.mkdir(parents=True, exist_ok=True)
    dump(artifacts, str(artifacts_path_p))
    print(f"✅ Model saved to {artifacts_path_p}")

    return model, artifacts, output_df, threshold_df


if __name__ == "__main__":
    train_xgb_tp_horizon_model()