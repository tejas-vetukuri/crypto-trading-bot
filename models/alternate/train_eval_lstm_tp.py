from __future__ import annotations

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
)

from models.alternate.lstm_tp import (
    train_lstm_tp_model,
    get_default_start_date,
    build_lstm_tp_save_paths,
)


def apply_ignore_zone(probs: np.ndarray, threshold: float):
    probs = np.asarray(probs, dtype=float).reshape(-1)
    pred = np.full_like(probs, fill_value=-1, dtype=np.int32)

    pred[probs >= threshold] = 1
    pred[probs < 1.0 - threshold] = 0

    used_mask = pred != -1
    return pred, used_mask


def _safe_roc_auc(y_true: np.ndarray, probs: np.ndarray):
    try:
        y_true = np.asarray(y_true).astype(int)
        probs = np.asarray(probs).astype(float)
        if len(np.unique(y_true)) < 2:
            return None
        return float(roc_auc_score(y_true, probs))
    except Exception:
        return None


def random_baseline_with_coverage(y_true: np.ndarray, coverage: float, seed: int = 42):
    rng = np.random.default_rng(seed)
    n = len(y_true)
    used_n = int(round(float(coverage) * n))
    if used_n <= 0:
        return np.nan

    used_idx = rng.choice(n, size=used_n, replace=False)
    pred = np.full(n, -1, dtype=np.int32)
    pred[used_idx] = rng.integers(0, 2, size=used_n)

    used_mask = pred != -1
    return accuracy_score(y_true[used_mask], pred[used_mask]) if used_mask.sum() > 0 else np.nan


def majority_baseline_with_coverage(y_true: np.ndarray, coverage: float):
    n = len(y_true)
    used_n = int(round(float(coverage) * n))
    if used_n <= 0:
        return np.nan

    used_mask = np.zeros(n, dtype=bool)
    used_mask[:used_n] = True

    majority_class = int(pd.Series(y_true).value_counts().idxmax())
    pred_used = np.full(used_n, majority_class, dtype=np.int32)
    return accuracy_score(y_true[used_mask], pred_used)


def build_threshold_summary(y_true: np.ndarray, probs: np.ndarray, threshold: float) -> dict:
    y_true = np.asarray(y_true).astype(int)
    probs = np.asarray(probs).astype(float)

    pred_raw, used_mask = apply_ignore_zone(probs, threshold=threshold)

    pred_all = np.full(len(probs), 2, dtype=int)
    pred_all[used_mask] = pred_raw[used_mask]

    coverage = float(used_mask.mean()) if len(used_mask) > 0 else 0.0
    used_samples = int(used_mask.sum())
    total_samples = int(len(y_true))

    result = {
        "threshold_type": "probability",
        "threshold": float(threshold),
        "roc_auc": _safe_roc_auc(y_true, probs),
        "coverage": coverage,
        "ignored_rate": 1.0 - coverage,
        "used_samples": used_samples,
        "total_samples": total_samples,
        "used_accuracy": None,
        "random_baseline_accuracy": None,
        "majority_baseline_accuracy": None,
        "confusion_matrix": None,
        "classification_report_df": None,
        "used_mask": used_mask,
        "pred_used": np.array([], dtype=int),
        "pred_all": pred_all,
    }

    if used_samples == 0:
        return result

    y_true_used = y_true[used_mask]
    y_pred_used = pred_all[used_mask]

    report_df = pd.DataFrame(
        classification_report(
            y_true_used,
            y_pred_used,
            target_names=["SL_FIRST", "TP_FIRST"],
            zero_division=0,
            output_dict=True,
        )
    ).transpose().reset_index().rename(columns={"index": "class"})

    result["used_accuracy"] = float((y_true_used == y_pred_used).mean())
    result["random_baseline_accuracy"] = random_baseline_with_coverage(
        y_true, coverage=coverage, seed=42
    )
    result["majority_baseline_accuracy"] = majority_baseline_with_coverage(
        y_true, coverage=coverage
    )
    result["confusion_matrix"] = confusion_matrix(y_true_used, y_pred_used, labels=[0, 1])
    result["classification_report_df"] = report_df
    result["pred_used"] = y_pred_used

    return result


def build_distribution_tables(y_true: np.ndarray, used_mask: np.ndarray, pred_used):
    actual_full = (
        pd.Series(y_true)
        .value_counts(normalize=True)
        .sort_index()
        .rename(index={0: "SL_FIRST(0)", 1: "TP_FIRST(1)"})
        .reset_index()
    )
    actual_full.columns = ["Class", "Share"]

    if used_mask.sum() == 0:
        actual_used = pd.DataFrame(columns=["Class", "Share"])
        pred_used_df = pd.DataFrame(columns=["Class", "Share"])
    else:
        actual_used = (
            pd.Series(y_true[used_mask])
            .value_counts(normalize=True)
            .sort_index()
            .rename(index={0: "SL_FIRST(0)", 1: "TP_FIRST(1)"})
            .reset_index()
        )
        actual_used.columns = ["Class", "Share"]

        pred_used_df = (
            pd.Series(pred_used)
            .value_counts(normalize=True)
            .sort_index()
            .rename(index={0: "SL_FIRST(0)", 1: "TP_FIRST(1)"})
            .reset_index()
        )
        pred_used_df.columns = ["Class", "Share"]

    return actual_full, actual_used, pred_used_df


def build_lstm_tp_threshold_sweep(
    y_true: np.ndarray,
    probs: np.ndarray,
    thresholds: tuple[float, ...] | list[float],
) -> pd.DataFrame:
    rows = []

    for threshold in thresholds:
        summary = build_threshold_summary(
            y_true=y_true,
            probs=probs,
            threshold=float(threshold),
        )

        if summary["used_samples"] > 0:
            y_true_used = y_true[summary["used_mask"]]
            y_pred_used = summary["pred_used"]

            precision = precision_score(y_true_used, y_pred_used, average="macro", zero_division=0)
            recall = recall_score(y_true_used, y_pred_used, average="macro", zero_division=0)
            f1 = f1_score(y_true_used, y_pred_used, average="macro", zero_division=0)
        else:
            precision = None
            recall = None
            f1 = None

        rows.append({
            "threshold": round(float(threshold), 4),
            "accuracy": summary["used_accuracy"],
            "precision_macro": precision,
            "recall_macro": recall,
            "f1_macro": f1,
            "coverage": summary["coverage"],
        })

    return pd.DataFrame(rows)


def main():
    symbol = "BTCUSDT"
    resolution = "1h"
    side = "long"

    start_date = get_default_start_date(resolution)
    save_paths = build_lstm_tp_save_paths(symbol, resolution, side=side)

    chosen_threshold = 0.30
    thresholds = (0.20, 0.25, 0.30, 0.35, 0.40, 0.50)

    RUN_TRAIN = True

    print("🚀 Starting LSTM TP-Before-SL Evaluation...\n")
    print(f"Symbol:      {symbol}")
    print(f"Resolution:  {resolution}")
    print(f"Side:        {side}")
    print(f"Start date:  {start_date}")
    print(f"Model path:  {save_paths['model_path']}")
    print(f"Artifacts:   {save_paths['artifacts_path']}")
    print(f"Test probs:  {save_paths['test_probs_path']}")
    print()

    if RUN_TRAIN:
        model, history, probs_df, metrics_df, scaler = train_lstm_tp_model(
            symbol=symbol,
            resolution=resolution,
            start_date=start_date,
            end_date=None,
            x_window_size=100,
            epochs=10,
            batch_size=64,
            train_ratio=0.80,
            validation_split=0.05,
            learning_rate=0.001,
            lstm_units=100,
            dropout_rate=0.20,
            early_stopping_patience=3,
            horizon=12,
            tp_pct=0.02,
            sl_pct=0.01,
            side=side,
            skip_ambiguous=True,
            model_path=save_paths["model_path"],
            scaler_path=save_paths["scaler_path"],
            artifacts_path=save_paths["artifacts_path"],
            test_probs_path=save_paths["test_probs_path"],
            metrics_path=save_paths["metrics_path"],
            thresholds=thresholds,
        )
    else:
        print("📂 Skipping training. Loading saved test probabilities...")
        probs_df = pd.read_csv(save_paths["test_probs_path"])
        history = None

    y_true = probs_df["y_true"].values.astype(int)
    probs = probs_df["p"].values.astype(float)

    auc = _safe_roc_auc(y_true, probs)

    print("\n📈 ================= PROBABILITY METRIC =================")
    if auc is None:
        print("⚠️ ROC-AUC could not be computed.")
    else:
        print(f"📈 ROC-AUC (full test, using probs): {auc:.4f}")

    raw_pred = (probs >= 0.5).astype(int)
    raw_acc = accuracy_score(y_true, raw_pred)
    raw_cm = confusion_matrix(y_true, raw_pred, labels=[0, 1])

    print("\n📊 ================= RAW RESULTS =================")
    print("Threshold:         0.50")
    print(f"Accuracy:          {raw_acc:.4f}")

    print("\n🔢 Confusion Matrix:")
    print(raw_cm)

    print("\n📄 Classification Report:")
    print(classification_report(
        y_true,
        raw_pred,
        labels=[0, 1],
        target_names=["SL_FIRST", "TP_FIRST"],
        zero_division=0,
    ))

    summary = build_threshold_summary(
        y_true=y_true,
        probs=probs,
        threshold=chosen_threshold,
    )

    print("\n📊 ================= CHOSEN CONFIGURATION =================")
    print(f"Threshold:         {chosen_threshold}")
    print(f"Coverage:          {summary['coverage']:.4f}")
    print(f"Ignored rate:      {summary['ignored_rate']:.4f}")
    print(f"Used samples:      {summary['used_samples']} / {summary['total_samples']}")
    print(
        f"Used accuracy:     {summary['used_accuracy']:.4f}"
        if summary["used_accuracy"] is not None
        else "Used accuracy:     None"
    )

    if summary["confusion_matrix"] is not None:
        print("\n🔢 Confusion Matrix (used only):")
        print(summary["confusion_matrix"])

    if summary["classification_report_df"] is not None:
        print("\n📄 Classification Report (used only):")
        print(summary["classification_report_df"])

    actual_full, actual_used, pred_used_df = build_distribution_tables(
        y_true=y_true,
        used_mask=summary["used_mask"],
        pred_used=summary["pred_used"],
    )

    print("\n📊 Actual distribution (full test):")
    print(actual_full)

    print("\n📊 Actual distribution (used only):")
    print(actual_used)

    print("\n📊 Predicted distribution (used only):")
    print(pred_used_df)

    print("\n📊 ================= ACCURACY ACROSS THRESHOLDS =================")
    sweep_df = build_lstm_tp_threshold_sweep(
        y_true=y_true,
        probs=probs,
        thresholds=thresholds,
    )
    print(sweep_df)

    print("\n📊 Test probs preview:")
    print(probs_df.head())
    print(f"\nTotal test samples: {len(probs_df)}")

    print("\n✅ Full LSTM TP-before-SL evaluation completed successfully")


if __name__ == "__main__":
    main()