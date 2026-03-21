from __future__ import annotations

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
)

from models.baselines.simple_rnn import (
    train_simple_rnn_model,
    get_default_start_date,
    combo_tag,
    build_simple_rnn_save_paths,
)


def apply_ignore_zone(probs: np.ndarray, threshold: float):
    """
    Returns:
      pred: array with {0,1} for decided samples, -1 for ignored (HOLD)
      used_mask: boolean mask where pred != -1
    """
    probs = np.asarray(probs, dtype=float).reshape(-1)
    pred = np.full_like(probs, fill_value=-1, dtype=np.int32)

    pred[probs >= threshold] = 1
    pred[probs < 1.0 - threshold] = 0

    used_mask = pred != -1
    return pred, used_mask


def print_sample_distribution(y_true: np.ndarray, pred: np.ndarray, used_mask: np.ndarray):
    print("\n📊 ================= SAMPLE DISTRIBUTION =================")

    actual = pd.Series(y_true).value_counts(normalize=True).sort_index()
    print("\n🔹 Actual Class Distribution (Full Test):")
    print(actual.rename(index={0: "DOWN(0)", 1: "UP(1)"}))

    coverage = float(used_mask.mean())
    print(f"\n🔹 Coverage (non-ignored): {coverage:.3f}  |  Ignored: {(1 - coverage):.3f}")

    if used_mask.sum() == 0:
        print("\n🔹 Actual Class Distribution (Used Only): (none)")
        print("🔹 Predicted Class Distribution (Used Only): (none)")
        return

    actual_used = pd.Series(y_true[used_mask]).value_counts(normalize=True).sort_index()
    print("\n🔹 Actual Class Distribution (Used Only):")
    print(actual_used.rename(index={0: "DOWN(0)", 1: "UP(1)"}))

    pred_used = pd.Series(pred[used_mask]).value_counts(normalize=True).sort_index()
    print("\n🔹 Predicted Class Distribution (Used Only):")
    print(pred_used.rename(index={0: "DOWN(0)", 1: "UP(1)"}))


def evaluate_threshold(y_true: np.ndarray, probs: np.ndarray, threshold: float):
    pred, used_mask = apply_ignore_zone(probs, threshold=threshold)

    print("\n📊 ================= THRESHOLD EVALUATION =================")
    print(f"🎚️ Threshold: {threshold}")

    coverage = float(used_mask.mean())
    print(f"✅ Coverage (used samples): {coverage:.3f}  |  Ignored: {(1 - coverage):.3f}")

    if used_mask.sum() == 0:
        print("⚠️ No samples used at this threshold (everything ignored).")
        print_sample_distribution(y_true, pred, used_mask)
        return coverage

    y_true_used = y_true[used_mask]
    y_pred_used = pred[used_mask]

    acc = accuracy_score(y_true_used, y_pred_used)
    print(f"\n✅ Accuracy (used only): {acc:.4f}")

    cm = confusion_matrix(y_true_used, y_pred_used, labels=[0, 1])
    print("\n🔢 Confusion Matrix (used only):")
    print(cm)

    tn, fp, fn, tp = cm.ravel()
    print(f"""
True Negatives  (Correct DOWN): {tn}
False Positives (DOWN → UP):    {fp}
False Negatives (UP → DOWN):    {fn}
True Positives  (Correct UP):   {tp}
""")

    print("\n📄 Classification Report (used only):")
    print(classification_report(
        y_true_used,
        y_pred_used,
        target_names=["DOWN", "UP"],
        zero_division=0,
    ))

    print_sample_distribution(y_true, pred, used_mask)
    return coverage


def evaluate_auc(y_true: np.ndarray, probs: np.ndarray):
    print("\n📈 ================= PROBABILITY METRIC =================")
    try:
        auc = roc_auc_score(y_true, probs)
        print(f"📈 ROC-AUC (full test, using probs): {auc:.4f}")
    except Exception:
        print("⚠️ ROC-AUC could not be computed (likely only one class present).")


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
    pred, used_mask = apply_ignore_zone(probs, threshold=threshold)

    coverage = float(used_mask.mean())
    used_samples = int(used_mask.sum())
    total_samples = int(len(y_true))

    result = {
        "threshold": float(threshold),
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
        "pred_used": None,
    }

    if used_samples == 0:
        return result

    y_true_used = y_true[used_mask]
    y_pred_used = pred[used_mask]

    used_accuracy = float((y_true_used == y_pred_used).mean())
    cm = confusion_matrix(y_true_used, y_pred_used, labels=[0, 1])

    report_df = pd.DataFrame(
        classification_report(
            y_true_used,
            y_pred_used,
            target_names=["DOWN", "UP"],
            zero_division=0,
            output_dict=True,
        )
    ).transpose()

    result["used_accuracy"] = used_accuracy
    result["random_baseline_accuracy"] = random_baseline_with_coverage(
        y_true, coverage=coverage, seed=42
    )
    result["majority_baseline_accuracy"] = majority_baseline_with_coverage(
        y_true, coverage=coverage
    )
    result["confusion_matrix"] = cm
    result["classification_report_df"] = report_df
    result["pred_used"] = y_pred_used

    return result


def build_distribution_tables(y_true: np.ndarray, used_mask: np.ndarray, pred_used):
    actual_full = (
        pd.Series(y_true)
        .value_counts(normalize=True)
        .sort_index()
        .rename(index={0: "DOWN(0)", 1: "UP(1)"})
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
            .rename(index={0: "DOWN(0)", 1: "UP(1)"})
            .reset_index()
        )
        actual_used.columns = ["Class", "Share"]

        pred_used_df = (
            pd.Series(pred_used)
            .value_counts(normalize=True)
            .sort_index()
            .rename(index={0: "DOWN(0)", 1: "UP(1)"})
            .reset_index()
        )
        pred_used_df.columns = ["Class", "Share"]

    return actual_full, actual_used, pred_used_df


def main():
    symbol = "BTCUSDT"
    resolution = "1h"

    start_date = get_default_start_date(resolution)
    tag = combo_tag(symbol, resolution)
    save_paths = build_simple_rnn_save_paths(symbol, resolution)

    thresholds = (0.50, 0.51, 0.52, 0.53, 0.54, 0.55, 0.56, 0.57, 0.58)

    print("🚀 Starting Simple RNN Baseline Evaluation...\n")
    print(f"Symbol:      {symbol}")
    print(f"Resolution:  {resolution}")
    print(f"Start date:  {start_date}")
    print(f"Combo tag:   {tag}")
    print(f"Model path:  {save_paths['model_path']}")
    print()

    model, history, probs_df, metrics_df, scaler = train_simple_rnn_model(
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
        rnn_units=100,
        dropout_rate=0.20,
        early_stopping_patience=3,
        model_path=save_paths["model_path"],
        scaler_path=save_paths["scaler_path"],
        artifacts_path=save_paths["artifacts_path"],
        test_probs_path=save_paths["test_probs_path"],
        metrics_path=save_paths["metrics_path"],
        thresholds=thresholds,
    )

    print("\n================ TRAINING SUMMARY ================")
    if "accuracy" in history.history:
        print(f"Final Training Accuracy:   {history.history['accuracy'][-1]:.4f}")
    if "val_accuracy" in history.history:
        print(f"Final Validation Accuracy: {history.history['val_accuracy'][-1]:.4f}")

    y_true = probs_df["y_true"].values.astype(int)
    probs = probs_df["p"].values.astype(float)

    evaluate_auc(y_true, probs)

    print("\n📊 ================= THRESHOLDS + BASELINES =================")
    for t in thresholds:
        print("\n" + "=" * 55)

        coverage = evaluate_threshold(y_true, probs, threshold=t)

        rand_acc = random_baseline_with_coverage(y_true, coverage=coverage, seed=42)
        maj_acc = majority_baseline_with_coverage(y_true, coverage=coverage)

        print("\n📌 Baselines (matched coverage):")
        print("Random Baseline Accuracy:  ", "nan" if np.isnan(rand_acc) else f"{rand_acc:.4f}")
        print("Majority Baseline Accuracy:", "nan" if np.isnan(maj_acc) else f"{maj_acc:.4f}")

    print("\n📊 Test probs preview:")
    print(probs_df.head())
    print(f"\nTotal test samples: {len(probs_df)}")

    print("\n✅ Full evaluation completed successfully")
    print(f"✅ Saved combination files for: {tag}")


if __name__ == "__main__":
    main()