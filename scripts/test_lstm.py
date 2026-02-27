import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
)

# ✅ UPDATED import: new lstm.py trainer (next-candle direction, gyusu-style normalization)
# Change the import name if you used a slightly different function name.
from models.lstm.lstm import train_lstm_model_next_direction_gyusu_norm


# -----------------------------
# Helper: apply ignore-zone thresholding
# -----------------------------
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
    """
    Distribution for true labels in full set, and predictions in used set.
    """
    print("\n📊 ================= SAMPLE DISTRIBUTION =================")

    # Actual distribution (full test set)
    actual = pd.Series(y_true).value_counts(normalize=True).sort_index()
    print("\n🔹 Actual Class Distribution (Full Test):")
    print(actual.rename(index={0: "DOWN(0)", 1: "UP(1)"}))

    # Used/ignored
    coverage = float(used_mask.mean())
    print(f"\n🔹 Coverage (non-ignored): {coverage:.3f}  |  Ignored: {(1-coverage):.3f}")

    if used_mask.sum() == 0:
        print("\n🔹 Actual Class Distribution (Used Only): (none)")
        print("🔹 Predicted Class Distribution (Used Only): (none)")
        return

    # Actual distribution among used samples only
    actual_used = pd.Series(y_true[used_mask]).value_counts(normalize=True).sort_index()
    print("\n🔹 Actual Class Distribution (Used Only):")
    print(actual_used.rename(index={0: "DOWN(0)", 1: "UP(1)"}))

    # Pred distribution among used samples
    pred_used = pd.Series(pred[used_mask]).value_counts(normalize=True).sort_index()
    print("\n🔹 Predicted Class Distribution (Used Only):")
    print(pred_used.rename(index={0: "DOWN(0)", 1: "UP(1)"}))


def evaluate_threshold(y_true: np.ndarray, probs: np.ndarray, threshold: float):
    """
    Evaluates performance using ignore-zone thresholding.
    Metrics are computed ONLY on used samples.
    """
    pred, used_mask = apply_ignore_zone(probs, threshold=threshold)

    print("\n📊 ================= THRESHOLD EVALUATION =================")
    print(f"🎚️ Threshold: {threshold}")

    coverage = float(used_mask.mean())
    print(f"✅ Coverage (used samples): {coverage:.3f}  |  Ignored: {(1-coverage):.3f}")

    if used_mask.sum() == 0:
        print("⚠️ No samples used at this threshold (everything ignored).")
        print_sample_distribution(y_true, pred, used_mask)
        return coverage

    y_true_used = y_true[used_mask]
    y_pred_used = pred[used_mask]

    # Accuracy
    acc = accuracy_score(y_true_used, y_pred_used)
    print(f"\n✅ Accuracy (used only): {acc:.4f}")

    # Confusion matrix
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

    # Classification report
    print("\n📄 Classification Report (used only):")
    print(classification_report(
        y_true_used, y_pred_used,
        target_names=["DOWN", "UP"],
        zero_division=0
    ))

    # Distribution
    print_sample_distribution(y_true, pred, used_mask)

    return coverage


def evaluate_auc(y_true: np.ndarray, probs: np.ndarray):
    """
    ROC-AUC on raw probabilities over the full test set (no ignore-zone).
    """
    print("\n📈 ================= PROBABILITY METRIC =================")
    try:
        auc = roc_auc_score(y_true, probs)
        print(f"📈 ROC-AUC (full test, using probs): {auc:.4f}")
    except Exception:
        print("⚠️ ROC-AUC could not be computed (likely only one class present).")


# -----------------------------
# Baselines with matched coverage
# -----------------------------
def random_baseline_with_coverage(y_true: np.ndarray, coverage: float, seed: int = 42):
    """
    Random predictions on a random subset of size coverage*N, ignore the rest.
    """
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
    """
    Majority class prediction on a subset of size coverage*N, ignore the rest.
    Takes first used_n chronologically to mimic time-ordered usage.
    """
    n = len(y_true)
    used_n = int(round(float(coverage) * n))
    if used_n <= 0:
        return np.nan

    used_mask = np.zeros(n, dtype=bool)
    used_mask[:used_n] = True

    majority_class = int(pd.Series(y_true).value_counts().idxmax())
    pred_used = np.full(used_n, majority_class, dtype=np.int32)

    return accuracy_score(y_true[used_mask], pred_used)


def main():
    print("🚀 Starting Next-Direction (Gyusu-Norm) LSTM Evaluation...\n")

    # ✅ UPDATED: call the new trainer
    model, history, probs_df, metrics_df = train_lstm_model_next_direction_gyusu_norm(
        symbol="BTCUSD",
        resolution="1h",            # keep what you had; 1m is harder but more standard
        start_date="2019-06-01",
        end_date=None,
        x_window_size=100,
        epochs=10,
        batch_size=64,
        model_path="lstm_next_direction_gyusu_norm.h5"
    )

    print("\n================ TRAINING SUMMARY ================")
    if "accuracy" in history.history:
        print(f"Final Training Accuracy:   {history.history['accuracy'][-1]:.4f}")
    if "val_accuracy" in history.history:
        print(f"Final Validation Accuracy: {history.history['val_accuracy'][-1]:.4f}")

    # Extract arrays
    y_true = probs_df["y_true"].values.astype(int)
    probs = probs_df["p"].values.astype(float)

    # AUC on probs
    evaluate_auc(y_true, probs)

    # Threshold evaluations + baselines with matched coverage
    print("\n📊 ================= THRESHOLDS + BASELINES =================")
    for t in [0.5, 0.51, 0.52, 0.53, 0.54]:
        print("\n" + "=" * 55)

        coverage = evaluate_threshold(y_true, probs, threshold=t)

        rand_acc = random_baseline_with_coverage(y_true, coverage=coverage, seed=42)
        maj_acc = majority_baseline_with_coverage(y_true, coverage=coverage)

        print("\n📌 Baselines (matched coverage):")
        if np.isnan(rand_acc):
            print("Random Baseline Accuracy:   nan")
        else:
            print(f"Random Baseline Accuracy:   {rand_acc:.4f}")

        if np.isnan(maj_acc):
            print("Majority Baseline Accuracy: nan")
        else:
            print(f"Majority Baseline Accuracy: {maj_acc:.4f}")

    print("\n📊 Test probs preview:")
    print(probs_df.head())
    print(f"\nTotal test samples: {len(probs_df)}")

    print("\n✅ Full evaluation completed successfully")


if __name__ == "__main__":
    main()