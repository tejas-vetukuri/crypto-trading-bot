import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
)

from models.lstm.lstm import train_lstm_model

from models.baselines.random_baseline import RandomBaseline
from models.baselines.majority_baseline import MajorityBaseline


def print_sample_distribution(results_df):
    """
    Print class distribution statistics.
    """

    print("\n📊 ================= SAMPLE DISTRIBUTION =================")

    # Overall test distribution (actual)
    print("\n🔹 Actual Class Distribution (Test Set):")
    actual_dist = results_df["actual"].value_counts(normalize=True)
    print(actual_dist)

    # Predicted distribution
    print("\n🔹 Predicted Class Distribution:")
    pred_dist = results_df["pred"].value_counts(normalize=True)
    print(pred_dist)

    # Raw counts
    print("\n🔹 Raw Counts (Actual):")
    print(results_df["actual"].value_counts())

    print("\n🔹 Raw Counts (Predicted):")
    print(results_df["pred"].value_counts())


def evaluate_results(results_df):
    """
    Perform full evaluation of LSTM test predictions.
    """

    y_true = results_df["actual"].values
    y_pred = results_df["pred"].values

    print("\n📊 ================= TEST METRICS =================")

    # -----------------------------
    # Accuracy
    # -----------------------------
    acc = accuracy_score(y_true, y_pred)
    print(f"\n✅ Test Accuracy: {acc:.4f}")

    # -----------------------------
    # Confusion Matrix
    # -----------------------------
    cm = confusion_matrix(y_true, y_pred)

    print("\n🔢 Confusion Matrix:")
    print(cm)

    tn, fp, fn, tp = cm.ravel()
    print(f"""
True Negatives  (Correct DOWN): {tn}
False Positives (DOWN → UP):    {fp}
False Negatives (UP → DOWN):    {fn}
True Positives  (Correct UP):   {tp}
""")

    # -----------------------------
    # Classification Report
    # -----------------------------
    print("\n📄 Classification Report:")
    print(classification_report(y_true, y_pred, target_names=["DOWN", "UP"]))

    # -----------------------------
    # ROC-AUC
    # -----------------------------
    try:
        auc = roc_auc_score(y_true, y_pred)
        print(f"📈 ROC-AUC Score: {auc:.4f}")
    except Exception:
        print("⚠️ ROC-AUC could not be computed.")

    # -----------------------------
    # Prediction distribution
    # -----------------------------
    print_sample_distribution(results_df)


def main():
    print("🚀 Starting Full LSTM Evaluation...\n")

    model, history, results_df = train_lstm_model(
        symbol="BTCUSD",
        resolution="4h",
        start_date="2019-01-01",
        end_date=None,
        sequence_length=20,
        epochs=10,
        batch_size=64,
        model_path="lstm_model_test.h5"
    )

    print("\n================ TRAINING SUMMARY ================")
    print(f"Final Training Accuracy:   {history.history['accuracy'][-1]:.4f}")
    print(f"Final Validation Accuracy: {history.history['val_accuracy'][-1]:.4f}")

    # Evaluate on test set
    evaluate_results(results_df)

    print("\n📊 Test results preview:")
    print(results_df.head())

    print(f"\nTotal test samples: {len(results_df)}")
    print("\n✅ Full evaluation completed successfully")

    # -----------------------------
    # Baseline accuracies (just summary)
    # -----------------------------
    y_true = results_df["actual"].values
    X_test = results_df.drop(columns=["actual"]).values  # Needed for baseline fit interface

    baselines = [
        ("Random Baseline", RandomBaseline()),
        ("Majority Baseline", MajorityBaseline())
    ]

    print("\n📊 ================= BASELINE ACCURACIES =================")
    for name, baseline in baselines:
        baseline.fit(X_test, y_true)
        y_pred = baseline.predict(X_test)
        acc = accuracy_score(y_true, y_pred)
        print(f"{name} Accuracy: {acc:.4f}")

    print("\n✅ Full evaluation completed successfully")


if __name__ == "__main__":
    main()
