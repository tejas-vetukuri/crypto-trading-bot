from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)

from data.binance import BinanceDataClient
from data.feature_engineering import feature_engineering_xgb
from models.lstm.lstm import get_default_start_date


class MajorityBaseline:
    def fit(self, X, y):
        counts = Counter(y)
        self.majority_class = counts.most_common(1)[0][0]
        self.classes_ = np.sort(np.unique(y))
        return self

    def predict(self, X):
        return np.full(len(X), self.majority_class, dtype=int)

    def predict_proba(self, X):
        """
        Degenerate probability output:
        majority class gets probability 1.0, others 0.0.
        """
        n = len(X)
        probs = np.zeros((n, len(self.classes_)), dtype=float)
        maj_idx = int(np.where(self.classes_ == self.majority_class)[0][0])
        probs[:, maj_idx] = 1.0
        return probs


def evaluate_majority_baseline(
    X_test,
    y_test,
    X_train=None,
    y_train=None,
    baseline: MajorityBaseline | None = None,
    verbose: bool = True,
) -> dict:
    if baseline is None:
        if y_train is None:
            raise ValueError("Provide either a fitted baseline or y_train for fitting.")
        baseline = MajorityBaseline().fit(X_train, y_train)

    y_test = np.asarray(y_test).astype(int)
    y_pred = baseline.predict(X_test)

    result = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "confusion_matrix": confusion_matrix(y_test, y_pred, labels=[0, 1]),
        "classification_report_df": pd.DataFrame(
            classification_report(
                y_test,
                y_pred,
                labels=[0, 1],
                target_names=["DOWN", "UP"],
                zero_division=0,
                output_dict=True,
            )
        ).transpose(),
        "roc_auc": None,
        "y_pred": y_pred,
    }

    try:
        y_prob_up = baseline.predict_proba(X_test)[:, 1]
        result["roc_auc"] = float(roc_auc_score(y_test, y_prob_up))
    except Exception:
        result["roc_auc"] = None

    if verbose:
        print("\n================ MAJORITY BASELINE EVALUATION ================")
        print(f"Accuracy: {result['accuracy']:.4f}")
        if result["roc_auc"] is not None:
            print(f"ROC-AUC:  {result['roc_auc']:.4f}")
        else:
            print("ROC-AUC:  could not be computed / not meaningful")

        print("\nConfusion Matrix:")
        print(result["confusion_matrix"])

        print("\nClassification Report:")
        print(result["classification_report_df"])

    return result


def load_xgb_style_dataset(
    symbol: str = "BTCUSDT",
    resolution: str = "1h",
    start_date: str | None = None,
    end_date: str | None = None,
    train_ratio: float = 0.80,
):
    """
    Loads a simple binary direction dataset using the XGBoost-style feature engineering.
    This is only for evaluating the baseline model on the same task style.
    """
    symbol = symbol.upper()
    if start_date is None:
        start_date = get_default_start_date(resolution)

    client = BinanceDataClient(market="spot")
    df_raw = client.get_candles(
        symbol=symbol,
        resolution=resolution,
        start_date=start_date,
        end_date=end_date,
    ).sort_values("timestamp").reset_index(drop=True)

    df_feat = feature_engineering_xgb(df_raw).copy()
    if "actual_trend" not in df_feat.columns:
        raise ValueError("Expected 'actual_trend' column not found after feature engineering.")

    df_feat = df_feat[df_feat["actual_trend"].isin(["down", "up"])].copy()
    df_feat["y"] = df_feat["actual_trend"].map({"down": 0, "up": 1}).astype(int)

    drop_cols = {"timestamp", "symbol", "resolution", "actual_trend", "y"}
    feature_cols = [c for c in df_feat.columns if c not in drop_cols]

    X = df_feat[feature_cols].copy()
    y = df_feat["y"].values.astype(int)

    n = len(df_feat)
    split = int(n * train_ratio)
    if split <= 0 or split >= n:
        raise ValueError(f"Invalid split: n={n}, split={split}")

    X_train = X.iloc[:split].copy()
    y_train = y[:split]
    X_test = X.iloc[split:].copy()
    y_test = y[split:]

    return X_train, y_train, X_test, y_test, df_feat


def main():
    symbol = "BTCUSDT"
    resolution = "1h"
    start_date = get_default_start_date(resolution)
    end_date = None
    train_ratio = 0.80

    print("\n================ MAJORITY BASELINE CONFIG ================")
    print(f"Symbol:      {symbol}")
    print(f"Resolution:  {resolution}")
    print(f"Start date:  {start_date}")
    print(f"End date:    {end_date}")
    print(f"Train ratio: {train_ratio}")
    print("=========================================================")

    X_train, y_train, X_test, y_test, df_all = load_xgb_style_dataset(
        symbol=symbol,
        resolution=resolution,
        start_date=start_date,
        end_date=end_date,
        train_ratio=train_ratio,
    )

    baseline = MajorityBaseline().fit(X_train, y_train)

    print(f"\nTrain samples: {len(X_train)}")
    print(f"Test samples:  {len(X_test)}")
    print(f"Majority class learned: {baseline.majority_class}")

    evaluate_majority_baseline(
        X_test=X_test,
        y_test=y_test,
        X_train=X_train,
        y_train=y_train,
        baseline=baseline,
        verbose=True,
    )

    print("\n✅ Majority baseline evaluation completed.")


if __name__ == "__main__":
    main()