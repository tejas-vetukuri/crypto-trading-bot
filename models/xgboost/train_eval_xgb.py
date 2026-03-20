from __future__ import annotations

import numpy as np
import pandas as pd

from sklearn.metrics import classification_report, confusion_matrix


def _safe_accuracy(y_true: np.ndarray, y_pred: np.ndarray):
    if len(y_true) == 0:
        return None
    return float((y_true == y_pred).mean())


def _safe_majority_baseline(y_true: np.ndarray):
    if len(y_true) == 0:
        return None
    vals, counts = np.unique(y_true, return_counts=True)
    return float(counts.max() / counts.sum())


def build_xgb_eval_summary(
    y_true: np.ndarray,
    p_up: np.ndarray,
    decision_boundary: float,
    margin_threshold: float,
) -> dict:
    y_true = np.asarray(y_true).astype(int)
    p_up = np.asarray(p_up).astype(float)

    margin = np.abs(p_up - float(decision_boundary))
    used_mask = margin >= float(margin_threshold)

    pred_all = np.full(len(p_up), 2, dtype=int)  # 2 = sideways
    pred_all[used_mask] = (p_up[used_mask] > float(decision_boundary)).astype(int)

    pred_used = pred_all[used_mask]
    y_used = y_true[used_mask]

    total_samples = int(len(y_true))
    used_samples = int(used_mask.sum())
    coverage = float(used_samples / total_samples) if total_samples > 0 else 0.0
    ignored_rate = float(1.0 - coverage)

    used_accuracy = _safe_accuracy(y_used, pred_used)
    random_baseline_accuracy = 0.5 if used_samples > 0 else None
    majority_baseline_accuracy = _safe_majority_baseline(y_used)

    if used_samples > 0:
        cm = confusion_matrix(y_used, pred_used, labels=[0, 1])
        cls_report = classification_report(
            y_used,
            pred_used,
            labels=[0, 1],
            target_names=["down", "up"],
            output_dict=True,
            zero_division=0,
        )
        cls_report_df = pd.DataFrame(cls_report).transpose().reset_index()
        cls_report_df = cls_report_df.rename(columns={"index": "class"})
    else:
        cm = None
        cls_report_df = None

    return {
        "decision_boundary": float(decision_boundary),
        "margin_threshold": float(margin_threshold),
        "coverage": coverage,
        "ignored_rate": ignored_rate,
        "used_samples": used_samples,
        "total_samples": total_samples,
        "used_accuracy": used_accuracy,
        "random_baseline_accuracy": random_baseline_accuracy,
        "majority_baseline_accuracy": majority_baseline_accuracy,
        "used_mask": used_mask,
        "pred_used": pred_used,
        "pred_all": pred_all,
        "confusion_matrix": cm,
        "classification_report_df": cls_report_df,
    }


def build_xgb_margin_sweep(
    y_true: np.ndarray,
    p_up: np.ndarray,
    decision_boundary: float,
    margins: tuple[float, ...] | list[float],
) -> pd.DataFrame:
    rows = []

    for margin_threshold in margins:
        summary = build_xgb_eval_summary(
            y_true=y_true,
            p_up=p_up,
            decision_boundary=float(decision_boundary),
            margin_threshold=float(margin_threshold),
        )

        rows.append({
            "margin_threshold": round(float(margin_threshold), 4),
            "coverage": summary["coverage"],
            "ignored_rate": summary["ignored_rate"],
            "used_samples": summary["used_samples"],
            "total_samples": summary["total_samples"],
            "used_accuracy": summary["used_accuracy"],
            "random_baseline_accuracy": summary["random_baseline_accuracy"],
            "majority_baseline_accuracy": summary["majority_baseline_accuracy"],
        })

    return pd.DataFrame(rows)


def build_xgb_distribution_tables(
    y_true: np.ndarray,
    used_mask: np.ndarray,
    pred_used: np.ndarray,
):
    y_true = np.asarray(y_true).astype(int)
    used_mask = np.asarray(used_mask).astype(bool)
    pred_used = np.asarray(pred_used).astype(int)

    actual_full = pd.DataFrame({
        "Class": ["DOWN", "UP"],
        "Count": [
            int((y_true == 0).sum()),
            int((y_true == 1).sum()),
        ],
    })

    y_used = y_true[used_mask]
    actual_used = pd.DataFrame({
        "Class": ["DOWN", "UP"],
        "Count": [
            int((y_used == 0).sum()),
            int((y_used == 1).sum()),
        ],
    })

    pred_used_df = pd.DataFrame({
        "Class": ["DOWN", "UP"],
        "Count": [
            int((pred_used == 0).sum()),
            int((pred_used == 1).sum()),
        ],
    })

    return actual_full, actual_used, pred_used_df