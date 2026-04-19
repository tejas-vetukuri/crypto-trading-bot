from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from joblib import load

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
)

from models.xgboost.xgb import train_xgb_model, build_xgb_save_paths, get_default_start_date


def _safe_accuracy(y_true: np.ndarray, y_pred: np.ndarray):
    if len(y_true) == 0:
        return None
    return float((y_true == y_pred).mean())


def _safe_majority_baseline(y_true: np.ndarray):
    if len(y_true) == 0:
        return None
    _, counts = np.unique(y_true, return_counts=True)
    return float(counts.max() / counts.sum())


def _safe_roc_auc(y_true: np.ndarray, probs: np.ndarray):
    try:
        y_true = np.asarray(y_true).astype(int)
        probs = np.asarray(probs).astype(float)
        if len(np.unique(y_true)) < 2:
            return None
        return float(roc_auc_score(y_true, probs))
    except Exception:
        return None


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
    roc_auc = _safe_roc_auc(y_true, p_up)
    random_baseline_accuracy = 0.5 if used_samples > 0 else None
    majority_baseline_accuracy = _safe_majority_baseline(y_used)

    if used_samples > 0:
        cm = confusion_matrix(y_used, pred_used, labels=[0, 1])
        cls_report = classification_report(
            y_used,
            pred_used,
            labels=[0, 1],
            target_names=["DOWN", "UP"],
            output_dict=True,
            zero_division=0,
        )
        cls_report_df = pd.DataFrame(cls_report).transpose().reset_index()
        cls_report_df = cls_report_df.rename(columns={"index": "class"})
    else:
        cm = None
        cls_report_df = None

    return {
        "threshold_type": "margin",
        "decision_boundary": float(decision_boundary),
        "margin_threshold": float(margin_threshold),
        "threshold": float(margin_threshold),
        "roc_auc": roc_auc,
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

    y_true = np.asarray(y_true).astype(int)

    for margin_threshold in margins:
        summary = build_xgb_eval_summary(
            y_true=y_true,
            p_up=p_up,
            decision_boundary=float(decision_boundary),
            margin_threshold=float(margin_threshold),
        )

        if summary["used_samples"] > 0:
            y_true_used = y_true[summary["used_mask"]]
            y_pred_used = summary["pred_used"]

            precision_macro = precision_score(
                y_true_used, y_pred_used, average="macro", zero_division=0
            )
            recall_macro = recall_score(
                y_true_used, y_pred_used, average="macro", zero_division=0
            )
            f1_macro = f1_score(
                y_true_used, y_pred_used, average="macro", zero_division=0
            )
        else:
            precision_macro = None
            recall_macro = None
            f1_macro = None

        rows.append({
            "margin_threshold": round(float(margin_threshold), 4),
            "decision_boundary": float(decision_boundary),
            "roc_auc": summary["roc_auc"],
            "used_accuracy": summary["used_accuracy"],
            "precision_macro": precision_macro,
            "recall_macro": recall_macro,
            "f1_macro": f1_macro,
            "coverage": summary["coverage"],
            "ignored_rate": summary["ignored_rate"],
            "used_samples": summary["used_samples"],
            "total_samples": summary["total_samples"],
        })

    return pd.DataFrame(rows).round(4)


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


from pathlib import Path
from joblib import load


def main():
    symbol = "BTCUSDT"
    resolution = "1h"

    start_date = get_default_start_date(resolution)
    save_paths = build_xgb_save_paths(symbol, resolution)

    chosen_boundary = 0.48
    chosen_margin = 0.10
    margins = (0.00, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16)

    RUN_TRAIN = False

    print("🚀 Starting XGBoost Evaluation...\n")
    print(f"Symbol:      {symbol}")
    print(f"Resolution:  {resolution}")
    print(f"Start date:  {start_date}")
    print(f"Artifacts:   {save_paths['artifacts_path']}")
    print(f"Preds CSV:   {save_paths['preds_csv_path']}")
    print()

    if RUN_TRAIN:
        model, artifacts, preds_df = train_xgb_model(
            symbol=symbol,
            resolution=resolution,
            start_date=start_date,
            end_date=None,
            train_ratio=0.80,
            decision_boundary=chosen_boundary,
            margin_threshold=chosen_margin,
            artifacts_path=save_paths["artifacts_path"],
            preds_csv_path=save_paths["preds_csv_path"],
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            use_sentiment_data=False,
        )

        print("\n⚙️ XGBoost Parameters (from training):")
        params = model.get_params()


    else:

        print("📂 Skipping training. Loading saved predictions + artifacts...\n")

        preds_df = pd.read_csv(save_paths["preds_csv_path"])

        artifacts = {"combo_tag": f"{symbol}_{resolution}"}

        artifacts_path = save_paths.get("artifacts_path")

        if artifacts_path and Path(artifacts_path).exists():

            saved_obj = load(artifacts_path)

            if isinstance(saved_obj, dict):

                model = saved_obj.get("model") or saved_obj.get("xgb_model")

                if model is not None:

                    params = model.get_params()

                    print("⚙️ XGBoost Parameters (from saved artifacts):")

                else:

                    print("⚠️ Artifacts file found, but no model key was present.")

                    print(f"Available keys: {list(saved_obj.keys())}")

                    params = None

            else:

                print("⚠️ Artifacts file is not a dictionary. Cannot extract model safely.")

                params = None

        else:

            print("⚠️ Saved artifacts file not found. Cannot print parameters.")

            params = None

    # 🔽 Print only key parameters (clean output)
    if params:
        important_params = [
            "n_estimators",
            "max_depth",
            "learning_rate",
            "min_child_weight",
            "gamma",
            "reg_alpha",
            "reg_lambda",
            "subsample",
            "colsample_bytree",
        ]

        for k in important_params:
            print(f"{k}: {params.get(k)}")

    # ============================
    # Continue with evaluation
    # ============================

    y_true = (preds_df["actual_trend"].astype(str).str.lower() == "up").astype(int).values
    p_up = preds_df["p_up"].astype(float).values


if __name__ == "__main__":
    main()