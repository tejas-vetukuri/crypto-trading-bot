import streamlit as st
import pandas as pd
import numpy as np

from datetime import date
from pathlib import Path
from sklearn.metrics import roc_auc_score

from models.lstm.lstm import (
    train_lstm_model,
    get_default_start_date,
    build_lstm_save_paths,
)
from models.lstm.train_eval_lstm import (
    build_threshold_summary,
    build_distribution_tables,
)
from models.xgboost.xgb import (
    train_xgb_model,
    build_xgb_save_paths,
)
from models.xgboost.train_eval_xgb import (
    build_xgb_eval_summary,
    build_xgb_margin_sweep,
    build_xgb_distribution_tables,
)


st.set_page_config(page_title="Model Evaluation Lab", layout="wide")


def safe_auc(y_true: np.ndarray, probs: np.ndarray):
    try:
        return float(roc_auc_score(y_true, probs))
    except Exception:
        return None


def render_lstm_results(results: dict):
    history = results.get("history")
    probs_df = results["probs_df"]
    metrics_df = results["metrics_df"]
    save_paths = results["save_paths"]
    threshold_summary = results["threshold_summary"]
    chosen_threshold = results["chosen_threshold"]

    y_true = probs_df["y_true"].values.astype(int)
    probs = probs_df["p"].values.astype(float)

    train_acc = None
    val_acc = None
    if history is not None:
        train_acc = history.history.get("accuracy", [None])[-1]
        val_acc = history.history.get("val_accuracy", [None])[-1]

    auc_val = safe_auc(y_true, probs)

    st.markdown("### Results")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Train Accuracy", f"{train_acc:.4f}" if train_acc is not None else "—")
    c2.metric("Validation Accuracy", f"{val_acc:.4f}" if val_acc is not None else "—")
    c3.metric("ROC-AUC", f"{auc_val:.4f}" if auc_val is not None else "—")
    c4.metric(f"Coverage @ {chosen_threshold:.2f}", f"{threshold_summary['coverage']:.4f}")

    summary_df = pd.DataFrame([
        {"Metric": "Threshold", "Value": f"{threshold_summary['threshold']:.2f}"},
        {"Metric": "Coverage", "Value": f"{threshold_summary['coverage']:.4f}"},
        {"Metric": "Ignored Rate", "Value": f"{threshold_summary['ignored_rate']:.4f}"},
        {"Metric": "Used Samples", "Value": threshold_summary["used_samples"]},
        {"Metric": "Total Samples", "Value": threshold_summary["total_samples"]},
        {
            "Metric": "Used Accuracy",
            "Value": "—" if threshold_summary["used_accuracy"] is None else f"{threshold_summary['used_accuracy']:.4f}",
        },
        {
            "Metric": "Random Baseline Accuracy",
            "Value": "—" if threshold_summary["random_baseline_accuracy"] is None else f"{threshold_summary['random_baseline_accuracy']:.4f}",
        },
        {
            "Metric": "Majority Baseline Accuracy",
            "Value": "—" if threshold_summary["majority_baseline_accuracy"] is None else f"{threshold_summary['majority_baseline_accuracy']:.4f}",
        },
    ])

    st.markdown("#### Threshold Evaluation Summary")
    st.dataframe(summary_df, use_container_width=True, hide_index=True)

    st.markdown("#### Threshold Sweep Metrics")
    st.dataframe(metrics_df, use_container_width=True, hide_index=True)

    st.markdown("#### Confusion Matrix")
    if threshold_summary["confusion_matrix"] is None:
        st.warning("No samples were used at this threshold.")
    else:
        cm_df = pd.DataFrame(
            threshold_summary["confusion_matrix"],
            index=["Actual DOWN", "Actual UP"],
            columns=["Pred DOWN", "Pred UP"],
        )
        st.dataframe(cm_df, use_container_width=True)

    st.markdown("#### Classification Report")
    if threshold_summary["classification_report_df"] is None:
        st.warning("No classification report available for this threshold.")
    else:
        st.dataframe(
            threshold_summary["classification_report_df"],
            use_container_width=True,
        )

    st.markdown("#### Sample Distribution")
    actual_full, actual_used, pred_used_df = build_distribution_tables(
        y_true=y_true,
        used_mask=threshold_summary["used_mask"],
        pred_used=threshold_summary["pred_used"],
    )

    d1, d2, d3 = st.columns(3)
    with d1:
        st.markdown("**Actual Class Distribution (Full Test)**")
        st.dataframe(actual_full, use_container_width=True, hide_index=True)
    with d2:
        st.markdown("**Actual Class Distribution (Used Only)**")
        st.dataframe(actual_used, use_container_width=True, hide_index=True)
    with d3:
        st.markdown("**Predicted Class Distribution (Used Only)**")
        st.dataframe(pred_used_df, use_container_width=True, hide_index=True)

    st.markdown("#### Test Probabilities Preview")
    st.dataframe(probs_df.head(50), use_container_width=True, hide_index=True)

    st.markdown("#### Saved Artifacts")
    paths_df = pd.DataFrame([
        {"Artifact": "Model Path", "Value": str(save_paths["model_path"])},
        {"Artifact": "Scaler Path", "Value": str(save_paths["scaler_path"])},
        {"Artifact": "Artifacts Path", "Value": str(save_paths["artifacts_path"])},
        {"Artifact": "Test Probs Path", "Value": str(save_paths["test_probs_path"])},
        {"Artifact": "Metrics Path", "Value": str(save_paths["metrics_path"])},
    ])
    st.dataframe(paths_df, use_container_width=True, hide_index=True)


def render_xgb_results(results: dict):
    output_df = results["output_df"]
    save_paths = results["save_paths"]
    xgb_summary = results["xgb_summary"]
    margin_sweep_df = results["margin_sweep_df"]
    chosen_margin = results["chosen_margin"]
    chosen_boundary = results["chosen_boundary"]

    y_true = output_df["y_true"].values.astype(int)
    p_up = output_df["p_up"].values.astype(float)

    auc_val = safe_auc(y_true, p_up)

    st.markdown("### Results")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("ROC-AUC", f"{auc_val:.4f}" if auc_val is not None else "—")
    c2.metric(f"Coverage @ Margin {chosen_margin:.2f}", f"{xgb_summary['coverage']:.4f}")
    c3.metric("Used Accuracy", "—" if xgb_summary["used_accuracy"] is None else f"{xgb_summary['used_accuracy']:.4f}")
    c4.metric("Ignored Rate", f"{xgb_summary['ignored_rate']:.4f}")

    summary_df = pd.DataFrame([
        {"Metric": "Decision Boundary", "Value": f"{chosen_boundary:.2f}"},
        {"Metric": "Margin Threshold", "Value": f"{chosen_margin:.2f}"},
        {"Metric": "Coverage", "Value": f"{xgb_summary['coverage']:.4f}"},
        {"Metric": "Ignored Rate", "Value": f"{xgb_summary['ignored_rate']:.4f}"},
        {"Metric": "Used Samples", "Value": xgb_summary["used_samples"]},
        {"Metric": "Total Samples", "Value": xgb_summary["total_samples"]},
        {
            "Metric": "Used Accuracy",
            "Value": "—" if xgb_summary["used_accuracy"] is None else f"{xgb_summary['used_accuracy']:.4f}",
        },
        {
            "Metric": "Random Baseline Accuracy",
            "Value": "—" if xgb_summary["random_baseline_accuracy"] is None else f"{xgb_summary['random_baseline_accuracy']:.4f}",
        },
        {
            "Metric": "Majority Baseline Accuracy",
            "Value": "—" if xgb_summary["majority_baseline_accuracy"] is None else f"{xgb_summary['majority_baseline_accuracy']:.4f}",
        },
    ])

    st.markdown("#### Margin Evaluation Summary")
    st.dataframe(summary_df, use_container_width=True, hide_index=True)

    st.markdown("#### Margin Sweep Metrics")
    st.dataframe(margin_sweep_df, use_container_width=True, hide_index=True)

    st.markdown("#### Confusion Matrix")
    if xgb_summary["confusion_matrix"] is None:
        st.warning("No samples were used at this margin threshold.")
    else:
        cm_df = pd.DataFrame(
            xgb_summary["confusion_matrix"],
            index=["Actual DOWN", "Actual UP"],
            columns=["Pred DOWN", "Pred UP"],
        )
        st.dataframe(cm_df, use_container_width=True)

    st.markdown("#### Classification Report")
    if xgb_summary["classification_report_df"] is None:
        st.warning("No classification report available for this margin threshold.")
    else:
        st.dataframe(
            xgb_summary["classification_report_df"],
            use_container_width=True,
            hide_index=True,
        )

    st.markdown("#### Sample Distribution")
    actual_full, actual_used, pred_used_df = build_xgb_distribution_tables(
        y_true=y_true,
        used_mask=xgb_summary["used_mask"],
        pred_used=xgb_summary["pred_used"],
    )

    d1, d2, d3 = st.columns(3)
    with d1:
        st.markdown("**Actual Class Distribution (Full Test)**")
        st.dataframe(actual_full, use_container_width=True, hide_index=True)
    with d2:
        st.markdown("**Actual Class Distribution (Used Only)**")
        st.dataframe(actual_used, use_container_width=True, hide_index=True)
    with d3:
        st.markdown("**Predicted Class Distribution (Used Only)**")
        st.dataframe(pred_used_df, use_container_width=True, hide_index=True)

    st.markdown("#### Test Predictions Preview")
    preview_cols = [
        "timestamp",
        "actual_trend",
        "prediction",
        "p_up",
        "decision_boundary",
        "margin",
        "used",
        "probability",
    ]
    st.dataframe(output_df[preview_cols].head(50), use_container_width=True, hide_index=True)

    st.markdown("#### Saved Artifacts")
    paths_df = pd.DataFrame([
        {"Artifact": "Artifacts Path", "Value": str(save_paths["artifacts_path"])},
        {"Artifact": "Predictions CSV Path", "Value": str(save_paths["preds_csv_path"])},
    ])
    st.dataframe(paths_df, use_container_width=True, hide_index=True)


if "coin" not in st.session_state:
    st.session_state.coin = "BTCUSDT"

if "interval" not in st.session_state:
    st.session_state.interval = "1h"

if "lstm_eval_results" not in st.session_state:
    st.session_state.lstm_eval_results = None

if "xgb_eval_results" not in st.session_state:
    st.session_state.xgb_eval_results = None

st.title("Model Evaluation Lab")

coin = st.session_state.coin
frequency = st.session_state.interval
default_start = get_default_start_date(frequency)

st.write("Coin:", coin)
st.write("Frequency:", frequency)

st.divider()

with st.expander("Global Evaluation Settings", expanded=True):
    r1c1, r1c2, r1c3, r1c4, r1c5 = st.columns(5)

    with r1c1:
        st.text_input("Coin", value=coin, disabled=True)

    with r1c2:
        st.text_input("Frequency", value=frequency, disabled=True)

    with r1c3:
        start_date = st.date_input(
            "Start Date",
            value=date.fromisoformat(default_start),
            key="global_start_date",
        )

    with r1c4:
        if "use_latest_data" not in st.session_state:
            st.session_state.use_latest_data = True

        if st.session_state.use_latest_data:
            end_date = None
            st.text_input("End Date", value="Latest", disabled=True)
        else:
            end_date = st.date_input(
                "End Date",
                value=date.today(),
                key="global_end_date",
            )

    with r1c5:
        st.text_input("Train/Test Split", value="80/20", disabled=True)

    r2c1, r2c2, r2c3, r2c4, r2c5 = st.columns(5)

    with r2c4:
        st.session_state.use_latest_data = st.checkbox(
            "Use latest available data",
            value=st.session_state.use_latest_data
        )

st.divider()

with st.expander("LSTM Evaluation", expanded=True):
    st.subheader("LSTM Parameters")

    with st.form("lstm_eval_form"):
        r1c1, r1c2, r1c3, r1c4 = st.columns(4)
        with r1c1:
            x_window_size = st.number_input("x_window_size", min_value=1, value=100, step=1)
        with r1c2:
            epochs = st.number_input("epochs", min_value=1, value=10, step=1)
        with r1c3:
            batch_size = st.number_input("batch_size", min_value=1, value=64, step=1)
        with r1c4:
            learning_rate = st.number_input(
                "learning_rate",
                min_value=0.0001,
                value=0.0010,
                step=0.0001,
                format="%.4f",
            )

        r2c1, r2c2, r2c3, r2c4, r2c5 = st.columns(5)
        with r2c1:
            lstm_units = st.number_input("lstm_units", min_value=1, value=100, step=1)
        with r2c2:
            dropout = st.number_input(
                "dropout",
                min_value=0.0,
                max_value=0.9,
                value=0.20,
                step=0.05,
                format="%.2f",
            )
        with r2c3:
            validation_split = st.number_input(
                "validation_split",
                min_value=0.01,
                max_value=0.50,
                value=0.05,
                step=0.01,
                format="%.2f",
                disabled=True,
            )
        with r2c4:
            early_stopping_patience = st.number_input(
                "early_stopping_patience",
                min_value=1,
                value=3,
                step=1,
            )
        with r2c5:
            threshold = st.number_input(
                "threshold",
                min_value=0.01,
                max_value=0.99,
                value=0.53,
                step=0.01,
                format="%.2f",
            )

        st.caption(
            "Note: Default parameters are set to the best-performing configuration identified "
            "through prior grid search tuning on BTCUSDT 1h data."
        )

        b1, b2 = st.columns(2)
        with b1:
            retrain_clicked = st.form_submit_button("Retrain LSTM", use_container_width=True)
        with b2:
            evaluate_clicked = st.form_submit_button("Evaluate LSTM", use_container_width=True)

    save_paths = build_lstm_save_paths(coin, frequency)

    if retrain_clicked:
        try:
            with st.spinner("Retraining LSTM..."):
                thresholds = tuple(sorted(set([
                    0.50, 0.51, 0.52, round(float(threshold), 2), 0.54, 0.55, 0.56, 0.57, 0.58
                ])))

                model, history, probs_df, metrics_df, scaler = train_lstm_model(
                    symbol=coin,
                    resolution=frequency,
                    start_date=start_date.isoformat(),
                    end_date=end_date.isoformat() if isinstance(end_date, date) else None,
                    x_window_size=int(x_window_size),
                    epochs=int(epochs),
                    batch_size=int(batch_size),
                    train_ratio=0.80,
                    validation_split=float(validation_split),
                    learning_rate=float(learning_rate),
                    lstm_units=int(lstm_units),
                    dropout_rate=float(dropout),
                    early_stopping_patience=int(early_stopping_patience),
                    model_path=save_paths["model_path"],
                    scaler_path=save_paths["scaler_path"],
                    artifacts_path=save_paths["artifacts_path"],
                    test_probs_path=save_paths["test_probs_path"],
                    metrics_path=save_paths["metrics_path"],
                    thresholds=thresholds,
                )

                y_true = probs_df["y_true"].values.astype(int)
                probs = probs_df["p"].values.astype(float)

                threshold_summary = build_threshold_summary(
                    y_true=y_true,
                    probs=probs,
                    threshold=float(threshold),
                )

                st.session_state.lstm_eval_results = {
                    "history": history,
                    "probs_df": probs_df,
                    "metrics_df": metrics_df,
                    "save_paths": save_paths,
                    "threshold_summary": threshold_summary,
                    "chosen_threshold": float(threshold),
                }

            st.success("LSTM retraining and evaluation completed successfully.")

        except Exception as e:
            st.error(f"LSTM retraining failed: {e}")

    if evaluate_clicked:
        try:
            probs_path = Path(save_paths["test_probs_path"])
            metrics_path = Path(save_paths["metrics_path"])

            if not probs_path.exists() or not metrics_path.exists():
                st.error("Saved LSTM evaluation files were not found. Retrain the model first.")
            else:
                probs_df = pd.read_csv(probs_path)
                metrics_df = pd.read_csv(metrics_path)

                y_true = probs_df["y_true"].values.astype(int)
                probs = probs_df["p"].values.astype(float)

                threshold_summary = build_threshold_summary(
                    y_true=y_true,
                    probs=probs,
                    threshold=float(threshold),
                )

                st.session_state.lstm_eval_results = {
                    "history": None,
                    "probs_df": probs_df,
                    "metrics_df": metrics_df,
                    "save_paths": save_paths,
                    "threshold_summary": threshold_summary,
                    "chosen_threshold": float(threshold),
                }

                st.success("Loaded saved LSTM evaluation results.")

        except Exception as e:
            st.error(f"LSTM evaluation failed: {e}")

    if st.session_state.lstm_eval_results is not None:
        render_lstm_results(st.session_state.lstm_eval_results)
    else:
        st.info("Use Retrain LSTM to train and save results, or Evaluate LSTM to load saved results.")



with st.expander("XGBoost Evaluation", expanded=False):
    st.subheader("XGBoost Parameters")

    with st.form("xgb_eval_form"):
        r1c1, r1c2, r1c3, r1c4 = st.columns(4)
        with r1c1:
            n_estimators = st.number_input(
                "n_estimators",
                min_value=10,
                value=300,
                step=10,
            )
        with r1c2:
            learning_rate_xgb = st.number_input(
                "learning_rate",
                min_value=0.001,
                value=0.05,
                step=0.001,
                format="%.3f",
            )
        with r1c3:
            max_depth = st.number_input(
                "max_depth",
                min_value=1,
                value=6,
                step=1,
            )
        with r1c4:
            subsample = st.number_input(
                "subsample",
                min_value=0.10,
                max_value=1.00,
                value=0.80,
                step=0.05,
                format="%.2f",
            )

        r2c1, r2c2, r2c3, r2c4 = st.columns(4)
        with r2c1:
            colsample_bytree = st.number_input(
                "colsample_bytree",
                min_value=0.10,
                max_value=1.00,
                value=0.80,
                step=0.05,
                format="%.2f",
            )
        with r2c2:
            decision_boundary = st.number_input(
                "decision_boundary",
                min_value=0.01,
                max_value=0.99,
                value=0.48,
                step=0.01,
                format="%.2f",
            )
        with r2c3:
            margin_threshold = st.number_input(
                "margin_threshold",
                min_value=0.00,
                max_value=0.49,
                value=0.10,
                step=0.01,
                format="%.2f",
            )
        is_sentiment_available = (coin.upper() == "BTCUSDT" and frequency == "1h")

        with r2c4:
            st.markdown("<div style='height: 8px'></div>", unsafe_allow_html=True)

            use_sentiment_data = st.checkbox(
                "use sentiment data",
                value=False,
                disabled=not is_sentiment_available,
                help="Available only for BTC 1h. Adds Bybit derivatives features (OI + LSR).",
            )

            if not is_sentiment_available:
                st.caption("Sentiment features currently supported only for BTC 1h experiments.")

        st.caption(
            "Note: Default parameters are set to the best-performing configuration identified "
            "through prior grid search tuning on BTCUSDT 1h data."
        )

        b1, b2 = st.columns(2)
        with b1:
            retrain_xgb_clicked = st.form_submit_button("Retrain XGBoost", use_container_width=True)
        with b2:
            evaluate_xgb_clicked = st.form_submit_button("Evaluate XGBoost", use_container_width=True)

st.divider()

with st.expander("LSTM + XGBoost Ensemble Evaluation"):
    st.write("Ensemble evaluation controls and outputs will go here.")

with st.expander("Ensemble + RL Filtering (Main Model)"):
    st.write("Main model evaluation controls and outputs will go here.")

with st.expander("LSTM (Alternative Target)"):
    st.write("Alternative LSTM evaluation controls and outputs will go here.")