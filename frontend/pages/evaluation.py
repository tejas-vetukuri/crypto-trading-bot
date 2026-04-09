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

from models.baselines.randomforest import build_rf_save_paths
from models.baselines.simple_rnn import build_simple_rnn_save_paths

from models.rl.eval_ensemble_only import evaluate_ensemble_only
from models.rl.rl_ensemble import (
    RiskConfig,
    train_rl_policy,
    build_combo_artifact_paths,
)
from models.rl.eval_rl import evaluate_rl_agent


st.set_page_config(page_title="Model Evaluation Lab", layout="wide")


def safe_auc(y_true: np.ndarray, probs: np.ndarray):
    try:
        return float(roc_auc_score(y_true, probs))
    except Exception:
        return None


def safe_binary_accuracy(y_true: np.ndarray, preds: np.ndarray):
    try:
        return float((y_true == preds).mean()) if len(y_true) else None
    except Exception:
        return None


def margin_to_lstm_threshold(margin: float, center: float = 0.5) -> float:
    return center + float(margin)


def margin_to_symmetric_bounds(margin: float, center: float = 0.5):
    margin = float(margin)
    return center - margin, center + margin


def load_simple_rnn_raw_accuracy(symbol: str, resolution: str):
    try:
        probs_path = Path(build_simple_rnn_save_paths(symbol, resolution)["test_probs_path"])
        if not probs_path.exists():
            return None

        probs_df = pd.read_csv(probs_path)
        y_true = probs_df["y_true"].astype(int).values
        probs = probs_df["p"].astype(float).values
        pred = (probs >= 0.5).astype(int)
        return safe_binary_accuracy(y_true, pred)
    except Exception:
        return None


def load_rf_raw_accuracy(symbol: str, resolution: str):
    try:
        preds_path = Path(build_rf_save_paths(symbol, resolution)["preds_csv_path"])
        if not preds_path.exists():
            return None

        output_df = pd.read_csv(preds_path)
        y_true = (output_df["actual_trend"].astype(str).str.lower() == "up").astype(int).values
        p_up = output_df["p_up"].astype(float).values
        pred = (p_up >= 0.5).astype(int)
        return safe_binary_accuracy(y_true, pred)
    except Exception:
        return None


def render_lstm_results(results: dict, coin: str, frequency: str):
    history = results.get("history")
    probs_df = results["probs_df"]
    metrics_df = results["metrics_df"]
    save_paths = results["save_paths"]
    threshold_summary = results["threshold_summary"]
    chosen_threshold = results["chosen_threshold"]

    y_true = probs_df["y_true"].astype(int).values
    probs = probs_df["p"].astype(float).values
    lstm_raw_accuracy = safe_binary_accuracy(y_true, (probs >= 0.5).astype(int))
    simple_rnn_raw_accuracy = load_simple_rnn_raw_accuracy(coin, frequency)
    auc_val = safe_auc(y_true, probs)

    train_acc = val_acc = None
    if history is not None:
        train_acc = history.history.get("accuracy", [None])[-1]
        val_acc = history.history.get("val_accuracy", [None])[-1]

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
            "Metric": "LSTM Used Accuracy",
            "Value": "—" if threshold_summary["used_accuracy"] is None else f"{threshold_summary['used_accuracy']:.4f}",
        },
        {
            "Metric": "LSTM Raw Accuracy",
            "Value": "—" if lstm_raw_accuracy is None else f"{lstm_raw_accuracy:.4f}",
        },
        {
            "Metric": "Simple RNN Accuracy",
            "Value": "—" if simple_rnn_raw_accuracy is None else f"{simple_rnn_raw_accuracy:.4f}",
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


def render_xgb_results(results: dict, coin: str, frequency: str):
    output_df = results["output_df"]
    save_paths = results["save_paths"]
    xgb_summary = results["xgb_summary"]
    margin_sweep_df = results["margin_sweep_df"]
    chosen_margin = results["chosen_margin"]
    chosen_boundary = results["chosen_boundary"]

    y_true = output_df["y_true"].astype(int).values
    p_up = output_df["p_up"].astype(float).values
    xgb_raw_accuracy = safe_binary_accuracy(y_true, (p_up >= 0.5).astype(int))
    rf_raw_accuracy = load_rf_raw_accuracy(coin, frequency)
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
            "Metric": "XGBoost Used Accuracy",
            "Value": "—" if xgb_summary["used_accuracy"] is None else f"{xgb_summary['used_accuracy']:.4f}",
        },
        {
            "Metric": "XGBoost Raw Accuracy",
            "Value": "—" if xgb_raw_accuracy is None else f"{xgb_raw_accuracy:.4f}",
        },
        {
            "Metric": "Random Forest Accuracy",
            "Value": "—" if rf_raw_accuracy is None else f"{rf_raw_accuracy:.4f}",
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
    preview_cols = [c for c in preview_cols if c in output_df.columns]
    st.dataframe(output_df[preview_cols].head(50), use_container_width=True, hide_index=True)

    st.markdown("#### Saved Artifacts")
    paths_df = pd.DataFrame([
        {"Artifact": "Artifacts Path", "Value": str(save_paths["artifacts_path"])},
        {"Artifact": "Predictions CSV Path", "Value": str(save_paths["preds_csv_path"])},
    ])
    st.dataframe(paths_df, use_container_width=True, hide_index=True)


def render_ensemble_results(results: dict):
    base_metrics = results["base_metrics"]
    ensemble_metrics = results["ensemble_metrics"]
    trade_with_fees = results["trade_metrics_with_fees"]
    trade_no_fees = results["trade_metrics_no_fees"]
    merged_preview = results["merged_preview"]

    st.markdown("### Results")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("XGB Acc (non-hold)", f"{base_metrics['xgb_accuracy_non_hold']:.4f}")
    c2.metric("LSTM Acc (non-hold)", f"{base_metrics['lstm_accuracy_non_hold']:.4f}")
    c3.metric("Ensemble 2-way Acc", f"{ensemble_metrics['ensemble_2way_accuracy']:.4f}")
    c4.metric("Ensemble 3-way Acc", f"{ensemble_metrics['ensemble_3way_accuracy_non_hold']:.4f}")

    summary_df = pd.DataFrame([
        {"Metric": "Rows Evaluated", "Value": results["rows_evaluated"]},
        {"Metric": "Train/Test Split", "Value": f"{results['train_ratio']:.2f} / {1 - results['train_ratio']:.2f}"},
        {"Metric": "XGB Non-hold Accuracy", "Value": f"{base_metrics['xgb_accuracy_non_hold']:.4f}"},
        {"Metric": "XGB Non-hold Sample Count", "Value": base_metrics["xgb_n"]},
        {"Metric": "LSTM Non-hold Accuracy", "Value": f"{base_metrics['lstm_accuracy_non_hold']:.4f}"},
        {"Metric": "LSTM Non-hold Sample Count", "Value": base_metrics["lstm_n"]},
        {"Metric": "Ensemble 2-way Accuracy", "Value": f"{ensemble_metrics['ensemble_2way_accuracy']:.4f}"},
        {"Metric": "Ensemble 2-way Sample Count", "Value": ensemble_metrics["ensemble_2way_n"]},
        {
            "Metric": "Ensemble 3-way Accuracy (non-hold)",
            "Value": f"{ensemble_metrics['ensemble_3way_accuracy_non_hold']:.4f}",
        },
        {"Metric": "Ensemble 3-way Non-hold Sample Count", "Value": ensemble_metrics["ensemble_3way_n_non_hold"]},
        {"Metric": "Agreement-only Accuracy", "Value": f"{ensemble_metrics['agreement_only_accuracy']:.4f}"},
        {"Metric": "Agreement-only Sample Count", "Value": ensemble_metrics["agreement_only_n"]},
        {"Metric": "XGB Weight", "Value": f"{ensemble_metrics['ensemble_weight_xgb']:.2f}"},
        {"Metric": "LSTM Weight", "Value": f"{ensemble_metrics['ensemble_weight_lstm']:.2f}"},
        {"Metric": "Ensemble Lower", "Value": f"{ensemble_metrics['ensemble_lower']:.2f}"},
        {"Metric": "Ensemble Upper", "Value": f"{ensemble_metrics['ensemble_upper']:.2f}"},
    ])
    st.markdown("#### Ensemble Summary")
    st.dataframe(summary_df, use_container_width=True, hide_index=True)

    t1, t2, t3, t4 = st.columns(4)
    t1.metric("Setups", trade_with_fees["setups"])
    t2.metric("Win Rate", f"{trade_with_fees['win_rate']:.4f}")
    t3.metric("Avg Net R / Trade", f"{trade_with_fees['avg_net_r_per_trade']:.4f}")
    t4.metric("Total Return", f"{trade_with_fees['total_return']:.4f}")

    trade_df = pd.DataFrame([
        {"Metric": "Setups", "With Fees": trade_with_fees["setups"], "No Fees": trade_no_fees["setups"]},
        {"Metric": "Taken", "With Fees": trade_with_fees["taken"], "No Fees": trade_no_fees["taken"]},
        {"Metric": "Take Rate", "With Fees": f"{trade_with_fees['take_rate']:.4f}", "No Fees": f"{trade_no_fees['take_rate']:.4f}"},
        {
            "Metric": "Directional Accuracy",
            "With Fees": f"{trade_with_fees['directional_accuracy']:.4f}",
            "No Fees": f"{trade_no_fees['directional_accuracy']:.4f}",
        },
        {"Metric": "Win Rate", "With Fees": f"{trade_with_fees['win_rate']:.4f}", "No Fees": f"{trade_no_fees['win_rate']:.4f}"},
        {
            "Metric": "Avg Gross R / Trade",
            "With Fees": f"{trade_with_fees['avg_gross_r_per_trade']:.4f}",
            "No Fees": f"{trade_no_fees['avg_gross_r_per_trade']:.4f}",
        },
        {
            "Metric": "Avg Net R / Trade",
            "With Fees": f"{trade_with_fees['avg_net_r_per_trade']:.4f}",
            "No Fees": f"{trade_no_fees['avg_net_r_per_trade']:.4f}",
        },
        {"Metric": "Total Return", "With Fees": f"{trade_with_fees['total_return']:.4f}", "No Fees": f"{trade_no_fees['total_return']:.4f}"},
        {"Metric": "Max Drawdown", "With Fees": f"{trade_with_fees['max_drawdown']:.4f}", "No Fees": f"{trade_no_fees['max_drawdown']:.4f}"},
        {"Metric": "TP Exits", "With Fees": trade_with_fees["tp_exits"], "No Fees": trade_no_fees["tp_exits"]},
        {"Metric": "SL Exits", "With Fees": trade_with_fees["sl_exits"], "No Fees": trade_no_fees["sl_exits"]},
        {"Metric": "Horizon Exits", "With Fees": trade_with_fees["horizon_exits"], "No Fees": trade_no_fees["horizon_exits"]},
        {"Metric": "Other Exits", "With Fees": trade_with_fees["other_exits"], "No Fees": trade_no_fees["other_exits"]},
    ])
    st.markdown("#### Trade Simulation")
    st.dataframe(trade_df, use_container_width=True, hide_index=True)

    st.markdown("#### Ensemble Preview")
    preview_cols = [
        "timestamp",
        "actual",
        "xgb_pred",
        "lstm_pred",
        "xgb_p_up",
        "lstm_p_up",
        "ensemble_p_up",
        "ensemble_pred_2way",
        "ensemble_pred_3way",
        "ensemble_signal",
    ]
    preview_cols = [c for c in preview_cols if c in merged_preview.columns]
    st.dataframe(merged_preview[preview_cols].head(50), use_container_width=True, hide_index=True)


def render_rl_results(results: dict):
    base = results["base_metrics"]
    rl_fees = results["rl_metrics_with_fees"]
    rl_no_fees = results["rl_metrics_no_fees"]
    cfg = results["config"]

    st.markdown("### Results")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Candidate Setups", int(rl_fees["setups"]))
    c2.metric("Taken by RL", int(rl_fees["taken"]))
    c3.metric("Take Rate", f"{rl_fees['take_rate']:.4f}")
    c4.metric("Win Rate", f"{rl_fees['win_rate']:.4f}")

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Avg Gross R / Trade", f"{rl_fees['avg_gross_r_per_trade']:.4f}")
    c6.metric("Avg Net R / Trade", f"{rl_fees['avg_net_r_per_trade']:.4f}")
    c7.metric("Total Return", f"{rl_fees['total_return']:.4f}")
    c8.metric("Max Drawdown", f"{rl_fees['max_drawdown']:.4f}")

    summary_df = pd.DataFrame([
        {"Metric": "Rows Evaluated", "Value": len(results["merged_df"])},
        {"Metric": "Train Ratio Used", "Value": f"{cfg['train_ratio']:.2f}"},
        {"Metric": "XGB Accuracy (non-hold)", "Value": f"{base['xgb_acc_non_hold']:.4f}"},
        {"Metric": "XGB Non-hold n", "Value": base["xgb_n"]},
        {"Metric": "LSTM Accuracy (non-hold)", "Value": f"{base['lstm_acc_non_hold']:.4f}"},
        {"Metric": "LSTM Non-hold n", "Value": base["lstm_n"]},
        {"Metric": "Ensemble 2-way Accuracy", "Value": f"{base['ens_acc_2way']:.4f}"},
        {"Metric": "Ensemble 2-way n", "Value": base["ens_n_2way"]},
        {"Metric": "Ensemble 3-way Accuracy (non-hold)", "Value": f"{base['ens_acc_non_hold']:.4f}"},
        {"Metric": "Ensemble 3-way Non-hold n", "Value": base["ens_n_non_hold"]},
        {"Metric": "Agreement-only Accuracy", "Value": f"{base['agree_acc']:.4f}"},
        {"Metric": "Agreement-only n", "Value": base["agree_n"]},
        {"Metric": "Ensemble Weight XGB", "Value": f"{cfg['ensemble_weight_xgb']:.2f}"},
        {"Metric": "Ensemble Weight LSTM", "Value": f"{cfg['ensemble_weight_lstm']:.2f}"},
        {"Metric": "Ensemble Lower", "Value": f"{cfg['ensemble_lower']:.2f}"},
        {"Metric": "Ensemble Upper", "Value": f"{cfg['ensemble_upper']:.2f}"},
        {"Metric": "Min Take Visits", "Value": cfg["min_take_visits"]},
        {"Metric": "Q Take Margin", "Value": f"{cfg['q_take_margin']:.4f}"},
    ])
    st.markdown("#### RL Filter Summary")
    st.dataframe(summary_df, use_container_width=True, hide_index=True)

    trade_df = pd.DataFrame([
        {"Metric": "Setups", "With Fees": rl_fees["setups"], "No Fees": rl_no_fees["setups"]},
        {"Metric": "Taken", "With Fees": rl_fees["taken"], "No Fees": rl_no_fees["taken"]},
        {"Metric": "Skipped", "With Fees": rl_fees["skipped"], "No Fees": rl_no_fees["skipped"]},
        {"Metric": "Take Rate", "With Fees": f"{rl_fees['take_rate']:.4f}", "No Fees": f"{rl_no_fees['take_rate']:.4f}"},
        {
            "Metric": "Directional Accuracy",
            "With Fees": f"{rl_fees['directional_accuracy']:.4f}",
            "No Fees": f"{rl_no_fees['directional_accuracy']:.4f}",
        },
        {"Metric": "Win Rate", "With Fees": f"{rl_fees['win_rate']:.4f}", "No Fees": f"{rl_no_fees['win_rate']:.4f}"},
        {
            "Metric": "Avg Gross R / Trade",
            "With Fees": f"{rl_fees['avg_gross_r_per_trade']:.4f}",
            "No Fees": f"{rl_no_fees['avg_gross_r_per_trade']:.4f}",
        },
        {
            "Metric": "Avg Net R / Trade",
            "With Fees": f"{rl_fees['avg_net_r_per_trade']:.4f}",
            "No Fees": f"{rl_no_fees['avg_net_r_per_trade']:.4f}",
        },
        {"Metric": "Total Return", "With Fees": f"{rl_fees['total_return']:.4f}", "No Fees": f"{rl_no_fees['total_return']:.4f}"},
        {"Metric": "Max Drawdown", "With Fees": f"{rl_fees['max_drawdown']:.4f}", "No Fees": f"{rl_no_fees['max_drawdown']:.4f}"},
        {"Metric": "TP Exits", "With Fees": rl_fees["tp_exits"], "No Fees": rl_no_fees["tp_exits"]},
        {"Metric": "SL Exits", "With Fees": rl_fees["sl_exits"], "No Fees": rl_no_fees["sl_exits"]},
        {"Metric": "Horizon Exits", "With Fees": rl_fees["horizon_exits"], "No Fees": rl_no_fees["horizon_exits"]},
        {"Metric": "Other Exits", "With Fees": rl_fees["other_exits"], "No Fees": rl_no_fees["other_exits"]},
    ])
    st.markdown("#### Trade Simulation")
    st.dataframe(trade_df, use_container_width=True, hide_index=True)

    st.markdown("#### RL Preview")
    preview_cols = [
        "timestamp",
        "actual",
        "xgb_pred",
        "lstm_pred",
        "xgb_p_up",
        "lstm_p_up",
        "close",
        "atr",
    ]
    preview_cols = [c for c in preview_cols if c in results["merged_df"].columns]
    st.dataframe(results["merged_df"][preview_cols].head(50), use_container_width=True, hide_index=True)

    st.markdown("#### Saved / Resolved Paths")
    paths_df = pd.DataFrame([
        {"Artifact": "XGB Artifacts Path", "Value": str(cfg["xgb_artifacts_path"])},
        {"Artifact": "LSTM Artifacts Path", "Value": str(cfg["lstm_artifacts_path"])},
        {"Artifact": "RL Agent Path", "Value": str(cfg["rl_agent_path"])},
    ])
    st.dataframe(paths_df, use_container_width=True, hide_index=True)


for key, value in {
    "coin": "BTCUSDT",
    "interval": "1h",
    "lstm_eval_results": None,
    "xgb_eval_results": None,
    "ensemble_eval_results": None,
    "rl_eval_results": None,
    "use_latest_data": True,
}.items():
    if key not in st.session_state:
        st.session_state[key] = value


st.title("Model Evaluation Lab")

coin = st.session_state.coin
frequency = st.session_state.interval
default_start = get_default_start_date(frequency)

st.caption(f"Using sidebar selection: {coin} | {frequency}")
st.divider()

with st.expander("Global Evaluation Settings", expanded=False):
    r1c1, r1c2, r1c3, r1c4, r1c5 = st.columns(5)

    with r1c1:
        start_date = st.date_input(
            "Start Date",
            value=date.fromisoformat(default_start),
            key="global_start_date",
        )

    with r1c2:
        if st.session_state.use_latest_data:
            end_date = None
            st.text_input("End Date", value="Latest", disabled=True)
        else:
            end_date = st.date_input(
                "End Date",
                value=date.today(),
                key="global_end_date",
            )

    with r1c3:
        capital = st.number_input("capital", min_value=1.0, value=5000.0, step=100.0, format="%.1f")

    with r1c4:
        rr = st.number_input("rr", min_value=0.10, value=1.25, step=0.05, format="%.2f")

    with r1c5:
        leverage = st.number_input("leverage", min_value=1.0, value=25.0, step=1.0, format="%.1f")

    r2c1, r2c2, r2c3, r2c4, r2c5 = st.columns(5)

    with r2c1:
        st.text_input("Train/Test Split", value="80/20", disabled=True)

    with r2c2:
        st.session_state.use_latest_data = st.checkbox(
            "Use latest available data",
            value=st.session_state.use_latest_data,
        )

    with r2c3:
        max_horizon = st.number_input("max_horizon", min_value=1, value=3, step=1)

    with r2c4:
        fee_bps = st.number_input("fee_bps", min_value=0.0, value=2.0, step=0.5, format="%.1f")

    with r2c5:
        trade_penalty_bps = st.number_input("trade_penalty_bps", min_value=0.0, value=2.0, step=0.5, format="%.1f")

st.divider()
st.markdown("Main Framework")

with st.expander("LSTM", expanded=False):
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
            lstm_margin = st.number_input(
                "margin_threshold",
                min_value=0.00,
                max_value=0.49,
                value=0.03,
                step=0.01,
                format="%.2f",
                help="Symmetric margin around 0.50. Example: 0.03 corresponds to an upper threshold of 0.53.",
            )

        lstm_threshold = margin_to_lstm_threshold(float(lstm_margin))

        st.caption(
            "Note: Default parameters are set to the best-performing configuration identified "
            "through prior grid search tuning on BTCUSDT 1h data."
        )
        st.caption(
            f"LSTM uses margin format. Current margin={float(lstm_margin):.2f}, "
            f"corresponding to threshold={float(lstm_threshold):.2f}."
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
                    0.50, 0.51, 0.52, round(float(lstm_threshold), 2), 0.54, 0.55, 0.56, 0.57, 0.58
                ])))

                _, history, probs_df, metrics_df, _ = train_lstm_model(
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

                y_true = probs_df["y_true"].astype(int).values
                probs = probs_df["p"].astype(float).values

                threshold_summary = build_threshold_summary(
                    y_true=y_true,
                    probs=probs,
                    threshold=float(lstm_threshold),
                )

                st.session_state.lstm_eval_results = {
                    "history": history,
                    "probs_df": probs_df,
                    "metrics_df": metrics_df,
                    "save_paths": save_paths,
                    "threshold_summary": threshold_summary,
                    "chosen_threshold": float(lstm_threshold),
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

                y_true = probs_df["y_true"].astype(int).values
                probs = probs_df["p"].astype(float).values

                threshold_summary = build_threshold_summary(
                    y_true=y_true,
                    probs=probs,
                    threshold=float(lstm_threshold),
                )

                st.session_state.lstm_eval_results = {
                    "history": None,
                    "probs_df": probs_df,
                    "metrics_df": metrics_df,
                    "save_paths": save_paths,
                    "threshold_summary": threshold_summary,
                    "chosen_threshold": float(lstm_threshold),
                }

                st.success("Loaded saved LSTM evaluation results.")

        except Exception as e:
            st.error(f"LSTM evaluation failed: {e}")

    if st.session_state.lstm_eval_results is not None:
        render_lstm_results(st.session_state.lstm_eval_results, coin, frequency)
    else:
        st.info("Use Retrain LSTM to train and save results, or Evaluate LSTM to load saved results.")


with st.expander("XGBoost", expanded=False):
    st.subheader("XGBoost Parameters")

    with st.form("xgb_eval_form"):
        r1c1, r1c2, r1c3, r1c4 = st.columns(4)
        with r1c1:
            n_estimators = st.number_input("n_estimators", min_value=10, value=300, step=10)
        with r1c2:
            learning_rate_xgb = st.number_input(
                "learning_rate",
                min_value=0.001,
                value=0.05,
                step=0.001,
                format="%.3f",
            )
        with r1c3:
            max_depth = st.number_input("max_depth", min_value=1, value=6, step=1)
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

        is_sentiment_available = coin.upper() == "BTCUSDT" and frequency == "1h"

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

    xgb_save_paths = build_xgb_save_paths(coin, frequency)

    if retrain_xgb_clicked:
        try:
            with st.spinner("Retraining XGBoost..."):
                _, _, output_df = train_xgb_model(
                    symbol=coin,
                    resolution=frequency,
                    start_date=start_date.isoformat(),
                    end_date=end_date.isoformat() if isinstance(end_date, date) else None,
                    train_ratio=0.80,
                    n_estimators=int(n_estimators),
                    learning_rate=float(learning_rate_xgb),
                    max_depth=int(max_depth),
                    subsample=float(subsample),
                    colsample_bytree=float(colsample_bytree),
                    decision_boundary=float(decision_boundary),
                    margin_threshold=float(margin_threshold),
                    artifacts_path=xgb_save_paths["artifacts_path"],
                    preds_csv_path=xgb_save_paths["preds_csv_path"],
                    use_sentiment_data=bool(use_sentiment_data),
                )

                output_df = output_df.copy()
                output_df["y_true"] = (output_df["actual_trend"].astype(str).str.lower() == "up").astype(int)

                xgb_summary = build_xgb_eval_summary(
                    y_true=output_df["y_true"].astype(int).values,
                    p_up=output_df["p_up"].astype(float).values,
                    decision_boundary=float(decision_boundary),
                    margin_threshold=float(margin_threshold),
                )

                margin_sweep_df = build_xgb_margin_sweep(
                    y_true=output_df["y_true"].astype(int).values,
                    p_up=output_df["p_up"].astype(float).values,
                    decision_boundary=float(decision_boundary),
                    margins=[0.00, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.15, 0.20],
                )

                st.session_state.xgb_eval_results = {
                    "output_df": output_df,
                    "save_paths": xgb_save_paths,
                    "xgb_summary": xgb_summary,
                    "margin_sweep_df": margin_sweep_df,
                    "chosen_margin": float(margin_threshold),
                    "chosen_boundary": float(decision_boundary),
                }

            st.success("XGBoost retraining and evaluation completed successfully.")

        except Exception as e:
            st.error(f"XGBoost retraining failed: {e}")

    if evaluate_xgb_clicked:
        try:
            preds_path = Path(xgb_save_paths["preds_csv_path"])
            artifacts_path = Path(xgb_save_paths["artifacts_path"])

            if not preds_path.exists() or not artifacts_path.exists():
                st.error("Saved XGBoost evaluation files were not found. Retrain the model first.")
            else:
                output_df = pd.read_csv(preds_path)
                output_df["y_true"] = (output_df["actual_trend"].astype(str).str.lower() == "up").astype(int)

                xgb_summary = build_xgb_eval_summary(
                    y_true=output_df["y_true"].astype(int).values,
                    p_up=output_df["p_up"].astype(float).values,
                    decision_boundary=float(decision_boundary),
                    margin_threshold=float(margin_threshold),
                )

                margin_sweep_df = build_xgb_margin_sweep(
                    y_true=output_df["y_true"].astype(int).values,
                    p_up=output_df["p_up"].astype(float).values,
                    decision_boundary=float(decision_boundary),
                    margins=[0.00, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.15, 0.20],
                )

                st.session_state.xgb_eval_results = {
                    "output_df": output_df,
                    "save_paths": xgb_save_paths,
                    "xgb_summary": xgb_summary,
                    "margin_sweep_df": margin_sweep_df,
                    "chosen_margin": float(margin_threshold),
                    "chosen_boundary": float(decision_boundary),
                }

                st.success("Loaded saved XGBoost evaluation results.")

        except Exception as e:
            st.error(f"XGBoost evaluation failed: {e}")

    if st.session_state.xgb_eval_results is not None:
        render_xgb_results(st.session_state.xgb_eval_results, coin, frequency)
    else:
        st.info("Use Retrain XGBoost to train and save results, or Evaluate XGBoost to load saved results.")


with st.expander("LSTM + XGBoost Ensemble", expanded=False):
    st.subheader("Ensemble Parameters")

    lstm_paths_ensemble = build_lstm_save_paths(coin, frequency)
    xgb_paths_ensemble = build_xgb_save_paths(coin, frequency)

    with st.form("ensemble_eval_form"):
        r1c1, r1c2, r1c3, r1c4 = st.columns(4)

        with r1c1:
            ensemble_weight_xgb = st.number_input(
                "ensemble_weight_xgb",
                min_value=0.00,
                max_value=1.00,
                value=0.80,
                step=0.05,
                format="%.2f",
            )

        with r1c2:
            ensemble_weight_lstm = st.number_input(
                "ensemble_weight_lstm",
                min_value=0.00,
                max_value=1.00,
                value=0.20,
                step=0.05,
                format="%.2f",
            )

        with r1c3:
            ensemble_margin = st.number_input(
                "ensemble_margin_threshold",
                min_value=0.00,
                max_value=0.49,
                value=0.10,
                step=0.01,
                format="%.2f",
                help="Symmetric margin around 0.50. Example: 0.10 gives lower=0.40 and upper=0.60.",
            )

        with r1c4:
            st.text_input("ensemble_bounds", value="auto from margin", disabled=True)

        ensemble_lower, ensemble_upper = margin_to_symmetric_bounds(float(ensemble_margin))

        r2c1 = st.columns(1)[0]
        with r2c1:
            lstm_margin_ensemble = st.number_input(
                "lstm_margin_threshold",
                min_value=0.00,
                max_value=0.49,
                value=0.03,
                step=0.01,
                format="%.2f",
                help="LSTM symmetric margin around 0.50.",
            )

        lstm_threshold_ensemble = margin_to_lstm_threshold(float(lstm_margin_ensemble))

        st.caption(
            "This section does not retrain models. It evaluates the saved LSTM and XGBoost artifacts "
            "for the currently selected coin/frequency combination."
        )
        st.caption(
            "Shared global trading settings applied here: "
            f"capital={float(capital):.1f}, rr={float(rr):.2f}, leverage={float(leverage):.1f}, "
            f"max_horizon={int(max_horizon)}, fee_bps={float(fee_bps):.1f}, "
            f"trade_penalty_bps={float(trade_penalty_bps):.1f}."
        )
        st.caption(
            f"Ensemble margin={float(ensemble_margin):.2f} -> lower={float(ensemble_lower):.2f}, "
            f"upper={float(ensemble_upper):.2f}. "
            f"LSTM margin={float(lstm_margin_ensemble):.2f} -> threshold={float(lstm_threshold_ensemble):.2f}."
        )
        st.caption(
            f"LSTM artifacts: {lstm_paths_ensemble['artifacts_path']}  |  "
            f"XGBoost artifacts: {xgb_paths_ensemble['artifacts_path']}"
        )

        evaluate_ensemble_clicked = st.form_submit_button("Evaluate Ensemble", use_container_width=True)

    if evaluate_ensemble_clicked:
        try:
            if (ensemble_weight_xgb + ensemble_weight_lstm) <= 0:
                st.error("At least one ensemble weight must be greater than zero.")
            elif not Path(lstm_paths_ensemble["artifacts_path"]).exists():
                st.error(
                    f"LSTM artifacts not found for {coin} {frequency}. "
                    "Train/evaluate LSTM first for this combination."
                )
            elif not Path(xgb_paths_ensemble["artifacts_path"]).exists():
                st.error(
                    f"XGBoost artifacts not found for {coin} {frequency}. "
                    "Train/evaluate XGBoost first for this combination."
                )
            else:
                with st.spinner("Evaluating ensemble..."):
                    st.session_state.ensemble_eval_results = evaluate_ensemble_only(
                        symbol=coin,
                        resolution=frequency,
                        start_date=start_date.isoformat(),
                        end_date=end_date.isoformat() if isinstance(end_date, date) else None,
                        train_ratio=0.80,
                        xgb_artifacts_path=str(xgb_paths_ensemble["artifacts_path"]),
                        lstm_artifacts_path=str(lstm_paths_ensemble["artifacts_path"]),
                        lstm_threshold=float(lstm_threshold_ensemble),
                        risk=RiskConfig(
                            capital_usd=float(capital),
                            risk_per_trade=0.02,
                            rr=float(rr),
                            leverage=float(leverage),
                            fee_bps=float(fee_bps),
                            trade_penalty_bps=float(trade_penalty_bps),
                            sl_atr_mult=1.0,
                            min_atr_pct=0.001,
                        ),
                        ensemble_weight_xgb=float(ensemble_weight_xgb),
                        ensemble_weight_lstm=float(ensemble_weight_lstm),
                        ensemble_upper=float(ensemble_upper),
                        ensemble_lower=float(ensemble_lower),
                        max_horizon=int(max_horizon),
                        verbose=False,
                    )

                st.success("Ensemble evaluation completed successfully.")

        except Exception as e:
            st.error(f"Ensemble evaluation failed: {e}")

    if st.session_state.ensemble_eval_results is not None:
        render_ensemble_results(st.session_state.ensemble_eval_results)
    else:
        st.info("This evaluates already-saved LSTM and XGBoost artifacts for the current coin/frequency combination.")


with st.expander("Ensemble + RL Filtering (Main Model)", expanded=False):
    st.subheader("Ensemble + RL Parameters")

    rl_combo_paths = build_combo_artifact_paths(coin, frequency)

    with st.form("rl_eval_form"):
        r1c1, r1c2, r1c3, r1c4 = st.columns(4)

        with r1c1:
            rl_ensemble_weight_xgb = st.number_input(
                "rl_ensemble_weight_xgb",
                min_value=0.00,
                max_value=1.00,
                value=0.80,
                step=0.05,
                format="%.2f",
            )

        with r1c2:
            rl_ensemble_weight_lstm = st.number_input(
                "rl_ensemble_weight_lstm",
                min_value=0.00,
                max_value=1.00,
                value=0.20,
                step=0.05,
                format="%.2f",
            )

        with r1c3:
            rl_ensemble_margin = st.number_input(
                "rl_ensemble_margin_threshold",
                min_value=0.00,
                max_value=0.49,
                value=0.10,
                step=0.01,
                format="%.2f",
                help="Symmetric margin around 0.50. Example: 0.10 gives lower=0.40 and upper=0.60.",
            )

        with r1c4:
            st.text_input("rl_ensemble_bounds", value="auto from margin", disabled=True)

        rl_ensemble_lower, rl_ensemble_upper = margin_to_symmetric_bounds(float(rl_ensemble_margin))

        r2c1, r2c2, r2c3 = st.columns(3)

        with r2c1:
            rl_lstm_margin = st.number_input(
                "rl_lstm_margin_threshold",
                min_value=0.00,
                max_value=0.49,
                value=0.03,
                step=0.01,
                format="%.2f",
                help="LSTM symmetric margin around 0.50.",
            )

        rl_lstm_threshold = margin_to_lstm_threshold(float(rl_lstm_margin))

        with r2c2:
            rl_alpha = st.number_input(
                "alpha",
                min_value=0.001,
                max_value=1.0,
                value=0.20,
                step=0.01,
                format="%.3f",
            )

        with r2c3:
            rl_gamma = st.number_input(
                "gamma",
                min_value=0.00,
                max_value=0.999,
                value=0.95,
                step=0.01,
                format="%.2f",
            )

        r3c1, r3c2, r3c3, r3c4 = st.columns(4)

        with r3c1:
            rl_eps = st.number_input(
                "eps",
                min_value=0.00,
                max_value=1.00,
                value=0.25,
                step=0.01,
                format="%.2f",
            )

        with r3c2:
            rl_episodes = st.number_input("episodes", min_value=1, value=60, step=1)

        with r3c3:
            rl_min_take_visits = st.number_input("min_take_visits", min_value=0, value=20, step=1)

        with r3c4:
            rl_q_take_margin = st.number_input(
                "q_take_margin",
                min_value=0.00,
                value=0.20,
                step=0.01,
                format="%.2f",
            )

        with st.columns(1)[0]:
            rl_skip_reward_scale = st.number_input(
                "skip_reward_scale",
                min_value=0.00,
                value=0.15,
                step=0.01,
                format="%.2f",
            )

        st.caption(
            "This section uses the saved LSTM and XGBoost artifacts for the selected coin/frequency, "
            "then trains or evaluates the RL trade filter on top of the ensemble."
        )
        st.caption(
            "Risk settings use the shared global configuration for capital, rr, leverage, "
            "max_horizon, fee_bps, and trade_penalty_bps. "
            "Fixed values remain: risk_per_trade=0.02, sl_atr_mult=1.0, min_atr_pct=0.001."
        )
        st.caption(
            f"RL ensemble margin={float(rl_ensemble_margin):.2f} -> lower={float(rl_ensemble_lower):.2f}, "
            f"upper={float(rl_ensemble_upper):.2f}. "
            f"LSTM margin={float(rl_lstm_margin):.2f} -> threshold={float(rl_lstm_threshold):.2f}."
        )
        st.caption(
            f"LSTM artifacts: {rl_combo_paths['lstm_artifacts_path']}  |  "
            f"XGBoost artifacts: {rl_combo_paths['xgb_artifacts_path']}  |  "
            f"RL agent: {rl_combo_paths['rl_agent_path']}"
        )

        b1, b2 = st.columns(2)
        with b1:
            train_rl_clicked = st.form_submit_button("Train RL Filter", use_container_width=True)
        with b2:
            eval_rl_clicked = st.form_submit_button("Evaluate RL Filter", use_container_width=True)

    rl_risk = RiskConfig(
        capital_usd=float(capital),
        risk_per_trade=0.02,
        rr=float(rr),
        leverage=float(leverage),
        fee_bps=float(fee_bps),
        trade_penalty_bps=float(trade_penalty_bps),
        sl_atr_mult=1.0,
        min_atr_pct=0.001,
    )

    if train_rl_clicked:
        try:
            if (rl_ensemble_weight_xgb + rl_ensemble_weight_lstm) <= 0:
                st.error("At least one RL ensemble weight must be greater than zero.")
            elif not Path(rl_combo_paths["lstm_artifacts_path"]).exists():
                st.error(
                    f"LSTM artifacts not found for {coin} {frequency}. "
                    "Train/evaluate LSTM first for this combination."
                )
            elif not Path(rl_combo_paths["xgb_artifacts_path"]).exists():
                st.error(
                    f"XGBoost artifacts not found for {coin} {frequency}. "
                    "Train/evaluate XGBoost first for this combination."
                )
            else:
                with st.spinner("Training RL filter..."):
                    train_rl_policy(
                        symbol=coin,
                        resolution=frequency,
                        start_date=start_date.isoformat(),
                        end_date=end_date.isoformat() if isinstance(end_date, date) else None,
                        train_ratio=0.80,
                        xgb_artifacts_path=str(rl_combo_paths["xgb_artifacts_path"]),
                        lstm_artifacts_path=str(rl_combo_paths["lstm_artifacts_path"]),
                        lstm_threshold=float(rl_lstm_threshold),
                        agent_out_path=str(rl_combo_paths["rl_agent_path"]),
                        alpha=float(rl_alpha),
                        gamma=float(rl_gamma),
                        eps=float(rl_eps),
                        episodes=int(rl_episodes),
                        risk=rl_risk,
                        ensemble_weight_xgb=float(rl_ensemble_weight_xgb),
                        ensemble_weight_lstm=float(rl_ensemble_weight_lstm),
                        ensemble_upper=float(rl_ensemble_upper),
                        ensemble_lower=float(rl_ensemble_lower),
                        max_horizon=int(max_horizon),
                        skip_reward_scale=float(rl_skip_reward_scale),
                    )

                st.success("RL filter training completed successfully.")

        except Exception as e:
            st.error(f"RL training failed: {e}")

    if eval_rl_clicked:
        try:
            if (rl_ensemble_weight_xgb + rl_ensemble_weight_lstm) <= 0:
                st.error("At least one RL ensemble weight must be greater than zero.")
            elif not Path(rl_combo_paths["lstm_artifacts_path"]).exists():
                st.error(
                    f"LSTM artifacts not found for {coin} {frequency}. "
                    "Train/evaluate LSTM first for this combination."
                )
            elif not Path(rl_combo_paths["xgb_artifacts_path"]).exists():
                st.error(
                    f"XGBoost artifacts not found for {coin} {frequency}. "
                    "Train/evaluate XGBoost first for this combination."
                )
            elif not Path(rl_combo_paths["rl_agent_path"]).exists():
                st.error(
                    f"RL agent not found for {coin} {frequency}. "
                    "Train the RL filter first for this combination."
                )
            else:
                with st.spinner("Evaluating RL filter..."):
                    st.session_state.rl_eval_results = evaluate_rl_agent(
                        symbol=coin,
                        resolution=frequency,
                        start_date=start_date.isoformat(),
                        end_date=end_date.isoformat() if isinstance(end_date, date) else None,
                        train_ratio=0.80,
                        xgb_artifacts_path=str(rl_combo_paths["xgb_artifacts_path"]),
                        lstm_artifacts_path=str(rl_combo_paths["lstm_artifacts_path"]),
                        lstm_threshold=float(rl_lstm_threshold),
                        rl_agent_path=str(rl_combo_paths["rl_agent_path"]),
                        risk=rl_risk,
                        ensemble_weight_xgb=float(rl_ensemble_weight_xgb),
                        ensemble_weight_lstm=float(rl_ensemble_weight_lstm),
                        ensemble_upper=float(rl_ensemble_upper),
                        ensemble_lower=float(rl_ensemble_lower),
                        max_horizon=int(max_horizon),
                        min_take_visits=int(rl_min_take_visits),
                        q_take_margin=float(rl_q_take_margin),
                    )

                st.success("RL filter evaluation completed successfully.")

        except Exception as e:
            st.error(f"RL evaluation failed: {e}")

    if st.session_state.rl_eval_results is not None:
        render_rl_results(st.session_state.rl_eval_results)
    else:
        st.info("Use Train RL Filter to train and save the RL agent, or Evaluate RL Filter to evaluate a saved RL agent.")