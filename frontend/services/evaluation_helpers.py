from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import roc_auc_score

from models.baselines.randomforest import build_rf_save_paths
from models.baselines.simple_rnn import build_simple_rnn_save_paths
from models.lstm.lstm import build_lstm_save_paths
from models.lstm.train_eval_lstm import build_distribution_tables, build_threshold_summary
from models.rl.eval_ensemble_only import evaluate_ensemble_only
from models.rl.eval_rl import evaluate_rl_agent
from models.rl.rl_ensemble import RiskConfig, build_combo_artifact_paths, train_rl_policy
from models.xgboost.train_eval_xgb import (
    build_xgb_distribution_tables,
    build_xgb_eval_summary,
    build_xgb_margin_sweep,
)
from models.xgboost.xgb import build_xgb_save_paths

DEFAULTS = {
    "train_ratio": 0.80,
    "validation_split": 0.05,
    "lstm_batch_size": 64,
    "lstm_dropout_rate": 0.20,
    "xgb_subsample": 0.80,
    "xgb_colsample_bytree": 0.80,
    "rl_alpha": 0.20,
    "rl_gamma": 0.95,
    "rl_eps": 0.25,
    "risk_per_trade": 0.02,
    "sl_atr_mult": 1.0,
    "min_atr_pct": 0.001,
}


SESSION_DEFAULTS = {
    "coin": "BTCUSDT",
    "interval": "1h",
    "lstm_eval_results": None,
    "xgb_eval_results": None,
    "ensemble_eval_results": None,
    "rl_eval_results": None,
    "use_latest_data": True,
}


def init_session_state() -> None:
    for key, value in SESSION_DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = value


def safe_auc(y_true: np.ndarray, probs: np.ndarray) -> float | None:
    try:
        return float(roc_auc_score(y_true, probs))
    except Exception:
        return None


def safe_binary_accuracy(y_true: np.ndarray, preds: np.ndarray) -> float | None:
    try:
        return float((y_true == preds).mean()) if len(y_true) else None
    except Exception:
        return None


def margin_to_lstm_threshold(margin: float, center: float = 0.5) -> float:
    return center + float(margin)


def margin_to_symmetric_bounds(margin: float, center: float = 0.5) -> tuple[float, float]:
    margin = float(margin)
    return center - margin, center + margin


def load_simple_rnn_raw_accuracy(symbol: str, resolution: str) -> float | None:
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


def load_rf_raw_accuracy(symbol: str, resolution: str) -> float | None:
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


def build_risk_config(
    capital: float,
    rr: float,
    leverage: float,
    fee_bps: float,
    trade_penalty_bps: float,
) -> RiskConfig:
    return RiskConfig(
        capital_usd=float(capital),
        risk_per_trade=float(DEFAULTS["risk_per_trade"]),
        rr=float(rr),
        leverage=float(leverage),
        fee_bps=float(fee_bps),
        trade_penalty_bps=float(trade_penalty_bps),
        sl_atr_mult=float(DEFAULTS["sl_atr_mult"]),
        min_atr_pct=float(DEFAULTS["min_atr_pct"]),
    )


def check_required_paths(path_map: dict[str, Any]) -> tuple[bool, str | None]:
    for label, path_value in path_map.items():
        if not Path(path_value).exists():
            return False, f"{label} not found: {path_value}"
    return True, None


def run_lstm_retrain(
    *,
    coin: str,
    frequency: str,
    start_date: str,
    end_date: str | None,
    x_window_size: int,
    epochs: int,
    lstm_units: int,
    learning_rate: float,
    early_stopping_patience: int,
    margin_threshold: float,
    train_lstm_model,
) -> dict[str, Any]:
    save_paths = build_lstm_save_paths(coin, frequency)
    threshold = margin_to_lstm_threshold(margin_threshold)
    thresholds = tuple(sorted(set([
        0.50, 0.51, 0.52, round(float(threshold), 2), 0.54, 0.55, 0.56, 0.57, 0.58,
    ])))

    _, history, probs_df, metrics_df, _ = train_lstm_model(
        symbol=coin,
        resolution=frequency,
        start_date=start_date,
        end_date=end_date,
        x_window_size=int(x_window_size),
        epochs=int(epochs),
        batch_size=int(DEFAULTS["lstm_batch_size"]),
        train_ratio=float(DEFAULTS["train_ratio"]),
        validation_split=float(DEFAULTS["validation_split"]),
        learning_rate=float(learning_rate),
        lstm_units=int(lstm_units),
        dropout_rate=float(DEFAULTS["lstm_dropout_rate"]),
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
    threshold_summary = build_threshold_summary(y_true=y_true, probs=probs, threshold=float(threshold))

    return {
        "history": history,
        "probs_df": probs_df,
        "metrics_df": metrics_df,
        "save_paths": save_paths,
        "threshold_summary": threshold_summary,
        "chosen_threshold": float(threshold),
        "chosen_margin": float(margin_threshold),
    }


def run_lstm_evaluate(
    *,
    coin: str,
    frequency: str,
    margin_threshold: float,
) -> dict[str, Any]:
    save_paths = build_lstm_save_paths(coin, frequency)
    probs_path = Path(save_paths["test_probs_path"])
    metrics_path = Path(save_paths["metrics_path"])

    if not probs_path.exists() or not metrics_path.exists():
        raise FileNotFoundError("Saved LSTM evaluation files were not found. Retrain the model first.")

    probs_df = pd.read_csv(probs_path)
    metrics_df = pd.read_csv(metrics_path)
    threshold = margin_to_lstm_threshold(margin_threshold)

    y_true = probs_df["y_true"].astype(int).values
    probs = probs_df["p"].astype(float).values
    threshold_summary = build_threshold_summary(y_true=y_true, probs=probs, threshold=float(threshold))

    return {
        "history": None,
        "probs_df": probs_df,
        "metrics_df": metrics_df,
        "save_paths": save_paths,
        "threshold_summary": threshold_summary,
        "chosen_threshold": float(threshold),
        "chosen_margin": float(margin_threshold),
    }


def run_xgb_retrain(
    *,
    coin: str,
    frequency: str,
    start_date: str,
    end_date: str | None,
    n_estimators: int,
    learning_rate: float,
    max_depth: int,
    decision_boundary: float,
    margin_threshold: float,
    use_sentiment_data: bool,
    train_xgb_model,
) -> dict[str, Any]:
    save_paths = build_xgb_save_paths(coin, frequency)

    _, _, output_df = train_xgb_model(
        symbol=coin,
        resolution=frequency,
        start_date=start_date,
        end_date=end_date,
        train_ratio=float(DEFAULTS["train_ratio"]),
        n_estimators=int(n_estimators),
        learning_rate=float(learning_rate),
        max_depth=int(max_depth),
        subsample=float(DEFAULTS["xgb_subsample"]),
        colsample_bytree=float(DEFAULTS["xgb_colsample_bytree"]),
        decision_boundary=float(decision_boundary),
        margin_threshold=float(margin_threshold),
        artifacts_path=save_paths["artifacts_path"],
        preds_csv_path=save_paths["preds_csv_path"],
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

    return {
        "output_df": output_df,
        "save_paths": save_paths,
        "xgb_summary": xgb_summary,
        "margin_sweep_df": margin_sweep_df,
        "chosen_margin": float(margin_threshold),
        "chosen_boundary": float(decision_boundary),
    }


def run_xgb_evaluate(
    *,
    coin: str,
    frequency: str,
    decision_boundary: float,
    margin_threshold: float,
) -> dict[str, Any]:
    save_paths = build_xgb_save_paths(coin, frequency)
    preds_path = Path(save_paths["preds_csv_path"])
    artifacts_path = Path(save_paths["artifacts_path"])

    if not preds_path.exists() or not artifacts_path.exists():
        raise FileNotFoundError("Saved XGBoost evaluation files were not found. Retrain the model first.")

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

    return {
        "output_df": output_df,
        "save_paths": save_paths,
        "xgb_summary": xgb_summary,
        "margin_sweep_df": margin_sweep_df,
        "chosen_margin": float(margin_threshold),
        "chosen_boundary": float(decision_boundary),
    }


def run_ensemble_evaluate(
    *,
    coin: str,
    frequency: str,
    start_date: str,
    end_date: str | None,
    capital: float,
    rr: float,
    leverage: float,
    fee_bps: float,
    trade_penalty_bps: float,
    max_horizon: int,
    ensemble_weight_xgb: float,
    ensemble_weight_lstm: float,
    ensemble_margin_threshold: float,
    lstm_margin_threshold: float,
) -> dict[str, Any]:
    lstm_paths = build_lstm_save_paths(coin, frequency)
    xgb_paths = build_xgb_save_paths(coin, frequency)

    ok, message = check_required_paths({
        "LSTM artifacts": lstm_paths["artifacts_path"],
        "XGBoost artifacts": xgb_paths["artifacts_path"],
    })
    if not ok:
        raise FileNotFoundError(message)

    if (ensemble_weight_xgb + ensemble_weight_lstm) <= 0:
        raise ValueError("At least one ensemble weight must be greater than zero.")

    ensemble_lower, ensemble_upper = margin_to_symmetric_bounds(ensemble_margin_threshold)
    lstm_threshold = margin_to_lstm_threshold(lstm_margin_threshold)
    risk = build_risk_config(capital, rr, leverage, fee_bps, trade_penalty_bps)

    return evaluate_ensemble_only(
        symbol=coin,
        resolution=frequency,
        start_date=start_date,
        end_date=end_date,
        train_ratio=float(DEFAULTS["train_ratio"]),
        xgb_artifacts_path=str(xgb_paths["artifacts_path"]),
        lstm_artifacts_path=str(lstm_paths["artifacts_path"]),
        lstm_threshold=float(lstm_threshold),
        risk=risk,
        ensemble_weight_xgb=float(ensemble_weight_xgb),
        ensemble_weight_lstm=float(ensemble_weight_lstm),
        ensemble_upper=float(ensemble_upper),
        ensemble_lower=float(ensemble_lower),
        max_horizon=int(max_horizon),
        verbose=False,
    )


def train_rl_filter(
    *,
    coin: str,
    frequency: str,
    start_date: str,
    end_date: str | None,
    capital: float,
    rr: float,
    leverage: float,
    fee_bps: float,
    trade_penalty_bps: float,
    max_horizon: int,
    rl_ensemble_weight_xgb: float,
    rl_ensemble_weight_lstm: float,
    rl_ensemble_margin_threshold: float,
    rl_lstm_margin_threshold: float,
    episodes: int,
    skip_reward_scale: float,
) -> dict[str, Any]:
    combo_paths = build_combo_artifact_paths(coin, frequency)
    ok, message = check_required_paths({
        "LSTM artifacts": combo_paths["lstm_artifacts_path"],
        "XGBoost artifacts": combo_paths["xgb_artifacts_path"],
    })
    if not ok:
        raise FileNotFoundError(message)

    if (rl_ensemble_weight_xgb + rl_ensemble_weight_lstm) <= 0:
        raise ValueError("At least one RL ensemble weight must be greater than zero.")

    rl_ensemble_lower, rl_ensemble_upper = margin_to_symmetric_bounds(rl_ensemble_margin_threshold)
    rl_lstm_threshold = margin_to_lstm_threshold(rl_lstm_margin_threshold)
    rl_risk = build_risk_config(capital, rr, leverage, fee_bps, trade_penalty_bps)

    train_rl_policy(
        symbol=coin,
        resolution=frequency,
        start_date=start_date,
        end_date=end_date,
        train_ratio=float(DEFAULTS["train_ratio"]),
        xgb_artifacts_path=str(combo_paths["xgb_artifacts_path"]),
        lstm_artifacts_path=str(combo_paths["lstm_artifacts_path"]),
        lstm_threshold=float(rl_lstm_threshold),
        agent_out_path=str(combo_paths["rl_agent_path"]),
        alpha=float(DEFAULTS["rl_alpha"]),
        gamma=float(DEFAULTS["rl_gamma"]),
        eps=float(DEFAULTS["rl_eps"]),
        episodes=int(episodes),
        risk=rl_risk,
        ensemble_weight_xgb=float(rl_ensemble_weight_xgb),
        ensemble_weight_lstm=float(rl_ensemble_weight_lstm),
        ensemble_upper=float(rl_ensemble_upper),
        ensemble_lower=float(rl_ensemble_lower),
        max_horizon=int(max_horizon),
        skip_reward_scale=float(skip_reward_scale),
    )

    return combo_paths


def run_rl_evaluate(
    *,
    coin: str,
    frequency: str,
    start_date: str,
    end_date: str | None,
    capital: float,
    rr: float,
    leverage: float,
    fee_bps: float,
    trade_penalty_bps: float,
    max_horizon: int,
    rl_ensemble_weight_xgb: float,
    rl_ensemble_weight_lstm: float,
    rl_ensemble_margin_threshold: float,
    rl_lstm_margin_threshold: float,
    min_take_visits: int,
    q_take_margin: float,
) -> dict[str, Any]:
    combo_paths = build_combo_artifact_paths(coin, frequency)
    ok, message = check_required_paths({
        "LSTM artifacts": combo_paths["lstm_artifacts_path"],
        "XGBoost artifacts": combo_paths["xgb_artifacts_path"],
        "RL agent": combo_paths["rl_agent_path"],
    })
    if not ok:
        raise FileNotFoundError(message)

    if (rl_ensemble_weight_xgb + rl_ensemble_weight_lstm) <= 0:
        raise ValueError("At least one RL ensemble weight must be greater than zero.")

    rl_ensemble_lower, rl_ensemble_upper = margin_to_symmetric_bounds(rl_ensemble_margin_threshold)
    rl_lstm_threshold = margin_to_lstm_threshold(rl_lstm_margin_threshold)
    rl_risk = build_risk_config(capital, rr, leverage, fee_bps, trade_penalty_bps)

    return evaluate_rl_agent(
        symbol=coin,
        resolution=frequency,
        start_date=start_date,
        end_date=end_date,
        train_ratio=float(DEFAULTS["train_ratio"]),
        xgb_artifacts_path=str(combo_paths["xgb_artifacts_path"]),
        lstm_artifacts_path=str(combo_paths["lstm_artifacts_path"]),
        lstm_threshold=float(rl_lstm_threshold),
        rl_agent_path=str(combo_paths["rl_agent_path"]),
        risk=rl_risk,
        ensemble_weight_xgb=float(rl_ensemble_weight_xgb),
        ensemble_weight_lstm=float(rl_ensemble_weight_lstm),
        ensemble_upper=float(rl_ensemble_upper),
        ensemble_lower=float(rl_ensemble_lower),
        max_horizon=int(max_horizon),
        min_take_visits=int(min_take_visits),
        q_take_margin=float(q_take_margin),
    )


def render_lstm_results(results: dict[str, Any], coin: str, frequency: str) -> None:
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
        {"Metric": "LSTM Raw Accuracy", "Value": "—" if lstm_raw_accuracy is None else f"{lstm_raw_accuracy:.4f}"},
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
        st.dataframe(threshold_summary["classification_report_df"], use_container_width=True)

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


def render_xgb_results(results: dict[str, Any], coin: str, frequency: str) -> None:
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
        {"Metric": "XGBoost Raw Accuracy", "Value": "—" if xgb_raw_accuracy is None else f"{xgb_raw_accuracy:.4f}"},
        {"Metric": "Random Forest Accuracy", "Value": "—" if rf_raw_accuracy is None else f"{rf_raw_accuracy:.4f}"},
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
        st.dataframe(xgb_summary["classification_report_df"], use_container_width=True, hide_index=True)

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


def render_ensemble_results(results: dict[str, Any]) -> None:
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
        {"Metric": "Directional Accuracy", "With Fees": f"{trade_with_fees['directional_accuracy']:.4f}", "No Fees": f"{trade_no_fees['directional_accuracy']:.4f}"},
        {"Metric": "Win Rate", "With Fees": f"{trade_with_fees['win_rate']:.4f}", "No Fees": f"{trade_no_fees['win_rate']:.4f}"},
        {"Metric": "Avg Gross R / Trade", "With Fees": f"{trade_with_fees['avg_gross_r_per_trade']:.4f}", "No Fees": f"{trade_no_fees['avg_gross_r_per_trade']:.4f}"},
        {"Metric": "Avg Net R / Trade", "With Fees": f"{trade_with_fees['avg_net_r_per_trade']:.4f}", "No Fees": f"{trade_no_fees['avg_net_r_per_trade']:.4f}"},
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


def render_rl_results(results: dict[str, Any]) -> None:
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
        {"Metric": "Directional Accuracy", "With Fees": f"{rl_fees['directional_accuracy']:.4f}", "No Fees": f"{rl_no_fees['directional_accuracy']:.4f}"},
        {"Metric": "Win Rate", "With Fees": f"{rl_fees['win_rate']:.4f}", "No Fees": f"{rl_no_fees['win_rate']:.4f}"},
        {"Metric": "Avg Gross R / Trade", "With Fees": f"{rl_fees['avg_gross_r_per_trade']:.4f}", "No Fees": f"{rl_no_fees['avg_gross_r_per_trade']:.4f}"},
        {"Metric": "Avg Net R / Trade", "With Fees": f"{rl_fees['avg_net_r_per_trade']:.4f}", "No Fees": f"{rl_no_fees['avg_net_r_per_trade']:.4f}"},
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
