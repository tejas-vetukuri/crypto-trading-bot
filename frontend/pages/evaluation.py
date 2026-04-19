from __future__ import annotations

from datetime import date

import streamlit as st

from frontend.services.evaluation_helpers import (
    DEFAULTS,
    init_session_state,
    margin_to_lstm_threshold,
    render_ensemble_results,
    render_lstm_results,
    render_rl_results,
    render_xgb_results,
    run_ensemble_evaluate,
    run_lstm_evaluate,
    run_lstm_retrain,
    run_rl_evaluate,
    run_xgb_evaluate,
    run_xgb_retrain,
    train_rl_filter,
)
from models.lstm.lstm import build_lstm_save_paths, get_default_start_date, train_lstm_model
from models.rl.rl_ensemble import build_combo_artifact_paths
from models.xgboost.xgb import build_xgb_save_paths, train_xgb_model


def fmt_float(value: float) -> str:
    return f"{float(value):.2f}"


def caption_fixed_defaults(**kwargs) -> str:
    parts = [f"{key}={value}" for key, value in kwargs.items()]
    return f"Only selected controls are exposed. Fixed defaults: {', '.join(parts)}."


def caption_decision_logic(
    decision_boundary: float,
    margin: float,
    lstm_margin: float | None = None,
    lstm_threshold: float | None = None,
) -> str:
    lower = float(decision_boundary) - float(margin)
    upper = float(decision_boundary) + float(margin)

    text = (
        f"Decision boundary={fmt_float(decision_boundary)}, "
        f"margin={fmt_float(margin)} \u2192 "
        f"lower={fmt_float(lower)}, upper={fmt_float(upper)}."
    )

    if lstm_margin is not None and lstm_threshold is not None:
        text += (
            f" LSTM margin={fmt_float(lstm_margin)} \u2192 "
            f"threshold={fmt_float(lstm_threshold)}."
        )

    return text


def caption_lstm_logic(lstm_margin: float, lstm_threshold: float) -> str:
    return (
        f"LSTM margin={fmt_float(lstm_margin)} \u2192 "
        f"threshold={fmt_float(lstm_threshold)}."
    )


def caption_artifacts(**kwargs) -> str:
    parts = [f"{key}={value}" for key, value in kwargs.items()]
    return f"Artifacts: {' | '.join(parts)}"


st.set_page_config(page_title="Model Evaluation Lab", layout="wide")
init_session_state()

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
            end_date = st.date_input("End Date", value=date.today(), key="global_end_date")

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
        trade_penalty_bps = st.number_input(
            "trade_penalty_bps",
            min_value=0.0,
            value=2.0,
            step=0.5,
            format="%.1f",
        )

st.divider()
st.markdown("Main Framework")

with st.expander("LSTM", expanded=False):
    st.subheader("LSTM Parameters")

    lstm_paths = build_lstm_save_paths(coin, frequency)

    with st.form("lstm_eval_form"):
        r1c1, r1c2, r1c3 = st.columns(3)
        with r1c1:
            x_window_size = st.number_input("x_window_size", min_value=1, value=100, step=1)
        with r1c2:
            epochs = st.number_input("epochs", min_value=1, value=10, step=1)
        with r1c3:
            lstm_units = st.number_input("lstm_units", min_value=1, value=100, step=1)

        r2c1, r2c2, r2c3 = st.columns(3)
        with r2c1:
            learning_rate = st.number_input(
                "learning_rate",
                min_value=0.0001,
                value=0.0010,
                step=0.0001,
                format="%.4f",
            )
        with r2c2:
            early_stopping_patience = st.number_input(
                "early_stopping_patience",
                min_value=1,
                value=3,
                step=1,
            )
        with r2c3:
            lstm_margin_threshold = st.number_input(
                "margin_threshold",
                min_value=0.00,
                max_value=0.49,
                value=0.08,
                step=0.01,
                format="%.2f",
            )

        lstm_threshold = margin_to_lstm_threshold(float(lstm_margin_threshold))

        st.caption(
            caption_fixed_defaults(
                batch_size=DEFAULTS["lstm_batch_size"],
                dropout=f"{DEFAULTS['lstm_dropout_rate']:.2f}",
                validation_split=f"{DEFAULTS['validation_split']:.2f}",
            )
        )
        st.caption(
            caption_lstm_logic(
                lstm_margin=float(lstm_margin_threshold),
                lstm_threshold=float(lstm_threshold),
            )
        )
        st.caption(
            caption_artifacts(
                LSTM=lstm_paths["artifacts_path"],
            )
        )

        b1, b2 = st.columns(2)
        with b1:
            retrain_clicked = st.form_submit_button("Retrain LSTM", use_container_width=True)
        with b2:
            evaluate_clicked = st.form_submit_button("Evaluate LSTM", use_container_width=True)

    if retrain_clicked:
        try:
            with st.spinner("Retraining LSTM..."):
                st.session_state.lstm_eval_results = run_lstm_retrain(
                    coin=coin,
                    frequency=frequency,
                    start_date=start_date.isoformat(),
                    end_date=end_date.isoformat() if isinstance(end_date, date) else None,
                    x_window_size=int(x_window_size),
                    epochs=int(epochs),
                    lstm_units=int(lstm_units),
                    learning_rate=float(learning_rate),
                    early_stopping_patience=int(early_stopping_patience),
                    margin_threshold=float(lstm_margin_threshold),
                    train_lstm_model=train_lstm_model,
                )
            st.success("LSTM retraining and evaluation completed successfully.")
        except Exception as e:
            st.error(f"LSTM retraining failed: {e}")

    if evaluate_clicked:
        try:
            st.session_state.lstm_eval_results = run_lstm_evaluate(
                coin=coin,
                frequency=frequency,
                margin_threshold=float(lstm_margin_threshold),
            )
            st.success("Loaded saved LSTM evaluation results.")
        except Exception as e:
            st.error(f"LSTM evaluation failed: {e}")

    if st.session_state.lstm_eval_results is not None:
        render_lstm_results(st.session_state.lstm_eval_results, coin, frequency)
    else:
        st.info("Use Retrain LSTM to train and save results, or Evaluate LSTM to load saved results.")


with st.expander("XGBoost", expanded=False):
    st.subheader("XGBoost Parameters")

    xgb_paths = build_xgb_save_paths(coin, frequency)

    with st.form("xgb_eval_form"):
        r1c1, r1c2, r1c3 = st.columns(3)
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

        r2c1, r2c2, r2c3 = st.columns(3)
        with r2c1:
            decision_boundary = st.number_input(
                "decision_boundary",
                min_value=0.01,
                max_value=0.99,
                value=0.48,
                step=0.01,
                format="%.2f",
            )
        with r2c2:
            margin_threshold = st.number_input(
                "margin_threshold",
                min_value=0.00,
                max_value=0.49,
                value=0.10,
                step=0.01,
                format="%.2f",
            )

        is_sentiment_available = coin.upper() == "BTCUSDT" and frequency == "1h"
        with r2c3:
            use_sentiment_data = st.checkbox(
                "use_sentiment_data",
                value=False,
                disabled=not is_sentiment_available,
                help="Available only for BTC 1h. Adds Bybit derivatives features (OI + LSR).",
            )

        st.caption(
            caption_fixed_defaults(
                subsample=f"{DEFAULTS['xgb_subsample']:.2f}",
                colsample_bytree=f"{DEFAULTS['xgb_colsample_bytree']:.2f}",
            )
        )
        st.caption(
            caption_decision_logic(
                decision_boundary=float(decision_boundary),
                margin=float(margin_threshold),
            )
        )
        st.caption(
            caption_artifacts(
                XGBoost=xgb_paths["artifacts_path"],
            )
        )

        if not is_sentiment_available:
            st.caption("Sentiment features are available only for BTCUSDT 1h.")

        b1, b2 = st.columns(2)
        with b1:
            retrain_xgb_clicked = st.form_submit_button("Retrain XGBoost", use_container_width=True)
        with b2:
            evaluate_xgb_clicked = st.form_submit_button("Evaluate XGBoost", use_container_width=True)

    if retrain_xgb_clicked:
        try:
            with st.spinner("Retraining XGBoost..."):
                st.session_state.xgb_eval_results = run_xgb_retrain(
                    coin=coin,
                    frequency=frequency,
                    start_date=start_date.isoformat(),
                    end_date=end_date.isoformat() if isinstance(end_date, date) else None,
                    n_estimators=int(n_estimators),
                    learning_rate=float(learning_rate_xgb),
                    max_depth=int(max_depth),
                    decision_boundary=float(decision_boundary),
                    margin_threshold=float(margin_threshold),
                    use_sentiment_data=bool(use_sentiment_data),
                    train_xgb_model=train_xgb_model,
                )
            st.success("XGBoost retraining and evaluation completed successfully.")
        except Exception as e:
            st.error(f"XGBoost retraining failed: {e}")

    if evaluate_xgb_clicked:
        try:
            st.session_state.xgb_eval_results = run_xgb_evaluate(
                coin=coin,
                frequency=frequency,
                decision_boundary=float(decision_boundary),
                margin_threshold=float(margin_threshold),
            )
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
        r1c1, r1c2, r1c3, r1c4, r1c5 = st.columns(5)
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
            ensemble_decision_boundary = st.number_input(
                "ensemble_decision_boundary",
                min_value=0.01,
                max_value=0.99,
                value=0.48,
                step=0.01,
                format="%.2f",
            )
        with r1c4:
            ensemble_margin_threshold = st.number_input(
                "ensemble_margin_threshold",
                min_value=0.00,
                max_value=0.49,
                value=0.10,
                step=0.01,
                format="%.2f",
            )
        with r1c5:
            lstm_margin_threshold_ensemble = st.number_input(
                "lstm_margin_threshold",
                min_value=0.00,
                max_value=0.49,
                value=0.08,
                step=0.01,
                format="%.2f",
            )

        lstm_threshold_ensemble = margin_to_lstm_threshold(float(lstm_margin_threshold_ensemble))

        st.caption(
            caption_decision_logic(
                decision_boundary=float(ensemble_decision_boundary),
                margin=float(ensemble_margin_threshold),
                lstm_margin=float(lstm_margin_threshold_ensemble),
                lstm_threshold=float(lstm_threshold_ensemble),
            )
        )
        st.caption(
            caption_artifacts(
                LSTM=lstm_paths_ensemble["artifacts_path"],
                XGBoost=xgb_paths_ensemble["artifacts_path"],
            )
        )

        evaluate_ensemble_clicked = st.form_submit_button("Evaluate Ensemble", use_container_width=True)

    if evaluate_ensemble_clicked:
        try:
            with st.spinner("Evaluating ensemble..."):
                st.session_state.ensemble_eval_results = run_ensemble_evaluate(
                    coin=coin,
                    frequency=frequency,
                    start_date=start_date.isoformat(),
                    end_date=end_date.isoformat() if isinstance(end_date, date) else None,
                    capital=float(capital),
                    rr=float(rr),
                    leverage=float(leverage),
                    fee_bps=float(fee_bps),
                    trade_penalty_bps=float(trade_penalty_bps),
                    max_horizon=int(max_horizon),
                    ensemble_weight_xgb=float(ensemble_weight_xgb),
                    ensemble_weight_lstm=float(ensemble_weight_lstm),
                    ensemble_decision_boundary=float(ensemble_decision_boundary),
                    ensemble_margin_threshold=float(ensemble_margin_threshold),
                    lstm_margin_threshold=float(lstm_margin_threshold_ensemble),
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
        r1c1, r1c2, r1c3, r1c4, r1c5 = st.columns(5)
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
            rl_ensemble_decision_boundary = st.number_input(
                "rl_ensemble_decision_boundary",
                min_value=0.01,
                max_value=0.99,
                value=0.48,
                step=0.01,
                format="%.2f",
            )
        with r1c4:
            rl_ensemble_margin_threshold = st.number_input(
                "rl_ensemble_margin_threshold",
                min_value=0.00,
                max_value=0.49,
                value=0.10,
                step=0.01,
                format="%.2f",
            )
        with r1c5:
            rl_lstm_margin_threshold = st.number_input(
                "rl_lstm_margin_threshold",
                min_value=0.00,
                max_value=0.49,
                value=0.08,
                step=0.01,
                format="%.2f",
            )

        r2c1, r2c2, r2c3, r2c4 = st.columns(4)
        with r2c1:
            episodes = st.number_input("episodes", min_value=1, value=30, step=1)
        with r2c2:
            skip_reward_scale = st.number_input(
                "skip_reward_scale",
                min_value=0.00,
                value=0.15,
                step=0.01,
                format="%.2f",
            )
        with r2c3:
            min_take_visits = st.number_input("min_take_visits", min_value=0, value=20, step=1)
        with r2c4:
            q_take_margin = st.number_input(
                "q_take_margin",
                min_value=0.00,
                value=0.27,
                step=0.01,
                format="%.2f",
            )

        rl_lstm_threshold = margin_to_lstm_threshold(float(rl_lstm_margin_threshold))

        st.caption(
            caption_fixed_defaults(
                alpha=f"{DEFAULTS['rl_alpha']:.2f}",
                gamma=f"{DEFAULTS['rl_gamma']:.2f}",
                eps=f"{DEFAULTS['rl_eps']:.2f}",
            )
        )
        st.caption(
            caption_decision_logic(
                decision_boundary=float(rl_ensemble_decision_boundary),
                margin=float(rl_ensemble_margin_threshold),
                lstm_margin=float(rl_lstm_margin_threshold),
                lstm_threshold=float(rl_lstm_threshold),
            )
        )
        st.caption(
            caption_artifacts(
                LSTM=rl_combo_paths["lstm_artifacts_path"],
                XGBoost=rl_combo_paths["xgb_artifacts_path"],
                RL=rl_combo_paths["rl_agent_path"],
            )
        )

        b1, b2 = st.columns(2)
        with b1:
            train_rl_clicked = st.form_submit_button("Train RL Filter", use_container_width=True)
        with b2:
            eval_rl_clicked = st.form_submit_button("Evaluate RL Filter", use_container_width=True)

    if train_rl_clicked:
        try:
            with st.spinner("Training RL filter..."):
                train_rl_filter(
                    coin=coin,
                    frequency=frequency,
                    start_date=start_date.isoformat(),
                    end_date=end_date.isoformat() if isinstance(end_date, date) else None,
                    capital=float(capital),
                    rr=float(rr),
                    leverage=float(leverage),
                    fee_bps=float(fee_bps),
                    trade_penalty_bps=float(trade_penalty_bps),
                    max_horizon=int(max_horizon),
                    rl_ensemble_weight_xgb=float(rl_ensemble_weight_xgb),
                    rl_ensemble_weight_lstm=float(rl_ensemble_weight_lstm),
                    rl_ensemble_decision_boundary=float(rl_ensemble_decision_boundary),
                    rl_ensemble_margin_threshold=float(rl_ensemble_margin_threshold),
                    rl_lstm_margin_threshold=float(rl_lstm_margin_threshold),
                    episodes=int(episodes),
                    skip_reward_scale=float(skip_reward_scale),
                )
            st.success("RL filter training completed successfully.")
        except Exception as e:
            st.error(f"RL training failed: {e}")

    if eval_rl_clicked:
        try:
            with st.spinner("Evaluating RL filter..."):
                st.session_state.rl_eval_results = run_rl_evaluate(
                    coin=coin,
                    frequency=frequency,
                    start_date=start_date.isoformat(),
                    end_date=end_date.isoformat() if isinstance(end_date, date) else None,
                    capital=float(capital),
                    rr=float(rr),
                    leverage=float(leverage),
                    fee_bps=float(fee_bps),
                    trade_penalty_bps=float(trade_penalty_bps),
                    max_horizon=int(max_horizon),
                    rl_ensemble_weight_xgb=float(rl_ensemble_weight_xgb),
                    rl_ensemble_weight_lstm=float(rl_ensemble_weight_lstm),
                    rl_ensemble_decision_boundary=float(rl_ensemble_decision_boundary),
                    rl_ensemble_margin_threshold=float(rl_ensemble_margin_threshold),
                    rl_lstm_margin_threshold=float(rl_lstm_margin_threshold),
                    min_take_visits=int(min_take_visits),
                    q_take_margin=float(q_take_margin),
                )
            st.success("RL filter evaluation completed successfully.")
        except Exception as e:
            st.error(f"RL evaluation failed: {e}")

    if st.session_state.rl_eval_results is not None:
        render_rl_results(st.session_state.rl_eval_results)
    else:
        st.info("Use Train RL Filter to train and save the RL agent, or Evaluate RL Filter to evaluate a saved RL agent.")