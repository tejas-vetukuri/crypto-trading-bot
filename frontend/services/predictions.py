from __future__ import annotations

from services.live_inference import run_live_inference_from_df
from services.market_data import fetch_klines
from services.model_loader import load_live_artifacts


def fetch_live_predictions(
    symbol: str,
    interval: str,
    risk=None,
) -> dict:
    artifacts = load_live_artifacts(symbol, interval)
    lstm_lookback = int(
        artifacts["lstm_artifacts"].get(
            "lookback",
            artifacts["lstm_artifacts"].get("x_window_size", 50),
        )
    )

    min_rows = max(250, lstm_lookback + 100)

    df = fetch_klines(
        symbol=symbol,
        interval=interval,
        limit=min_rows,
    )

    result = run_live_inference_from_df(
        df=df,
        symbol=symbol,
        interval=interval,
        risk=risk,
    )

    rl_result = result.get("rl", {})
    risk_result = result.get("risk", {})
    xgb_result = result.get("xgb", {})
    lstm_result = result.get("lstm", {})
    ensemble_result = result.get("ensemble", {})

    return {
        "symbol": result["symbol"],
        "interval": result["interval"],
        "timestamp": result["timestamp"],
        "entry": result["entry"],
        "close_price": result["close_price"],

        "xgb_label": xgb_result.get("pred_label"),
        "xgb_prob_up": xgb_result.get("p_up"),
        "xgb_confidence": xgb_result.get("confidence"),

        "lstm_label": lstm_result.get("pred_label"),
        "lstm_prob_up": lstm_result.get("p_up"),
        "lstm_confidence": lstm_result.get("confidence"),

        "ensemble_prob_up": ensemble_result.get("p_up"),
        "ensemble_direction": ensemble_result.get("direction"),
        "ensemble_confidence": ensemble_result.get("confidence"),

        "rl_used": rl_result.get("used_rl"),
        "rl_decision": rl_result.get("decision"),
        "rl_reason": rl_result.get("reason"),
        "rl_state": rl_result.get("state"),
        "q_skip": rl_result.get("q_skip"),
        "q_take": rl_result.get("q_take"),
        "q_gap_take_minus_skip": rl_result.get("q_gap_take_minus_skip"),
        "policy_score": rl_result.get("policy_score"),
        "take_visits": rl_result.get("take_visits"),
        "skip_visits": rl_result.get("skip_visits"),

        "final_action": result.get("final_action"),
        "confidence": result.get("confidence"),

        "stop_loss": risk_result.get("stop_loss"),
        "take_profit": risk_result.get("take_profit"),
        "position_size_usd": risk_result.get("position_size_usd"),
        "atr": risk_result.get("atr"),
        "atr_pct": risk_result.get("atr_pct"),
        "leverage": risk_result.get("leverage"),
    }