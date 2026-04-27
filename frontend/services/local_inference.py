from __future__ import annotations


NEUTRAL_DIRECTIONS = {"SIDEWAYS", "HOLD", "NEUTRAL", "MIXED", ""}


def normalise_direction(label: object) -> str:
    """
    Model outputs should be represented as UP / DOWN / SIDEWAYS.
    Trading actions should remain LONG / SHORT / HOLD.
    """
    label = str(label or "").upper().strip()

    mapping = {
        "LONG": "UP",
        "BUY": "UP",
        "BULLISH": "UP",
        "BULL": "UP",
        "SHORT": "DOWN",
        "SELL": "DOWN",
        "BEARISH": "DOWN",
        "BEAR": "DOWN",
        "HOLD": "SIDEWAYS",
        "NEUTRAL": "SIDEWAYS",
        "MIXED": "SIDEWAYS",
        "": "SIDEWAYS",
    }

    return mapping.get(label, label)


def normalise_action(label: object) -> str:
    """
    Final RL/trade action should be represented as LONG / SHORT / HOLD.
    """
    label = str(label or "").upper().strip()

    mapping = {
        "UP": "LONG",
        "BUY": "LONG",
        "BULLISH": "LONG",
        "BULL": "LONG",
        "DOWN": "SHORT",
        "SELL": "SHORT",
        "BEARISH": "SHORT",
        "BEAR": "SHORT",
        "SIDEWAYS": "HOLD",
        "NEUTRAL": "HOLD",
        "MIXED": "HOLD",
        "": "HOLD",
    }

    return mapping.get(label, label)


def to_float(value: object) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def safe_sum_r(recent_trades) -> float:
    total = 0.0
    for trade in recent_trades:
        value = to_float(trade.get("r", 0.0))
        total += value if value is not None else 0.0
    return total


def is_directional(label: str) -> bool:
    return label in {"UP", "DOWN"}


def generate_local_inference(predictions, sentiment=None, recent_trades=None):
    predictions = predictions or {}

    lstm = normalise_direction(predictions.get("lstm_label"))
    xgb = normalise_direction(predictions.get("xgb_label"))
    ensemble = normalise_direction(predictions.get("ensemble_direction"))
    final_action = normalise_action(predictions.get("final_action"))
    rl_reason = str(predictions.get("rl_reason", "")).lower().strip()

    # Enforce consistency if RL was skipped or unavailable.
    if rl_reason in {"ensemble_no_setup", "rl_agent_not_loaded"}:
        final_action = "HOLD"

    lstm_conf = to_float(predictions.get("lstm_confidence"))
    xgb_conf = to_float(predictions.get("xgb_confidence"))
    ens_conf = to_float(predictions.get("ensemble_confidence"))
    policy_score = to_float(predictions.get("policy_score"))

    headline = "Mixed setup"
    summary_parts = []
    lstm_reason_parts = []
    xgb_reason_parts = []
    ensemble_reason_parts = []
    rl_reason_parts = []
    sentiment_reason_parts = []
    trade_reason_parts = []

    confidence_note = "Overall conviction is moderate."
    risk_note = (
        "This is a rule-based interpretation of model outputs and recent replay context, "
        "not a guarantee of market outcome."
    )

    # Headline
    if final_action in {"LONG", "SHORT"}:
        headline = f"RL approves {final_action} setup"
    elif is_directional(ensemble) and final_action == "HOLD":
        headline = "Directional setup filtered by RL"
    elif ensemble == "SIDEWAYS":
        headline = "Low-conviction setup"
    else:
        headline = "Mixed setup"

    # Summary
    if lstm == xgb and is_directional(lstm):
        summary_parts.append(
            f"LSTM and XGBoost both lean {lstm}, so the raw directional signal is aligned."
        )
    elif is_directional(lstm) and is_directional(xgb) and lstm != xgb:
        summary_parts.append(
            f"LSTM ({lstm}) and XGBoost ({xgb}) disagree, which reduces clarity."
        )
    else:
        summary_parts.append(
            "At least one model is sideways, so the setup lacks strong directional agreement."
        )

    if ensemble == "UP":
        summary_parts.append("The ensemble resolves to UP.")
    elif ensemble == "DOWN":
        summary_parts.append("The ensemble resolves to DOWN.")
    else:
        summary_parts.append("The ensemble remains SIDEWAYS.")

    if final_action in {"LONG", "SHORT"}:
        summary_parts.append(f"RL allows a {final_action} action.")
    else:
        summary_parts.append("RL does not approve an active trade.")

    # LSTM reason
    lstm_reason_parts.append(f"LSTM output is {lstm}.")
    if lstm == "SIDEWAYS":
        lstm_reason_parts.append(
            "This suggests the sequence model does not see a strong short-term directional edge."
        )
    elif lstm == "UP":
        lstm_reason_parts.append(
            "This suggests the sequence model sees short-term upward momentum or continuation."
        )
    elif lstm == "DOWN":
        lstm_reason_parts.append(
            "This suggests the sequence model sees short-term downward momentum or continuation."
        )
    else:
        lstm_reason_parts.append(
            "This output is not recognised as a standard UP, DOWN, or SIDEWAYS direction."
        )

    if lstm_conf is not None:
        if lstm == "SIDEWAYS":
            lstm_reason_parts.append(
                "Confidence is not emphasised because the prediction is sideways."
            )
        elif is_directional(lstm):
            if lstm_conf >= 0.70:
                lstm_reason_parts.append(f"Confidence is relatively strong at {lstm_conf:.2%}.")
            elif lstm_conf >= 0.55:
                lstm_reason_parts.append(f"Confidence is moderate at {lstm_conf:.2%}.")
            else:
                lstm_reason_parts.append(
                    f"Confidence is weak at {lstm_conf:.2%}, so conviction is limited."
                )

    # XGBoost reason
    xgb_reason_parts.append(f"XGBoost output is {xgb}.")
    if xgb == "SIDEWAYS":
        xgb_reason_parts.append(
            "This suggests the tabular feature set does not currently favour a clear directional trade."
        )
    elif xgb == "UP":
        xgb_reason_parts.append(
            "This suggests the engineered feature snapshot favours upward movement."
        )
    elif xgb == "DOWN":
        xgb_reason_parts.append(
            "This suggests the engineered feature snapshot favours downward movement."
        )
    else:
        xgb_reason_parts.append(
            "This output is not recognised as a standard UP, DOWN, or SIDEWAYS direction."
        )

    if xgb_conf is not None:
        if xgb == "SIDEWAYS":
            xgb_reason_parts.append(
                "Confidence is not emphasised because the prediction is sideways."
            )
        elif is_directional(xgb):
            if xgb_conf >= 0.70:
                xgb_reason_parts.append(f"Confidence is relatively strong at {xgb_conf:.2%}.")
            elif xgb_conf >= 0.55:
                xgb_reason_parts.append(f"Confidence is moderate at {xgb_conf:.2%}.")
            else:
                xgb_reason_parts.append(
                    f"Confidence is weak at {xgb_conf:.2%}, so the edge is not strong."
                )

    # Ensemble reason
    if lstm == xgb and is_directional(lstm):
        ensemble_reason_parts.append(
            "Because both base models point the same way, the ensemble has cleaner directional agreement."
        )
    elif is_directional(lstm) and is_directional(xgb) and lstm != xgb:
        ensemble_reason_parts.append(
            "Because the base models disagree, the ensemble result should be treated with more caution."
        )
    else:
        ensemble_reason_parts.append(
            "Sideways output from at least one base model lowers ensemble conviction."
        )

    if ens_conf is not None:
        if ensemble == "SIDEWAYS":
            ensemble_reason_parts.append(
                "Ensemble confidence is not emphasised because the combined output is sideways."
            )
        elif is_directional(ensemble):
            if ens_conf >= 0.70:
                ensemble_reason_parts.append(f"Ensemble confidence is strong at {ens_conf:.2%}.")
            elif ens_conf >= 0.55:
                ensemble_reason_parts.append(f"Ensemble confidence is moderate at {ens_conf:.2%}.")
            else:
                ensemble_reason_parts.append(f"Ensemble confidence is weak at {ens_conf:.2%}.")

    if ensemble == "UP":
        ensemble_reason_parts.append("The combined directional outcome is UP.")
    elif ensemble == "DOWN":
        ensemble_reason_parts.append("The combined directional outcome is DOWN.")
    else:
        ensemble_reason_parts.append("The combined directional outcome is SIDEWAYS.")

    # RL reason
    if rl_reason == "ensemble_no_setup":
        rl_reason_parts.append(
            "RL is skipped because the ensemble did not produce a valid directional setup."
        )
    elif rl_reason == "rl_agent_not_loaded":
        rl_reason_parts.append(
            "RL filtering is unavailable, so the output should be interpreted without learned RL validation."
        )
    else:
        if final_action in {"LONG", "SHORT"}:
            rl_reason_parts.append(
                f"RL accepts a {final_action} trade, meaning the learned policy considers the current state tradable."
            )
        else:
            rl_reason_parts.append(
                "RL rejects active execution here, meaning the current state is not strong enough for a trade."
            )

        if policy_score is not None:
            rl_reason_parts.append(f"Policy score is {policy_score:.2f}.")

    # Sentiment reason
    if sentiment:
        bias = str(sentiment.get("sentiment", "UNKNOWN")).upper().strip()
        oi_change = to_float(sentiment.get("oi_change_pct"))
        lsr = to_float(sentiment.get("lsr_ratio"))
        long_ratio = to_float(sentiment.get("long_ratio"))
        short_ratio = to_float(sentiment.get("short_ratio"))
        price_change = to_float(sentiment.get("price_change_pct"))
        bias_reason = sentiment.get("reason", "")

        sentiment_reason_parts.append(f"Sentiment block reads {bias}.")

        if oi_change is not None:
            if oi_change > 0:
                sentiment_reason_parts.append(
                    f"Open interest is rising ({oi_change:+.2f}%), which suggests participation is increasing."
                )
            elif oi_change < 0:
                sentiment_reason_parts.append(
                    f"Open interest is falling ({oi_change:+.2f}%), which suggests participation is cooling."
                )
            else:
                sentiment_reason_parts.append("Open interest is broadly flat.")

        if lsr is not None:
            if lsr > 1:
                sentiment_reason_parts.append("Long positioning is stronger than short positioning.")
            elif lsr < 1:
                sentiment_reason_parts.append("Short positioning is stronger than long positioning.")
            else:
                sentiment_reason_parts.append("Long and short positioning are balanced.")

        if long_ratio is not None and short_ratio is not None:
            sentiment_reason_parts.append(
                f"Long share is {long_ratio:.2%} versus short share of {short_ratio:.2%}."
            )

        if price_change is not None:
            sentiment_reason_parts.append(f"Latest price change is {price_change:+.2f}%.")

        if bias_reason:
            sentiment_reason_parts.append(str(bias_reason))
    else:
        sentiment_reason_parts.append(
            "Sentiment data is unavailable, so no external futures-positioning context is added."
        )

    # Trade history reason
    if recent_trades:
        total = len(recent_trades)
        wins = sum(1 for t in recent_trades if str(t.get("outcome", "")).upper() == "WIN")
        losses = sum(1 for t in recent_trades if str(t.get("outcome", "")).upper() == "LOSS")
        breakevens = sum(
            1 for t in recent_trades if str(t.get("outcome", "")).upper() == "BREAKEVEN"
        )

        net_r = safe_sum_r(recent_trades)
        avg_r = net_r / total if total else 0.0
        win_pct = wins / total * 100.0 if total else 0.0

        trade_reason_parts.append(
            f"Across the last {total} RL replay trades, win rate is {win_pct:.1f}% "
            f"with net R of {net_r:+.2f}R and average R of {avg_r:+.2f}R."
        )
        trade_reason_parts.append(
            f"Resolved outcomes: {wins} wins, {losses} losses, {breakevens} breakevens."
        )

        if net_r > 0 and win_pct >= 50:
            trade_reason_parts.append(
                "Recent replay performance supports the policy, although replay is still only contextual evidence."
            )
        elif net_r < 0:
            trade_reason_parts.append(
                "Recent replay performance weakens confidence in the current policy state."
            )
        else:
            trade_reason_parts.append(
                "Recent replay performance is mixed, so it should not materially raise conviction."
            )
    else:
        trade_reason_parts.append("No recent RL replay trades are available for contextual validation.")

    # Confidence note
    if final_action in {"LONG", "SHORT"}:
        if ens_conf is not None and ens_conf >= 0.70 and lstm == xgb and is_directional(lstm):
            confidence_note = (
                "Overall conviction is relatively strong because model agreement and ensemble confidence both support the setup."
            )
        elif ens_conf is not None and ens_conf >= 0.55:
            confidence_note = (
                "Overall conviction is moderate: there is enough alignment for a setup, but not overwhelming strength."
            )
        else:
            confidence_note = (
                "Overall conviction is modest because RL allows the trade, but model confidence is not especially strong."
            )
    else:
        confidence_note = (
            "Overall conviction is low because the pipeline does not currently justify an active trade."
        )

    return {
        "headline": headline,
        "summary": " ".join(summary_parts),
        "lstm_reason": " ".join(lstm_reason_parts),
        "xgb_reason": " ".join(xgb_reason_parts),
        "ensemble_reason": " ".join(ensemble_reason_parts),
        "rl_reason": " ".join(rl_reason_parts),
        "sentiment_reason": " ".join(sentiment_reason_parts),
        "trade_history_reason": " ".join(trade_reason_parts),
        "confidence_note": confidence_note,
        "risk_note": risk_note,
    }