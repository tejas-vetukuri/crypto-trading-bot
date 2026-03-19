from __future__ import annotations


def generate_local_inference(predictions, sentiment=None, recent_trades=None):
    lstm = str(predictions.get("lstm_label", "")).upper().strip()
    xgb = str(predictions.get("xgb_label", "")).upper().strip()
    ensemble = str(predictions.get("ensemble_direction", "")).upper().strip()
    final_action = str(predictions.get("final_action", "")).upper().strip()
    rl_reason = str(predictions.get("rl_reason", "")).lower().strip()

    lstm_conf = predictions.get("lstm_confidence")
    xgb_conf = predictions.get("xgb_confidence")
    ens_conf = predictions.get("ensemble_confidence")
    policy_score = predictions.get("policy_score")

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
    if final_action in ["LONG", "SHORT"]:
        headline = f"RL approves {final_action} setup"
    elif ensemble in ["UP", "DOWN"] and final_action in ["HOLD", "SIDEWAYS", ""]:
        headline = "Directional setup filtered by RL"
    elif ensemble in ["SIDEWAYS", "HOLD", "NEUTRAL", "MIXED", ""]:
        headline = "Low-conviction setup"
    else:
        headline = "Mixed setup"

    # Summary
    if lstm == xgb and lstm not in ["SIDEWAYS", "HOLD", "NEUTRAL", "MIXED", ""]:
        summary_parts.append(
            f"LSTM and XGBoost both lean {lstm}, so the raw directional signal is aligned."
        )
    elif lstm != xgb:
        summary_parts.append(
            f"LSTM ({lstm}) and XGBoost ({xgb}) disagree, which reduces clarity."
        )
    else:
        summary_parts.append(
            "At least one model is neutral, so the setup lacks strong directional agreement."
        )

    if ensemble in ["UP", "DOWN"]:
        summary_parts.append(f"The ensemble resolves to {ensemble}.")
    else:
        summary_parts.append("The ensemble remains sideways/neutral.")

    if final_action in ["LONG", "SHORT"]:
        summary_parts.append(f"RL allows a {final_action} action.")
    else:
        summary_parts.append("RL does not approve an active trade.")

    # LSTM reason
    lstm_reason_parts.append(f"LSTM output is {lstm}.")
    if lstm in ["SIDEWAYS", "HOLD", "NEUTRAL", "MIXED", ""]:
        lstm_reason_parts.append(
            "This suggests the sequence model does not see a strong short-term directional edge."
        )
    else:
        lstm_reason_parts.append(
            "This suggests the sequence model sees short-term momentum or continuation in that direction."
        )

    if lstm_conf is not None:
        if lstm in ["SIDEWAYS", "HOLD", "NEUTRAL", "MIXED"]:
            lstm_reason_parts.append("Confidence is intentionally suppressed for a neutral prediction.")
        elif lstm_conf >= 0.70:
            lstm_reason_parts.append(f"Confidence is relatively strong at {lstm_conf:.2%}.")
        elif lstm_conf >= 0.55:
            lstm_reason_parts.append(f"Confidence is moderate at {lstm_conf:.2%}.")
        else:
            lstm_reason_parts.append(f"Confidence is weak at {lstm_conf:.2%}, so conviction is limited.")

    # XGB reason
    xgb_reason_parts.append(f"XGBoost output is {xgb}.")
    if xgb in ["SIDEWAYS", "HOLD", "NEUTRAL", "MIXED", ""]:
        xgb_reason_parts.append(
            "This suggests the tabular feature set does not currently favor a clear directional trade."
        )
    else:
        xgb_reason_parts.append(
            "This suggests the engineered feature snapshot favors that direction."
        )

    if xgb_conf is not None:
        if xgb in ["SIDEWAYS", "HOLD", "NEUTRAL", "MIXED"]:
            xgb_reason_parts.append("Confidence is intentionally suppressed for a neutral prediction.")
        elif xgb_conf >= 0.70:
            xgb_reason_parts.append(f"Confidence is relatively strong at {xgb_conf:.2%}.")
        elif xgb_conf >= 0.55:
            xgb_reason_parts.append(f"Confidence is moderate at {xgb_conf:.2%}.")
        else:
            xgb_reason_parts.append(f"Confidence is weak at {xgb_conf:.2%}, so the edge is not strong.")

    # Ensemble reason
    if lstm == xgb and lstm not in ["SIDEWAYS", "HOLD", "NEUTRAL", "MIXED", ""]:
        ensemble_reason_parts.append(
            "Because both base models point the same way, the ensemble has cleaner directional agreement."
        )
    elif lstm != xgb:
        ensemble_reason_parts.append(
            "Because the base models disagree, the ensemble result should be treated with more caution."
        )
    else:
        ensemble_reason_parts.append(
            "Neutral output from at least one base model lowers ensemble conviction."
        )

    if ens_conf is not None:
        if ensemble in ["SIDEWAYS", "HOLD", "NEUTRAL", "MIXED", ""]:
            ensemble_reason_parts.append("Ensemble confidence is hidden because the combined output is neutral.")
        elif ens_conf >= 0.70:
            ensemble_reason_parts.append(f"Ensemble confidence is strong at {ens_conf:.2%}.")
        elif ens_conf >= 0.55:
            ensemble_reason_parts.append(f"Ensemble confidence is moderate at {ens_conf:.2%}.")
        else:
            ensemble_reason_parts.append(f"Ensemble confidence is weak at {ens_conf:.2%}.")

    if ensemble in ["UP", "DOWN"]:
        ensemble_reason_parts.append(f"The combined directional outcome is {ensemble}.")
    else:
        ensemble_reason_parts.append("The combined directional outcome is sideways/hold.")

    # RL reason
    if rl_reason == "ensemble_no_setup":
        rl_reason_parts.append(
            "RL is skipped because the ensemble did not produce a valid directional setup."
        )
    elif rl_reason == "rl_agent_not_loaded":
        rl_reason_parts.append(
            "RL filtering is unavailable, so the final display falls back to the ensemble signal."
        )
    else:
        if final_action in ["LONG", "SHORT"]:
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
        oi_change = sentiment.get("oi_change_pct")
        lsr = sentiment.get("lsr_ratio")
        long_ratio = sentiment.get("long_ratio")
        short_ratio = sentiment.get("short_ratio")
        price_change = sentiment.get("price_change_pct")
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
        breakevens = sum(1 for t in recent_trades if str(t.get("outcome", "")).upper() == "BREAKEVEN")
        net_r = sum(float(t.get("r", 0.0)) for t in recent_trades)
        avg_r = net_r / total if total else 0.0
        win_pct = (wins / total * 100.0) if total else 0.0

        trade_reason_parts.append(
            f"Across the last {total} RL replay trades, win rate is {win_pct:.1f}% with net R of {net_r:+.2f}R and average R of {avg_r:+.2f}R."
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
    if final_action in ["LONG", "SHORT"]:
        if ens_conf is not None and ens_conf >= 0.70 and lstm == xgb:
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