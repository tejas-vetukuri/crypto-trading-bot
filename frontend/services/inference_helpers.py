from __future__ import annotations


def predict_xgb_latest(X_latest, xgb_artifacts: dict) -> dict:
    model = xgb_artifacts["model"]
    decision_boundary = float(xgb_artifacts.get("decision_boundary", 0.48))
    margin_threshold = float(xgb_artifacts.get("margin_threshold", 0.10))

    prob_up = float(model.predict_proba(X_latest)[0, 1])
    margin = abs(prob_up - decision_boundary)

    if margin < margin_threshold:
        pred_class = 2
        pred_label = "SIDEWAYS"
        used = 0
        probability = float(decision_boundary)
        confidence = 0.0
    else:
        pred_class = int(prob_up > decision_boundary)  # 1=up, 0=down
        pred_label = "LONG" if pred_class == 1 else "SHORT"
        used = 1
        probability = float(prob_up)
        confidence = float(prob_up if pred_class == 1 else (1.0 - prob_up))

    return {
        "pred_class": pred_class,                  # 0=SHORT, 1=LONG, 2=SIDEWAYS
        "pred_label": pred_label,
        "prob_up": prob_up,
        "confidence": confidence,
        "decision_boundary": decision_boundary,
        "margin_threshold": margin_threshold,
        "margin": float(margin),
        "used": used,
        "probability": probability,
    }


def predict_lstm_latest(X_latest_window, lstm_model, threshold: float = 0.52) -> dict:
    prob_up = float(lstm_model.predict(X_latest_window, verbose=0)[0][0])

    upper = float(threshold)
    lower = float(1.0 - threshold)

    if prob_up >= upper:
        pred_class = 1
        pred_label = "LONG"
        confidence = float(prob_up)
        used = 1
    elif prob_up <= lower:
        pred_class = 0
        pred_label = "SHORT"
        confidence = float(1.0 - prob_up)
        used = 1
    else:
        pred_class = 2
        pred_label = "SIDEWAYS"
        confidence = 0.0
        used = 0

    return {
        "pred_class": pred_class,
        "pred_label": pred_label,
        "prob_up": prob_up,
        "confidence": confidence,
        "threshold": upper,
        "lower_threshold": lower,
        "used": used,
    }


def resolve_ensemble_params(
    xgb_artifacts: dict,
    lstm_artifacts: dict | None = None,
) -> dict:
    lstm_artifacts = lstm_artifacts or {}

    return {
        "xgb_weight": float(xgb_artifacts.get("ensemble_weight_xgb", 0.8)),
        "lstm_weight": float(xgb_artifacts.get("ensemble_weight_lstm", 0.2)),
        "hold_low": float(xgb_artifacts.get("ensemble_lower", 0.4)),
        "hold_high": float(xgb_artifacts.get("ensemble_upper", 0.6)),
        "lstm_threshold": float(
            lstm_artifacts.get(
                "ignore_zone_threshold_for_sideways",
                xgb_artifacts.get("lstm_threshold", 0.53),
            )
        ),
    }