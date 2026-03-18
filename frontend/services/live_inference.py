from __future__ import annotations

import math
from dataclasses import dataclass
import pandas as pd

from services.model_loader import load_live_artifacts
from services.feature_builders import (
    build_latest_lstm_window,
    build_latest_xgb_features,
)
from services.inference_helpers import (
    predict_lstm_latest,
    predict_xgb_latest,
    resolve_ensemble_params,
)


@dataclass
class RiskConfig:
    capital_usd: float = 5000.0
    risk_per_trade: float = 0.02
    rr: float = 1.25
    leverage: float = 1.0
    fee_bps: float = 2.0
    trade_penalty_bps: float = 2.0
    sl_atr_mult: float = 1.0
    min_atr_pct: float = 0.001


def _clip01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))


def _bucket(x: float, edges: tuple[float, ...]) -> int:
    for i, e in enumerate(edges):
        if x < e:
            return i
    return len(edges)


def build_direction_from_ensemble(
    xgb_p: float,
    lstm_p: float,
    xgb_weight: float = 0.8,
    lstm_weight: float = 0.2,
    upper: float = 0.60,
    lower: float = 0.40,
) -> tuple[str, float, int]:
    w_sum = float(xgb_weight) + float(lstm_weight)
    if w_sum <= 0:
        raise ValueError("Ensemble weights must sum to > 0")

    wx = float(xgb_weight) / w_sum
    wl = float(lstm_weight) / w_sum

    p_ens = wx * float(xgb_p) + wl * float(lstm_p)

    if p_ens >= float(upper):
        return "long", float(p_ens), 1
    if p_ens <= float(lower):
        return "short", float(p_ens), -1
    return "hold", float(p_ens), 0


def build_filter_state_index(
    p_ens: float,
    xgb_p: float,
    lstm_p: float,
    atr_pct: float,
    side: int,
) -> int:
    conf_ens = abs(float(p_ens) - 0.5) * 2.0
    dis = abs(float(xgb_p) - float(lstm_p))
    agree = int((float(xgb_p) >= 0.5) == (float(lstm_p) >= 0.5))
    atr_pct = max(0.0, float(atr_pct))

    b_conf = _bucket(conf_ens, (0.25, 0.50, 0.75))
    b_dis = _bucket(dis, (0.05, 0.15, 0.30))
    b_atr = _bucket(atr_pct, (0.004, 0.012))

    idx = b_conf
    idx += 4 * b_dis
    idx += 16 * b_atr
    idx += 48 * int(agree)
    idx += 96 * int(side)
    return int(idx)


def choose_action_with_margin(
    agent,
    state_idx: int,
    min_take_visits: int = 20,
    q_take_margin: float = 0.0,
) -> tuple[int, str, dict]:
    s = int(state_idx)

    q_skip = float(agent.Q[s, 0])
    q_take = float(agent.Q[s, 1])

    skip_visits = int(agent.visits[s, 0])
    take_visits = int(agent.visits[s, 1])

    if take_visits < min_take_visits and skip_visits < min_take_visits:
        a_idx = 0
        decision = "skip"
    elif take_visits < min_take_visits:
        a_idx = 0
        decision = "skip"
    else:
        if q_take >= q_skip + float(q_take_margin):
            a_idx = 1
            decision = "take"
        else:
            a_idx = 0
            decision = "skip"

    info = {
        "q_skip": q_skip,
        "q_take": q_take,
        "q_gap_take_minus_skip": float(q_take - q_skip),
        "q_take_margin": float(q_take_margin),
        "skip_visits": skip_visits,
        "take_visits": take_visits,
    }
    return int(a_idx), str(decision), info


def _compute_atr_pct(row: pd.Series) -> float:
    if "atr" in row and pd.notna(row["atr"]) and float(row["close"]) > 0:
        return float(row["atr"]) / float(row["close"])
    return 0.0


def _sl_tp_from_atr(
    entry: float,
    atr: float,
    action: str,
    rr: float,
    sl_atr_mult: float,
):
    if not (atr and atr > 0 and entry > 0):
        return None, None

    sl_dist = sl_atr_mult * atr
    tp_dist = rr * sl_dist

    if action == "long":
        return entry - sl_dist, entry + tp_dist
    if action == "short":
        return entry + sl_dist, entry - tp_dist
    return None, None


def _position_size_from_risk(
    capital: float,
    risk_pct: float,
    entry: float,
    stop_loss,
) -> float:
    if stop_loss is None:
        return 0.0
    risk_amount = float(capital) * float(risk_pct)
    sl_dist = abs(float(entry) - float(stop_loss))
    if sl_dist <= 0:
        return 0.0
    return float(max(0.0, risk_amount * float(entry) / sl_dist))


def _extract_timestamp(df: pd.DataFrame):
    for col in ["timestamp", "open_time", "datetime", "date"]:
        if col in df.columns:
            return df.iloc[-1][col]
    return None


def _extract_price(df: pd.DataFrame):
    for col in ["close", "Close"]:
        if col in df.columns:
            return float(df.iloc[-1][col])
    return None


def _resolve_rl_params(xgb_artifacts: dict) -> dict:
    return {
        "min_take_visits": int(xgb_artifacts.get("rl_min_take_visits", 20)),
        "q_take_margin": float(xgb_artifacts.get("rl_q_take_margin", 0.0)),
    }


def run_live_inference_from_df(
    df: pd.DataFrame,
    symbol: str,
    interval: str,
    risk=None,
) -> dict:
    if df is None or df.empty:
        raise ValueError("Input dataframe is empty.")

    if risk is None:
        risk = RiskConfig()

    artifacts = load_live_artifacts(symbol, interval)

    xgb_artifacts = artifacts["xgb"]
    lstm_artifacts = artifacts["lstm_artifacts"]
    lstm_model = artifacts["lstm_model"]
    rl_agent = artifacts["rl"]

    xgb_latest_row, X_xgb_latest = build_latest_xgb_features(df, xgb_artifacts)
    lstm_latest_row, X_lstm_latest = build_latest_lstm_window(df, lstm_artifacts)

    latest_row = xgb_latest_row.iloc[0]

    ensemble_params = resolve_ensemble_params(xgb_artifacts, lstm_artifacts)
    rl_params = _resolve_rl_params(xgb_artifacts)

    xgb_result = predict_xgb_latest(X_xgb_latest, xgb_artifacts)
    lstm_result = predict_lstm_latest(
        X_lstm_latest,
        lstm_model,
        threshold=ensemble_params["lstm_threshold"],
    )

    direction_name, p_ens, direction_sign = build_direction_from_ensemble(
        xgb_p=float(xgb_result["prob_up"]),
        lstm_p=float(lstm_result["prob_up"]),
        xgb_weight=float(ensemble_params["xgb_weight"]),
        lstm_weight=float(ensemble_params["lstm_weight"]),
        upper=float(ensemble_params["hold_high"]),
        lower=float(ensemble_params["hold_low"]),
    )

    entry = _extract_price(df)

    atr_pct = _compute_atr_pct(latest_row)
    if "atr" in latest_row and pd.notna(latest_row["atr"]):
        atr = float(latest_row["atr"])
    elif entry is not None:
        atr = float(entry) * float(risk.min_atr_pct)
    else:
        atr = 0.0

    if direction_name == "long":
        ensemble_conf = _clip01(float(p_ens))
    elif direction_name == "short":
        ensemble_conf = _clip01(1.0 - float(p_ens))
    else:
        ensemble_conf = 0.0

    rl_result = {
        "used_rl": False,
        "decision": None,
        "state": None,
        "q_skip": None,
        "q_take": None,
        "q_gap_take_minus_skip": None,
        "policy_score": None,
        "take_visits": None,
        "skip_visits": None,
        "reason": None,
    }

    stop_loss = None
    take_profit = None
    position_size_usd = 0.0
    final_action = "hold"
    confidence = ensemble_conf

    if direction_name == "hold":
        rl_result["reason"] = "ensemble_no_setup"
    else:
        if rl_agent is None:
            rl_result["reason"] = "rl_agent_not_loaded"
            final_action = direction_name
        else:
            side = 1 if direction_name == "long" else 0

            state_idx = build_filter_state_index(
                p_ens=float(p_ens),
                xgb_p=float(xgb_result["prob_up"]),
                lstm_p=float(lstm_result["prob_up"]),
                atr_pct=float(atr_pct),
                side=side,
            )

            a_idx, decision, decision_info = choose_action_with_margin(
                agent=rl_agent,
                state_idx=state_idx,
                min_take_visits=rl_params["min_take_visits"],
                q_take_margin=rl_params["q_take_margin"],
            )

            q_skip = float(decision_info["q_skip"])
            q_take = float(decision_info["q_take"])
            q_gap = float(abs(q_take - q_skip))
            q_conf = _clip01(1.0 - math.exp(-3.0 * max(0.0, q_gap)))
            policy_score = float(q_conf)

            dis = abs(float(xgb_result["prob_up"]) - float(lstm_result["prob_up"]))
            agreement_bonus = 1.0 if (
                (float(xgb_result["prob_up"]) >= 0.5) ==
                (float(lstm_result["prob_up"]) >= 0.5)
            ) else 0.0
            disagreement_penalty = _clip01(dis * 2.0)

            confidence = _clip01(
                0.45 * ensemble_conf
                + 0.25 * q_conf
                + 0.20 * agreement_bonus
                - 0.10 * disagreement_penalty
            )

            rl_result = {
                "used_rl": True,
                "decision": decision,
                "action_idx": int(a_idx),
                "state": int(state_idx),
                "q_skip": q_skip,
                "q_take": q_take,
                "q_gap_take_minus_skip": float(decision_info["q_gap_take_minus_skip"]),
                "policy_score": float(policy_score),
                "q_take_margin": float(decision_info["q_take_margin"]),
                "take_visits": int(decision_info["take_visits"]),
                "skip_visits": int(decision_info["skip_visits"]),
                "reason": None,
            }

            if decision == "skip":
                final_action = "hold"
            else:
                final_action = direction_name

                stop_loss, take_profit = _sl_tp_from_atr(
                    entry=float(entry),
                    atr=float(atr),
                    action=final_action,
                    rr=risk.rr,
                    sl_atr_mult=risk.sl_atr_mult,
                )

                position_size_usd = _position_size_from_risk(
                    capital=risk.capital_usd,
                    risk_pct=risk.risk_per_trade,
                    entry=float(entry),
                    stop_loss=stop_loss,
                )

    return {
        "symbol": symbol.upper(),
        "interval": interval.lower(),
        "timestamp": _extract_timestamp(df),
        "entry": entry,
        "close_price": entry,
        "xgb": {
            **xgb_result,
            "p_up": float(xgb_result["prob_up"]),
        },
        "lstm": {
            **lstm_result,
            "p_up": float(lstm_result["prob_up"]),
        },
        "ensemble": {
            "p_up": float(p_ens),
            "direction": direction_name,
            "confidence": float(ensemble_conf),
            "direction_sign": 0 if direction_name == "hold" else int(direction_sign),
            "xgb_weight": float(ensemble_params["xgb_weight"]),
            "lstm_weight": float(ensemble_params["lstm_weight"]),
            "upper": float(ensemble_params["hold_high"]),
            "lower": float(ensemble_params["hold_low"]),
        },
        "rl": rl_result,
        "final_action": final_action,
        "confidence": float(confidence),
        "risk": {
            "atr": float(atr),
            "atr_pct": float(atr_pct),
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "position_size_usd": float(position_size_usd),
            "leverage": float(risk.leverage),
            "rr": float(risk.rr),
            "risk_per_trade": float(risk.risk_per_trade),
        },
        "latest_xgb_row": xgb_latest_row.to_dict(orient="records")[0],
        "latest_lstm_row": lstm_latest_row.to_dict(orient="records")[0],
    }