from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List

import pandas as pd
import streamlit as st

from data.binance import BinanceDataClient
from models.rl.rl_ensemble import (
    RiskConfig,
    QTableAgent,
    build_combo_artifact_paths,
    build_merged_dataset,
    build_direction_from_ensemble,
    build_filter_state_index,
    choose_action_with_margin,
    simulate_trade_outcome,
    _compute_atr_pct,
    resolve_artifact_path,
)


def _pnl_from_r(r_multiple: float, capital_usd: float, risk_per_trade: float) -> float:
    risk_amount = float(capital_usd) * float(risk_per_trade)
    return float(r_multiple) * risk_amount


def _recent_start_date_for_interval(interval: str) -> str:
    now = datetime.now(timezone.utc)

    lookback_days = {
        "5m": 14,
        "15m": 45,
        "1h": 180,
        "4h": 365,
    }.get(str(interval).lower(), 120)

    start_dt = now - timedelta(days=lookback_days)
    return start_dt.strftime("%Y-%m-%d")


@st.cache_resource(show_spinner=False)
def _load_rl_agent_cached(symbol: str, interval: str):
    combo_paths = build_combo_artifact_paths(symbol.upper(), interval)
    rl_agent_path = combo_paths["rl_agent_path"]
    return QTableAgent.load(resolve_artifact_path(rl_agent_path))


@st.cache_data(show_spinner=False, ttl=300)
def fetch_recent_rl_replay_trades(
    symbol: str,
    interval: str,
    limit: int = 10,
    start_date: str | None = None,
    end_date: str | None = None,
    lstm_threshold: float = 0.52,
    capital_usd: float = 5000.0,
    risk_per_trade: float = 0.02,
    rr: float = 1.25,
    leverage: float = 25.0,
    fee_bps: float = 2.0,
    trade_penalty_bps: float = 2.0,
    sl_atr_mult: float = 1.0,
    min_atr_pct: float = 0.001,
    ensemble_weight_xgb: float = 0.8,
    ensemble_weight_lstm: float = 0.2,
    ensemble_upper: float = 0.60,
    ensemble_lower: float = 0.40,
    min_take_visits: int = 20,
    q_take_margin: float = 0.0,
    max_horizon: int = 3,
) -> List[Dict[str, Any]]:
    symbol = symbol.upper()
    interval = str(interval).lower()

    if start_date is None:
        start_date = _recent_start_date_for_interval(interval)

    risk = RiskConfig(
        capital_usd=capital_usd,
        risk_per_trade=risk_per_trade,
        rr=rr,
        leverage=leverage,
        fee_bps=fee_bps,
        trade_penalty_bps=trade_penalty_bps,
        sl_atr_mult=sl_atr_mult,
        min_atr_pct=min_atr_pct,
    )

    combo_paths = build_combo_artifact_paths(symbol, interval)
    xgb_artifacts_path = combo_paths["xgb_artifacts_path"]
    lstm_artifacts_path = combo_paths["lstm_artifacts_path"]

    agent = _load_rl_agent_cached(symbol, interval)

    client = BinanceDataClient(market="spot")
    df_raw = client.get_candles(
        symbol=symbol,
        resolution=interval,
        start_date=start_date,
        end_date=end_date,
    ).sort_values("timestamp").reset_index(drop=True)

    if df_raw.empty:
        return []

    merged = build_merged_dataset(
        df_raw=df_raw,
        xgb_artifacts_path=xgb_artifacts_path,
        lstm_artifacts_path=lstm_artifacts_path,
        lstm_threshold=lstm_threshold,
    )

    trades: List[Dict[str, Any]] = []

    # exclude unresolved rows near the end
    last_usable_idx = len(merged) - 1 - int(max_horizon)
    if last_usable_idx <= 0:
        return []

    t = 0
    while t <= last_usable_idx:
        row = merged.iloc[t]

        direction_name, p_ens, _ = build_direction_from_ensemble(
            xgb_p=float(row["xgb_p_up"]),
            lstm_p=float(row["lstm_p_up"]),
            xgb_weight=ensemble_weight_xgb,
            lstm_weight=ensemble_weight_lstm,
            upper=ensemble_upper,
            lower=ensemble_lower,
        )

        if direction_name == "hold":
            t += 1
            continue

        side = 1 if direction_name == "long" else 0

        state_idx = build_filter_state_index(
            p_ens=float(p_ens),
            xgb_p=float(row["xgb_p_up"]),
            lstm_p=float(row["lstm_p_up"]),
            atr_pct=_compute_atr_pct(row),
            side=side,
        )

        _, decision, info = choose_action_with_margin(
            agent=agent,
            state_idx=state_idx,
            min_take_visits=min_take_visits,
            q_take_margin=q_take_margin,
        )

        if decision == "skip":
            t += 1
            continue

        sim = simulate_trade_outcome(
            merged=merged,
            idx=t,
            direction_name=direction_name,
            risk=risk,
            max_horizon=max_horizon,
        )

        reward_r = float(sim["reward_r"])
        exit_index = int(sim["exit_index"])

        if reward_r > 0:
            outcome = "WIN"
        elif reward_r < 0:
            outcome = "LOSS"
        else:
            outcome = "BREAKEVEN"

        pnl = _pnl_from_r(reward_r, risk.capital_usd, risk.risk_per_trade)

        trades.append(
            {
                "time": pd.to_datetime(row["timestamp"], utc=True).strftime("%Y-%m-%d %H:%M"),
                "decision": "TRADE",
                "direction": "LONG" if direction_name == "long" else "SHORT",
                "outcome": outcome,
                "r": reward_r,
                "pnl": pnl,
                "entry_price": float(sim["entry"]),
                "exit_price": float(sim["exit_price"]),
                "policy_score": float(info["q_take"] - info["q_skip"]),
            }
        )

        # skip ahead so replayed trades do not overlap
        t = exit_index + 1

    if not trades:
        return []

    trades = sorted(trades, key=lambda x: x["time"], reverse=True)
    return trades[:limit]