from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from data.delta_exchange import DeltaDataClient

from models.rl.rl_ensemble import (
    QTableAgent,
    RiskConfig,
    build_merged_dataset,
    build_direction_from_ensemble,
    build_filter_state_index,
    _compute_atr_pct,
    simulate_trade_outcome,
    choose_action_with_margin,
)


def _safe_mean(x):
    return float(np.mean(x)) if len(x) else 0.0


def _evaluate_binary_model(actual, pred, mask=None):
    if mask is None:
        mask = np.ones(len(actual), dtype=bool)
    actual = np.asarray(actual)[mask]
    pred = np.asarray(pred)[mask]
    if len(actual) == 0:
        return 0.0, 0
    return float((actual == pred).mean()), int(len(actual))


def _evaluate_three_way_non_hold(actual, pred_3way):
    actual = np.asarray(actual)
    pred_3way = np.asarray(pred_3way)

    mask = pred_3way != 2
    if mask.sum() == 0:
        return 0.0, 0

    acc = float((actual[mask] == pred_3way[mask]).mean())
    return acc, int(mask.sum())


def _run_rl_trade_simulation(
    merged: pd.DataFrame,
    agent: QTableAgent,
    risk: RiskConfig,
    ensemble_weight_xgb: float,
    ensemble_weight_lstm: float,
    ensemble_upper: float,
    ensemble_lower: float,
    max_horizon: int,
    min_take_visits: int,
    q_take_margin: float,
) -> dict:
    setups = 0
    taken = 0
    skipped = 0
    rewards_r = []
    gross_rewards_r = []
    correct_dirs = []

    tp_exits = 0
    sl_exits = 0
    horizon_exits = 0
    other_exits = 0

    equity = 1.0
    equity_curve = [equity]

    i = 0
    while i < len(merged) - 1:
        row = merged.iloc[i]

        direction_name, p_ens_i, _ = build_direction_from_ensemble(
            xgb_p=float(row["xgb_p_up"]),
            lstm_p=float(row["lstm_p_up"]),
            xgb_weight=ensemble_weight_xgb,
            lstm_weight=ensemble_weight_lstm,
            upper=ensemble_upper,
            lower=ensemble_lower,
        )

        if direction_name == "hold":
            i += 1
            continue

        setups += 1
        side = 1 if direction_name == "long" else 0

        s = build_filter_state_index(
            p_ens=float(p_ens_i),
            xgb_p=float(row["xgb_p_up"]),
            lstm_p=float(row["lstm_p_up"]),
            atr_pct=_compute_atr_pct(row),
            side=side,
        )

        _, decision, _ = choose_action_with_margin(
            agent=agent,
            state_idx=s,
            min_take_visits=min_take_visits,
            q_take_margin=q_take_margin,
        )

        if decision == "skip":
            skipped += 1
            i += 1
            continue

        taken += 1

        sim = simulate_trade_outcome(
            merged=merged,
            idx=i,
            direction_name=direction_name,
            risk=risk,
            max_horizon=max_horizon,
        )

        reward_r = float(sim["reward_r"])
        gross_r = float(sim["gross_r"])
        exit_reason = str(sim["exit_reason"])

        rewards_r.append(reward_r)
        gross_rewards_r.append(gross_r)
        correct_dirs.append(int(gross_r > 0))

        if exit_reason == "tp":
            tp_exits += 1
        elif exit_reason in ("sl", "sl_tp_same_bar_sl_first"):
            sl_exits += 1
        elif exit_reason == "horizon":
            horizon_exits += 1
        else:
            other_exits += 1

        equity *= (1.0 + float(risk.risk_per_trade) * reward_r)
        equity_curve.append(equity)

        i = int(sim["exit_index"]) + 1

    direction_accuracy = _safe_mean(correct_dirs)
    win_rate = _safe_mean([r > 0 for r in rewards_r])
    avg_r_per_trade = _safe_mean(rewards_r)
    avg_gross_r_per_trade = _safe_mean(gross_rewards_r)
    total_return = float(equity - 1.0)

    eq = pd.Series(equity_curve)
    max_drawdown = float(((eq / eq.cummax()) - 1.0).min()) if len(eq) else 0.0

    return {
        "setups": setups,
        "taken": taken,
        "skipped": skipped,
        "take_rate": (taken / setups) if setups else 0.0,
        "directional_accuracy": direction_accuracy,
        "win_rate": win_rate,
        "avg_gross_r_per_trade": avg_gross_r_per_trade,
        "avg_net_r_per_trade": avg_r_per_trade,
        "total_return": total_return,
        "max_drawdown": max_drawdown,
        "tp_exits": tp_exits,
        "sl_exits": sl_exits,
        "horizon_exits": horizon_exits,
        "other_exits": other_exits,
    }


def _print_trade_block(
    title: str,
    metrics: dict,
    risk: RiskConfig,
    max_horizon: int,
    min_take_visits: int,
    q_take_margin: float,
):
    print(f"\n---------------- {title} ----------------")
    print(f"Candidate setups from ensemble:  {metrics['setups']}")
    print(f"Taken by RL:                     {metrics['taken']}")
    print(f"Skipped by RL:                   {metrics['skipped']}")
    print(f"Take rate on setups:             {metrics['take_rate']:.4f}")
    print(f"Directional Accuracy (taken):    {metrics['directional_accuracy']:.4f}")
    print(f"Win Rate (taken):                {metrics['win_rate']:.4f}")
    print(f"Average Gross R / trade:         {metrics['avg_gross_r_per_trade']:.4f}")
    print(f"Average Net R / trade:           {metrics['avg_net_r_per_trade']:.4f}")
    print(f"Total Return:                    {metrics['total_return']:.4f}")
    print(f"Max Drawdown:                    {metrics['max_drawdown']:.4f}")
    print(f"TP exits:                        {metrics['tp_exits']}")
    print(f"SL exits:                        {metrics['sl_exits']}")
    print(f"Horizon exits:                   {metrics['horizon_exits']}")
    print(f"Other exits:                     {metrics['other_exits']}")
    print(f"Max horizon:                     {max_horizon}")
    print(f"Min take visits:                 {min_take_visits}")
    print(f"Q take margin:                   {q_take_margin:.4f}")
    print(f"RR target:                       {risk.rr:.2f}")
    print(f"SL ATR multiplier:               {risk.sl_atr_mult:.2f}")
    print(f"Risk per trade:                  {risk.risk_per_trade:.4f}")
    print(f"Fee bps:                         {risk.fee_bps}")
    print(f"Trade penalty bps:               {risk.trade_penalty_bps}")


def evaluate_rl_agent(
    symbol: str = "BTCUSD",
    resolution: str = "1h",
    start_date: str = "2019-06-01",
    end_date: str | None = None,
    train_ratio: float = 0.80,
    xgb_artifacts_path: str = "models/xgboost/xgb_trend_artifacts.joblib",
    lstm_artifacts_path: str = "models/lstm/lstm_artifacts.joblib",
    lstm_threshold: float = 0.52,
    rl_agent_path: str = "models/rl/rl_qtable_agent.joblib",
    risk: RiskConfig = RiskConfig(),
    ensemble_weight_xgb: float = 0.8,
    ensemble_weight_lstm: float = 0.2,
    ensemble_upper: float = 0.60,
    ensemble_lower: float = 0.40,
    max_horizon: int = 3,
    min_take_visits: int = 20,
    q_take_margin: float = 0.0,
):
    agent = QTableAgent.load(rl_agent_path)

    client = DeltaDataClient()
    df_raw = client.get_candles(
        symbol=symbol,
        resolution=resolution,
        start_date=start_date,
        end_date=end_date,
    ).sort_values("timestamp").reset_index(drop=True)

    n = len(df_raw)
    split = int(n * train_ratio)
    if split <= 0 or split >= n - 2:
        raise ValueError(f"Invalid split: n={n}, split={split}")

    test_raw = df_raw.iloc[split:].copy()

    merged = build_merged_dataset(
        df_raw=test_raw,
        xgb_artifacts_path=xgb_artifacts_path,
        lstm_artifacts_path=lstm_artifacts_path,
        lstm_threshold=lstm_threshold,
    )

    actual = merged["actual"].values

    xgb_mask = merged["xgb_pred"].values != 2
    xgb_acc_non_hold, xgb_n = _evaluate_binary_model(
        actual=actual,
        pred=merged["xgb_pred"].values,
        mask=xgb_mask,
    )

    lstm_mask = merged["lstm_pred"].values != 2
    lstm_acc_non_hold, lstm_n = _evaluate_binary_model(
        actual=actual,
        pred=merged["lstm_pred"].values,
        mask=lstm_mask,
    )

    p_ens = (
        ensemble_weight_xgb * merged["xgb_p_up"].values
        + ensemble_weight_lstm * merged["lstm_p_up"].values
    ) / (ensemble_weight_xgb + ensemble_weight_lstm)

    ens_pred_2way = (p_ens >= 0.5).astype(int)
    ens_acc_2way, ens_n_2way = _evaluate_binary_model(actual=actual, pred=ens_pred_2way)

    ens_pred_3way = np.where(
        p_ens >= ensemble_upper,
        1,
        np.where(p_ens <= ensemble_lower, 0, 2),
    )
    ens_acc_non_hold, ens_n_non_hold = _evaluate_three_way_non_hold(actual, ens_pred_3way)

    xgb_dir = (merged["xgb_p_up"].values >= 0.5).astype(int)
    lstm_dir = (merged["lstm_p_up"].values >= 0.5).astype(int)
    agree_mask = xgb_dir == lstm_dir
    agree_pred = xgb_dir
    agree_acc, agree_n = _evaluate_binary_model(actual=actual, pred=agree_pred, mask=agree_mask)

    risk_no_fees = replace(risk, fee_bps=0.0, trade_penalty_bps=0.0)

    metrics_with_fees = _run_rl_trade_simulation(
        merged=merged,
        agent=agent,
        risk=risk,
        ensemble_weight_xgb=ensemble_weight_xgb,
        ensemble_weight_lstm=ensemble_weight_lstm,
        ensemble_upper=ensemble_upper,
        ensemble_lower=ensemble_lower,
        max_horizon=max_horizon,
        min_take_visits=min_take_visits,
        q_take_margin=q_take_margin,
    )

    metrics_no_fees = _run_rl_trade_simulation(
        merged=merged,
        agent=agent,
        risk=risk_no_fees,
        ensemble_weight_xgb=ensemble_weight_xgb,
        ensemble_weight_lstm=ensemble_weight_lstm,
        ensemble_upper=ensemble_upper,
        ensemble_lower=ensemble_lower,
        max_horizon=max_horizon,
        min_take_visits=min_take_visits,
        q_take_margin=q_take_margin,
    )

    print("\n================ TEST SET SUMMARY ================")
    print(f"Rows evaluated:                  {len(merged)}")
    print(f"Train ratio used:                {train_ratio:.2f}")

    print("\n---------------- Base Models ----------------")
    print(f"XGB accuracy (non-hold):         {xgb_acc_non_hold:.4f}  | n={xgb_n}")
    print(f"LSTM accuracy (non-hold):        {lstm_acc_non_hold:.4f} | n={lstm_n}")

    print("\n---------------- Raw Ensemble ----------------")
    print(f"Ensemble 2-way accuracy:         {ens_acc_2way:.4f}  | n={ens_n_2way}")
    print(f"Ensemble 3-way acc (non-hold):   {ens_acc_non_hold:.4f} | n={ens_n_non_hold}")
    print(f"Agreement-only accuracy:         {agree_acc:.4f}  | n={agree_n}")
    print(f"Ensemble weights:                xgb={ensemble_weight_xgb:.2f}, lstm={ensemble_weight_lstm:.2f}")
    print(f"Ensemble hold band:              [{ensemble_lower:.2f}, {ensemble_upper:.2f}]")

    _print_trade_block(
        title="RL Trade Filter (WITH FEES)",
        metrics=metrics_with_fees,
        risk=risk,
        max_horizon=max_horizon,
        min_take_visits=min_take_visits,
        q_take_margin=q_take_margin,
    )

    _print_trade_block(
        title="RL Trade Filter (NO FEES)",
        metrics=metrics_no_fees,
        risk=risk_no_fees,
        max_horizon=max_horizon,
        min_take_visits=min_take_visits,
        q_take_margin=q_take_margin,
    )


if __name__ == "__main__":
    evaluate_rl_agent(
        risk=RiskConfig(
            capital_usd=5000.0,
            risk_per_trade=0.02,
            rr=1.25,
            leverage=25.0,
            fee_bps=2.0,
            trade_penalty_bps=2.0,
            sl_atr_mult=1.0,
            min_atr_pct=0.001,
        ),
        ensemble_upper=0.60,
        ensemble_lower=0.40,
        max_horizon=3,
        min_take_visits=20,
        q_take_margin=0.20,
    )