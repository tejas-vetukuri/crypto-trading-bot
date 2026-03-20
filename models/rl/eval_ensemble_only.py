from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from data.binance import BinanceDataClient

from models.rl.rl_ensemble import (
    RiskConfig,
    build_merged_dataset,
    build_direction_from_ensemble,
    simulate_trade_outcome,
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


def _run_ensemble_trade_simulation(
    merged: pd.DataFrame,
    risk: RiskConfig,
    ensemble_weight_xgb: float,
    ensemble_weight_lstm: float,
    ensemble_upper: float,
    ensemble_lower: float,
    max_horizon: int,
) -> dict:
    setups = 0
    taken = 0
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
        "skipped": 0,
        "take_rate": 1.0 if setups else 0.0,
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
        "equity_curve": equity_curve,
    }


def _print_trade_block(
    title: str,
    metrics_with_fees: dict,
    metrics_no_fees: dict,
    risk: RiskConfig,
    max_horizon: int,
):
    print(f"\n---------------- {title} ----------------")
    print(f"Candidate setups from ensemble:  {metrics_with_fees['setups']}")
    print(f"Taken by ensemble:               {metrics_with_fees['taken']}")
    print(f"Skipped by ensemble:             {metrics_with_fees['skipped']}")
    print(f"Take rate on setups:             {metrics_with_fees['take_rate']:.4f}")
    print(f"Directional Accuracy (taken):    {metrics_with_fees['directional_accuracy']:.4f}")
    print(f"Win Rate:                        {metrics_with_fees['win_rate']:.4f}")
    print(f"Average Gross R / trade:         {metrics_with_fees['avg_gross_r_per_trade']:.4f}")
    print(f"Average Net R / trade:           {metrics_with_fees['avg_net_r_per_trade']:.4f}")
    print(f"Total Return:                    {metrics_with_fees['total_return']:.4f}")
    print(f"Total Return (no fees):          {metrics_no_fees['total_return']:.4f}")
    print(f"Max Drawdown:                    {metrics_with_fees['max_drawdown']:.4f}")
    print(f"Max Drawdown (no fees):          {metrics_no_fees['max_drawdown']:.4f}")
    print(f"TP exits:                        {metrics_with_fees['tp_exits']}")
    print(f"SL exits:                        {metrics_with_fees['sl_exits']}")
    print(f"Horizon exits:                   {metrics_with_fees['horizon_exits']}")
    print(f"Other exits:                     {metrics_with_fees['other_exits']}")
    print(f"Max horizon:                     {max_horizon}")
    print(f"RR target:                       {risk.rr:.2f}")
    print(f"SL ATR multiplier:               {risk.sl_atr_mult:.2f}")
    print(f"Risk per trade:                  {risk.risk_per_trade:.4f}")
    print(f"Fee bps:                         {risk.fee_bps}")
    print(f"Trade penalty bps:               {risk.trade_penalty_bps}")


def evaluate_ensemble_only(
    symbol: str = "BTCUSDT",
    resolution: str = "1h",
    start_date: str = "2019-06-01",
    end_date: str | None = None,
    train_ratio: float = 0.80,
    xgb_artifacts_path: str = "models/xgboost/xgb_trend_artifacts.joblib",
    lstm_artifacts_path: str = "models/lstm/lstm_artifacts.joblib",
    lstm_threshold: float = 0.52,
    risk: RiskConfig = RiskConfig(),
    ensemble_weight_xgb: float = 0.8,
    ensemble_weight_lstm: float = 0.2,
    ensemble_upper: float = 0.60,
    ensemble_lower: float = 0.40,
    max_horizon: int = 3,
    verbose: bool = True,
) -> dict:
    client = BinanceDataClient()
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
    ).copy()

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

    weight_sum = ensemble_weight_xgb + ensemble_weight_lstm
    if weight_sum <= 0:
        raise ValueError("ensemble_weight_xgb + ensemble_weight_lstm must be > 0")

    p_ens = (
        ensemble_weight_xgb * merged["xgb_p_up"].values
        + ensemble_weight_lstm * merged["lstm_p_up"].values
    ) / weight_sum

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

    merged["ensemble_p_up"] = p_ens
    merged["ensemble_pred_2way"] = ens_pred_2way
    merged["ensemble_pred_3way"] = ens_pred_3way
    merged["ensemble_signal"] = np.where(
        ens_pred_3way == 1,
        "up",
        np.where(ens_pred_3way == 0, "down", "hold"),
    )

    risk_no_fees = replace(risk, fee_bps=0.0, trade_penalty_bps=0.0)

    metrics_with_fees = _run_ensemble_trade_simulation(
        merged=merged,
        risk=risk,
        ensemble_weight_xgb=ensemble_weight_xgb,
        ensemble_weight_lstm=ensemble_weight_lstm,
        ensemble_upper=ensemble_upper,
        ensemble_lower=ensemble_lower,
        max_horizon=max_horizon,
    )

    metrics_no_fees = _run_ensemble_trade_simulation(
        merged=merged,
        risk=risk_no_fees,
        ensemble_weight_xgb=ensemble_weight_xgb,
        ensemble_weight_lstm=ensemble_weight_lstm,
        ensemble_upper=ensemble_upper,
        ensemble_lower=ensemble_lower,
        max_horizon=max_horizon,
    )

    result = {
        "rows_evaluated": len(merged),
        "train_ratio": train_ratio,
        "base_metrics": {
            "xgb_accuracy_non_hold": xgb_acc_non_hold,
            "xgb_n": xgb_n,
            "lstm_accuracy_non_hold": lstm_acc_non_hold,
            "lstm_n": lstm_n,
        },
        "ensemble_metrics": {
            "ensemble_2way_accuracy": ens_acc_2way,
            "ensemble_2way_n": ens_n_2way,
            "ensemble_3way_accuracy_non_hold": ens_acc_non_hold,
            "ensemble_3way_n_non_hold": ens_n_non_hold,
            "agreement_only_accuracy": agree_acc,
            "agreement_only_n": agree_n,
            "ensemble_weight_xgb": ensemble_weight_xgb,
            "ensemble_weight_lstm": ensemble_weight_lstm,
            "ensemble_upper": ensemble_upper,
            "ensemble_lower": ensemble_lower,
        },
        "trade_metrics_with_fees": metrics_with_fees,
        "trade_metrics_no_fees": metrics_no_fees,
        "merged_preview": merged,
    }

    if verbose:
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
            title="Ensemble-Only Trade Simulation",
            metrics_with_fees=metrics_with_fees,
            metrics_no_fees=metrics_no_fees,
            risk=risk,
            max_horizon=max_horizon,
        )

    return result


if __name__ == "__main__":
    evaluate_ensemble_only(
        symbol="BTCUSDT",
        resolution="1h",
        start_date="2017-09-01",
        end_date=None,
        train_ratio=0.80,
        xgb_artifacts_path="models/xgboost/xgb_trend_artifacts.joblib",
        lstm_artifacts_path="models/lstm/lstm_artifacts.joblib",
        lstm_threshold=0.53,
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
        ensemble_weight_xgb=0.8,
        ensemble_weight_lstm=0.2,
        ensemble_upper=0.60,
        ensemble_lower=0.40,
        max_horizon=3,
        verbose=True,
    )