from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from data.binance import BinanceDataClient

from models.rl.rl_ensemble import (
    RiskConfig,
    build_merged_dataset,
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


def combine_probs_plain_weighted(
    xgb_p: np.ndarray,
    lstm_p: np.ndarray,
    xgb_weight: float,
    lstm_weight: float,
    eps: float = 1e-12,
) -> np.ndarray:
    denom = max(xgb_weight + lstm_weight, eps)
    p = (xgb_weight * xgb_p + lstm_weight * lstm_p) / denom
    return np.clip(p, 0.0, 1.0)


def combine_probs_confidence_weighted(
    xgb_p: np.ndarray,
    lstm_p: np.ndarray,
    xgb_weight: float,
    lstm_weight: float,
    eps: float = 1e-12,
) -> np.ndarray:
    xgb_p = np.asarray(xgb_p, dtype=float)
    lstm_p = np.asarray(lstm_p, dtype=float)

    xgb_conf = np.abs(xgb_p - 0.5)
    lstm_conf = np.abs(lstm_p - 0.5)

    wx = xgb_weight * xgb_conf
    wl = lstm_weight * lstm_conf

    denom = np.maximum(wx + wl, eps)
    p = (wx * xgb_p + wl * lstm_p) / denom
    return np.clip(p, 0.0, 1.0)


def combine_probs_confidence_weighted_agreement(
    xgb_p: np.ndarray,
    lstm_p: np.ndarray,
    xgb_weight: float,
    lstm_weight: float,
    disagreement_shrink: float = 0.5,
    eps: float = 1e-12,
) -> np.ndarray:
    xgb_p = np.asarray(xgb_p, dtype=float)
    lstm_p = np.asarray(lstm_p, dtype=float)

    xgb_conf = np.abs(xgb_p - 0.5)
    lstm_conf = np.abs(lstm_p - 0.5)

    wx = xgb_weight * xgb_conf
    wl = lstm_weight * lstm_conf

    denom = np.maximum(wx + wl, eps)
    p = (wx * xgb_p + wl * lstm_p) / denom

    xgb_dir = xgb_p >= 0.5
    lstm_dir = lstm_p >= 0.5
    disagree = xgb_dir != lstm_dir

    p[disagree] = 0.5 + disagreement_shrink * (p[disagree] - 0.5)
    return np.clip(p, 0.0, 1.0)


def combine_probs_xgb_only(
    xgb_p: np.ndarray,
    lstm_p: np.ndarray,
    xgb_weight: float,
    lstm_weight: float,
) -> np.ndarray:
    return np.clip(np.asarray(xgb_p, dtype=float), 0.0, 1.0)


def combine_probs_lstm_only(
    xgb_p: np.ndarray,
    lstm_p: np.ndarray,
    xgb_weight: float,
    lstm_weight: float,
) -> np.ndarray:
    return np.clip(np.asarray(lstm_p, dtype=float), 0.0, 1.0)


def probs_to_3way_preds(
    p_up: np.ndarray,
    upper: float,
    lower: float,
) -> np.ndarray:
    return np.where(
        p_up >= upper,
        1,
        np.where(p_up <= lower, 0, 2),
    )


def probs_to_direction_name(
    p_up: float,
    upper: float,
    lower: float,
) -> str:
    if p_up >= upper:
        return "long"
    if p_up <= lower:
        return "short"
    return "hold"


def _run_signal_trade_simulation(
    merged: pd.DataFrame,
    risk: RiskConfig,
    p_sig: np.ndarray,
    upper: float,
    lower: float,
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
        direction_name = probs_to_direction_name(
            p_up=float(p_sig[i]),
            upper=upper,
            lower=lower,
        )

        if direction_name not in {"long", "short", "hold"}:
            raise ValueError(f"Unexpected direction_name: {direction_name}")

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
    }


def _print_trade_block(title: str, metrics: dict, risk: RiskConfig, max_horizon: int):
    print(f"\n---------------- {title} ----------------")
    print(f"Candidate setups from signal:    {metrics['setups']}")
    print(f"Taken:                           {metrics['taken']}")
    print(f"Skipped:                         {metrics['skipped']}")
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
    print(f"RR target:                       {risk.rr:.2f}")
    print(f"SL ATR multiplier:               {risk.sl_atr_mult:.2f}")
    print(f"Risk per trade:                  {risk.risk_per_trade:.4f}")
    print(f"Fee bps:                         {risk.fee_bps}")
    print(f"Trade penalty bps:               {risk.trade_penalty_bps}")


def _evaluate_signal_family(
    title: str,
    merged: pd.DataFrame,
    actual: np.ndarray,
    risk: RiskConfig,
    p_sig: np.ndarray,
    ensemble_upper: float,
    ensemble_lower: float,
    max_horizon: int,
):
    pred_2way = (p_sig >= 0.5).astype(int)
    acc_2way, n_2way = _evaluate_binary_model(actual=actual, pred=pred_2way)

    pred_3way = probs_to_3way_preds(
        p_up=p_sig,
        upper=ensemble_upper,
        lower=ensemble_lower,
    )
    acc_non_hold, n_non_hold = _evaluate_three_way_non_hold(actual, pred_3way)

    risk_no_fees = replace(risk, fee_bps=0.0, trade_penalty_bps=0.0)

    metrics_with_fees = _run_signal_trade_simulation(
        merged=merged,
        risk=risk,
        p_sig=p_sig,
        upper=ensemble_upper,
        lower=ensemble_lower,
        max_horizon=max_horizon,
    )

    metrics_no_fees = _run_signal_trade_simulation(
        merged=merged,
        risk=risk_no_fees,
        p_sig=p_sig,
        upper=ensemble_upper,
        lower=ensemble_lower,
        max_horizon=max_horizon,
    )

    print(f"\n================ {title.upper()} ================")
    print(f"2-way accuracy:                  {acc_2way:.4f} | n={n_2way}")
    print(f"3-way accuracy (non-hold):       {acc_non_hold:.4f} | n={n_non_hold}")
    print(f"Mean p_up:                       {float(np.mean(p_sig)):.4f}")
    print(f"Std p_up:                        {float(np.std(p_sig)):.4f}")

    _print_trade_block(
        title=f"{title} Trade Simulation (WITH FEES)",
        metrics=metrics_with_fees,
        risk=risk,
        max_horizon=max_horizon,
    )

    _print_trade_block(
        title=f"{title} Trade Simulation (NO FEES)",
        metrics=metrics_no_fees,
        risk=risk_no_fees,
        max_horizon=max_horizon,
    )

    return {
        "title": title,
        "acc_2way": acc_2way,
        "n_2way": n_2way,
        "acc_3way_non_hold": acc_non_hold,
        "n_3way_non_hold": n_non_hold,
        "mean_p_up": float(np.mean(p_sig)),
        "std_p_up": float(np.std(p_sig)),
        "with_fees": metrics_with_fees,
        "no_fees": metrics_no_fees,
    }


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
    ensemble_weight_xgb: float = 0.2,
    ensemble_weight_lstm: float = 0.8,
    ensemble_upper: float = 0.60,
    ensemble_lower: float = 0.40,
    max_horizon: int = 3,
    disagreement_shrink: float = 0.5,
):
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
    )

    actual = merged["actual"].values
    xgb_p = merged["xgb_p_up"].values.astype(float)
    lstm_p = merged["lstm_p_up"].values.astype(float)

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

    xgb_dir = (xgb_p >= 0.5).astype(int)
    lstm_dir = (lstm_p >= 0.5).astype(int)
    agree_mask = xgb_dir == lstm_dir
    agree_pred = xgb_dir
    agree_acc, agree_n = _evaluate_binary_model(actual=actual, pred=agree_pred, mask=agree_mask)

    print("\n================ TEST SET SUMMARY ================")
    print(f"Rows evaluated:                  {len(merged)}")
    print(f"Train ratio used:                {train_ratio:.2f}")

    print("\n---------------- Base Models ----------------")
    print(f"XGB accuracy (non-hold):         {xgb_acc_non_hold:.4f} | n={xgb_n}")
    print(f"LSTM accuracy (non-hold):        {lstm_acc_non_hold:.4f} | n={lstm_n}")
    print(f"Agreement-only accuracy:         {agree_acc:.4f} | n={agree_n}")

    results = []

    p_plain = combine_probs_plain_weighted(
        xgb_p=xgb_p,
        lstm_p=lstm_p,
        xgb_weight=ensemble_weight_xgb,
        lstm_weight=ensemble_weight_lstm,
    )
    results.append(_evaluate_signal_family(
        title=f"Plain Weighted Avg ({ensemble_weight_xgb:.2f}/{ensemble_weight_lstm:.2f})",
        merged=merged,
        actual=actual,
        risk=risk,
        p_sig=p_plain,
        ensemble_upper=ensemble_upper,
        ensemble_lower=ensemble_lower,
        max_horizon=max_horizon,
    ))

    p_conf = combine_probs_confidence_weighted(
        xgb_p=xgb_p,
        lstm_p=lstm_p,
        xgb_weight=ensemble_weight_xgb,
        lstm_weight=ensemble_weight_lstm,
    )
    results.append(_evaluate_signal_family(
        title=f"Confidence-Weighted Avg ({ensemble_weight_xgb:.2f}/{ensemble_weight_lstm:.2f})",
        merged=merged,
        actual=actual,
        risk=risk,
        p_sig=p_conf,
        ensemble_upper=ensemble_upper,
        ensemble_lower=ensemble_lower,
        max_horizon=max_horizon,
    ))

    p_conf_agree = combine_probs_confidence_weighted_agreement(
        xgb_p=xgb_p,
        lstm_p=lstm_p,
        xgb_weight=ensemble_weight_xgb,
        lstm_weight=ensemble_weight_lstm,
        disagreement_shrink=disagreement_shrink,
    )
    results.append(_evaluate_signal_family(
        title=f"Confidence+Agreement Avg ({ensemble_weight_xgb:.2f}/{ensemble_weight_lstm:.2f})",
        merged=merged,
        actual=actual,
        risk=risk,
        p_sig=p_conf_agree,
        ensemble_upper=ensemble_upper,
        ensemble_lower=ensemble_lower,
        max_horizon=max_horizon,
    ))

    p_xgb = combine_probs_xgb_only(
        xgb_p=xgb_p,
        lstm_p=lstm_p,
        xgb_weight=1.0,
        lstm_weight=0.0,
    )
    results.append(_evaluate_signal_family(
        title="XGB Only",
        merged=merged,
        actual=actual,
        risk=risk,
        p_sig=p_xgb,
        ensemble_upper=ensemble_upper,
        ensemble_lower=ensemble_lower,
        max_horizon=max_horizon,
    ))

    p_lstm = combine_probs_lstm_only(
        xgb_p=xgb_p,
        lstm_p=lstm_p,
        xgb_weight=0.0,
        lstm_weight=1.0,
    )
    results.append(_evaluate_signal_family(
        title="LSTM Only",
        merged=merged,
        actual=actual,
        risk=risk,
        p_sig=p_lstm,
        ensemble_upper=ensemble_upper,
        ensemble_lower=ensemble_lower,
        max_horizon=max_horizon,
    ))

    summary_rows = []
    for r in results:
        wf = r["with_fees"]
        summary_rows.append({
            "method": r["title"],
            "2way_acc": r["acc_2way"],
            "3way_non_hold_acc": r["acc_3way_non_hold"],
            "trades_with_fees": wf["taken"],
            "dir_acc_with_fees": wf["directional_accuracy"],
            "win_rate_with_fees": wf["win_rate"],
            "avg_net_r_with_fees": wf["avg_net_r_per_trade"],
            "total_return_with_fees": wf["total_return"],
            "max_dd_with_fees": wf["max_drawdown"],
        })

    summary_df = pd.DataFrame(summary_rows).sort_values(
        by=["total_return_with_fees", "avg_net_r_with_fees"],
        ascending=[False, False],
    )

    print("\n================ FINAL COMPARISON SUMMARY ================")
    print(summary_df.to_string(index=False))

    return {
        "summary_df": summary_df,
        "best_method": summary_df.iloc[0]["method"],
        "best_total_return_with_fees": float(summary_df.iloc[0]["total_return_with_fees"]),
        "best_avg_net_r_with_fees": float(summary_df.iloc[0]["avg_net_r_with_fees"]),
        "best_max_dd_with_fees": float(summary_df.iloc[0]["max_dd_with_fees"]),
        "best_trades_with_fees": int(summary_df.iloc[0]["trades_with_fees"]),
    }


if __name__ == "__main__":
    evaluate_ensemble_only(
        symbol="BTCUSDT",
        resolution="1h",
        start_date="2017-09-01",
        end_date=None,
        train_ratio=0.80,
        xgb_artifacts_path="models/xgboost/xgb_trend_artifacts.joblib",
        lstm_artifacts_path="models/lstm/lstm_artifacts.joblib",
        lstm_threshold=0.52,
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
        ensemble_weight_xgb=0.20,
        ensemble_weight_lstm=0.80,
        ensemble_upper=0.60,
        ensemble_lower=0.40,
        max_horizon=3,
        disagreement_shrink=0.5,
    )