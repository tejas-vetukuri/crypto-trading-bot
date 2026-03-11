from __future__ import annotations

import itertools
from dataclasses import replace

import numpy as np
import pandas as pd

from data.delta_exchange import DeltaDataClient
from models.rl.rl_ensemble import (
    RiskConfig,
    build_merged_dataset,
    build_direction_from_ensemble,
    simulate_trade_outcome,
)


def _safe_mean(x):
    return float(np.mean(x)) if len(x) else 0.0


def evaluate_ensemble_only_config(
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
        "max_horizon": max_horizon,
        "sl_atr_mult": risk.sl_atr_mult,
        "rr": risk.rr,
        "fee_bps": risk.fee_bps,
        "trade_penalty_bps": risk.trade_penalty_bps,
        "risk_per_trade": risk.risk_per_trade,
    }


def run_parameter_sweep(
    symbol: str = "BTCUSD",
    resolution: str = "1h",
    start_date: str = "2019-06-01",
    end_date: str | None = None,
    train_ratio: float = 0.80,
    xgb_artifacts_path: str = "models/xgboost/xgb_trend_artifacts.joblib",
    lstm_artifacts_path: str = "models/lstm/lstm_artifacts.joblib",
    lstm_threshold: float = 0.52,
    base_risk: RiskConfig = RiskConfig(
        capital_usd=5000.0,
        risk_per_trade=0.02,
        rr=2.0,
        leverage=25.0,
        fee_bps=2.0,
        trade_penalty_bps=2.0,
        sl_atr_mult=1.5,
        min_atr_pct=0.001,
    ),
    ensemble_weight_xgb: float = 0.8,
    ensemble_weight_lstm: float = 0.2,
    ensemble_upper: float = 0.60,
    ensemble_lower: float = 0.40,
    max_horizon_values: tuple[int, ...] = (1, 2, 3),
    sl_atr_mult_values: tuple[float, ...] = (0.5, 0.8, 1.0),
    rr_values: tuple[float, ...] = (0.75, 1.0, 1.25),
    output_csv: str = "short_horizon_sweep_results.csv",
):
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

    results = []

    grid = list(itertools.product(max_horizon_values, sl_atr_mult_values, rr_values))
    total = len(grid)

    print(f"Running {total} parameter combinations...\n")

    for idx, (max_horizon, sl_atr_mult, rr) in enumerate(grid, start=1):
        risk = replace(
            base_risk,
            sl_atr_mult=float(sl_atr_mult),
            rr=float(rr),
        )

        metrics = evaluate_ensemble_only_config(
            merged=merged,
            risk=risk,
            ensemble_weight_xgb=ensemble_weight_xgb,
            ensemble_weight_lstm=ensemble_weight_lstm,
            ensemble_upper=ensemble_upper,
            ensemble_lower=ensemble_lower,
            max_horizon=int(max_horizon),
        )

        results.append(metrics)

        print(
            f"[{idx:02d}/{total}] "
            f"h={max_horizon}, sl_atr={sl_atr_mult:.2f}, rr={rr:.2f} | "
            f"grossR={metrics['avg_gross_r_per_trade']:.4f}, "
            f"netR={metrics['avg_net_r_per_trade']:.4f}, "
            f"ret={metrics['total_return']:.4f}, "
            f"trades={metrics['taken']}"
        )

    results_df = pd.DataFrame(results)

    # Rank primarily by avg gross R, then net R, then total return
    results_df = results_df.sort_values(
        by=["avg_gross_r_per_trade", "avg_net_r_per_trade", "total_return"],
        ascending=[False, False, False],
    ).reset_index(drop=True)

    results_df.to_csv(output_csv, index=False)

    print("\n================ TOP 10 CONFIGS ================\n")
    cols_to_show = [
        "max_horizon",
        "sl_atr_mult",
        "rr",
        "taken",
        "directional_accuracy",
        "win_rate",
        "avg_gross_r_per_trade",
        "avg_net_r_per_trade",
        "total_return",
        "max_drawdown",
        "tp_exits",
        "sl_exits",
        "horizon_exits",
    ]
    print(results_df[cols_to_show].head(10).to_string(index=False))

    print(f"\nSaved full results to: {output_csv}")
    return results_df


if __name__ == "__main__":
    run_parameter_sweep(
        symbol="BTCUSD",
        resolution="1h",
        start_date="2019-06-01",
        end_date=None,
        train_ratio=0.80,
        xgb_artifacts_path="models/xgboost/xgb_trend_artifacts.joblib",
        lstm_artifacts_path="models/lstm/lstm_artifacts.joblib",
        lstm_threshold=0.52,
        base_risk=RiskConfig(
            capital_usd=5000.0,
            risk_per_trade=0.02,
            rr=2.0,              # overwritten by sweep
            leverage=25.0,
            fee_bps=2.0,
            trade_penalty_bps=2.0,
            sl_atr_mult=1.5,     # overwritten by sweep
            min_atr_pct=0.001,
        ),
        ensemble_weight_xgb=0.8,
        ensemble_weight_lstm=0.2,
        ensemble_upper=0.60,
        ensemble_lower=0.40,
        max_horizon_values=(1, 2, 3),
        sl_atr_mult_values=(0.5, 0.8, 1.0),
        rr_values=(0.75, 1.0, 1.25),
        output_csv="short_horizon_sweep_results.csv",
    )