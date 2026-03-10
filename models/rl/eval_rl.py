# eval_rl.py

import numpy as np
import pandas as pd

from data.delta_exchange import DeltaDataClient

from models.rl.rl_ensemble import (
    QTableAgent,
    ACTIONS,
    ACTION_TO_IDX,
    build_state_index,
    predict_xgb_series,
    predict_lstm_series,
    _compute_atr_pct,
    _reward_from_next_return,
)


def evaluate_rl_agent(
    symbol: str = "BTCUSD",
    resolution: str = "1h",
    start_date: str = "2019-06-01",
    end_date: str | None = None,
    xgb_artifacts_path: str = "models/xgboost/xgb_trend_artifacts.joblib",
    lstm_artifacts_path: str = "models/lstm/lstm_artifacts.joblib",
    lstm_threshold: float = 0.52,
    rl_agent_path: str = "models/rl/rl_qtable_agent.joblib",
    fee_bps: float = 2.0,
    trade_penalty_bps: float = 2.0,  # ✅ NEW: match RL reward signature
    hard_gate_sideways_hold: bool = True,  # ✅ NEW: match training/inference
):
    agent = QTableAgent.load(rl_agent_path)

    client = DeltaDataClient()
    df_raw = client.get_candles(
        symbol=symbol,
        resolution=resolution,
        start_date=start_date,
        end_date=end_date,
    ).sort_values("timestamp").reset_index(drop=True)

    xgb_df = predict_xgb_series(df_raw, xgb_artifacts_path)
    lstm_df = predict_lstm_series(df_raw, lstm_artifacts_path, lstm_threshold)

    merged = pd.merge(xgb_df, lstm_df, on=["timestamp", "close"], how="inner")
    merged = merged.sort_values("timestamp").reset_index(drop=True)

    merged["close_next"] = merged["close"].shift(-1)
    merged["ret_next"] = (merged["close_next"] / merged["close"]) - 1.0
    merged = merged.dropna(subset=["ret_next"]).reset_index(drop=True)

    actions = []
    rewards = []
    correct_dirs = []
    equity = 1.0
    equity_curve = [equity]

    for i in range(len(merged)):
        row = merged.iloc[i]
        atr_pct = _compute_atr_pct(row)

        s = build_state_index(
            xgb_p=row["xgb_p_up"],
            lstm_p=row["lstm_p_up"],
            xgb_used=row["xgb_used"],
            lstm_used=row["lstm_used"],
            atr_pct=atr_pct,
        )

        # ✅ Hard gate: if BOTH models are sideways => HOLD
        if hard_gate_sideways_hold and int(row["xgb_used"]) == 0 and int(row["lstm_used"]) == 0:
            a_idx = ACTION_TO_IDX["hold"]
        else:
            a_idx = agent.act(s, greedy=True)

        action = ACTIONS[a_idx]
        ret_next = float(row["ret_next"])

        # ✅ Updated reward signature
        reward = _reward_from_next_return(
            a_idx,
            ret_next,
            fee_bps=fee_bps,
            trade_penalty_bps=trade_penalty_bps,
        )

        actions.append(action)
        rewards.append(reward)

        if action != "hold":
            correct = (
                (action == "long" and ret_next > 0) or
                (action == "short" and ret_next < 0)
            )
            correct_dirs.append(int(correct))

        equity *= (1 + reward)
        equity_curve.append(equity)

    non_hold_accuracy = float(np.mean(correct_dirs)) if len(correct_dirs) else 0.0
    traded_rewards = [r for a, r in zip(actions, rewards) if a != "hold"]
    win_rate = float(np.mean([r > 0 for r in traded_rewards])) if len(traded_rewards) else 0.0
    total_return = float(equity - 1.0)

    eq = pd.Series(equity_curve)
    max_drawdown = float(((eq / eq.cummax()) - 1.0).min())

    print("\n📊 RL Evaluation")
    print(f"Directional Accuracy (non-hold): {non_hold_accuracy:.4f}")
    print(f"Win Rate (non-hold):            {win_rate:.4f}")
    print(f"Total Return:                   {total_return:.4f}")
    print(f"Max Drawdown:                   {max_drawdown:.4f}")
    print(f"Trades taken:                   {sum(a != 'hold' for a in actions)} / {len(actions)}")
    print(f"Hard-gate sideways hold:        {int(hard_gate_sideways_hold)}")
    print(f"Fee bps:                        {fee_bps}")
    print(f"Trade penalty bps:              {trade_penalty_bps}")


if __name__ == "__main__":
    evaluate_rl_agent()