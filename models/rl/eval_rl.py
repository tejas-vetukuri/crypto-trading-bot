import numpy as np
import pandas as pd

from data.delta_exchange import DeltaDataClient

from models.rl.rl_ensemble import (
    QTableAgent,
    ACTIONS,
    ACTION_TO_IDX,
    build_state_index,
    build_merged_dataset,
    _compute_atr_pct,
    _reward_from_next_return,
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
    fee_bps: float = 2.0,
    trade_penalty_bps: float = 2.0,
    hard_gate_sideways_hold: bool = True,
    ensemble_weight_xgb: float = 0.50,
    ensemble_weight_lstm: float = 0.50,
    ensemble_upper: float = 0.55,
    ensemble_lower: float = 0.45,
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

    # -------------------------
    # Base model evaluation
    # -------------------------
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

    # 2-way raw ensemble
    p_ens = (
        ensemble_weight_xgb * merged["xgb_p_up"].values
        + ensemble_weight_lstm * merged["lstm_p_up"].values
    )
    ens_pred_2way = (p_ens >= 0.5).astype(int)
    ens_acc_2way, ens_n_2way = _evaluate_binary_model(actual=actual, pred=ens_pred_2way)

    # 3-way raw ensemble with hold zone
    ens_pred_3way = np.where(
        p_ens >= ensemble_upper,
        1,
        np.where(p_ens <= ensemble_lower, 0, 2),
    )
    ens_acc_non_hold, ens_n_non_hold = _evaluate_three_way_non_hold(actual, ens_pred_3way)

    # agreement-only ensemble
    xgb_dir = (merged["xgb_p_up"].values >= 0.5).astype(int)
    lstm_dir = (merged["lstm_p_up"].values >= 0.5).astype(int)
    agree_mask = xgb_dir == lstm_dir
    agree_pred = xgb_dir
    agree_acc, agree_n = _evaluate_binary_model(actual=actual, pred=agree_pred, mask=agree_mask)

    # -------------------------
    # RL evaluation
    # -------------------------
    actions = []
    rewards = []
    correct_dirs = []
    equity = 1.0
    equity_curve = [equity]

    for i in range(len(merged)):
        row = merged.iloc[i]
        atr_pct = _compute_atr_pct(row)

        s = build_state_index(
            xgb_p=float(row["xgb_p_up"]),
            lstm_p=float(row["lstm_p_up"]),
            xgb_used=int(row["xgb_used"]),
            lstm_used=int(row["lstm_used"]),
            xgb_pred=int(row["xgb_pred"]),
            lstm_pred=int(row["lstm_pred"]),
            atr_pct=float(atr_pct),
        )

        if hard_gate_sideways_hold and int(row["xgb_used"]) == 0 and int(row["lstm_used"]) == 0:
            a_idx = ACTION_TO_IDX["hold"]
        else:
            a_idx = agent.act(s, greedy=True)

        action = ACTIONS[a_idx]
        ret_next = float(row["ret_next"])

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

    rl_non_hold_accuracy = _safe_mean(correct_dirs)
    traded_rewards = [r for a, r in zip(actions, rewards) if a != "hold"]
    rl_win_rate = _safe_mean([r > 0 for r in traded_rewards])
    total_return = float(equity - 1.0)

    eq = pd.Series(equity_curve)
    max_drawdown = float(((eq / eq.cummax()) - 1.0).min())

    # -------------------------
    # Print summary
    # -------------------------
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

    print("\n---------------- RL Controller ----------------")
    print(f"Directional Accuracy (non-hold): {rl_non_hold_accuracy:.4f}")
    print(f"Win Rate (non-hold):             {rl_win_rate:.4f}")
    print(f"Total Return:                    {total_return:.4f}")
    print(f"Max Drawdown:                    {max_drawdown:.4f}")
    print(f"Trades taken:                    {sum(a != 'hold' for a in actions)} / {len(actions)}")
    print(f"Hard-gate sideways hold:         {int(hard_gate_sideways_hold)}")
    print(f"Fee bps:                         {fee_bps}")
    print(f"Trade penalty bps:               {trade_penalty_bps}")


if __name__ == "__main__":
    evaluate_rl_agent()