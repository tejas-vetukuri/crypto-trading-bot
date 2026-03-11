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


def evaluate_ensemble_only(
    symbol: str = "BTCUSD",
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
    max_horizon: int = 12,
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

    actual = merged["actual"].values

    # -------------------------
    # Base models
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

    # -------------------------
    # Raw weighted ensemble classification view
    # -------------------------
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

    # -------------------------
    # Ensemble-only trade simulation
    # Non-overlapping trades
    # -------------------------
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

        direction_name, p_ens_i, direction_sign = build_direction_from_ensemble(
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
    max_drawdown = float(((eq / eq.cummax()) - 1.0).min())

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

    print("\n---------------- Ensemble-Only Trade Simulation ----------------")
    print(f"Candidate setups from ensemble:  {setups}")
    print(f"Taken:                           {taken}")
    print(f"Directional Accuracy (taken):    {direction_accuracy:.4f}")
    print(f"Win Rate (taken):                {win_rate:.4f}")
    print(f"Average Gross R / trade:         {avg_gross_r_per_trade:.4f}")
    print(f"Average Net R / trade:           {avg_r_per_trade:.4f}")
    print(f"Total Return:                    {total_return:.4f}")
    print(f"Max Drawdown:                    {max_drawdown:.4f}")
    print(f"TP exits:                        {tp_exits}")
    print(f"SL exits:                        {sl_exits}")
    print(f"Horizon exits:                   {horizon_exits}")
    print(f"Other exits:                     {other_exits}")
    print(f"Max horizon:                     {max_horizon}")
    print(f"Risk per trade:                  {risk.risk_per_trade:.4f}")
    print(f"RR target:                       {risk.rr:.2f}")
    print(f"Fee bps:                         {risk.fee_bps}")
    print(f"Trade penalty bps:               {risk.trade_penalty_bps}")


if __name__ == "__main__":
    evaluate_ensemble_only(
        symbol="BTCUSD",
        resolution="1h",
        start_date="2019-06-01",
        end_date=None,
        train_ratio=0.80,
        xgb_artifacts_path="models/xgboost/xgb_trend_artifacts.joblib",
        lstm_artifacts_path="models/lstm/lstm_artifacts.joblib",
        lstm_threshold=0.52,
        risk=RiskConfig(
            capital_usd=5000.0,
            risk_per_trade=0.02,
            rr=2.0,
            leverage=25.0,
            fee_bps=2.0,
            trade_penalty_bps=2.0,
            sl_atr_mult=1.5,
            min_atr_pct=0.001,
        ),
        ensemble_weight_xgb=0.8,
        ensemble_weight_lstm=0.2,
        ensemble_upper=0.60,
        ensemble_lower=0.40,
        max_horizon=12,
    )