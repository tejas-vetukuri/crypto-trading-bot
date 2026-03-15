# eval_lstm_robustness.py

from dataclasses import replace
import itertools
import pandas as pd

from models.rl.eval_ensemble_only import evaluate_ensemble_only
from models.rl.rl_ensemble import RiskConfig


def run_lstm_robustness():
    splits = [0.75, 0.80, 0.85]

    hold_bands = [
        (0.40, 0.60),
        (0.42, 0.58),
        (0.45, 0.55),
    ]

    results = []

    for split, band in itertools.product(splits, hold_bands):
        lower, upper = band

        print("\n==============================")
        print(f"SPLIT={split} | BAND={band}")
        print("==============================")

        metrics = evaluate_ensemble_only(
            symbol="BTCUSDT",
            resolution="1h",
            start_date="2017-09-01",
            end_date=None,
            train_ratio=split,
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
            ensemble_weight_xgb=0.0,
            ensemble_weight_lstm=1.0,
            ensemble_lower=lower,
            ensemble_upper=upper,
            disagreement_shrink=0.5,
        )

        summary_df = metrics["summary_df"]
        lstm_row = summary_df.loc[summary_df["method"] == "LSTM Only"].iloc[0]

        results.append({
            "split": split,
            "lower": lower,
            "upper": upper,
            "method": "LSTM Only",
            "total_return_with_fees": float(lstm_row["total_return_with_fees"]),
            "avg_net_r_with_fees": float(lstm_row["avg_net_r_with_fees"]),
            "max_dd_with_fees": float(lstm_row["max_dd_with_fees"]),
            "trades_with_fees": int(lstm_row["trades_with_fees"]),
            "2way_acc": float(lstm_row["2way_acc"]),
            "3way_non_hold_acc": float(lstm_row["3way_non_hold_acc"]),
            "dir_acc_with_fees": float(lstm_row["dir_acc_with_fees"]),
            "win_rate_with_fees": float(lstm_row["win_rate_with_fees"]),
        })

    df = pd.DataFrame(results)

    sort_cols = [
        "total_return_with_fees",
        "avg_net_r_with_fees",
        "dir_acc_with_fees",
    ]
    df_sorted = df.sort_values(sort_cols, ascending=[False, False, False]).reset_index(drop=True)

    df_sorted.to_csv("lstm_robustness_results.csv", index=False)

    print("\n================ ROBUSTNESS SUMMARY ================")
    print(df_sorted.to_string(index=False))
    print("\n✅ Saved robustness results: lstm_robustness_results.csv")


if __name__ == "__main__":
    run_lstm_robustness()
