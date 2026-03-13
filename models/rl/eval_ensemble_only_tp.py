from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _safe_mean(x):
    return float(np.mean(x)) if len(x) else 0.0


@dataclass
class FilterRiskConfig:
    capital_usd: float = 5000.0
    risk_per_trade: float = 0.02
    fee_bps: float = 2.0
    trade_penalty_bps: float = 2.0


def _binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())

    acc = float((y_true == y_pred).mean()) if len(y_true) else 0.0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "n": int(len(y_true)),
    }


def _print_side_metrics(title: str, y_true: np.ndarray, p: np.ndarray, threshold: float):
    y_pred = (p >= threshold).astype(int)
    m = _binary_metrics(y_true, y_pred)

    print(f"\n---------------- {title} ----------------")
    print(f"Threshold:                       {threshold:.2f}")
    print(f"Accuracy:                        {m['accuracy']:.4f}")
    print(f"Precision:                       {m['precision']:.4f}")
    print(f"Recall:                          {m['recall']:.4f}")
    print(f"F1:                              {m['f1']:.4f}")
    print(f"TP: {m['tp']} | FP: {m['fp']} | TN: {m['tn']} | FN: {m['fn']} | n={m['n']}")


def _resolve_path(path_like: str | Path) -> Path:
    p = Path(path_like)
    if p.is_absolute():
        return p
    return get_project_root() / p


def _print_overlap(a: pd.DataFrame, b: pd.DataFrame, a_name: str, b_name: str):
    a_ts = set(a["timestamp"])
    b_ts = set(b["timestamp"])
    inter = a_ts & b_ts
    print(f"Overlap {a_name} vs {b_name}: {len(inter)}")


def load_side_csv(path_like: str | Path, prob_col_candidates: list[str], name: str):
    path = _resolve_path(path_like)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    df = pd.read_csv(path)

    print(f"\n===== LOADING {name} =====")
    print(f"Path: {path}")
    print(f"Columns: {list(df.columns)}")
    print(f"Rows before processing: {len(df)}")

    ts_col = None
    for c in ["timestamp", "time", "datetime", "Date", "date", "open_time", "close_time"]:
        if c in df.columns:
            ts_col = c
            break

    if ts_col is None:
        raise ValueError(
            f"{path.name} must contain a timestamp-like column "
            f"(one of: timestamp, time, datetime, Date, date, open_time, close_time)."
        )

    if "y_true" not in df.columns:
        raise ValueError(f"{path.name} must contain a 'y_true' column.")

    prob_col = None
    for c in prob_col_candidates:
        if c in df.columns:
            prob_col = c
            break

    if prob_col is None:
        raise ValueError(
            f"{path.name} must contain one of probability columns: {prob_col_candidates}"
        )

    df = df.copy()
    raw_ts_preview = df[[ts_col]].head(5).copy()
    df["timestamp"] = pd.to_datetime(df[ts_col], utc=True, errors="coerce")

    print("\nRaw -> parsed timestamp preview:")
    preview = raw_ts_preview.copy()
    preview["parsed_timestamp"] = df["timestamp"].head(5).values
    print(preview)

    bad_ts = int(df["timestamp"].isna().sum())
    if bad_ts > 0:
        print(f"{name}: unparsable timestamps dropped = {bad_ts}")

    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    # normalize to hour for hourly candles
    df["timestamp"] = df["timestamp"].dt.floor("h")

    dupes = int(df["timestamp"].duplicated().sum())
    if dupes > 0:
        print(f"{name}: duplicate timestamps found = {dupes}, keeping first")
        df = df.drop_duplicates(subset=["timestamp"], keep="first").reset_index(drop=True)

    print(f"Rows after timestamp parse: {len(df)}")
    if len(df) > 0:
        print(f"{name} min timestamp: {df['timestamp'].min()}")
        print(f"{name} max timestamp: {df['timestamp'].max()}")
        print(f"{name} first 5 timestamps: {df['timestamp'].head().tolist()}")
        print(f"{name} last 5 timestamps: {df['timestamp'].tail().tolist()}")

    return df, prob_col, path


def load_and_merge_dual_side_predictions(
    xgb_long_csv: str | Path,
    xgb_short_csv: str | Path,
    lstm_long_csv: str | Path,
    lstm_short_csv: str | Path,
) -> pd.DataFrame:
    xgb_long, xgb_long_prob_col, xgb_long_path = load_side_csv(
        xgb_long_csv, ["p_tp_first", "p_long_tp_first", "p"], "XGB_LONG"
    )
    xgb_short, xgb_short_prob_col, xgb_short_path = load_side_csv(
        xgb_short_csv, ["p_tp_first", "p_short_tp_first", "p"], "XGB_SHORT"
    )
    lstm_long, lstm_long_prob_col, lstm_long_path = load_side_csv(
        lstm_long_csv, ["p_tp_first", "p_long_tp_first", "p"], "LSTM_LONG"
    )
    lstm_short, lstm_short_prob_col, lstm_short_path = load_side_csv(
        lstm_short_csv, ["p_tp_first", "p_short_tp_first", "p"], "LSTM_SHORT"
    )

    print("\n===== PAIRWISE OVERLAP CHECK =====")
    _print_overlap(xgb_long, xgb_short, "XGB_LONG", "XGB_SHORT")
    _print_overlap(xgb_long, lstm_long, "XGB_LONG", "LSTM_LONG")
    _print_overlap(xgb_long, lstm_short, "XGB_LONG", "LSTM_SHORT")
    _print_overlap(xgb_short, lstm_long, "XGB_SHORT", "LSTM_LONG")
    _print_overlap(xgb_short, lstm_short, "XGB_SHORT", "LSTM_SHORT")
    _print_overlap(lstm_long, lstm_short, "LSTM_LONG", "LSTM_SHORT")

    xgb_long = xgb_long.rename(columns={"y_true": "xgb_long_y_true", xgb_long_prob_col: "xgb_p_long"})
    xgb_short = xgb_short.rename(columns={"y_true": "xgb_short_y_true", xgb_short_prob_col: "xgb_p_short"})
    lstm_long = lstm_long.rename(columns={"y_true": "lstm_long_y_true", lstm_long_prob_col: "lstm_p_long"})
    lstm_short = lstm_short.rename(columns={"y_true": "lstm_short_y_true", lstm_short_prob_col: "lstm_p_short"})

    merged = xgb_long[["timestamp", "xgb_long_y_true", "xgb_p_long"]].merge(
        xgb_short[["timestamp", "xgb_short_y_true", "xgb_p_short"]],
        on="timestamp",
        how="inner",
    )
    print(f"\nAfter merge xgb_long + xgb_short: {len(merged)}")

    merged = merged.merge(
        lstm_long[["timestamp", "lstm_long_y_true", "lstm_p_long"]],
        on="timestamp",
        how="inner",
    )
    print(f"After merge + lstm_long: {len(merged)}")

    merged = merged.merge(
        lstm_short[["timestamp", "lstm_short_y_true", "lstm_p_short"]],
        on="timestamp",
        how="inner",
    )
    print(f"After merge + lstm_short: {len(merged)}")

    if len(merged) == 0:
        common = (
            set(xgb_long["timestamp"])
            & set(xgb_short["timestamp"])
            & set(lstm_long["timestamp"])
            & set(lstm_short["timestamp"])
        )
        print("\n===== DEBUG: COMMON TIMESTAMPS ACROSS ALL 4 =====")
        print(f"Common timestamps across all 4 files: {len(common)}")

        raise ValueError(
            "Merged dataset is empty. The four files do not share common timestamps after normalization."
        )

    merged = merged.sort_values("timestamp").reset_index(drop=True)

    long_match = (
        (merged["xgb_long_y_true"].values == merged["lstm_long_y_true"].values).mean()
    )
    short_match = (
        (merged["xgb_short_y_true"].values == merged["lstm_short_y_true"].values).mean()
    )

    print(f"\nLong label match:                {long_match:.4f}")
    print(f"Short label match:               {short_match:.4f}")

    if long_match < 0.999:
        raise ValueError("Long-side labels do not align between XGB and LSTM.")
    if short_match < 0.999:
        raise ValueError("Short-side labels do not align between XGB and LSTM.")

    merged["actual_long"] = merged["xgb_long_y_true"].astype(int)
    merged["actual_short"] = merged["xgb_short_y_true"].astype(int)

    print("\nLoaded files:")
    print(f"  XGB long : {xgb_long_path}")
    print(f"  XGB short: {xgb_short_path}")
    print(f"  LSTM long: {lstm_long_path}")
    print(f"  LSTM short: {lstm_short_path}")

    print(f"\nMerged rows:                     {len(merged)}")
    print(f"Merged min timestamp:            {merged['timestamp'].min()}")
    print(f"Merged max timestamp:            {merged['timestamp'].max()}")

    return merged


def build_dual_side_decisions(
    merged: pd.DataFrame,
    xgb_weight: float,
    lstm_weight: float,
    entry_threshold: float,
    min_edge_gap: float,
) -> pd.DataFrame:
    out = merged.copy()

    out["ens_p_long"] = (
        xgb_weight * out["xgb_p_long"].values + lstm_weight * out["lstm_p_long"].values
    ) / (xgb_weight + lstm_weight)

    out["ens_p_short"] = (
        xgb_weight * out["xgb_p_short"].values + lstm_weight * out["lstm_p_short"].values
    ) / (xgb_weight + lstm_weight)

    decisions = []
    chosen_prob = []

    for p_long, p_short in zip(out["ens_p_long"].values, out["ens_p_short"].values):
        if p_long >= entry_threshold and p_long >= p_short + min_edge_gap:
            decisions.append(1)  # long
            chosen_prob.append(float(p_long))
        elif p_short >= entry_threshold and p_short >= p_long + min_edge_gap:
            decisions.append(0)  # short
            chosen_prob.append(float(p_short))
        else:
            decisions.append(2)  # hold
            chosen_prob.append(max(float(p_long), float(p_short)))

    out["decision"] = decisions  # 1=long, 0=short, 2=hold
    out["chosen_prob"] = chosen_prob

    chosen_actual = []
    for d, a_long, a_short in zip(out["decision"], out["actual_long"], out["actual_short"]):
        if d == 1:
            chosen_actual.append(int(a_long))
        elif d == 0:
            chosen_actual.append(int(a_short))
        else:
            chosen_actual.append(np.nan)

    out["chosen_actual"] = chosen_actual
    return out


def run_dual_side_trade_simulation(
    merged: pd.DataFrame,
    risk: FilterRiskConfig,
) -> dict:
    trade_df = merged[merged["decision"] != 2].copy()

    setups = len(merged)
    taken = len(trade_df)
    skipped = setups - taken

    if taken == 0:
        return {
            "setups": setups,
            "taken": 0,
            "skipped": skipped,
            "take_rate": 0.0,
            "win_rate": 0.0,
            "avg_gross_r_per_trade": 0.0,
            "avg_net_r_per_trade": 0.0,
            "total_return": 0.0,
            "max_drawdown": 0.0,
            "long_taken": 0,
            "short_taken": 0,
            "long_win_rate": 0.0,
            "short_win_rate": 0.0,
            "tp_wins": 0,
            "sl_losses": 0,
        }

    gross_r = np.where(trade_df["chosen_actual"].values == 1, 1.0, -1.0)

    cost_fraction = (risk.fee_bps + risk.trade_penalty_bps) / 10000.0
    cost_r = cost_fraction / max(risk.risk_per_trade, 1e-12)
    net_r = gross_r - cost_r

    equity = 1.0
    equity_curve = [equity]
    for r in net_r:
        next_equity = equity * (1.0 + risk.risk_per_trade * float(r))
        equity = max(next_equity, 0.0)
        equity_curve.append(equity)

    eq = pd.Series(equity_curve)
    max_drawdown = float(((eq / eq.cummax()) - 1.0).min()) if len(eq) else 0.0

    long_df = trade_df[trade_df["decision"] == 1]
    short_df = trade_df[trade_df["decision"] == 0]

    long_win_rate = float((long_df["chosen_actual"].values == 1).mean()) if len(long_df) else 0.0
    short_win_rate = float((short_df["chosen_actual"].values == 1).mean()) if len(short_df) else 0.0

    return {
        "setups": setups,
        "taken": taken,
        "skipped": skipped,
        "take_rate": taken / setups if setups else 0.0,
        "win_rate": float((gross_r > 0).mean()),
        "avg_gross_r_per_trade": float(np.mean(gross_r)),
        "avg_net_r_per_trade": float(np.mean(net_r)),
        "total_return": float(equity - 1.0),
        "max_drawdown": max_drawdown,
        "long_taken": int((trade_df["decision"].values == 1).sum()),
        "short_taken": int((trade_df["decision"].values == 0).sum()),
        "long_win_rate": long_win_rate,
        "short_win_rate": short_win_rate,
        "tp_wins": int((trade_df["chosen_actual"].values == 1).sum()),
        "sl_losses": int((trade_df["chosen_actual"].values == 0).sum()),
    }


def _print_trade_block(title: str, metrics: dict, risk: FilterRiskConfig, entry_threshold: float, min_edge_gap: float):
    print(f"\n---------------- {title} ----------------")
    print(f"Candidate setups:                 {metrics['setups']}")
    print(f"Taken:                            {metrics['taken']}")
    print(f"Skipped:                          {metrics['skipped']}")
    print(f"Take rate on setups:              {metrics['take_rate']:.4f}")
    print(f"Win Rate (taken):                 {metrics['win_rate']:.4f}")
    print(f"Average Gross R / trade:          {metrics['avg_gross_r_per_trade']:.4f}")
    print(f"Average Net R / trade:            {metrics['avg_net_r_per_trade']:.4f}")
    print(f"Total Return:                     {metrics['total_return']:.4f}")
    print(f"Max Drawdown:                     {metrics['max_drawdown']:.4f}")
    print(f"Long trades taken:                {metrics['long_taken']}")
    print(f"Short trades taken:               {metrics['short_taken']}")
    print(f"Long win rate:                    {metrics['long_win_rate']:.4f}")
    print(f"Short win rate:                   {metrics['short_win_rate']:.4f}")
    print(f"TP wins:                          {metrics['tp_wins']}")
    print(f"SL losses:                        {metrics['sl_losses']}")
    print(f"Entry threshold:                  {entry_threshold:.2f}")
    print(f"Min edge gap:                     {min_edge_gap:.2f}")
    print(f"Risk per trade:                   {risk.risk_per_trade:.4f}")
    print(f"Fee bps:                          {risk.fee_bps}")
    print(f"Trade penalty bps:                {risk.trade_penalty_bps}")


def evaluate_dual_side_ensemble(
    xgb_long_csv: str | Path = "xgb_tp_horizon_predictions_long.csv",
    xgb_short_csv: str | Path = "xgb_tp_horizon_predictions_short.csv",
    lstm_long_csv: str | Path = "lstm_tp_horizon_eval_test_probs_long.csv",
    lstm_short_csv: str | Path = "lstm_tp_horizon_eval_test_probs_short.csv",
    risk: FilterRiskConfig = FilterRiskConfig(),
    ensemble_weight_xgb: float = 0.8,
    ensemble_weight_lstm: float = 0.2,
    entry_threshold: float = 0.60,
    min_edge_gap: float = 0.10,
    base_eval_threshold: float = 0.50,
):
    merged = load_and_merge_dual_side_predictions(
        xgb_long_csv=xgb_long_csv,
        xgb_short_csv=xgb_short_csv,
        lstm_long_csv=lstm_long_csv,
        lstm_short_csv=lstm_short_csv,
    )

    print("\n================ SIDE MODELS =================")
    _print_side_metrics("XGB LONG", merged["actual_long"].values, merged["xgb_p_long"].values, base_eval_threshold)
    _print_side_metrics("LSTM LONG", merged["actual_long"].values, merged["lstm_p_long"].values, base_eval_threshold)
    _print_side_metrics("XGB SHORT", merged["actual_short"].values, merged["xgb_p_short"].values, base_eval_threshold)
    _print_side_metrics("LSTM SHORT", merged["actual_short"].values, merged["lstm_p_short"].values, base_eval_threshold)

    decisions = build_dual_side_decisions(
        merged=merged,
        xgb_weight=ensemble_weight_xgb,
        lstm_weight=ensemble_weight_lstm,
        entry_threshold=entry_threshold,
        min_edge_gap=min_edge_gap,
    )

    trade_metrics = run_dual_side_trade_simulation(decisions, risk)

    risk_no_fees = FilterRiskConfig(
        capital_usd=risk.capital_usd,
        risk_per_trade=risk.risk_per_trade,
        fee_bps=0.0,
        trade_penalty_bps=0.0,
    )
    trade_metrics_no_fees = run_dual_side_trade_simulation(decisions, risk_no_fees)

    print("\n================ DUAL-SIDE ENSEMBLE =================")
    print(f"Rows evaluated:                   {len(decisions)}")
    print(f"Ensemble weights:                 xgb={ensemble_weight_xgb:.2f}, lstm={ensemble_weight_lstm:.2f}")

    _print_trade_block(
        title="Dual-Side Ensemble Simulation (WITH FEES)",
        metrics=trade_metrics,
        risk=risk,
        entry_threshold=entry_threshold,
        min_edge_gap=min_edge_gap,
    )

    _print_trade_block(
        title="Dual-Side Ensemble Simulation (NO FEES)",
        metrics=trade_metrics_no_fees,
        risk=risk_no_fees,
        entry_threshold=entry_threshold,
        min_edge_gap=min_edge_gap,
    )

    out_path = _resolve_path("ensemble_dual_side_merged.csv")
    decisions.to_csv(out_path, index=False)
    print(f"\n✅ Saved merged decisions: {out_path}")

    taken_path = _resolve_path("ensemble_dual_side_taken_trades.csv")
    decisions[decisions["decision"] != 2].to_csv(taken_path, index=False)
    print(f"✅ Saved taken trades:     {taken_path}")


if __name__ == "__main__":
    PROJECT_ROOT = get_project_root()

    evaluate_dual_side_ensemble(
        xgb_long_csv=PROJECT_ROOT / "models/xgboost/xgb_tp_horizon_predictions_long.csv",
        xgb_short_csv=PROJECT_ROOT / "models/xgboost/xgb_tp_horizon_predictions_short.csv",
        lstm_long_csv=PROJECT_ROOT / "models/lstm/lstm_tp_horizon_eval_test_probs_long.csv",
        lstm_short_csv=PROJECT_ROOT / "models/lstm/lstm_tp_horizon_eval_test_probs_short.csv",
        risk=FilterRiskConfig(
            capital_usd=5000.0,
            risk_per_trade=0.02,
            fee_bps=2.0,
            trade_penalty_bps=2.0,
        ),
        ensemble_weight_xgb=0.8,
        ensemble_weight_lstm=0.2,
        entry_threshold=0.60,
        min_edge_gap=0.10,
        base_eval_threshold=0.50,
    )