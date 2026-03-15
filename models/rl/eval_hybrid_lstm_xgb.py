# models/rl/eval_hybrid_lstm_xgb.py

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from data.binance import BinanceDataClient
from models.rl.rl_ensemble import RiskConfig


# ============================================================
# 0) Project-root path helpers
# ============================================================

THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[2]   # .../crypto-trading-bot
MODELS_DIR = PROJECT_ROOT / "models"
LSTM_DIR = MODELS_DIR / "lstm"
XGB_DIR = MODELS_DIR / "xgboost"


def resolve_from_root(path_str: str | Path) -> Path:
    """
    Resolve a path robustly from project root.

    Rules:
    - absolute path -> returned as-is
    - relative path starting with 'models/' -> PROJECT_ROOT / relative_path
    - other relative path -> PROJECT_ROOT / relative_path
    """
    p = Path(path_str)
    if p.is_absolute():
        return p
    return (PROJECT_ROOT / p).resolve()


def ensure_exists(path_str: str | Path, label: str) -> Path:
    p = resolve_from_root(path_str)
    if not p.exists():
        raise FileNotFoundError(
            f"{label} not found:\n"
            f"  requested: {path_str}\n"
            f"  resolved : {p}"
        )
    return p


# ============================================================
# 1) Helpers
# ============================================================

def _to_datetime_col(df: pd.DataFrame, col: str = "timestamp") -> pd.DataFrame:
    df = df.copy()
    df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")
    df = df.dropna(subset=[col]).sort_values(col).reset_index(drop=True)
    return df


def _safe_float(x: Any, default: float = np.nan) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _load_lstm_predictions(preds_csv_path: str | Path) -> pd.DataFrame:
    csv_path = ensure_exists(preds_csv_path, "LSTM predictions CSV")
    print(f"✅ Loading LSTM predictions from: {csv_path}")

    df = pd.read_csv(csv_path)
    required = {"timestamp", "p_up"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"LSTM predictions missing required columns: {missing}")

    df = _to_datetime_col(df, "timestamp")

    if "y_true" in df.columns:
        df["y_true"] = pd.to_numeric(df["y_true"], errors="coerce")
    else:
        df["y_true"] = np.nan

    if "actual_trend" not in df.columns:
        df["actual_trend"] = np.where(df["y_true"] == 1, "up", "down")

    return df[["timestamp", "p_up", "y_true", "actual_trend"]].copy()


def _load_xgb_predictions(preds_csv_path: str | Path) -> pd.DataFrame:
    csv_path = ensure_exists(preds_csv_path, "XGB predictions CSV")
    print(f"✅ Loading XGB predictions from: {csv_path}")

    df = pd.read_csv(csv_path)
    required = {"timestamp", "p_up"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"XGB predictions missing required columns: {missing}")

    df = _to_datetime_col(df, "timestamp")

    if "actual_trend" not in df.columns:
        df["actual_trend"] = np.nan

    return df[["timestamp", "p_up", "actual_trend"]].copy()


def _fetch_price_df(
    symbol: str,
    resolution: str,
    start_date: str,
    end_date: str | None,
) -> pd.DataFrame:
    client = BinanceDataClient(market="spot")
    df = client.get_candles(
        symbol=symbol,
        resolution=resolution,
        start_date=start_date,
        end_date=end_date,
    )
    required = {"timestamp", "open", "high", "low", "close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Candle data missing required columns: {missing}")

    df = _to_datetime_col(df, "timestamp")
    return df[["timestamp", "open", "high", "low", "close"]].copy()


def add_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    df = df.copy()
    prev_close = df["close"].shift(1)
    tr1 = df["high"] - df["low"]
    tr2 = (df["high"] - prev_close).abs()
    tr3 = (df["low"] - prev_close).abs()
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    df["atr"] = pd.Series(tr, index=df.index).rolling(period).mean()
    return df


# ============================================================
# 2) Hybrid rule
# ============================================================

def hybrid_signal_from_probs(
    lstm_p_up: float,
    xgb_p_up: float,
    strong_long: float = 0.60,
    moderate_long: float = 0.55,
    moderate_short: float = 0.45,
    strong_short: float = 0.40,
    xgb_long_confirm: float = 0.55,
    xgb_short_confirm: float = 0.45,
) -> int:
    if lstm_p_up >= strong_long:
        return 1

    if lstm_p_up >= moderate_long:
        return 1 if xgb_p_up >= xgb_long_confirm else 0

    if lstm_p_up <= strong_short:
        return -1

    if lstm_p_up <= moderate_short:
        return -1 if xgb_p_up <= xgb_short_confirm else 0

    return 0


def build_hybrid_df(
    lstm_df: pd.DataFrame,
    xgb_df: pd.DataFrame,
    price_df: pd.DataFrame,
    strong_long: float = 0.60,
    moderate_long: float = 0.55,
    moderate_short: float = 0.45,
    strong_short: float = 0.40,
    xgb_long_confirm: float = 0.55,
    xgb_short_confirm: float = 0.45,
) -> pd.DataFrame:
    merged = pd.merge(
        lstm_df.rename(columns={"p_up": "lstm_p_up", "actual_trend": "lstm_actual_trend"}),
        xgb_df.rename(columns={"p_up": "xgb_p_up", "actual_trend": "xgb_actual_trend"}),
        on="timestamp",
        how="inner",
    )

    merged = pd.merge(
        merged,
        price_df,
        on="timestamp",
        how="inner",
    ).sort_values("timestamp").reset_index(drop=True)

    if len(merged) == 0:
        raise ValueError("Merged dataset is empty. Check timestamp alignment of LSTM, XGB, and candles.")

    merged["hybrid_signal"] = merged.apply(
        lambda row: hybrid_signal_from_probs(
            lstm_p_up=float(row["lstm_p_up"]),
            xgb_p_up=float(row["xgb_p_up"]),
            strong_long=strong_long,
            moderate_long=moderate_long,
            moderate_short=moderate_short,
            strong_short=strong_short,
            xgb_long_confirm=xgb_long_confirm,
            xgb_short_confirm=xgb_short_confirm,
        ),
        axis=1,
    )

    if merged["y_true"].notna().any():
        merged["y_true_bin"] = merged["y_true"].astype("Int64")
    else:
        merged["y_true_bin"] = np.where(merged["lstm_actual_trend"] == "up", 1, 0)

    return merged


# ============================================================
# 3) Evaluation metrics
# ============================================================

def classification_summary_from_signal(df: pd.DataFrame) -> dict[str, float]:
    out: dict[str, float] = {}

    non_hold = df["hybrid_signal"] != 0
    n_non_hold = int(non_hold.sum())

    if n_non_hold > 0:
        pred_up = (df.loc[non_hold, "hybrid_signal"] == 1).astype(int).values
        true_up = df.loc[non_hold, "y_true_bin"].astype(int).values
        acc_non_hold = float((pred_up == true_up).mean())
    else:
        acc_non_hold = np.nan

    out["3way_non_hold_acc"] = acc_non_hold
    out["predicted_non_hold_n"] = n_non_hold
    out["2way_acc"] = acc_non_hold
    out["n_long"] = int((df["hybrid_signal"] == 1).sum())
    out["n_short"] = int((df["hybrid_signal"] == -1).sum())
    out["n_hold"] = int((df["hybrid_signal"] == 0).sum())

    return out


# ============================================================
# 4) Trade simulation
# ============================================================

def simulate_hybrid_trades(
    df: pd.DataFrame,
    risk: RiskConfig,
    max_horizon: int = 3,
) -> tuple[pd.DataFrame, dict[str, float]]:
    sim_df = add_atr(df, period=14).copy()
    sim_df = sim_df.dropna(subset=["atr"]).reset_index(drop=True)

    trades: list[dict[str, Any]] = []

    fee_r = (risk.fee_bps + risk.trade_penalty_bps) / 10000.0 * risk.leverage

    for i in range(len(sim_df) - max_horizon):
        row = sim_df.iloc[i]
        signal = int(row["hybrid_signal"])
        if signal == 0:
            continue

        entry = float(row["close"])
        atr = float(row["atr"])
        atr_pct = atr / entry if entry > 0 else np.nan

        if not np.isfinite(atr_pct) or atr_pct < risk.min_atr_pct:
            continue

        stop_dist = max(atr * risk.sl_atr_mult, entry * risk.min_atr_pct)

        if signal == 1:
            sl_price = entry - stop_dist
            tp_price = entry + (risk.rr * stop_dist)
        else:
            sl_price = entry + stop_dist
            tp_price = entry - (risk.rr * stop_dist)

        exit_idx = None
        exit_price = None
        exit_reason = None

        for step in range(1, max_horizon + 1):
            future = sim_df.iloc[i + step]
            high_ = float(future["high"])
            low_ = float(future["low"])
            close_ = float(future["close"])

            if signal == 1:
                hit_tp = high_ >= tp_price
                hit_sl = low_ <= sl_price
            else:
                hit_tp = low_ <= tp_price
                hit_sl = high_ >= sl_price

            if hit_sl:
                exit_idx = i + step
                exit_price = sl_price
                exit_reason = "SL"
                break
            if hit_tp:
                exit_idx = i + step
                exit_price = tp_price
                exit_reason = "TP"
                break

            if step == max_horizon:
                exit_idx = i + step
                exit_price = close_
                exit_reason = "HORIZON"

        if exit_idx is None:
            continue

        if signal == 1:
            gross_r = (exit_price - entry) / stop_dist
        else:
            gross_r = (entry - exit_price) / stop_dist

        net_r = gross_r - fee_r
        win = int(net_r > 0.0)
        dir_correct = int(
            (signal == 1 and sim_df.iloc[exit_idx]["close"] >= entry) or
            (signal == -1 and sim_df.iloc[exit_idx]["close"] <= entry)
        )

        trades.append({
            "entry_timestamp": row["timestamp"],
            "exit_timestamp": sim_df.iloc[exit_idx]["timestamp"],
            "signal": signal,
            "entry_close": entry,
            "exit_price": exit_price,
            "atr": atr,
            "sl_price": sl_price,
            "tp_price": tp_price,
            "exit_reason": exit_reason,
            "gross_r": gross_r,
            "net_r": net_r,
            "win": win,
            "dir_correct": dir_correct,
        })

    trades_df = pd.DataFrame(trades)

    if len(trades_df) == 0:
        summary = {
            "trades_with_fees": 0,
            "dir_acc_with_fees": np.nan,
            "win_rate_with_fees": np.nan,
            "avg_gross_r_with_fees": np.nan,
            "avg_net_r_with_fees": np.nan,
            "total_return_with_fees": np.nan,
            "max_dd_with_fees": np.nan,
            "tp_exits": 0,
            "sl_exits": 0,
            "horizon_exits": 0,
            "other_exits": 0,
        }
        return trades_df, summary

    equity = [1.0]
    for r in trades_df["net_r"].values:
        equity.append(equity[-1] * (1.0 + risk.risk_per_trade * r))
    equity = pd.Series(equity[1:])

    peak = equity.cummax()
    dd = (equity / peak) - 1.0

    summary = {
        "trades_with_fees": int(len(trades_df)),
        "dir_acc_with_fees": float(trades_df["dir_correct"].mean()),
        "win_rate_with_fees": float(trades_df["win"].mean()),
        "avg_gross_r_with_fees": float(trades_df["gross_r"].mean()),
        "avg_net_r_with_fees": float(trades_df["net_r"].mean()),
        "total_return_with_fees": float(equity.iloc[-1] - 1.0),
        "max_dd_with_fees": float(dd.min()),
        "tp_exits": int((trades_df["exit_reason"] == "TP").sum()),
        "sl_exits": int((trades_df["exit_reason"] == "SL").sum()),
        "horizon_exits": int((trades_df["exit_reason"] == "HORIZON").sum()),
        "other_exits": int((~trades_df["exit_reason"].isin(["TP", "SL", "HORIZON"])).sum()),
    }

    return trades_df, summary


# ============================================================
# 5) Main evaluation
# ============================================================

def evaluate_lstm_xgb_hybrid(
    symbol: str = "BTCUSDT",
    resolution: str = "1h",
    start_date: str = "2017-09-01",
    end_date: str | None = None,

    lstm_preds_csv_path: str = "models/lstm/lstm_vol_adj_target_test_probs.csv",
    xgb_preds_csv_path: str = "models/xgboost/xgb_predictions.csv",

    risk: RiskConfig = RiskConfig(
        capital_usd=5000.0,
        risk_per_trade=0.02,
        rr=1.25,
        leverage=25.0,
        fee_bps=2.0,
        trade_penalty_bps=2.0,
        sl_atr_mult=1.0,
        min_atr_pct=0.001,
    ),
    max_horizon: int = 3,

    strong_long: float = 0.60,
    moderate_long: float = 0.55,
    moderate_short: float = 0.45,
    strong_short: float = 0.40,
    xgb_long_confirm: float = 0.55,
    xgb_short_confirm: float = 0.45,

    save_prefix: str = "lstm_xgb_hybrid",
) -> dict[str, Any]:
    lstm_df = _load_lstm_predictions(lstm_preds_csv_path)
    xgb_df = _load_xgb_predictions(xgb_preds_csv_path)
    price_df = _fetch_price_df(
        symbol=symbol,
        resolution=resolution,
        start_date=start_date,
        end_date=end_date,
    )

    merged = build_hybrid_df(
        lstm_df=lstm_df,
        xgb_df=xgb_df,
        price_df=price_df,
        strong_long=strong_long,
        moderate_long=moderate_long,
        moderate_short=moderate_short,
        strong_short=strong_short,
        xgb_long_confirm=xgb_long_confirm,
        xgb_short_confirm=xgb_short_confirm,
    )

    cls_summary = classification_summary_from_signal(merged)
    trades_df, trade_summary = simulate_hybrid_trades(
        df=merged,
        risk=risk,
        max_horizon=max_horizon,
    )

    summary_row = {
        "method": "LSTM strong + XGB confirms moderate",
        **cls_summary,
        **trade_summary,
        "strong_long": strong_long,
        "moderate_long": moderate_long,
        "moderate_short": moderate_short,
        "strong_short": strong_short,
        "xgb_long_confirm": xgb_long_confirm,
        "xgb_short_confirm": xgb_short_confirm,
        "max_horizon": max_horizon,
    }
    summary_df = pd.DataFrame([summary_row])

    save_prefix_path = resolve_from_root(save_prefix)
    save_dir = save_prefix_path.parent
    save_stem = save_prefix_path.name
    save_dir.mkdir(parents=True, exist_ok=True)

    merged.to_csv(save_dir / f"{save_stem}_merged_signals.csv", index=False)
    trades_df.to_csv(save_dir / f"{save_stem}_trades.csv", index=False)
    summary_df.to_csv(save_dir / f"{save_stem}_summary.csv", index=False)

    print("\n================ HYBRID SUMMARY ================")
    print(summary_df.to_string(index=False))

    print("\n---------------- Trade Simulation (WITH FEES) ----------------")
    print(f"Candidate setups from signal:    {int((merged['hybrid_signal'] != 0).sum())}")
    print(f"Taken:                           {int(len(trades_df))}")
    print(f"Directional Accuracy (taken):    {_safe_float(summary_row['dir_acc_with_fees']):.4f}")
    print(f"Win Rate (taken):                {_safe_float(summary_row['win_rate_with_fees']):.4f}")
    print(f"Average Gross R / trade:         {_safe_float(summary_row['avg_gross_r_with_fees']):.4f}")
    print(f"Average Net R / trade:           {_safe_float(summary_row['avg_net_r_with_fees']):.4f}")
    print(f"Total Return:                    {_safe_float(summary_row['total_return_with_fees']):.4f}")
    print(f"Max Drawdown:                    {_safe_float(summary_row['max_dd_with_fees']):.4f}")
    print(f"TP exits:                        {int(summary_row['tp_exits'])}")
    print(f"SL exits:                        {int(summary_row['sl_exits'])}")
    print(f"Horizon exits:                   {int(summary_row['horizon_exits'])}")
    print(f"Other exits:                     {int(summary_row['other_exits'])}")
    print(f"Max horizon:                     {max_horizon}")
    print(f"RR target:                       {risk.rr:.2f}")
    print(f"SL ATR multiplier:               {risk.sl_atr_mult:.2f}")
    print(f"Risk per trade:                  {risk.risk_per_trade:.4f}")
    print(f"Fee bps:                         {risk.fee_bps:.1f}")
    print(f"Trade penalty bps:               {risk.trade_penalty_bps:.1f}")

    print("\nSaved files:")
    print(f"  {(save_dir / f'{save_stem}_merged_signals.csv')}")
    print(f"  {(save_dir / f'{save_stem}_trades.csv')}")
    print(f"  {(save_dir / f'{save_stem}_summary.csv')}")

    return {
        "merged_df": merged,
        "trades_df": trades_df,
        "summary_df": summary_df,
    }


if __name__ == "__main__":
    evaluate_lstm_xgb_hybrid(
        symbol="BTCUSDT",
        resolution="1h",
        start_date="2017-09-01",
        end_date=None,
        lstm_preds_csv_path="models/lstm/lstm_vol_adj_target_test_probs.csv",
        xgb_preds_csv_path="models/xgboost/xgb_predictions.csv",
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
        max_horizon=3,
        strong_long=0.60,
        moderate_long=0.55,
        moderate_short=0.45,
        strong_short=0.40,
        xgb_long_confirm=0.55,
        xgb_short_confirm=0.45,
        save_prefix="models/rl/lstm_xgb_hybrid",
    )
