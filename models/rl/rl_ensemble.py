#models/rl/rl_ensemble.py
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd

from joblib import load, dump
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler

from data.binance import BinanceDataClient
from data.feature_engineering import feature_engineering_xgb, feature_engineering_lstm
from models.lstm.sequence_builder import make_windows


# -----------------------------
# Path helpers
# -----------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RL_DIR = PROJECT_ROOT / "models" / "rl"
RL_SAVED_DIR = RL_DIR / "saved"

START_DATES_BY_INTERVAL = {
    "5m": "2025-01-01",
    "15m": "2024-01-01",
    "1h": "2017-09-01",
    "4h": "2017-09-01",
}


def resolve_artifact_path(path_like: str | Path) -> str:
    p = Path(path_like)
    if p.is_absolute():
        return str(p)
    return str((PROJECT_ROOT / p).resolve())


def get_default_start_date(resolution: str) -> str:
    if resolution not in START_DATES_BY_INTERVAL:
        raise ValueError(
            f"Unsupported resolution '{resolution}'. "
            f"Expected one of: {list(START_DATES_BY_INTERVAL.keys())}"
        )
    return START_DATES_BY_INTERVAL[resolution]


def symbol_tag(symbol: str) -> str:
    symbol = symbol.upper()
    if symbol.endswith("USDT"):
        return symbol[:-4]
    return symbol


def combo_tag(symbol: str, resolution: str) -> str:
    return f"{symbol_tag(symbol)}_{resolution}"


def build_combo_artifact_paths(symbol: str, resolution: str) -> dict[str, Path]:
    tag = combo_tag(symbol, resolution)
    RL_SAVED_DIR.mkdir(parents=True, exist_ok=True)

    return {
        "xgb_artifacts_path": PROJECT_ROOT / "models" / "xgboost" / "saved" / f"xgb_artifacts_{tag}.joblib",
        "lstm_artifacts_path": PROJECT_ROOT / "models" / "lstm" / "saved" / f"lstm_artifacts_{tag}.joblib",
        "rl_agent_path": RL_SAVED_DIR / f"rl_qtable_agent_{tag}.joblib",
    }


# -----------------------------
# Output / Risk config
# -----------------------------

@dataclass
class TradeSignal:
    action: str
    confidence: float
    entry: float
    stop_loss: Optional[float]
    take_profit: Optional[float]
    position_size_usd: float
    leverage: float
    meta: Dict[str, float | str | int]


@dataclass
class RiskConfig:
    capital_usd: float = 5000.0
    risk_per_trade: float = 0.02
    rr: float = 1.25
    leverage: float = 1.0
    fee_bps: float = 2.0
    trade_penalty_bps: float = 2.0
    sl_atr_mult: float = 1.0
    min_atr_pct: float = 0.001


# -----------------------------
# RL filter agent
# -----------------------------

ACTIONS = ["skip", "take"]
ACTION_TO_IDX = {a: i for i, a in enumerate(ACTIONS)}


def _clip01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))


def _bucket(x: float, edges: Tuple[float, ...]) -> int:
    for i, e in enumerate(edges):
        if x < e:
            return i
    return len(edges)


class QTableAgent:
    def __init__(
        self,
        n_states: int,
        n_actions: int = 2,
        alpha: float = 0.10,
        gamma: float = 0.95,
        eps: float = 0.10,
        eps_decay: float = 0.999,
        eps_min: float = 0.02,
        seed: int = 42,
    ):
        self.n_states = int(n_states)
        self.n_actions = int(n_actions)
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.eps = float(eps)
        self.eps_decay = float(eps_decay)
        self.eps_min = float(eps_min)
        self.rng = np.random.default_rng(seed)
        self.Q = np.zeros((self.n_states, self.n_actions), dtype=np.float32)
        self.visits = np.zeros((self.n_states, self.n_actions), dtype=np.int32)

    def act(self, s: int, greedy: bool = False) -> int:
        s = int(s)
        if greedy or (self.rng.random() > self.eps):
            return int(np.argmax(self.Q[s]))
        return int(self.rng.integers(0, self.n_actions))

    def update(self, s: int, a: int, r: float, s2: int, done: bool):
        s = int(s)
        a = int(a)
        s2 = int(s2)
        r = float(r)

        self.visits[s, a] += 1

        best_next = 0.0 if done else float(np.max(self.Q[s2]))
        td_target = r + self.gamma * best_next
        td_error = td_target - float(self.Q[s, a])

        lr = self.alpha / max(1.0, np.sqrt(float(self.visits[s, a])))
        self.Q[s, a] = float(self.Q[s, a] + lr * td_error)

    def decay_eps(self):
        self.eps = max(self.eps_min, self.eps * self.eps_decay)

    def save(self, path: str | Path):
        dump(
            {
                "Q": self.Q,
                "visits": self.visits,
                "meta": {
                    "n_states": self.n_states,
                    "n_actions": self.n_actions,
                    "alpha": self.alpha,
                    "gamma": self.gamma,
                    "eps": self.eps,
                    "eps_decay": self.eps_decay,
                    "eps_min": self.eps_min,
                },
            },
            resolve_artifact_path(path),
        )

    @staticmethod
    def load(path: str | Path) -> "QTableAgent":
        obj = load(resolve_artifact_path(path))
        Q = obj["Q"]
        visits = obj.get("visits", np.zeros_like(Q, dtype=np.int32))
        meta = obj["meta"]

        agent = QTableAgent(
            n_states=int(meta["n_states"]),
            n_actions=int(meta["n_actions"]),
            alpha=float(meta["alpha"]),
            gamma=float(meta["gamma"]),
            eps=float(meta["eps"]),
            eps_decay=float(meta["eps_decay"]),
            eps_min=float(meta["eps_min"]),
            seed=42,
        )
        agent.Q = Q
        agent.visits = visits
        return agent


# -----------------------------
# Decision helper
# -----------------------------

def choose_action_with_margin(
    agent: QTableAgent,
    state_idx: int,
    min_take_visits: int = 20,
    q_take_margin: float = 0.0,
) -> tuple[int, str, dict]:
    s = int(state_idx)

    q_skip = float(agent.Q[s, ACTION_TO_IDX["skip"]])
    q_take = float(agent.Q[s, ACTION_TO_IDX["take"]])

    skip_visits = int(agent.visits[s, ACTION_TO_IDX["skip"]])
    take_visits = int(agent.visits[s, ACTION_TO_IDX["take"]])

    if take_visits < min_take_visits and skip_visits < min_take_visits:
        a_idx = ACTION_TO_IDX["skip"]
        decision = "skip"
    elif take_visits < min_take_visits:
        a_idx = ACTION_TO_IDX["skip"]
        decision = "skip"
    else:
        if q_take >= q_skip + float(q_take_margin):
            a_idx = ACTION_TO_IDX["take"]
            decision = "take"
        else:
            a_idx = ACTION_TO_IDX["skip"]
            decision = "skip"

    info = {
        "q_skip": q_skip,
        "q_take": q_take,
        "q_gap_take_minus_skip": float(q_take - q_skip),
        "q_take_margin": float(q_take_margin),
        "skip_visits": skip_visits,
        "take_visits": take_visits,
    }
    return int(a_idx), str(decision), info


# -----------------------------
# Prediction wrappers
# -----------------------------

def predict_xgb_series(
    df_raw: pd.DataFrame,
    xgb_artifacts_path: str | Path,
) -> pd.DataFrame:
    artifacts = load(resolve_artifact_path(xgb_artifacts_path))
    model = artifacts["model"]
    features = artifacts["features"]
    decision_boundary = float(artifacts.get("decision_boundary", 0.5))
    margin_threshold = float(artifacts.get("margin_threshold", 0.0))

    df = df_raw.sort_values("timestamp").reset_index(drop=True).copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = feature_engineering_xgb(df)

    keep_market_cols = [c for c in ["open", "high", "low", "close", "atr"] if c in df.columns]
    df = df.dropna(subset=features + ["close"]).reset_index(drop=True)

    df["close_next_raw"] = df["close"].shift(-1)
    df["actual"] = (df["close_next_raw"] > df["close"]).astype(int)
    df = df.dropna(subset=["close_next_raw"]).reset_index(drop=True)

    probs = model.predict_proba(df[features].values)
    p_up = probs[:, 1]

    margin = np.abs(p_up - decision_boundary)
    xgb_used = (margin >= margin_threshold).astype(int)

    xgb_pred = np.full(len(p_up), 2, dtype=int)
    confident = xgb_used.astype(bool)
    xgb_pred[confident] = (p_up[confident] > decision_boundary).astype(int)

    keep_cols = ["timestamp"] + keep_market_cols + ["close_next_raw", "actual"]
    out = df[keep_cols].copy()
    out["xgb_p_up"] = p_up.astype(float)
    out["xgb_used"] = xgb_used.astype(int)
    out["xgb_pred"] = xgb_pred.astype(int)
    out["xgb_margin"] = margin.astype(float)
    out["xgb_boundary"] = decision_boundary
    out["xgb_margin_threshold"] = margin_threshold
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True)
    return out


def predict_lstm_series(
    df_raw: pd.DataFrame,
    lstm_artifacts_path: str | Path,
    threshold: float = 0.52,
    batch_size: int = 256,
) -> pd.DataFrame:
    art = load(resolve_artifact_path(lstm_artifacts_path))
    model_path = resolve_artifact_path(art["model_path"])
    scaler_path = resolve_artifact_path(art["scaler_path"])
    x_window_size = int(art["x_window_size"])
    feature_cols = list(art["feature_cols"])

    df = df_raw.sort_values("timestamp").reset_index(drop=True).copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = feature_engineering_lstm(df)
    df = df.dropna(subset=feature_cols + ["close"]).reset_index(drop=True)

    X, _ = make_windows(df, x_window_size=x_window_size, feature_cols=feature_cols)
    X = np.asarray(X, dtype=np.float32)
    if len(X) == 0:
        raise ValueError("No LSTM windows could be created (check data length / dropna).")

    scaler: StandardScaler = load(scaler_path)
    model = load_model(model_path)

    n_features = X.shape[-1]
    Xs = scaler.transform(X.reshape(-1, n_features)).reshape(X.shape).astype(np.float32)
    p = model.predict(Xs, batch_size=batch_size, verbose=0).reshape(-1).astype(float)

    lower = 1.0 - float(threshold)
    upper = float(threshold)

    lstm_pred = np.full(len(p), 2, dtype=int)
    lstm_pred[p >= upper] = 1
    lstm_pred[p < lower] = 0
    lstm_used = (lstm_pred != 2).astype(int)

    signal_idx = np.arange(x_window_size - 1, x_window_size - 1 + len(p))
    target_idx = signal_idx + 1

    keep_market_cols = [c for c in ["open", "high", "low", "close", "atr"] if c in df.columns]

    out = df.loc[signal_idx, ["timestamp"] + keep_market_cols].copy().reset_index(drop=True)
    out["close_next_raw"] = df.loc[target_idx, "close"].values.astype(float)
    out["actual"] = (
        df.loc[target_idx, "close"].values > df.loc[signal_idx, "close"].values
    ).astype(int)
    out["lstm_p_up"] = p
    out["lstm_pred"] = lstm_pred
    out["lstm_used"] = lstm_used
    out["lstm_threshold"] = float(threshold)
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True)
    return out


# -----------------------------
# Ensemble helpers
# -----------------------------

def build_direction_from_ensemble(
    xgb_p: float,
    lstm_p: float,
    xgb_weight: float = 0.8,
    lstm_weight: float = 0.2,
    upper: float = 0.60,
    lower: float = 0.40,
) -> tuple[str, float, int]:
    w_sum = float(xgb_weight) + float(lstm_weight)
    if w_sum <= 0:
        raise ValueError("Ensemble weights must sum to > 0")

    wx = float(xgb_weight) / w_sum
    wl = float(lstm_weight) / w_sum

    p_ens = wx * float(xgb_p) + wl * float(lstm_p)

    if p_ens >= float(upper):
        return "long", float(p_ens), 1
    if p_ens <= float(lower):
        return "short", float(p_ens), -1
    return "hold", float(p_ens), 0


# -----------------------------
# State builder for trade filter
# conf_ens(4) * dis(4) * atr(3) * agree(2) * side(2) = 192 states
# -----------------------------

def build_filter_state_index(
    p_ens: float,
    xgb_p: float,
    lstm_p: float,
    atr_pct: float,
    side: int,
) -> int:
    conf_ens = abs(float(p_ens) - 0.5) * 2.0
    dis = abs(float(xgb_p) - float(lstm_p))
    agree = int((float(xgb_p) >= 0.5) == (float(lstm_p) >= 0.5))
    atr_pct = max(0.0, float(atr_pct))

    b_conf = _bucket(conf_ens, (0.25, 0.50, 0.75))
    b_dis = _bucket(dis, (0.05, 0.15, 0.30))
    b_atr = _bucket(atr_pct, (0.004, 0.012))

    idx = b_conf
    idx += 4 * b_dis
    idx += 16 * b_atr
    idx += 48 * int(agree)
    idx += 96 * int(side)
    return int(idx)


N_STATES = 4 * 4 * 3 * 2 * 2


# -----------------------------
# Reward + trade parameterization
# -----------------------------

def _compute_atr_pct(row: pd.Series) -> float:
    if "atr" in row and pd.notna(row["atr"]) and float(row["close"]) > 0:
        return float(row["atr"]) / float(row["close"])
    return 0.0


def _sl_tp_from_atr(
    entry: float,
    atr: float,
    action: str,
    rr: float,
    sl_atr_mult: float,
) -> Tuple[Optional[float], Optional[float]]:
    if not (atr and atr > 0 and entry > 0):
        return None, None

    sl_dist = sl_atr_mult * atr
    tp_dist = rr * sl_dist

    if action == "long":
        return entry - sl_dist, entry + tp_dist
    if action == "short":
        return entry + sl_dist, entry - tp_dist
    return None, None


def _position_size_from_risk(
    capital: float,
    risk_pct: float,
    entry: float,
    stop_loss: Optional[float],
) -> float:
    if stop_loss is None:
        return 0.0
    risk_amount = float(capital) * float(risk_pct)
    sl_dist = abs(float(entry) - float(stop_loss))
    if sl_dist <= 0:
        return 0.0
    return float(max(0.0, risk_amount * float(entry) / sl_dist))


def _cost_in_r(
    entry: float,
    stop_loss: float,
    fee_bps: float,
    trade_penalty_bps: float,
) -> float:
    sl_pct = abs(float(entry) - float(stop_loss)) / float(entry)
    if sl_pct <= 0:
        return 0.0

    round_trip_fee_pct = 2.0 * (float(fee_bps) / 10000.0)
    extra_penalty_pct = float(trade_penalty_bps) / 10000.0
    total_cost_pct = round_trip_fee_pct + extra_penalty_pct
    return float(total_cost_pct / sl_pct)


def _trade_r_multiple_from_exit(
    entry: float,
    exit_price: float,
    stop_loss: float,
    direction_sign: int,
) -> float:
    risk_per_unit = abs(float(entry) - float(stop_loss))
    if risk_per_unit <= 0:
        return 0.0
    pnl_per_unit = float(direction_sign) * (float(exit_price) - float(entry))
    return float(pnl_per_unit / risk_per_unit)


def simulate_trade_outcome(
    merged: pd.DataFrame,
    idx: int,
    direction_name: str,
    risk: RiskConfig,
    max_horizon: int = 3,
) -> Dict[str, float | int | str | None]:
    row = merged.iloc[idx]
    entry = float(row["close"])

    if "atr" in merged.columns and pd.notna(row.get("atr", np.nan)):
        atr = float(row["atr"])
    else:
        atr = entry * float(risk.min_atr_pct)

    stop_loss, take_profit = _sl_tp_from_atr(
        entry=entry,
        atr=atr,
        action=direction_name,
        rr=risk.rr,
        sl_atr_mult=risk.sl_atr_mult,
    )

    if stop_loss is None or take_profit is None:
        return {
            "reward_r": 0.0,
            "gross_r": 0.0,
            "exit_index": int(min(idx + 1, len(merged) - 1)),
            "exit_reason": "invalid_sl_tp",
            "entry": entry,
            "exit_price": entry,
            "stop_loss": None,
            "take_profit": None,
            "atr": atr,
        }

    direction_sign = 1 if direction_name == "long" else -1
    last_j = min(len(merged) - 1, idx + int(max_horizon))
    exit_index = last_j
    exit_reason = "horizon"
    exit_price = float(merged.iloc[last_j]["close"])

    for j in range(idx + 1, last_j + 1):
        future = merged.iloc[j]
        high = (
            float(future["high"])
            if "high" in merged.columns and pd.notna(future["high"])
            else float(future["close"])
        )
        low = (
            float(future["low"])
            if "low" in merged.columns and pd.notna(future["low"])
            else float(future["close"])
        )

        if direction_name == "long":
            sl_hit = low <= stop_loss
            tp_hit = high >= take_profit
        else:
            sl_hit = high >= stop_loss
            tp_hit = low <= take_profit

        if sl_hit and tp_hit:
            exit_index = j
            exit_reason = "sl_tp_same_bar_sl_first"
            exit_price = float(stop_loss)
            break
        if sl_hit:
            exit_index = j
            exit_reason = "sl"
            exit_price = float(stop_loss)
            break
        if tp_hit:
            exit_index = j
            exit_reason = "tp"
            exit_price = float(take_profit)
            break

    gross_r = _trade_r_multiple_from_exit(
        entry=entry,
        exit_price=exit_price,
        stop_loss=float(stop_loss),
        direction_sign=direction_sign,
    )
    cost_r = _cost_in_r(
        entry=entry,
        stop_loss=float(stop_loss),
        fee_bps=risk.fee_bps,
        trade_penalty_bps=risk.trade_penalty_bps,
    )
    reward_r = float(gross_r - cost_r)

    return {
        "reward_r": reward_r,
        "gross_r": float(gross_r),
        "exit_index": int(exit_index),
        "exit_reason": str(exit_reason),
        "entry": float(entry),
        "exit_price": float(exit_price),
        "stop_loss": float(stop_loss),
        "take_profit": float(take_profit),
        "atr": float(atr),
    }


def reward_for_action(
    take_action_idx: int,
    merged: pd.DataFrame,
    idx: int,
    direction_name: str,
    risk: RiskConfig,
    max_horizon: int = 3,
    skip_reward_scale: float = 0.15,
) -> Tuple[float, int, Dict[str, float | int | str | None]]:
    sim = simulate_trade_outcome(
        merged=merged,
        idx=idx,
        direction_name=direction_name,
        risk=risk,
        max_horizon=max_horizon,
    )

    take_reward = float(sim["reward_r"])

    if int(take_action_idx) == ACTION_TO_IDX["take"]:
        next_idx = int(sim["exit_index"]) + 1
        return take_reward, next_idx, sim

    skip_reward = -float(skip_reward_scale) * take_reward
    next_idx = idx + 1
    return float(skip_reward), next_idx, sim


# -----------------------------
# Shared merged-data builder
# -----------------------------

def build_merged_dataset(
    df_raw: pd.DataFrame,
    xgb_artifacts_path: str | Path,
    lstm_artifacts_path: str | Path,
    lstm_threshold: float,
) -> pd.DataFrame:
    xgb_df = predict_xgb_series(df_raw, xgb_artifacts_path=xgb_artifacts_path)
    lstm_df = predict_lstm_series(
        df_raw,
        lstm_artifacts_path=lstm_artifacts_path,
        threshold=lstm_threshold,
    )

    merged = pd.merge(
        xgb_df,
        lstm_df,
        on="timestamp",
        how="inner",
        suffixes=("_xgb", "_lstm"),
    )
    merged = merged.sort_values("timestamp").reset_index(drop=True)

    if len(merged) < 5:
        raise ValueError("Not enough aligned samples between XGB and LSTM outputs.")

    close_match = np.isclose(
        merged["close_xgb"].values,
        merged["close_lstm"].values,
        rtol=1e-10,
        atol=1e-10,
        equal_nan=False,
    ).mean()

    label_match = (merged["actual_xgb"].values == merged["actual_lstm"].values).mean()

    print(f"Close match on merged rows: {close_match:.4f}")
    print(f"Label match on merged rows: {label_match:.4f}")

    merged["close"] = merged["close_lstm"]
    merged["close_next_raw"] = merged["close_next_raw_lstm"]
    merged["actual"] = merged["actual_lstm"]
    merged["ret_next"] = (merged["close_next_raw"] / merged["close"]) - 1.0

    for col in ["open", "high", "low"]:
        if f"{col}_lstm" in merged.columns:
            merged[col] = merged[f"{col}_lstm"]
        elif f"{col}_xgb" in merged.columns:
            merged[col] = merged[f"{col}_xgb"]

    if "atr_lstm" in merged.columns and merged["atr_lstm"].notna().any():
        merged["atr"] = merged["atr_lstm"]
    elif "atr_xgb" in merged.columns and merged["atr_xgb"].notna().any():
        merged["atr"] = merged["atr_xgb"]

    return merged


# -----------------------------
# State helpers
# -----------------------------

def _state_from_row(
    row: pd.Series,
    ensemble_weight_xgb: float,
    ensemble_weight_lstm: float,
    ensemble_upper: float,
    ensemble_lower: float,
) -> Tuple[Optional[int], Optional[str], Optional[float], Optional[int]]:
    direction_name, p_ens, direction_sign = build_direction_from_ensemble(
        xgb_p=float(row["xgb_p_up"]),
        lstm_p=float(row["lstm_p_up"]),
        xgb_weight=ensemble_weight_xgb,
        lstm_weight=ensemble_weight_lstm,
        upper=ensemble_upper,
        lower=ensemble_lower,
    )

    if direction_name == "hold":
        return None, None, None, None

    side = 1 if direction_name == "long" else 0
    s = build_filter_state_index(
        p_ens=float(p_ens),
        xgb_p=float(row["xgb_p_up"]),
        lstm_p=float(row["lstm_p_up"]),
        atr_pct=_compute_atr_pct(row),
        side=side,
    )
    return int(s), direction_name, float(p_ens), int(direction_sign)


def _next_state_from_index(
    merged: pd.DataFrame,
    start_idx: int,
    ensemble_weight_xgb: float,
    ensemble_weight_lstm: float,
    ensemble_upper: float,
    ensemble_lower: float,
) -> Tuple[int, bool]:
    n = len(merged)
    i = int(start_idx)

    while i < n:
        s, _, _, _ = _state_from_row(
            row=merged.iloc[i],
            ensemble_weight_xgb=ensemble_weight_xgb,
            ensemble_weight_lstm=ensemble_weight_lstm,
            ensemble_upper=ensemble_upper,
            ensemble_lower=ensemble_lower,
        )
        if s is not None:
            return int(s), False
        i += 1

    return 0, True


# -----------------------------
# Training
# -----------------------------

def train_rl_policy(
    symbol: str = "BTCUSDT",
    resolution: str = "1h",
    start_date: str | None = None,
    end_date: str | None = None,
    train_ratio: float = 0.80,
    xgb_artifacts_path: str | Path | None = None,
    lstm_artifacts_path: str | Path | None = None,
    lstm_threshold: float = 0.52,
    agent_out_path: str | Path | None = None,
    alpha: float = 0.20,
    gamma: float = 0.95,
    eps: float = 0.25,
    episodes: int = 60,
    risk: RiskConfig = RiskConfig(),
    ensemble_weight_xgb: float = 0.8,
    ensemble_weight_lstm: float = 0.2,
    ensemble_upper: float = 0.60,
    ensemble_lower: float = 0.40,
    max_horizon: int = 3,
    skip_reward_scale: float = 0.15,
) -> QTableAgent:
    symbol = symbol.upper()
    if start_date is None:
        start_date = get_default_start_date(resolution)

    combo_paths = build_combo_artifact_paths(symbol, resolution)

    xgb_artifacts_path_p = (
        Path(resolve_artifact_path(xgb_artifacts_path))
        if xgb_artifacts_path
        else combo_paths["xgb_artifacts_path"]
    )
    lstm_artifacts_path_p = (
        Path(resolve_artifact_path(lstm_artifacts_path))
        if lstm_artifacts_path
        else combo_paths["lstm_artifacts_path"]
    )
    agent_out_path_p = (
        Path(resolve_artifact_path(agent_out_path))
        if agent_out_path
        else combo_paths["rl_agent_path"]
    )

    agent_out_path_p.parent.mkdir(parents=True, exist_ok=True)

    print("\n================ RL TRAIN CONFIG ================")
    print(f"Symbol:             {symbol}")
    print(f"Resolution:         {resolution}")
    print(f"Start date:         {start_date}")
    print(f"End date:           {end_date}")
    print(f"Train ratio:        {train_ratio}")
    print(f"Combo tag:          {combo_tag(symbol, resolution)}")
    print(f"XGB artifacts:      {xgb_artifacts_path_p}")
    print(f"LSTM artifacts:     {lstm_artifacts_path_p}")
    print(f"RL agent out:       {agent_out_path_p}")
    print("=================================================\n")

    client = BinanceDataClient(market="spot")
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

    train_raw = df_raw.iloc[:split].copy()

    merged = build_merged_dataset(
        df_raw=train_raw,
        xgb_artifacts_path=xgb_artifacts_path_p,
        lstm_artifacts_path=lstm_artifacts_path_p,
        lstm_threshold=lstm_threshold,
    )

    agent = QTableAgent(
        n_states=N_STATES,
        n_actions=len(ACTIONS),
        alpha=alpha,
        gamma=gamma,
        eps=eps,
        eps_decay=0.999,
        eps_min=0.02,
        seed=42,
    )

    for ep in range(int(episodes)):
        t = 0
        setups = 0
        taken = 0
        skipped = 0
        total_reward_r = 0.0

        while t < len(merged) - 1:
            row = merged.iloc[t]

            s, direction_name, _, _ = _state_from_row(
                row=row,
                ensemble_weight_xgb=ensemble_weight_xgb,
                ensemble_weight_lstm=ensemble_weight_lstm,
                ensemble_upper=ensemble_upper,
                ensemble_lower=ensemble_lower,
            )

            if s is None:
                t += 1
                continue

            setups += 1
            a = agent.act(s, greedy=False)

            reward_r, next_idx, _ = reward_for_action(
                take_action_idx=a,
                merged=merged,
                idx=t,
                direction_name=str(direction_name),
                risk=risk,
                max_horizon=max_horizon,
                skip_reward_scale=skip_reward_scale,
            )

            if a == ACTION_TO_IDX["take"]:
                taken += 1
            else:
                skipped += 1

            total_reward_r += reward_r

            s2, done = _next_state_from_index(
                merged=merged,
                start_idx=next_idx,
                ensemble_weight_xgb=ensemble_weight_xgb,
                ensemble_weight_lstm=ensemble_weight_lstm,
                ensemble_upper=ensemble_upper,
                ensemble_lower=ensemble_lower,
            )

            agent.update(s, a, reward_r, s2, done)
            t = next_idx

        agent.decay_eps()
        print(
            f"Episode {ep + 1}/{episodes} complete | "
            f"eps={agent.eps:.4f} | setups={setups} | taken={taken} | "
            f"skipped={skipped} | total_R={total_reward_r:.2f}"
        )

    agent.save(agent_out_path_p)
    print(f"✅ Saved RL agent to {resolve_artifact_path(agent_out_path_p)}")
    return agent


# -----------------------------
# Inference
# -----------------------------

def get_trade_signal_rl(
    symbol: str = "BTCUSDT",
    resolution: str = "1h",
    start_date: str | None = None,
    end_date: str | None = None,
    xgb_artifacts_path: str | Path | None = None,
    lstm_artifacts_path: str | Path | None = None,
    lstm_threshold: float = 0.52,
    rl_agent_path: str | Path | None = None,
    risk: RiskConfig = RiskConfig(),
    ensemble_weight_xgb: float = 0.8,
    ensemble_weight_lstm: float = 0.2,
    ensemble_upper: float = 0.60,
    ensemble_lower: float = 0.40,
    min_take_visits: int = 20,
    q_take_margin: float = 0.0,
) -> TradeSignal:
    symbol = symbol.upper()
    if start_date is None:
        start_date = get_default_start_date(resolution)

    combo_paths = build_combo_artifact_paths(symbol, resolution)

    xgb_artifacts_path_p = (
        Path(resolve_artifact_path(xgb_artifacts_path))
        if xgb_artifacts_path
        else combo_paths["xgb_artifacts_path"]
    )
    lstm_artifacts_path_p = (
        Path(resolve_artifact_path(lstm_artifacts_path))
        if lstm_artifacts_path
        else combo_paths["lstm_artifacts_path"]
    )
    rl_agent_path_p = (
        Path(resolve_artifact_path(rl_agent_path))
        if rl_agent_path
        else combo_paths["rl_agent_path"]
    )

    agent = QTableAgent.load(rl_agent_path_p)

    client = BinanceDataClient(market="spot")
    df_raw = client.get_candles(
        symbol=symbol,
        resolution=resolution,
        start_date=start_date,
        end_date=end_date,
    ).sort_values("timestamp").reset_index(drop=True)

    merged = build_merged_dataset(
        df_raw=df_raw,
        xgb_artifacts_path=xgb_artifacts_path_p,
        lstm_artifacts_path=lstm_artifacts_path_p,
        lstm_threshold=lstm_threshold,
    )

    last = merged.iloc[-1]

    direction_name, p_ens, direction_sign = build_direction_from_ensemble(
        xgb_p=float(last["xgb_p_up"]),
        lstm_p=float(last["lstm_p_up"]),
        xgb_weight=ensemble_weight_xgb,
        lstm_weight=ensemble_weight_lstm,
        upper=ensemble_upper,
        lower=ensemble_lower,
    )

    entry = float(last["close"])

    if "atr" in merged.columns and pd.notna(last.get("atr", np.nan)):
        atr = float(last["atr"])
    else:
        atr = entry * float(risk.min_atr_pct)

    atr_pct = atr / entry if entry > 0 else 0.0

    if direction_name == "hold":
        return TradeSignal(
            action="hold",
            confidence=_clip01(abs(float(p_ens) - 0.5) * 2.0),
            entry=entry,
            stop_loss=None,
            take_profit=None,
            position_size_usd=0.0,
            leverage=risk.leverage,
            meta={
                "symbol": symbol,
                "resolution": resolution,
                "combo_tag": combo_tag(symbol, resolution),
                "timestamp": str(last["timestamp"]),
                "xgb_artifacts_path": str(xgb_artifacts_path_p),
                "lstm_artifacts_path": str(lstm_artifacts_path_p),
                "rl_agent_path": str(rl_agent_path_p),
                "xgb_p_up": float(last["xgb_p_up"]),
                "lstm_p_up": float(last["lstm_p_up"]),
                "ensemble_p_up": float(p_ens),
                "ensemble_direction": "hold",
                "ensemble_upper": float(ensemble_upper),
                "ensemble_lower": float(ensemble_lower),
                "skip_reason": "ensemble_no_setup",
            },
        )

    side = 1 if direction_name == "long" else 0

    s = build_filter_state_index(
        p_ens=float(p_ens),
        xgb_p=float(last["xgb_p_up"]),
        lstm_p=float(last["lstm_p_up"]),
        atr_pct=float(atr_pct),
        side=side,
    )

    a_idx, decision, decision_info = choose_action_with_margin(
        agent=agent,
        state_idx=s,
        min_take_visits=min_take_visits,
        q_take_margin=q_take_margin,
    )

    q_skip = float(decision_info["q_skip"])
    q_take = float(decision_info["q_take"])
    q_gap = float(abs(q_take - q_skip))
    q_conf = _clip01(1.0 - math.exp(-3.0 * max(0.0, q_gap)))

    conf_ens = abs(float(p_ens) - 0.5) * 2.0
    dis = abs(float(last["xgb_p_up"]) - float(last["lstm_p_up"]))
    agreement_bonus = 1.0 if ((float(last["xgb_p_up"]) >= 0.5) == (float(last["lstm_p_up"]) >= 0.5)) else 0.0
    disagreement_penalty = _clip01(dis * 2.0)

    confidence = _clip01(
        0.45 * conf_ens
        + 0.25 * q_conf
        + 0.20 * agreement_bonus
        - 0.10 * disagreement_penalty
    )

    if decision == "skip":
        return TradeSignal(
            action="hold",
            confidence=confidence,
            entry=entry,
            stop_loss=None,
            take_profit=None,
            position_size_usd=0.0,
            leverage=risk.leverage,
            meta={
                "symbol": symbol,
                "resolution": resolution,
                "combo_tag": combo_tag(symbol, resolution),
                "timestamp": str(last["timestamp"]),
                "xgb_artifacts_path": str(xgb_artifacts_path_p),
                "lstm_artifacts_path": str(lstm_artifacts_path_p),
                "rl_agent_path": str(rl_agent_path_p),
                "xgb_p_up": float(last["xgb_p_up"]),
                "lstm_p_up": float(last["lstm_p_up"]),
                "ensemble_p_up": float(p_ens),
                "ensemble_direction": direction_name,
                "rl_decision": "skip",
                "state": int(s),
                "q_skip": q_skip,
                "q_take": q_take,
                "q_gap_take_minus_skip": float(decision_info["q_gap_take_minus_skip"]),
                "q_take_margin": float(decision_info["q_take_margin"]),
                "take_visits": int(decision_info["take_visits"]),
                "skip_visits": int(decision_info["skip_visits"]),
                "atr": float(atr),
                "atr_pct": float(atr_pct),
                "ensemble_upper": float(ensemble_upper),
                "ensemble_lower": float(ensemble_lower),
            },
        )

    action = direction_name

    sl, tp = _sl_tp_from_atr(
        entry=entry,
        atr=atr,
        action=action,
        rr=risk.rr,
        sl_atr_mult=risk.sl_atr_mult,
    )

    pos_usd = _position_size_from_risk(
        capital=risk.capital_usd,
        risk_pct=risk.risk_per_trade,
        entry=entry,
        stop_loss=sl,
    )

    return TradeSignal(
        action=action,
        confidence=confidence,
        entry=entry,
        stop_loss=sl,
        take_profit=tp,
        position_size_usd=pos_usd,
        leverage=risk.leverage,
        meta={
            "symbol": symbol,
            "resolution": resolution,
            "combo_tag": combo_tag(symbol, resolution),
            "timestamp": str(last["timestamp"]),
            "xgb_artifacts_path": str(xgb_artifacts_path_p),
            "lstm_artifacts_path": str(lstm_artifacts_path_p),
            "rl_agent_path": str(rl_agent_path_p),
            "xgb_p_up": float(last["xgb_p_up"]),
            "lstm_p_up": float(last["lstm_p_up"]),
            "ensemble_p_up": float(p_ens),
            "ensemble_direction": direction_name,
            "rl_decision": "take",
            "direction_sign": int(direction_sign),
            "state": int(s),
            "q_skip": q_skip,
            "q_take": q_take,
            "q_gap_take_minus_skip": float(decision_info["q_gap_take_minus_skip"]),
            "q_take_margin": float(decision_info["q_take_margin"]),
            "take_visits": int(decision_info["take_visits"]),
            "skip_visits": int(decision_info["skip_visits"]),
            "atr": float(atr),
            "atr_pct": float(atr_pct),
            "ensemble_upper": float(ensemble_upper),
            "ensemble_lower": float(ensemble_lower),
            "fee_bps": float(risk.fee_bps),
            "trade_penalty_bps": float(risk.trade_penalty_bps),
        },
    )