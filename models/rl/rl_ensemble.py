# models/rl/rl_ensemble.py
# RL meta-controller that calls XGBoost + LSTM predictions and outputs LONG/SHORT/HOLD
# Includes:
#  - Robust artifact path resolving relative to project root
#  - UTC-normalized timestamps for safe merges
#  - LSTM auto-detect via lstm_artifacts.joblib (model_path, scaler_path, window, feature_cols)
#  - LSTM produces up/down/sideways with ignore-zone threshold=0.52
#  - State includes xgb_used + lstm_used
#  - ✅ Minimal RL fixes:
#       (1) Hard gate: if BOTH models are sideways => force HOLD
#       (2) Extra trade penalty (bps) in reward to discourage overtrading

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

from data.delta_exchange import DeltaDataClient
from data.feature_engineering import feature_engineering_xgb, feature_engineering_lstm
from models.lstm.sequence_builder import make_windows


# -----------------------------
# Path helpers (PROJECT ROOT)
# -----------------------------

# This file is: <root>/models/rl/rl_ensemble.py
# parents[0]=rl, [1]=models, [2]=<root>
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def resolve_artifact_path(path_like: str) -> str:
    """
    Resolve artifact paths robustly.
    - If absolute => return as-is
    - If relative => interpret relative to PROJECT_ROOT
    """
    p = Path(path_like)
    if p.is_absolute():
        return str(p)
    return str((PROJECT_ROOT / p).resolve())


# -----------------------------
# Output / Risk config
# -----------------------------

@dataclass
class TradeSignal:
    action: str               # "long" | "short" | "hold"
    confidence: float         # 0..1
    entry: float
    stop_loss: Optional[float]
    take_profit: Optional[float]
    position_size_usd: float
    leverage: float
    meta: Dict[str, float | str | int]


@dataclass
class RiskConfig:
    capital_usd: float = 5000.0
    risk_per_trade: float = 0.02           # 2%
    rr: float = 2.0                        # 1:2 RR
    leverage: float = 1.0                  # set 25 if you want
    fee_bps: float = 2.0                   # rough (entry+exit). adjust
    trade_penalty_bps: float = 2.0         # ✅ extra discouragement per trade
    sl_atr_mult: float = 1.5               # SL distance = sl_atr_mult * ATR
    min_atr_pct: float = 0.001             # fallback ATR% if ATR missing


# -----------------------------
# Q-learning agent
# -----------------------------

ACTIONS = ["hold", "long", "short"]
ACTION_TO_IDX = {a: i for i, a in enumerate(ACTIONS)}


def _clip01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))


def _bucket(x: float, edges: Tuple[float, ...]) -> int:
    """Return bucket index 0..len(edges) for ascending edges."""
    for i, e in enumerate(edges):
        if x < e:
            return i
    return len(edges)


class QTableAgent:
    """Tabular Q-learning over discretized state space."""

    def __init__(
        self,
        n_states: int,
        n_actions: int = 3,
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
        best_next = 0.0 if done else float(np.max(self.Q[s2]))
        td_target = r + self.gamma * best_next
        td_error = td_target - float(self.Q[s, a])
        self.Q[s, a] = float(self.Q[s, a] + self.alpha * td_error)

    def decay_eps(self):
        self.eps = max(self.eps_min, self.eps * self.eps_decay)

    def save(self, path: str):
        dump(
            {
                "Q": self.Q,
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
    def load(path: str) -> "QTableAgent":
        obj = load(resolve_artifact_path(path))
        Q = obj["Q"]
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
        return agent


# -----------------------------
# Prediction wrappers
# -----------------------------

def predict_xgb_series(
    df_raw: pd.DataFrame,
    xgb_artifacts_path: str,
) -> pd.DataFrame:
    """
    Returns df aligned to df_raw AFTER FE and dropna, with:
      timestamp, close, atr(if exists), xgb_p_up, xgb_used, xgb_pred (0/1/2)
    """
    artifacts = load(resolve_artifact_path(xgb_artifacts_path))
    model = artifacts["model"]
    features = artifacts["features"]
    decision_boundary = float(artifacts.get("decision_boundary", 0.5))
    margin_threshold = float(artifacts.get("margin_threshold", 0.0))

    df = df_raw.sort_values("timestamp").reset_index(drop=True).copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = feature_engineering_xgb(df)

    keep_cols = ["timestamp", "close"]
    if "atr" in df.columns:
        keep_cols.append("atr")

    df = df.dropna(subset=features + ["close"]).reset_index(drop=True)

    probs = model.predict_proba(df[features].values)
    p_up = probs[:, 1]

    margin = np.abs(p_up - decision_boundary)
    xgb_used = (margin >= margin_threshold).astype(int)

    xgb_pred = np.full(len(p_up), 2, dtype=int)
    confident = xgb_used.astype(bool)
    xgb_pred[confident] = (p_up[confident] > decision_boundary).astype(int)

    out = df[keep_cols].copy()
    out["xgb_p_up"] = p_up.astype(float)
    out["xgb_used"] = xgb_used.astype(int)
    out["xgb_pred"] = xgb_pred.astype(int)
    out["xgb_margin"] = margin.astype(float)
    out["xgb_boundary"] = decision_boundary
    out["xgb_margin_threshold"] = margin_threshold
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True)  # ✅ harden dtype
    return out


def predict_lstm_series(
    df_raw: pd.DataFrame,
    lstm_artifacts_path: str = "models/lstm/lstm_artifacts.joblib",
    threshold: float = 0.52,
    batch_size: int = 256,
) -> pd.DataFrame:
    """
    Returns df aligned to END timestamp of each window:
      timestamp, close, lstm_p_up, lstm_pred (0/1/2), lstm_used (1 if not sideways)

    Sideways defined by ignore zone around 0.5 with threshold=0.52:
      up   if p >= 0.52
      down if p <= 0.48
      sideways otherwise
    """
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

    lower = 1.0 - float(threshold)  # 0.48
    upper = float(threshold)        # 0.52

    lstm_pred = np.full(len(p), 2, dtype=int)
    lstm_pred[p >= upper] = 1
    lstm_pred[p <= lower] = 0
    lstm_used = (lstm_pred != 2).astype(int)

    end_idx = np.arange(x_window_size, x_window_size + len(p))

    out = pd.DataFrame({
        "timestamp": df.loc[end_idx, "timestamp"].to_list(),  # ✅ avoid .values tz edge cases
        "close": df.loc[end_idx, "close"].values.astype(float),
        "lstm_p_up": p,
        "lstm_pred": lstm_pred,
        "lstm_used": lstm_used,
        "lstm_threshold": float(threshold),
    })
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True)  # ✅ harden dtype
    return out


# -----------------------------
# State builder (includes lstm_used)
# -----------------------------

def build_state_index(
    xgb_p: float,
    lstm_p: float,
    xgb_used: int,
    lstm_used: int,
    atr_pct: float,
) -> int:
    """
    dims:
      conf(5) * dis(5) * atr(5) * agree(2) * xgb_used(2) * lstm_used(2) = 1000
    """
    xgb_p = float(xgb_p)
    lstm_p = float(lstm_p)
    xgb_used = int(xgb_used)
    lstm_used = int(lstm_used)
    atr_pct = max(0.0, float(atr_pct))

    conf_x = abs(xgb_p - 0.5) * 2.0
    conf_l = abs(lstm_p - 0.5) * 2.0
    conf_avg = 0.5 * (conf_x + conf_l)

    side_x = 1 if xgb_p >= 0.5 else 0
    side_l = 1 if lstm_p >= 0.5 else 0
    agree = 1 if side_x == side_l else 0

    dis = abs(xgb_p - lstm_p)

    b_conf = _bucket(conf_avg, (0.2, 0.4, 0.6, 0.8))
    b_dis = _bucket(dis, (0.05, 0.10, 0.20, 0.35))
    b_atr = _bucket(atr_pct, (0.002, 0.004, 0.008, 0.015))

    idx = (
        b_conf
        + 5 * b_dis
        + 25 * b_atr
        + 125 * int(agree)
        + 250 * int(xgb_used)
        + 500 * int(lstm_used)
    )
    return int(idx)


N_STATES = 5 * 5 * 5 * 2 * 2 * 2  # 1000


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


def _reward_from_next_return(
    action_idx: int,
    ret_next: float,
    fee_bps: float,
    trade_penalty_bps: float,
) -> float:
    """
    Reward proxy:
      long  => +ret_next
      short => -ret_next
      hold  => 0
    minus fees + extra trade penalty if trade taken.
    """
    if int(action_idx) == ACTION_TO_IDX["hold"]:
        return 0.0

    direction = 1.0 if int(action_idx) == ACTION_TO_IDX["long"] else -1.0
    pnl = direction * float(ret_next)
    fee = float(fee_bps) / 10000.0
    pen = float(trade_penalty_bps) / 10000.0
    return float(pnl - fee - pen)


# -----------------------------
# Training
# -----------------------------

def train_rl_policy(
    symbol: str = "BTCUSD",
    resolution: str = "1h",
    start_date: str = "2019-06-01",
    end_date: str | None = None,
    train_ratio: float = 0.80,

    xgb_artifacts_path: str = "models/xgboost/xgb_trend_artifacts.joblib",
    lstm_artifacts_path: str = "models/lstm/lstm_artifacts.joblib",
    lstm_threshold: float = 0.52,

    agent_out_path: str = "models/rl/rl_qtable_agent.joblib",

    alpha: float = 0.10,
    gamma: float = 0.95,
    eps: float = 0.20,
    episodes: int = 3,

    risk: RiskConfig = RiskConfig(),
) -> QTableAgent:

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

    xgb_df = predict_xgb_series(test_raw, xgb_artifacts_path=xgb_artifacts_path)
    lstm_df = predict_lstm_series(
        test_raw,
        lstm_artifacts_path=lstm_artifacts_path,
        threshold=lstm_threshold,
    )

    merged = pd.merge(xgb_df, lstm_df, on=["timestamp", "close"], how="inner")
    merged = merged.sort_values("timestamp").reset_index(drop=True)
    if len(merged) < 5:
        raise ValueError("Not enough aligned samples between XGB and LSTM outputs.")

    merged["close_next"] = merged["close"].shift(-1)
    merged["ret_next"] = (merged["close_next"] / merged["close"]) - 1.0
    merged = merged.dropna(subset=["ret_next"]).reset_index(drop=True)

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

    for _ep in range(int(episodes)):
        for t in range(len(merged) - 1):
            row = merged.iloc[t]
            row2 = merged.iloc[t + 1]

            s = build_state_index(
                xgb_p=float(row["xgb_p_up"]),
                lstm_p=float(row["lstm_p_up"]),
                xgb_used=int(row["xgb_used"]),
                lstm_used=int(row["lstm_used"]),
                atr_pct=_compute_atr_pct(row),
            )

            # ✅ Fix (1): hard gate - if both models are sideways => HOLD
            if int(row["xgb_used"]) == 0 and int(row["lstm_used"]) == 0:
                a = ACTION_TO_IDX["hold"]
            else:
                a = agent.act(s, greedy=False)

            # ✅ Fix (2): trade penalty in reward
            r = _reward_from_next_return(
                action_idx=a,
                ret_next=float(row["ret_next"]),
                fee_bps=risk.fee_bps,
                trade_penalty_bps=risk.trade_penalty_bps,
            )

            s2 = build_state_index(
                xgb_p=float(row2["xgb_p_up"]),
                lstm_p=float(row2["lstm_p_up"]),
                xgb_used=int(row2["xgb_used"]),
                lstm_used=int(row2["lstm_used"]),
                atr_pct=_compute_atr_pct(row2),
            )

            done = (t == len(merged) - 2)
            agent.update(s, a, r, s2, done)

        agent.decay_eps()

    agent.save(agent_out_path)
    print(f"✅ Saved RL agent to {resolve_artifact_path(agent_out_path)}")
    return agent


# -----------------------------
# Inference
# -----------------------------

def get_trade_signal_rl(
    symbol: str = "BTCUSD",
    resolution: str = "1h",
    start_date: str = "2019-06-01",
    end_date: str | None = None,

    xgb_artifacts_path: str = "models/xgboost/xgb_trend_artifacts.joblib",
    lstm_artifacts_path: str = "models/lstm/lstm_artifacts.joblib",
    lstm_threshold: float = 0.52,

    rl_agent_path: str = "models/rl/rl_qtable_agent.joblib",
    risk: RiskConfig = RiskConfig(),
) -> TradeSignal:

    agent = QTableAgent.load(rl_agent_path)

    client = DeltaDataClient()
    df_raw = client.get_candles(
        symbol=symbol,
        resolution=resolution,
        start_date=start_date,
        end_date=end_date,
    ).sort_values("timestamp").reset_index(drop=True)

    xgb_df = predict_xgb_series(df_raw, xgb_artifacts_path=xgb_artifacts_path)
    lstm_df = predict_lstm_series(
        df_raw,
        lstm_artifacts_path=lstm_artifacts_path,
        threshold=lstm_threshold,
    )

    merged = pd.merge(xgb_df, lstm_df, on=["timestamp", "close"], how="inner")
    merged = merged.sort_values("timestamp").reset_index(drop=True)
    if len(merged) == 0:
        raise ValueError("No aligned timestamp between XGB and LSTM outputs for inference.")

    last = merged.iloc[-1]
    entry = float(last["close"])

    if "atr" in merged.columns and pd.notna(last.get("atr", np.nan)):
        atr = float(last["atr"])
    else:
        atr = entry * float(risk.min_atr_pct)

    atr_pct = atr / entry if entry > 0 else 0.0

    s = build_state_index(
        xgb_p=float(last["xgb_p_up"]),
        lstm_p=float(last["lstm_p_up"]),
        xgb_used=int(last["xgb_used"]),
        lstm_used=int(last["lstm_used"]),
        atr_pct=float(atr_pct),
    )

    # ✅ Fix (1): hard gate - if both models are sideways => HOLD
    if int(last["xgb_used"]) == 0 and int(last["lstm_used"]) == 0:
        a_idx = ACTION_TO_IDX["hold"]
    else:
        a_idx = agent.act(s, greedy=True)

    action = ACTIONS[a_idx]

    # Confidence: combine model confidence + agreement + Q gap
    q = agent.Q[s]
    q_sorted = np.sort(q)
    q_gap = float(q_sorted[-1] - q_sorted[-2]) if len(q_sorted) >= 2 else float(q_sorted[-1])
    q_conf = _clip01(1.0 - math.exp(-3.0 * max(0.0, q_gap)))

    xgb_p = float(last["xgb_p_up"])
    lstm_p = float(last["lstm_p_up"])
    conf_x = abs(xgb_p - 0.5) * 2.0
    conf_l = abs(lstm_p - 0.5) * 2.0
    model_conf = 0.5 * (conf_x + conf_l)

    side_agree = 1.0 if ((xgb_p >= 0.5) == (lstm_p >= 0.5)) else 0.0
    used_bonus = 0.5 * (float(last["xgb_used"]) + float(last["lstm_used"]))

    confidence = _clip01(0.35 * model_conf + 0.30 * side_agree + 0.20 * q_conf + 0.15 * used_bonus)

    if action == "hold":
        sl, tp, pos_usd = None, None, 0.0
    else:
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
            "timestamp": str(last["timestamp"]),
            "xgb_p_up": xgb_p,
            "lstm_p_up": lstm_p,
            "xgb_pred": int(last["xgb_pred"]),
            "lstm_pred": int(last["lstm_pred"]),
            "xgb_used": int(last["xgb_used"]),
            "lstm_used": int(last["lstm_used"]),
            "atr": float(atr),
            "atr_pct": float(atr_pct),
            "state": int(s),
            "q_hold": float(agent.Q[s, ACTION_TO_IDX["hold"]]),
            "q_long": float(agent.Q[s, ACTION_TO_IDX["long"]]),
            "q_short": float(agent.Q[s, ACTION_TO_IDX["short"]]),
            "lstm_threshold": float(lstm_threshold),
            "hard_gate_hold": int(int(last["xgb_used"]) == 0 and int(last["lstm_used"]) == 0),
            "fee_bps": float(risk.fee_bps),
            "trade_penalty_bps": float(risk.trade_penalty_bps),
        },
    )