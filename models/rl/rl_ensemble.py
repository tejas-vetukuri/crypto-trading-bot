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
# Path helpers
# -----------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def resolve_artifact_path(path_like: str) -> str:
    p = Path(path_like)
    if p.is_absolute():
        return str(p)
    return str((PROJECT_ROOT / p).resolve())


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
    rr: float = 2.0
    leverage: float = 1.0
    fee_bps: float = 2.0
    trade_penalty_bps: float = 2.0
    sl_atr_mult: float = 1.5
    min_atr_pct: float = 0.001


# -----------------------------
# Q-learning agent
# -----------------------------

ACTIONS = ["hold", "long", "short"]
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
    Returns XGB predictions aligned to the CURRENT decision candle:
      timestamp       = candle t timestamp
      close           = close[t]
      close_next_raw  = close[t+1]
      actual          = 1 if close[t+1] > close[t] else 0

    Plus:
      atr (if available)
      xgb_p_up, xgb_used, xgb_pred, xgb_margin, etc.
    """
    artifacts = load(resolve_artifact_path(xgb_artifacts_path))
    model = artifacts["model"]
    features = artifacts["features"]
    decision_boundary = float(artifacts.get("decision_boundary", 0.5))
    margin_threshold = float(artifacts.get("margin_threshold", 0.0))

    df = df_raw.sort_values("timestamp").reset_index(drop=True).copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = feature_engineering_xgb(df)

    # Keep only rows where model features + close exist
    df = df.dropna(subset=features + ["close"]).reset_index(drop=True)

    # Build next-candle aligned target BEFORE prediction output
    df["close_next_raw"] = df["close"].shift(-1)
    df["actual"] = (df["close_next_raw"] > df["close"]).astype(int)

    # Last row has no next candle target
    df = df.dropna(subset=["close_next_raw"]).reset_index(drop=True)

    probs = model.predict_proba(df[features].values)
    p_up = probs[:, 1]

    margin = np.abs(p_up - decision_boundary)
    xgb_used = (margin >= margin_threshold).astype(int)

    xgb_pred = np.full(len(p_up), 2, dtype=int)
    confident = xgb_used.astype(bool)
    xgb_pred[confident] = (p_up[confident] > decision_boundary).astype(int)

    keep_cols = ["timestamp", "close", "close_next_raw", "actual"]
    if "atr" in df.columns:
        keep_cols.append("atr")

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
    lstm_artifacts_path: str = "models/lstm/lstm_artifacts.joblib",
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

    # Correct alignment for make_windows():
    # window ends at i-1, target is movement from i-1 -> i
    signal_idx = np.arange(x_window_size - 1, x_window_size - 1 + len(p))
    target_idx = signal_idx + 1

    out = pd.DataFrame({
        "timestamp": df.loc[signal_idx, "timestamp"].to_list(),
        "close": df.loc[signal_idx, "close"].values.astype(float),
        "close_next_raw": df.loc[target_idx, "close"].values.astype(float),
        "actual": (
            df.loc[target_idx, "close"].values > df.loc[signal_idx, "close"].values
        ).astype(int),
        "lstm_p_up": p,
        "lstm_pred": lstm_pred,
        "lstm_used": lstm_used,
        "lstm_threshold": float(threshold),
    })
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True)
    return out


# -----------------------------
# State builder
# -----------------------------
# New state includes:
#   conf(5) * dis(5) * atr(5) * agree(2) * xgb_used(2) * lstm_used(2) * xgb_pred(3) * lstm_pred(3)
# = 9000 states

def build_state_index(
    xgb_p: float,
    lstm_p: float,
    xgb_used: int,
    lstm_used: int,
    xgb_pred: int,
    lstm_pred: int,
    atr_pct: float,
) -> int:
    xgb_p = float(xgb_p)
    lstm_p = float(lstm_p)
    xgb_used = int(xgb_used)
    lstm_used = int(lstm_used)
    xgb_pred = int(xgb_pred)
    lstm_pred = int(lstm_pred)
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

    idx = b_conf
    idx += 5 * b_dis
    idx += 25 * b_atr
    idx += 125 * agree
    idx += 250 * xgb_used
    idx += 500 * lstm_used
    idx += 1000 * xgb_pred
    idx += 3000 * lstm_pred
    return int(idx)


N_STATES = 5 * 5 * 5 * 2 * 2 * 2 * 3 * 3  # 9000


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
    if int(action_idx) == ACTION_TO_IDX["hold"]:
        return 0.0

    direction = 1.0 if int(action_idx) == ACTION_TO_IDX["long"] else -1.0
    pnl = direction * float(ret_next)
    fee = float(fee_bps) / 10000.0
    pen = float(trade_penalty_bps) / 10000.0
    return float(pnl - fee - pen)


# -----------------------------
# Shared merged-data builder
# -----------------------------

def build_merged_dataset(
    df_raw: pd.DataFrame,
    xgb_artifacts_path: str,
    lstm_artifacts_path: str,
    lstm_threshold: float,
) -> pd.DataFrame:
    """
    Build a clean merged dataset where both XGB and LSTM are aligned to the same
    decision candle and same next-candle target.

    We use the model-provided aligned targets instead of reconstructing target
    from merged.shift(-1), which can introduce label drift.
    """
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

    # Sanity checks
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

    # Optional hard checks if you want strict validation:
    # if close_match < 0.999:
    #     raise ValueError(f"Close mismatch after merge: {close_match:.4f}")
    # if label_match < 0.999:
    #     raise ValueError(f"Label mismatch after merge: {label_match:.4f}")

    # Use one consistent aligned target definition
    # LSTM-side target is fine now that it is aligned correctly
    merged["close"] = merged["close_lstm"]
    merged["close_next_raw"] = merged["close_next_raw_lstm"]
    merged["actual"] = merged["actual_lstm"]
    merged["ret_next"] = (merged["close_next_raw"] / merged["close"]) - 1.0

    # Keep ATR from XGB side if present
    if "atr" in merged.columns:
        merged["atr"] = merged["atr"]

    return merged


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
    episodes: int = 20,
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

    train_raw = df_raw.iloc[:split].copy()

    merged = build_merged_dataset(
        df_raw=train_raw,
        xgb_artifacts_path=xgb_artifacts_path,
        lstm_artifacts_path=lstm_artifacts_path,
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
        for t in range(len(merged) - 1):
            row = merged.iloc[t]
            row2 = merged.iloc[t + 1]

            s = build_state_index(
                xgb_p=float(row["xgb_p_up"]),
                lstm_p=float(row["lstm_p_up"]),
                xgb_used=int(row["xgb_used"]),
                lstm_used=int(row["lstm_used"]),
                xgb_pred=int(row["xgb_pred"]),
                lstm_pred=int(row["lstm_pred"]),
                atr_pct=_compute_atr_pct(row),
            )

            if int(row["xgb_used"]) == 0 and int(row["lstm_used"]) == 0:
                a = ACTION_TO_IDX["hold"]
            else:
                a = agent.act(s, greedy=False)

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
                xgb_pred=int(row2["xgb_pred"]),
                lstm_pred=int(row2["lstm_pred"]),
                atr_pct=_compute_atr_pct(row2),
            )

            done = (t == len(merged) - 2)
            agent.update(s, a, r, s2, done)

        agent.decay_eps()
        print(f"Episode {ep + 1}/{episodes} complete | eps={agent.eps:.4f}")

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

    merged = build_merged_dataset(
        df_raw=df_raw,
        xgb_artifacts_path=xgb_artifacts_path,
        lstm_artifacts_path=lstm_artifacts_path,
        lstm_threshold=lstm_threshold,
    )

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
        xgb_pred=int(last["xgb_pred"]),
        lstm_pred=int(last["lstm_pred"]),
        atr_pct=float(atr_pct),
    )

    if int(last["xgb_used"]) == 0 and int(last["lstm_used"]) == 0:
        a_idx = ACTION_TO_IDX["hold"]
    else:
        a_idx = agent.act(s, greedy=True)

    action = ACTIONS[a_idx]

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