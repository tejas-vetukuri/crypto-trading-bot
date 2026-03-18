from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace

from joblib import load
from tensorflow.keras.models import load_model

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _normalize_symbol(symbol: str) -> str:
    return symbol.upper().strip()


def _normalize_interval(interval: str) -> str:
    return interval.lower().strip()


def _combo_tag(symbol: str, interval: str) -> str:
    symbol = _normalize_symbol(symbol)
    if symbol.endswith("USDT"):
        symbol = symbol[:-4]
    return f"{symbol}_{_normalize_interval(interval)}"


def _build_model_paths(symbol: str, interval: str) -> dict:
    tag = _combo_tag(symbol, interval)

    return {
        "xgb_artifacts_path": PROJECT_ROOT / "models" / "xgboost" / "saved" / f"xgb_artifacts_{tag}.joblib",
        "lstm_artifacts_path": PROJECT_ROOT / "models" / "lstm" / "saved" / f"lstm_artifacts_{tag}.joblib",
        "rl_agent_path": PROJECT_ROOT / "models" / "rl" / "saved" / f"rl_qtable_agent_{tag}.joblib",
    }


def _resolve_path(path_like) -> Path:
    p = Path(path_like)
    if p.is_absolute():
        return p
    return (PROJECT_ROOT / p).resolve()


def _load_rl_agent_for_inference(path: Path):
    """
    Load the saved RL artifact without importing QTableAgent.
    Live inference only needs:
      - agent.Q
      - agent.visits
    """
    obj = load(path)

    if "Q" not in obj:
        raise ValueError(f"Invalid RL artifact, missing 'Q': {path}")

    q = obj["Q"]
    visits = obj.get("visits")

    if visits is None:
        import numpy as np
        visits = np.zeros_like(q, dtype=int)

    return SimpleNamespace(
        Q=q,
        visits=visits,
        meta=obj.get("meta", {}),
    )


@lru_cache(maxsize=32)
def load_live_artifacts(symbol: str, interval: str) -> dict:
    paths = _build_model_paths(symbol, interval)

    if not paths["xgb_artifacts_path"].exists():
        raise FileNotFoundError(f"Missing XGB artifacts: {paths['xgb_artifacts_path']}")
    if not paths["lstm_artifacts_path"].exists():
        raise FileNotFoundError(f"Missing LSTM artifacts: {paths['lstm_artifacts_path']}")
    if not paths["rl_agent_path"].exists():
        raise FileNotFoundError(f"Missing RL agent: {paths['rl_agent_path']}")

    xgb_artifacts = load(paths["xgb_artifacts_path"])
    lstm_artifacts = load(paths["lstm_artifacts_path"])

    model_path = _resolve_path(lstm_artifacts["model_path"])
    lstm_model = load_model(model_path)

    rl_agent = _load_rl_agent_for_inference(paths["rl_agent_path"])

    return {
        "symbol": symbol.upper(),
        "interval": interval.lower(),
        "xgb": xgb_artifacts,
        "lstm_artifacts": lstm_artifacts,
        "lstm_model": lstm_model,
        "rl": rl_agent,
        "paths": paths,
    }