import numpy as np
import pandas as pd

from models.rl.rl_ensemble import (
    QTableAgent,
    RiskConfig,
    ACTION_TO_IDX,
    get_trade_signal_rl,
)


def make_raw_price_df(n=20):
    timestamps = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    base = np.linspace(100, 120, n)

    return pd.DataFrame({
        "timestamp": timestamps,
        "open": base,
        "high": base + 2.0,
        "low": base - 2.0,
        "close": base + 0.5,
        "volume": np.linspace(1000, 2000, n),
    })


def fake_predict_xgb_series(df_raw, xgb_artifacts_path):
    timestamps = pd.date_range("2024-01-01", periods=8, freq="h", tz="UTC")
    close = np.array([100, 101, 102, 103, 104, 105, 106, 107], dtype=float)

    return pd.DataFrame({
        "timestamp": timestamps,
        "open": close - 0.5,
        "high": close + 1.5,
        "low": close - 1.5,
        "close": close,
        "atr": np.full(len(close), 2.0),
        "close_next_raw": close + 1.0,
        "actual": np.array([1, 1, 1, 1, 1, 1, 1, 1], dtype=int),
        "xgb_p_up": np.array([0.70, 0.72, 0.74, 0.76, 0.78, 0.80, 0.82, 0.84], dtype=float),
        "xgb_used": np.ones(len(close), dtype=int),
        "xgb_pred": np.ones(len(close), dtype=int),
        "xgb_margin": np.full(len(close), 0.20),
        "xgb_boundary": np.full(len(close), 0.48),
        "xgb_margin_threshold": np.full(len(close), 0.10),
    })


def fake_predict_lstm_series(df_raw, lstm_artifacts_path, threshold=0.52, batch_size=256):
    timestamps = pd.date_range("2024-01-01", periods=8, freq="h", tz="UTC")
    close = np.array([100, 101, 102, 103, 104, 105, 106, 107], dtype=float)

    return pd.DataFrame({
        "timestamp": timestamps,
        "open": close - 0.5,
        "high": close + 1.5,
        "low": close - 1.5,
        "close": close,
        "close_next_raw": close + 1.0,
        "actual": np.array([1, 1, 1, 1, 1, 1, 1, 1], dtype=int),
        "lstm_p_up": np.array([0.60, 0.62, 0.64, 0.66, 0.68, 0.70, 0.72, 0.74], dtype=float),
        "lstm_pred": np.ones(len(close), dtype=int),
        "lstm_used": np.ones(len(close), dtype=int),
        "lstm_threshold": np.full(len(close), threshold),
    })


def fake_agent_with_take_preference():
    agent = QTableAgent(n_states=192, n_actions=2, seed=42)

    # Make "take" clearly preferable and sufficiently visited
    agent.Q[:, ACTION_TO_IDX["skip"]] = 0.1
    agent.Q[:, ACTION_TO_IDX["take"]] = 1.0
    agent.visits[:, ACTION_TO_IDX["skip"]] = 25
    agent.visits[:, ACTION_TO_IDX["take"]] = 25
    return agent


def test_ensemble_rl_integration_returns_trade_signal(monkeypatch):
    """
    Integration test for:
    raw fetch -> model prediction wrappers -> merged dataset -> ensemble decision
    -> RL decision -> final trade signal
    """
    raw_df = make_raw_price_df(20)

    def fake_get_candles(self, symbol, resolution, start_date, end_date):
        return raw_df.copy()

    monkeypatch.setattr(
        "data.binance.BinanceDataClient.get_candles",
        fake_get_candles,
    )

    monkeypatch.setattr(
        "models.rl.rl_ensemble.predict_xgb_series",
        fake_predict_xgb_series,
    )

    monkeypatch.setattr(
        "models.rl.rl_ensemble.predict_lstm_series",
        fake_predict_lstm_series,
    )

    monkeypatch.setattr(
        "models.rl.rl_ensemble.QTableAgent.load",
        lambda path: fake_agent_with_take_preference(),
    )

    risk = RiskConfig(
        capital_usd=5000.0,
        risk_per_trade=0.02,
        rr=1.25,
        leverage=25.0,
        fee_bps=2.0,
        trade_penalty_bps=2.0,
        sl_atr_mult=1.0,
        min_atr_pct=0.001,
    )

    signal = get_trade_signal_rl(
        symbol="BTCUSDT",
        resolution="1h",
        start_date="2024-01-01",
        end_date="2024-01-03",
        xgb_artifacts_path="dummy_xgb.joblib",
        lstm_artifacts_path="dummy_lstm.joblib",
        rl_agent_path="dummy_rl.joblib",
        lstm_threshold=0.52,
        risk=risk,
        ensemble_weight_xgb=0.8,
        ensemble_weight_lstm=0.2,
        ensemble_upper=0.60,
        ensemble_lower=0.40,
        min_take_visits=20,
        q_take_margin=0.0,
    )

    # Ensemble should be long on the last row:
    # p_ens = 0.8 * 0.84 + 0.2 * 0.74 = 0.82
    assert signal.action == "long"
    assert 0.0 <= signal.confidence <= 1.0
    assert signal.entry > 0
    assert signal.stop_loss is not None
    assert signal.take_profit is not None
    assert signal.position_size_usd > 0
    assert signal.leverage == 25.0

    # Basic directional sanity
    assert signal.take_profit > signal.entry
    assert signal.stop_loss < signal.entry

    # Metadata sanity
    assert signal.meta["symbol"] == "BTCUSDT"
    assert signal.meta["resolution"] == "1h"
    assert signal.meta["ensemble_direction"] == "long"
    assert signal.meta["rl_decision"] == "take"
    assert signal.meta["xgb_p_up"] > 0
    assert signal.meta["lstm_p_up"] > 0
    assert signal.meta["ensemble_p_up"] > 0.60