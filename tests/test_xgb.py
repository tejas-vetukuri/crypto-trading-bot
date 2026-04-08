import pandas as pd
import numpy as np

from models.xgboost.xgb import train_xgb_model


class FakeXGBClassifier:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def fit(self, X, y):
        return self

    def predict_proba(self, X):
        """
        Return probabilities matching the number of test rows.
        First three rows are chosen to test:
        - sideways
        - down
        - up
        Remaining rows default to a confident up prediction.
        """
        n = len(X)

        base_p_up = [0.50, 0.20, 0.80]
        if n > 3:
            base_p_up.extend([0.80] * (n - 3))

        p_up = np.array(base_p_up[:n], dtype=float)
        p_down = 1.0 - p_up
        return np.column_stack([p_down, p_up])


def make_raw_price_df(n=20):
    timestamps = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    return pd.DataFrame({
        "timestamp": timestamps,
        "open": np.linspace(100, 119, n),
        "high": np.linspace(101, 120, n),
        "low": np.linspace(99, 118, n),
        "close": np.linspace(100.5, 119.5, n),
        "volume": np.linspace(1000, 2000, n),
    })


def fake_feature_engineering_xgb(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a minimal engineered dataframe containing all required feature columns
    and actual_trend, without depending on the real feature engineering pipeline.
    """
    out = df.copy().reset_index(drop=True)

    feature_cols = [
        "ema_20", "ema_50", "rsi", "atr",
        "log_ret_1", "ret_1", "ret_3", "ret_5", "ret_10",
        "body", "range", "upper_wick", "lower_wick", "body_pct", "range_pct", "clv",
        "ema_spread", "ema20_dist", "ema50_dist", "ema20_slope_3", "ema50_slope_3",
        "atr_pct", "rsi_delta", "rsi_ma_10", "rsi_dist",
        "volatility_5", "vol_10", "vol_30", "vol_ratio",
        "vol_chg_1", "vol_chg_5", "vol_z20",
    ]

    for i, col in enumerate(feature_cols, start=1):
        out[col] = float(i)

    # Make sure both classes exist in training data
    trends = ["up", "down"] * (len(out) // 2) + (["up"] if len(out) % 2 else [])
    out["actual_trend"] = trends[:len(out)]

    return out


def test_train_xgb_model_applies_decision_boundary_and_sideways_logic(monkeypatch, tmp_path):
    """
    Verifies that train_xgb_model():
    - applies the decision boundary correctly
    - marks low-margin predictions as sideways (2)
    - marks confident predictions as down (0) / up (1)
    """
    raw_df = make_raw_price_df(20)

    # Patch candle fetch
    def fake_get_candles(self, symbol, resolution, start_date, end_date):
        return raw_df

    monkeypatch.setattr(
        "data.binance.BinanceDataClient.get_candles",
        fake_get_candles,
    )

    # Patch feature engineering
    monkeypatch.setattr(
        "models.xgboost.xgb.feature_engineering_xgb",
        fake_feature_engineering_xgb,
    )

    # Patch model class
    monkeypatch.setattr(
        "models.xgboost.xgb.XGBClassifier",
        FakeXGBClassifier,
    )

    # Patch tuned artifact loading to avoid external dependency
    monkeypatch.setattr(
        "models.xgboost.xgb.load",
        lambda *args, **kwargs: {"best_params": {}},
    )

    artifacts_path = tmp_path / "xgb_artifacts.joblib"
    preds_csv_path = tmp_path / "xgb_predictions.csv"

    _, artifacts, output_df = train_xgb_model(
        symbol="BTCUSDT",
        resolution="1h",
        start_date="2024-01-01",
        end_date="2024-01-02",
        train_ratio=0.80,              # 16 train, 4 test before fake FE
        decision_boundary=0.48,
        margin_threshold=0.10,
        tuned_artifacts_path=tmp_path / "dummy_tuned.joblib",
        artifacts_path=artifacts_path,
        preds_csv_path=preds_csv_path,
        use_sentiment_data=False,
    )

    # Fake predict_proba returns p_up = [0.50, 0.20, 0.80]
    # With boundary=0.48 and margin=0.10:
    # 0.50 -> |0.50 - 0.48| = 0.02 < 0.10 -> sideways (2)
    # 0.20 -> |0.20 - 0.48| = 0.28 >= 0.10 -> down (0)
    # 0.80 -> |0.80 - 0.48| = 0.32 >= 0.10 -> up (1)

    expected_preds = [2, 0, 1]
    assert output_df["prediction"].tolist()[:3] == expected_preds

    assert output_df["used"].tolist()[:3] == [0, 1, 1]
    assert np.isclose(output_df["decision_boundary"].iloc[0], 0.48)
    assert np.isclose(artifacts["margin_threshold"], 0.10)