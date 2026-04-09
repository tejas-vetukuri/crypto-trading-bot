#tests/test_integration_xgb.py
import numpy as np
import pandas as pd
from pathlib import Path

from models.xgboost.xgb import train_xgb_model


class FakeXGBClassifier:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.fit_called = False

    def fit(self, X, y):
        self.fit_called = True

        # Integration assertions: real feature engineering should produce
        # a 2D feature matrix and aligned labels
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, np.ndarray)
        assert len(X) == len(y)
        assert X.shape[1] > 0
        return self

    def predict_proba(self, X):
        n = len(X)

        # First three rows test:
        # sideways, down, up
        p_up = [0.50, 0.20, 0.80]
        if n > 3:
            p_up.extend([0.80] * (n - 3))

        p_up = np.array(p_up[:n], dtype=float)
        p_down = 1.0 - p_up
        return np.column_stack([p_down, p_up])


def make_raw_price_df(n=320):
    timestamps = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")

    base = np.linspace(100, 220, n)
    wave = np.sin(np.linspace(0, 10, n))

    df = pd.DataFrame({
        "timestamp": timestamps,
        "open": base + 0.2 * wave,
        "high": base + 1.5 + 0.2 * wave,
        "low": base - 1.5 + 0.2 * wave,
        "close": base + 0.8 * wave,
        "volume": np.linspace(1000, 3000, n),
    })
    return df


def test_xgb_pipeline_integration_runs_end_to_end(monkeypatch):
    raw_df = make_raw_price_df(320)

    # Mock Binance fetch only
    def fake_get_candles(self, symbol, resolution, start_date, end_date):
        return raw_df.copy()

    monkeypatch.setattr(
        "data.binance.BinanceDataClient.get_candles",
        fake_get_candles,
    )

    # Mock tuned artifact loading
    monkeypatch.setattr(
        "models.xgboost.xgb.load",
        lambda *args, **kwargs: {"best_params": {}},
    )

    # Mock XGB model only
    monkeypatch.setattr(
        "models.xgboost.xgb.XGBClassifier",
        FakeXGBClassifier,
    )

    test_output_dir = Path("tests/_tmp_xgb")
    test_output_dir.mkdir(parents=True, exist_ok=True)

    artifacts_path = test_output_dir / "xgb_artifacts.joblib"
    preds_csv_path = test_output_dir / "xgb_predictions.csv"

    model, artifacts, output_df = train_xgb_model(
        symbol="BTCUSDT",
        resolution="1h",
        start_date="2024-01-01",
        end_date="2024-01-10",
        train_ratio=0.80,
        decision_boundary=0.48,
        margin_threshold=0.10,
        tuned_artifacts_path=test_output_dir / "dummy_tuned.joblib",
        artifacts_path=artifacts_path,
        preds_csv_path=preds_csv_path,
        use_sentiment_data=False,
    )

    # Check model training path ran
    assert model.fit_called is True

    # Check artifacts returned
    assert isinstance(artifacts, dict)
    assert "features" in artifacts
    assert "decision_boundary" in artifacts
    assert "margin_threshold" in artifacts
    assert artifacts["decision_boundary"] == 0.48
    assert artifacts["margin_threshold"] == 0.10

    # Check output dataframe
    assert not output_df.empty
    expected_cols = {
        "timestamp",
        "symbol",
        "resolution",
        "prediction",
        "actual_trend",
        "p_up",
        "decision_boundary",
        "margin",
        "used",
        "probability",
    }
    assert expected_cols.issubset(output_df.columns)

    assert output_df["symbol"].eq("BTCUSDT").all()
    assert output_df["resolution"].eq("1h").all()
    assert output_df["p_up"].between(0, 1).all()
    assert output_df["prediction"].isin([0, 1, 2]).all()
    assert output_df["used"].isin([0, 1]).all()

    # First three rows check the intended boundary behaviour
    # p_up = [0.50, 0.20, 0.80]
    # boundary=0.48, margin_threshold=0.10
    # -> [sideways(2), down(0), up(1)]
    assert output_df["prediction"].tolist()[:3] == [2, 0, 1]
    assert output_df["used"].tolist()[:3] == [0, 1, 1]

    # Check output files exist
    assert artifacts_path.exists()
    assert preds_csv_path.exists()

    saved_preds = pd.read_csv(preds_csv_path)
    assert not saved_preds.empty
    assert len(saved_preds) == len(output_df)