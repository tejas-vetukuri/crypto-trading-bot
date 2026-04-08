import numpy as np
import pandas as pd
from pathlib import Path

from models.lstm.lstm import train_lstm_model


class FakeHistory:
    def __init__(self):
        self.history = {
            "loss": [0.6],
            "accuracy": [0.7],
            "val_loss": [0.65],
            "val_accuracy": [0.68],
        }


class FakeModel:
    def __init__(self, layers):
        self.layers = layers
        self.compiled = False
        self.saved_path = None
        self.fit_called = False
        self.predict_called = False

    def compile(self, optimizer=None, loss=None, metrics=None):
        self.compiled = True

    def fit(
        self,
        X,
        y,
        epochs=None,
        batch_size=None,
        validation_split=None,
        shuffle=None,
        callbacks=None,
        verbose=None,
    ):
        self.fit_called = True
        assert isinstance(X, np.ndarray)
        assert isinstance(y, np.ndarray)
        assert X.ndim == 3
        assert len(X) == len(y)
        assert X.dtype == np.float32
        assert y.dtype == np.int32
        return FakeHistory()

    def save(self, path):
        self.saved_path = str(path)
        with open(path, "w", encoding="utf-8") as f:
            f.write("fake model file")

    def predict(self, X, batch_size=None, verbose=None):
        self.predict_called = True
        assert isinstance(X, np.ndarray)
        assert X.ndim == 3
        return np.full((len(X), 1), 0.6, dtype=np.float32)


def make_raw_candle_df(n=180):
    """
    Build enough rows so that:
    - feature_engineering_lstm() rolling features have valid rows
    - make_windows() with x_window_size=20 has enough samples
    """
    timestamps = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")

    base = np.linspace(100, 220, n)
    wave = np.sin(np.linspace(0, 12, n))

    df = pd.DataFrame({
        "timestamp": timestamps,
        "open": base + 0.2 * wave,
        "high": base + 1.5 + 0.2 * wave,
        "low": base - 1.5 + 0.2 * wave,
        "close": base + 0.8 * wave,
        "volume": np.linspace(1000, 3000, n),
    })
    return df


def test_lstm_pipeline_integration_runs_end_to_end(monkeypatch):
    raw_df = make_raw_candle_df(180)

    def fake_get_candles(self, symbol, resolution, start_date, end_date):
        return raw_df.copy()

    monkeypatch.setattr(
        "data.binance.BinanceDataClient.get_candles",
        fake_get_candles,
    )

    monkeypatch.setattr(
        "models.lstm.lstm.Sequential",
        FakeModel,
    )

    monkeypatch.setattr(
        "models.lstm.lstm.eval_with_ignore_zone",
        lambda y_true, p, threshold: {
            "threshold": float(threshold),
            "used_count": int(len(y_true)),
            "accuracy": 0.5,
            "coverage": 1.0,
        },
    )

    test_output_dir = Path("tests/_tmp_lstm")
    test_output_dir.mkdir(parents=True, exist_ok=True)

    model_path = test_output_dir / "lstm_test.keras"
    scaler_path = test_output_dir / "lstm_scaler.joblib"
    artifacts_path = test_output_dir / "lstm_artifacts.joblib"
    test_probs_path = test_output_dir / "lstm_test_probs.csv"
    metrics_path = test_output_dir / "lstm_metrics.csv"

    model, history, out_df, metrics_df, scaler = train_lstm_model(
        symbol="BTCUSDT",
        resolution="1h",
        start_date="2024-01-01",
        end_date="2024-01-10",
        x_window_size=20,
        epochs=1,
        batch_size=16,
        train_ratio=0.80,
        validation_split=0.05,
        learning_rate=0.001,
        lstm_units=16,
        dropout_rate=0.10,
        early_stopping_patience=1,
        model_path=model_path,
        scaler_path=scaler_path,
        artifacts_path=artifacts_path,
        test_probs_path=test_probs_path,
        metrics_path=metrics_path,
        thresholds=(0.50, 0.55),
    )

    assert model.fit_called is True
    assert model.predict_called is True
    assert model.compiled is True

    assert hasattr(history, "history")
    assert scaler is not None

    assert not out_df.empty
    assert {"symbol", "resolution", "y_true", "p"}.issubset(out_df.columns)
    assert out_df["symbol"].eq("BTCUSDT").all()
    assert out_df["resolution"].eq("1h").all()
    assert out_df["p"].between(0, 1).all()

    assert not metrics_df.empty
    assert {"symbol", "resolution", "threshold"}.issubset(metrics_df.columns)
    assert metrics_df["symbol"].eq("BTCUSDT").all()
    assert metrics_df["resolution"].eq("1h").all()

    assert model_path.exists()
    assert scaler_path.exists()
    assert artifacts_path.exists()
    assert test_probs_path.exists()
    assert metrics_path.exists()

    saved_probs = pd.read_csv(test_probs_path)
    saved_metrics = pd.read_csv(metrics_path)

    assert not saved_probs.empty
    assert not saved_metrics.empty
    assert len(saved_probs) == len(out_df)
    assert len(saved_metrics) == len(metrics_df)