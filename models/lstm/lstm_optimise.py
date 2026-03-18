# lstm_optimise.py
#
# Small (non-huge) time-series-safe tuning for your LSTM:
# - Chronological split: train/test (default 95/5)
# - Inside train: subtrain/val (default last 10% of train as val)
# - Train-only StandardScaler (fit on subtrain windows)
# - Small random search over a compact space (window, units, dropout, lr, batch)
# - Select best by validation logloss (probability quality)
# - Retrain best model on full train (subtrain+val) WITHOUT early stopping
# - Saves:
#   - lstm_tuning_results.csv (all trials)
#   - lstm_best_model.keras
#   - lstm_best_artifacts.joblib (best_params, scaler, feature_cols, etc.)
#   - lstm_best_test_probs.csv + lstm_best_test_metrics.csv

import os
import random
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from joblib import dump

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss, roc_auc_score

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

from data.binance import BinanceDataClient
from data.feature_engineering import feature_engineering_lstm
from models.lstm.sequence_builder import make_windows
from models.lstm.confidence_threshold import eval_with_ignore_zone


# -----------------------------
# Reproducibility helpers
# -----------------------------
def set_global_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


# -----------------------------
# Model builder
# -----------------------------
def build_lstm_model(
    window_size: int,
    n_features: int,
    units: int,
    dropout: float,
    lr: float,
) -> tf.keras.Model:
    model = Sequential([
        Input(shape=(window_size, n_features)),
        LSTM(units, return_sequences=False),
        Dropout(dropout),
        Dense(1, activation="sigmoid"),
    ])
    model.compile(
        optimizer=Adam(learning_rate=lr),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


# -----------------------------
# Data prep (one pass per window size)
# -----------------------------
def fetch_and_engineer(
    symbol: str,
    resolution: str,
    start_date: str,
    end_date: str | None,
) -> pd.DataFrame:
    client = BinanceDataClient()
    df = client.get_candles(
        symbol=symbol,
        resolution=resolution,
        start_date=start_date,
        end_date=end_date,
    )

    required = {"open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.dropna(subset=list(required)).reset_index(drop=True)
    df = feature_engineering_lstm(df)
    return df


def make_scaled_splits_for_window(
    df: pd.DataFrame,
    feature_cols: list[str],
    x_window_size: int,
    train_ratio: float,
    val_ratio_within_train: float,
) -> dict[str, Any]:
    df = df.dropna(subset=feature_cols).reset_index(drop=True)

    X, y = make_windows(df, x_window_size=x_window_size, feature_cols=feature_cols)
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int32)

    if X.ndim != 3:
        raise ValueError(f"Expected X with shape (N, T, F), got {X.shape}")
    if X.shape[1] != x_window_size:
        raise ValueError(f"Expected T={x_window_size}, got {X.shape[1]}")
    if X.shape[2] != len(feature_cols):
        raise ValueError(f"Expected F={len(feature_cols)}, got {X.shape[2]}")

    n = len(X)
    train_end = int(n * train_ratio)
    if train_end <= 0 or train_end >= n:
        raise ValueError(f"Not enough samples after windowing. n={n}, train_end={train_end}")

    X_train_full, X_test = X[:train_end], X[train_end:]
    y_train_full, y_test = y[:train_end], y[train_end:]

    subtrain_end = int(len(X_train_full) * (1.0 - val_ratio_within_train))
    if subtrain_end <= 0 or subtrain_end >= len(X_train_full):
        raise ValueError(
            f"Invalid val split inside train. n_train={len(X_train_full)}, subtrain_end={subtrain_end}"
        )

    X_sub, X_val = X_train_full[:subtrain_end], X_train_full[subtrain_end:]
    y_sub, y_val = y_train_full[:subtrain_end], y_train_full[subtrain_end:]

    n_features = X_sub.shape[-1]
    scaler = StandardScaler()
    scaler.fit(X_sub.reshape(-1, n_features))

    def scale(X_3d: np.ndarray) -> np.ndarray:
        X2 = X_3d.reshape(-1, n_features)
        Xs = scaler.transform(X2).reshape(X_3d.shape).astype(np.float32)
        return Xs

    return {
        "X_sub": scale(X_sub),
        "y_sub": y_sub,
        "X_val": scale(X_val),
        "y_val": y_val,
        "X_train_full": scale(X_train_full),
        "y_train_full": y_train_full,
        "X_test": scale(X_test),
        "y_test": y_test,
        "scaler": scaler,
        "n_features": n_features,
        "n_total": n,
        "n_train": len(X_train_full),
        "n_val": len(X_val),
        "n_test": len(X_test),
    }


# -----------------------------
# Tuning
# -----------------------------
@dataclass(frozen=True)
class TrialConfig:
    window_size: int
    units: int
    dropout: float
    lr: float
    batch_size: int
    seed: int


def sample_trials(
    n_trials: int,
    seed: int = 42,
) -> list[TrialConfig]:
    """
    Compact search space (not huge):
      window_size: 72/96/120   (3)
      units: 64/100            (2)
      dropout: 0.1/0.2/0.3     (3)
      lr: 1e-3/7e-4            (2)  <-- adjusted for short training (5 epochs)
      batch: 64/128            (2)

    Total combos = 72; we sample n_trials randomly WITHOUT duplicates.
    """
    rng = np.random.default_rng(seed)

    window_sizes = [72, 96, 120]
    units_list = [64, 100]
    dropouts = [0.1, 0.2, 0.3]
    lrs = [1e-3, 7e-4]          # changed from 3e-4
    batches = [64, 128]

    trials: list[TrialConfig] = []
    seen: set[tuple[int, int, float, float, int]] = set()

    max_unique = len(window_sizes) * len(units_list) * len(dropouts) * len(lrs) * len(batches)
    n_trials = min(n_trials, max_unique)

    while len(trials) < n_trials:
        cfg = TrialConfig(
            window_size=int(rng.choice(window_sizes)),
            units=int(rng.choice(units_list)),
            dropout=float(rng.choice(dropouts)),
            lr=float(rng.choice(lrs)),
            batch_size=int(rng.choice(batches)),
            seed=int(seed + len(trials)),  # deterministic per trial index
        )
        key = (cfg.window_size, cfg.units, cfg.dropout, cfg.lr, cfg.batch_size)
        if key in seen:
            continue
        seen.add(key)
        trials.append(cfg)

    return trials


def run_trial(
    cfg: TrialConfig,
    splits: dict[str, Any],
    epochs: int,
    patience: int,
    verbose: int,
) -> dict[str, Any]:
    set_global_seed(cfg.seed)

    model = build_lstm_model(
        window_size=cfg.window_size,
        n_features=splits["n_features"],
        units=cfg.units,
        dropout=cfg.dropout,
        lr=cfg.lr,
    )

    es = EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True)

    history = model.fit(
        splits["X_sub"],
        splits["y_sub"],
        validation_data=(splits["X_val"], splits["y_val"]),
        epochs=epochs,
        batch_size=cfg.batch_size,
        shuffle=False,
        callbacks=[es],
        verbose=verbose,
    )

    p_val = model.predict(splits["X_val"], batch_size=cfg.batch_size, verbose=0).reshape(-1)
    y_val = splits["y_val"].astype(int)

    val_ll = float(log_loss(y_val, p_val, labels=[0, 1]))
    try:
        val_auc = float(roc_auc_score(y_val, p_val))
    except ValueError:
        val_auc = float("nan")

    return {
        "window_size": cfg.window_size,
        "units": cfg.units,
        "dropout": cfg.dropout,
        "lr": cfg.lr,
        "batch_size": cfg.batch_size,
        "seed": cfg.seed,
        "val_logloss": val_ll,
        "val_auc": val_auc,
        "epochs_ran": len(history.history.get("loss", [])),
        "best_val_loss": float(np.min(history.history.get("val_loss", [np.inf]))),
    }


def train_lstm_optimise(
    symbol: str = "BTCUSDT",
    resolution: str = "1h",
    start_date: str = "2017-09-01",
    end_date: str | None = None,

    # Outer splits
    train_ratio: float = 0.95,
    val_ratio_within_train: float = 0.10,

    # Search control
    n_trials: int = 12,
    epochs: int = 5,
    patience: int = 2,

    thresholds: tuple[float, ...] = (0.5, 0.55, 0.6),

    results_csv: str = "lstm_tuning_results.csv",
    best_model_path: str = "lstm_best_model.keras",
    best_artifacts_path: str = "lstm_best_artifacts.joblib",
    best_test_probs_csv: str = "lstm_best_test_probs.csv",
    best_test_metrics_csv: str = "lstm_best_test_metrics.csv",

    verbose: int = 0,
    base_seed: int = 42,
):
    # -----------------------------
    # 1) Fetch + FE once
    # -----------------------------
    df = fetch_and_engineer(symbol, resolution, start_date, end_date)

    feature_cols = [
        "open", "high", "low", "close", "volume",
        "log_ret_1", "body", "range", "upper_wick", "lower_wick", "clv",
        "vol_10", "vol_30",
    ]

    # -----------------------------
    # 2) Sample trial configs (no duplicates)
    # -----------------------------
    trials = sample_trials(n_trials=n_trials, seed=base_seed)

    # Cache windowed+scaled splits per window_size
    split_cache: dict[int, dict[str, Any]] = {}

    all_results: list[dict[str, Any]] = []

    # -----------------------------
    # 3) Run trials
    # -----------------------------
    for i, cfg in enumerate(trials, start=1):
        if cfg.window_size not in split_cache:
            split_cache[cfg.window_size] = make_scaled_splits_for_window(
                df=df,
                feature_cols=feature_cols,
                x_window_size=cfg.window_size,
                train_ratio=train_ratio,
                val_ratio_within_train=val_ratio_within_train,
            )

        splits = split_cache[cfg.window_size]

        res = run_trial(
            cfg=cfg,
            splits=splits,
            epochs=epochs,
            patience=patience,
            verbose=verbose,
        )
        res["trial"] = i
        res["n_total"] = splits["n_total"]
        res["n_train"] = splits["n_train"]
        res["n_val"] = splits["n_val"]
        res["n_test"] = splits["n_test"]
        all_results.append(res)

        print(
            f"🧪 Trial {i}/{len(trials)} | win={cfg.window_size} units={cfg.units} "
            f"drop={cfg.dropout} lr={cfg.lr} bs={cfg.batch_size} "
            f"=> val_logloss={res['val_logloss']:.5f} "
            f"val_auc={res['val_auc'] if not np.isnan(res['val_auc']) else 'nan'} "
            f"(epochs_ran={res['epochs_ran']})"
        )

    results_df = pd.DataFrame(all_results).sort_values(["val_logloss", "val_auc"], ascending=[True, False])
    results_df.to_csv(results_csv, index=False)
    print(f"\n✅ Saved tuning results: {results_csv}")

    # -----------------------------
    # 4) Select best trial (min val_logloss)
    # -----------------------------
    best_row = results_df.iloc[0].to_dict()
    best_cfg = TrialConfig(
        window_size=int(best_row["window_size"]),
        units=int(best_row["units"]),
        dropout=float(best_row["dropout"]),
        lr=float(best_row["lr"]),
        batch_size=int(best_row["batch_size"]),
        seed=int(best_row["seed"]),
    )
    print("\n🏆 Best config:", best_cfg)

    # -----------------------------
    # 5) Retrain best on full train (subtrain+val), evaluate on test
    # NOTE: No early stopping here (prevents mismatch + keeps deterministic)
    # -----------------------------
    best_splits = split_cache[best_cfg.window_size]

    set_global_seed(best_cfg.seed)

    best_model = build_lstm_model(
        window_size=best_cfg.window_size,
        n_features=best_splits["n_features"],
        units=best_cfg.units,
        dropout=best_cfg.dropout,
        lr=best_cfg.lr,
    )

    best_model.fit(
        best_splits["X_train_full"],
        best_splits["y_train_full"],
        epochs=epochs,
        batch_size=best_cfg.batch_size,
        shuffle=False,
        verbose=0,
    )

    best_model.save(best_model_path)
    print(f"✅ Saved best model: {best_model_path}")

    # Test probabilities
    p_test = best_model.predict(best_splits["X_test"], batch_size=best_cfg.batch_size, verbose=0).reshape(-1)
    y_test = best_splits["y_test"].astype(int)

    out_df = pd.DataFrame({"y_true": y_test, "p": p_test.astype(float)})
    out_df.to_csv(best_test_probs_csv, index=False)
    print(f"✅ Saved: {best_test_probs_csv}")

    metrics = [eval_with_ignore_zone(y_test, p_test, threshold=t) for t in thresholds]
    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(best_test_metrics_csv, index=False)
    print(f"✅ Saved: {best_test_metrics_csv}")

    print("\n📊 Label balance:")
    print(f"Train UP rate: {float(best_splits['y_train_full'].mean()):.4f}")
    print(f"Test  UP rate: {float(y_test.mean()):.4f}")

    # -----------------------------
    # 6) Save artifacts (includes scaler + params)
    # -----------------------------
    artifacts = {
        "best_params": {
            "window_size": best_cfg.window_size,
            "units": best_cfg.units,
            "dropout": best_cfg.dropout,
            "lr": best_cfg.lr,
            "batch_size": best_cfg.batch_size,
            "seed": best_cfg.seed,
            "epochs": epochs,
            "patience": patience,
            "train_ratio": train_ratio,
            "val_ratio_within_train": val_ratio_within_train,
        },
        "feature_cols": feature_cols,
        "scaler": best_splits["scaler"],
        "thresholds": thresholds,
        "results_csv": results_csv,
        "best_model_path": best_model_path,
        "best_test_probs_csv": best_test_probs_csv,
        "best_test_metrics_csv": best_test_metrics_csv,
    }

    dump(artifacts, best_artifacts_path)
    print(f"✅ Saved best artifacts: {best_artifacts_path}")

    return best_model, artifacts, results_df, out_df, metrics_df


if __name__ == "__main__":
    train_lstm_optimise(
        symbol="BTCUSDT",
        resolution="1h",
        start_date="2017-09-01",
        end_date=None,
        n_trials=12,
        epochs=8,
        patience=2,
        verbose=0,
    )