import os
import random
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from joblib import dump

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    log_loss,
    roc_auc_score,
    balanced_accuracy_score,
)
from sklearn.utils.class_weight import compute_class_weight

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
# Target creation (same as XGB)
# -----------------------------
def create_vol_adjusted_binary_target(
    df: pd.DataFrame,
    horizon: int = 12,
    vol_window: int = 24,
) -> pd.DataFrame:
    df = df.copy()

    df["log_ret_1_raw"] = np.log(df["close"] / df["close"].shift(1))
    df["sigma_t"] = df["log_ret_1_raw"].rolling(vol_window).std()
    df["sigma_h"] = df["sigma_t"] * np.sqrt(horizon)

    df["future_close"] = df["close"].shift(-horizon)
    df["forward_log_return"] = np.log(df["future_close"] / df["close"])

    eps = 1e-12
    df["target_score"] = df["forward_log_return"] / (df["sigma_h"] + eps)

    df["actual_trend"] = np.where(df["target_score"] > 0, "up", "down")
    df["target"] = (df["actual_trend"] == "up").astype(np.int32)

    return df


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
# Data prep
# -----------------------------
def fetch_and_engineer(
    symbol: str,
    resolution: str,
    start_date: str,
    end_date: str | None,
    horizon: int,
    vol_window: int,
) -> pd.DataFrame:
    client = BinanceDataClient(market="spot")
    df = client.get_candles(
        symbol=symbol,
        resolution=resolution,
        start_date=start_date,
        end_date=end_date,
    )

    required = {"timestamp", "open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.sort_values("timestamp").reset_index(drop=True)
    df = df.dropna(subset=["open", "high", "low", "close", "volume"]).reset_index(drop=True)

    df = create_vol_adjusted_binary_target(
        df=df,
        horizon=horizon,
        vol_window=vol_window,
    )

    df = df.dropna(
        subset=[
            "future_close",
            "forward_log_return",
            "sigma_t",
            "sigma_h",
            "target_score",
            "actual_trend",
            "target",
        ]
    ).reset_index(drop=True)

    df = feature_engineering_lstm(df)

    return df


def make_scaled_splits_for_window(
    df: pd.DataFrame,
    feature_cols: list[str],
    x_window_size: int,
    train_ratio: float,
    val_ratio_within_train: float,
) -> dict[str, Any]:
    df = df.dropna(subset=feature_cols + ["target"]).reset_index(drop=True)

    X, y = make_windows(
        df,
        x_window_size=x_window_size,
        feature_cols=feature_cols,
        target_col="target",
    )
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int32)

    if X.ndim != 3:
        raise ValueError(f"Expected X with shape (N, T, F), got {X.shape}")
    if X.shape[1] != x_window_size:
        raise ValueError(f"Expected T={x_window_size}, got {X.shape[1]}")
    if X.shape[2] != len(feature_cols):
        raise ValueError(f"Expected F={len(feature_cols)}, got {X.shape[2]}")

    meta_df = df.iloc[x_window_size:].reset_index(drop=True)
    if len(meta_df) != len(y):
        raise ValueError(f"meta_df length {len(meta_df)} does not match y length {len(y)}")

    n = len(X)
    train_end = int(n * train_ratio)
    if train_end <= 0 or train_end >= n:
        raise ValueError(f"Not enough samples after windowing. n={n}, train_end={train_end}")

    X_train_full, X_test = X[:train_end], X[train_end:]
    y_train_full, y_test = y[:train_end], y[train_end:]
    meta_train_full, meta_test = meta_df.iloc[:train_end].copy(), meta_df.iloc[train_end:].copy()

    subtrain_end = int(len(X_train_full) * (1.0 - val_ratio_within_train))
    if subtrain_end <= 0 or subtrain_end >= len(X_train_full):
        raise ValueError(
            f"Invalid val split inside train. n_train={len(X_train_full)}, subtrain_end={subtrain_end}"
        )

    X_sub, X_val = X_train_full[:subtrain_end], X_train_full[subtrain_end:]
    y_sub, y_val = y_train_full[:subtrain_end], y_train_full[subtrain_end:]
    meta_sub, meta_val = meta_train_full.iloc[:subtrain_end].copy(), meta_train_full.iloc[subtrain_end:].copy()

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
        "meta_sub": meta_sub,
        "X_val": scale(X_val),
        "y_val": y_val,
        "meta_val": meta_val,
        "X_train_full": scale(X_train_full),
        "y_train_full": y_train_full,
        "meta_train_full": meta_train_full,
        "X_test": scale(X_test),
        "y_test": y_test,
        "meta_test": meta_test,
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
    rng = np.random.default_rng(seed)

    # slightly broader / safer search
    window_sizes = [48, 72, 96, 120]
    units_list = [32, 64, 100]
    dropouts = [0.1, 0.2, 0.3]
    lrs = [3e-4, 7e-4, 1e-3]
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
            seed=int(seed + len(trials)),
        )
        key = (cfg.window_size, cfg.units, cfg.dropout, cfg.lr, cfg.batch_size)
        if key in seen:
            continue
        seen.add(key)
        trials.append(cfg)

    return trials


def get_class_weight_dict(y: np.ndarray) -> dict[int, float]:
    classes = np.array([0, 1])
    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y.astype(int),
    )
    return {0: float(weights[0]), 1: float(weights[1])}


def collapse_penalty_from_probs(p: np.ndarray) -> float:
    """
    Penalize very one-sided predictions.
    Uses predicted positive rate at 0.5 threshold.
    """
    pred = (p >= 0.5).astype(int)
    pos_rate = float(pred.mean())

    # no penalty in a reasonable band
    if 0.35 <= pos_rate <= 0.65:
        return 0.0

    # moderate penalty outside band
    return abs(pos_rate - 0.5) * 2.0


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

    class_weight = get_class_weight_dict(splits["y_sub"])

    history = model.fit(
        splits["X_sub"],
        splits["y_sub"],
        validation_data=(splits["X_val"], splits["y_val"]),
        epochs=epochs,
        batch_size=cfg.batch_size,
        shuffle=False,
        callbacks=[es],
        verbose=verbose,
        class_weight=class_weight,
    )

    p_val = model.predict(splits["X_val"], batch_size=cfg.batch_size, verbose=0).reshape(-1)
    y_val = splits["y_val"].astype(int)

    val_ll = float(log_loss(y_val, p_val, labels=[0, 1]))

    try:
        val_auc = float(roc_auc_score(y_val, p_val))
    except ValueError:
        val_auc = float("nan")

    y_pred_05 = (p_val >= 0.5).astype(int)
    val_bal_acc = float(balanced_accuracy_score(y_val, y_pred_05))
    pred_pos_rate = float(y_pred_05.mean())
    collapse_penalty = float(collapse_penalty_from_probs(p_val))

    # lower is better
    selection_score = val_ll + collapse_penalty

    return {
        "window_size": cfg.window_size,
        "units": cfg.units,
        "dropout": cfg.dropout,
        "lr": cfg.lr,
        "batch_size": cfg.batch_size,
        "seed": cfg.seed,
        "val_logloss": val_ll,
        "val_auc": val_auc,
        "val_bal_acc_05": val_bal_acc,
        "val_pred_pos_rate_05": pred_pos_rate,
        "collapse_penalty": collapse_penalty,
        "selection_score": selection_score,
        "epochs_ran": len(history.history.get("loss", [])),
        "best_val_loss": float(np.min(history.history.get("val_loss", [np.inf]))),
    }


def train_lstm_optimise(
    symbol: str = "BTCUSDT",
    resolution: str = "1h",
    start_date: str = "2017-09-01",
    end_date: str | None = None,

    horizon: int = 12,
    vol_window: int = 24,

    train_ratio: float = 0.95,
    val_ratio_within_train: float = 0.10,

    n_trials: int = 16,
    epochs: int = 12,
    patience: int = 3,

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
    df = fetch_and_engineer(
        symbol=symbol,
        resolution=resolution,
        start_date=start_date,
        end_date=end_date,
        horizon=horizon,
        vol_window=vol_window,
    )

    feature_cols = [
        "open", "high", "low", "close", "volume",
        "log_ret_1", "body", "range", "upper_wick", "lower_wick", "clv",
        "vol_10", "vol_30",
    ]

    # -----------------------------
    # 2) Sample trial configs
    # -----------------------------
    trials = sample_trials(n_trials=n_trials, seed=base_seed)

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
            f"🧪 Trial {i}/{len(trials)} | "
            f"win={cfg.window_size} units={cfg.units} drop={cfg.dropout} "
            f"lr={cfg.lr} bs={cfg.batch_size} | "
            f"score={res['selection_score']:.5f} "
            f"logloss={res['val_logloss']:.5f} "
            f"bal_acc={res['val_bal_acc_05']:.4f} "
            f"auc={res['val_auc'] if not np.isnan(res['val_auc']) else 'nan'} "
            f"pos_rate={res['val_pred_pos_rate_05']:.3f}"
        )

    results_df = pd.DataFrame(all_results).sort_values(
        ["selection_score", "val_bal_acc_05", "val_auc"],
        ascending=[True, False, False],
    )
    results_df.to_csv(results_csv, index=False)
    print(f"\n✅ Saved tuning results: {results_csv}")

    # -----------------------------
    # 4) Select best trial
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
    # 5) Retrain best on full train
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

    class_weight_full = get_class_weight_dict(best_splits["y_train_full"])

    best_model.fit(
        best_splits["X_train_full"],
        best_splits["y_train_full"],
        epochs=epochs,
        batch_size=best_cfg.batch_size,
        shuffle=False,
        verbose=0,
        class_weight=class_weight_full,
    )

    best_model.save(best_model_path)
    print(f"✅ Saved best model: {best_model_path}")

    # -----------------------------
    # 6) Test probabilities
    # -----------------------------
    p_test = best_model.predict(best_splits["X_test"], batch_size=best_cfg.batch_size, verbose=0).reshape(-1)
    y_test = best_splits["y_test"].astype(int)
    meta_test = best_splits["meta_test"].copy()

    out_df = pd.DataFrame({
        "timestamp": meta_test["timestamp"].values if "timestamp" in meta_test.columns else np.arange(len(meta_test)),
        "y_true": y_test,
        "actual_trend": np.where(y_test == 1, "up", "down"),
        "p_down": 1.0 - p_test,
        "p_up": p_test.astype(float),
        "probability": p_test.astype(float),
        "future_close": meta_test["future_close"].values,
        "forward_log_return": meta_test["forward_log_return"].values,
        "sigma_t": meta_test["sigma_t"].values,
        "sigma_h": meta_test["sigma_h"].values,
        "target_score": meta_test["target_score"].values,
    })
    out_df.to_csv(best_test_probs_csv, index=False)
    print(f"✅ Saved: {best_test_probs_csv}")

    metrics = [eval_with_ignore_zone(y_test, p_test, threshold=t) for t in thresholds]
    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(best_test_metrics_csv, index=False)
    print(f"✅ Saved: {best_test_metrics_csv}")

    print("\n📊 Label balance:")
    print(f"Train UP rate: {float(best_splits['y_train_full'].mean()):.4f}")
    print(f"Test  UP rate: {float(y_test.mean()):.4f}")

    print("\n📊 Test probability summary:")
    print(pd.Series(p_test).describe())

    print("\n📊 Test prediction balance @0.5:")
    print(pd.Series((p_test >= 0.5).astype(int)).value_counts(normalize=True).sort_index())

    # -----------------------------
    # 7) Save artifacts
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
            "horizon": horizon,
            "vol_window": vol_window,
        },
        "feature_cols": feature_cols,
        "scaler": best_splits["scaler"],
        "thresholds": thresholds,
        "results_csv": results_csv,
        "best_model_path": best_model_path,
        "best_test_probs_csv": best_test_probs_csv,
        "best_test_metrics_csv": best_test_metrics_csv,
        "target_definition": (
            "binary target: up if forward_log_return / sigma_h > 0, down otherwise"
        ),
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
        horizon=12,
        vol_window=24,
        n_trials=16,
        epochs=12,
        patience=3,
        verbose=0,
    )