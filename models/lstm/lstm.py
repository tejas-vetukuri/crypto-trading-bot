# models/lstm/lstm.py

import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

from joblib import dump

from data.delta_exchange import DeltaDataClient
from data.feature_engineering import feature_engineering_lstm
from models.lstm.confidence_threshold import eval_with_ignore_zone


def build_tp_horizon_windows(
    df: pd.DataFrame,
    x_window_size: int,
    feature_cols: list[str],
    horizon: int = 12,
    tp_pct: float = 0.02,
    sl_pct: float = 0.01,
    side: str = "long",
    skip_ambiguous: bool = True,
):
    """
    side='long':
      y = 1 if long TP hit before long SL within horizon
      y = 0 if long SL hit before long TP within horizon

    side='short':
      y = 1 if short TP hit before short SL within horizon
      y = 0 if short SL hit before short TP within horizon
    """
    if side not in {"long", "short"}:
        raise ValueError(f"side must be 'long' or 'short', got {side}")

    X_list = []
    y_list = []
    meta_rows = []

    n = len(df)
    if n < x_window_size + horizon:
        raise ValueError(
            f"Not enough rows for x_window_size={x_window_size} and horizon={horizon}. Got n={n}"
        )

    for end_idx in range(x_window_size, n - horizon + 1):
        window = df.iloc[end_idx - x_window_size:end_idx][feature_cols].values.astype(np.float32)

        entry_idx = end_idx - 1
        entry_row = df.iloc[entry_idx]
        entry_price = float(entry_row["close"])
        timestamp = entry_row["timestamp"] if "timestamp" in df.columns else entry_idx

        if side == "long":
            tp_barrier = entry_price * (1.0 + tp_pct)
            sl_barrier = entry_price * (1.0 - sl_pct)
        else:
            tp_barrier = entry_price * (1.0 - tp_pct)
            sl_barrier = entry_price * (1.0 + sl_pct)

        future_slice = df.iloc[end_idx:end_idx + horizon]

        label = None
        hit_step = None
        outcome = "unresolved"

        for step, row in enumerate(future_slice.itertuples(index=False), start=1):
            high = float(row.high)
            low = float(row.low)

            if side == "long":
                hit_tp = high >= tp_barrier
                hit_sl = low <= sl_barrier
            else:
                hit_tp = low <= tp_barrier
                hit_sl = high >= sl_barrier

            if hit_tp and hit_sl:
                hit_step = step
                outcome = "ambiguous_same_bar"
                label = None
                break

            if hit_tp:
                label = 1
                hit_step = step
                outcome = "tp_first"
                break

            if hit_sl:
                label = 0
                hit_step = step
                outcome = "sl_first"
                break

        if label is None:
            if skip_ambiguous or outcome != "ambiguous_same_bar":
                continue
            continue

        X_list.append(window)
        y_list.append(label)
        meta_rows.append(
            {
                "timestamp": timestamp,
                "entry_idx": entry_idx,
                "entry_close": entry_price,
                "tp_barrier": tp_barrier,
                "sl_barrier": sl_barrier,
                "hit_step": hit_step,
                "label": label,
                "outcome": outcome,
                "side": side,
            }
        )

    X = np.asarray(X_list, dtype=np.float32)
    y = np.asarray(y_list, dtype=np.int32)
    meta_df = pd.DataFrame(meta_rows)

    return X, y, meta_df


def _print_label_stats(name: str, y: np.ndarray):
    y = np.asarray(y).astype(int)
    pos_rate = float(y.mean()) if len(y) > 0 else float("nan")
    neg_rate = 1.0 - pos_rate if len(y) > 0 else float("nan")
    print(f"\n📊 {name} label stats")
    print(f"Samples:        {len(y)}")
    print(f"Positive rate:  {pos_rate:.4f}")
    print(f"Negative rate:  {neg_rate:.4f}")
    print(f"Pos count:      {int((y == 1).sum())}")
    print(f"Neg count:      {int((y == 0).sum())}")


def train_lstm_model(
    symbol: str = "BTCUSD",
    resolution: str = "1h",
    start_date: str = "2019-06-01",
    end_date: str | None = None,
    x_window_size: int = 100,
    epochs: int = 10,
    batch_size: int = 64,
    train_ratio: float = 0.80,

    horizon: int = 12,
    tp_pct: float = 0.02,
    sl_pct: float = 0.01,
    side: str = "long",
    skip_ambiguous: bool = True,

    model_path: str | None = None,
    scaler_path: str | None = None,
    artifacts_path: str | None = None,

    thresholds: tuple[float, ...] = (0.20, 0.25, 0.30, 0.35, 0.40, 0.50),
):
    if side not in {"long", "short"}:
        raise ValueError(f"side must be 'long' or 'short', got {side}")

    if not (0.0 < train_ratio < 1.0):
        raise ValueError(f"train_ratio must be in (0,1), got {train_ratio}")

    side_suffix = side

    if model_path is None:
        model_path = f"models/lstm/lstm_tp_horizon_stdscale_{side_suffix}.keras"
    if scaler_path is None:
        scaler_path = f"models/lstm/lstm_tp_horizon_scaler_{side_suffix}.joblib"
    if artifacts_path is None:
        artifacts_path = f"models/lstm/lstm_tp_horizon_artifacts_{side_suffix}.joblib"

    out_probs_path = f"models/lstm/lstm_tp_horizon_test_probs_{side_suffix}.csv"
    out_metrics_path = f"models/lstm/lstm_tp_horizon_metrics_{side_suffix}.csv"

    client = DeltaDataClient()
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

    feature_cols = [
        "open", "high", "low", "close", "volume",
        "log_ret_1", "body", "range", "upper_wick", "lower_wick", "clv",
        "vol_10", "vol_30",
    ]
    df = df.dropna(subset=feature_cols).reset_index(drop=True)

    X, y, meta_df = build_tp_horizon_windows(
        df=df,
        x_window_size=x_window_size,
        feature_cols=feature_cols,
        horizon=horizon,
        tp_pct=tp_pct,
        sl_pct=sl_pct,
        side=side,
        skip_ambiguous=skip_ambiguous,
    )

    if len(X) == 0:
        raise ValueError("No valid samples after TP-horizon labeling.")

    n = len(X)
    train_end = int(n * train_ratio)
    if train_end <= 0 or train_end >= n:
        raise ValueError(f"Not enough labeled samples after windowing. n={n}, train_end={train_end}")

    X_train, X_test = X[:train_end], X[train_end:]
    y_train, y_test = y[:train_end], y[train_end:]
    meta_test = meta_df.iloc[train_end:].reset_index(drop=True)

    print(f"\nTrain ratio used: {train_ratio:.2f}")
    _print_label_stats("TRAIN", y_train)
    _print_label_stats("TEST", y_test)

    n_features = X_train.shape[-1]
    scaler = StandardScaler()
    X_train_2d = X_train.reshape(-1, n_features)
    scaler.fit(X_train_2d)

    scaler_path_p = Path(scaler_path)
    scaler_path_p.parent.mkdir(parents=True, exist_ok=True)
    dump(scaler, str(scaler_path_p))
    print(f"\n✅ Saved scaler: {scaler_path_p}")

    X_train_s = scaler.transform(X_train_2d).reshape(X_train.shape).astype(np.float32)
    X_test_s = scaler.transform(X_test.reshape(-1, n_features)).reshape(X_test.shape).astype(np.float32)

    classes = np.array([0, 1])
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    class_weight = {0: float(weights[0]), 1: float(weights[1])}

    print("\n⚖️ Class weights")
    print(class_weight)

    model = Sequential([
        Input(shape=(x_window_size, n_features)),
        LSTM(100, return_sequences=False),
        Dropout(0.2),
        Dense(1, activation="sigmoid"),
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    history = model.fit(
        X_train_s,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.05,
        shuffle=False,
        callbacks=[EarlyStopping(patience=3, restore_best_weights=True)],
        verbose=1,
        class_weight=class_weight,
    )

    model_path_p = Path(model_path)
    model_path_p.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(model_path_p))
    print(f"\n✅ Saved model: {model_path_p}")

    p_test = model.predict(X_test_s, batch_size=batch_size, verbose=0).reshape(-1)

    out_df = meta_test.copy()
    out_df["y_true"] = y_test.astype(int)
    out_df["p"] = p_test.astype(float)

    Path(out_probs_path).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_probs_path, index=False)

    metrics = [eval_with_ignore_zone(y_test, p_test, threshold=t) for t in thresholds]
    metrics_df = pd.DataFrame(metrics)
    Path(out_metrics_path).parent.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(out_metrics_path, index=False)

    lstm_artifacts = {
        "model_path": str(model_path_p),
        "scaler_path": str(scaler_path_p),
        "x_window_size": int(x_window_size),
        "feature_cols": feature_cols,
        "symbol": symbol,
        "resolution": resolution,
        "start_date": start_date,
        "end_date": end_date,
        "train_ratio": float(train_ratio),
        "thresholds_eval": thresholds,
        "target_type": "tp_before_sl_within_horizon",
        "side": side,
        "horizon": int(horizon),
        "tp_pct": float(tp_pct),
        "sl_pct": float(sl_pct),
        "skip_ambiguous": bool(skip_ambiguous),
        "class_weight": class_weight,
    }

    artifacts_path_p = Path(artifacts_path)
    artifacts_path_p.parent.mkdir(parents=True, exist_ok=True)
    dump(lstm_artifacts, str(artifacts_path_p))
    print(f"✅ Saved artifacts: {artifacts_path_p}")
    print(f"✅ Saved: {out_probs_path}")
    print(f"✅ Saved: {out_metrics_path}")

    print("\n📈 Test probability summary")
    print(pd.Series(p_test).describe())

    return model, history, out_df, metrics_df, scaler# models/lstm/lstm.py

import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

from joblib import dump

from data.delta_exchange import DeltaDataClient
from data.feature_engineering import feature_engineering_lstm
from models.lstm.confidence_threshold import eval_with_ignore_zone


def build_tp_horizon_windows(
    df: pd.DataFrame,
    x_window_size: int,
    feature_cols: list[str],
    horizon: int = 12,
    tp_pct: float = 0.02,
    sl_pct: float = 0.01,
    side: str = "long",
    skip_ambiguous: bool = True,
):
    """
    side='long':
      y = 1 if long TP hit before long SL within horizon
      y = 0 if long SL hit before long TP within horizon

    side='short':
      y = 1 if short TP hit before short SL within horizon
      y = 0 if short SL hit before short TP within horizon
    """
    if side not in {"long", "short"}:
        raise ValueError(f"side must be 'long' or 'short', got {side}")

    X_list = []
    y_list = []
    meta_rows = []

    n = len(df)
    if n < x_window_size + horizon:
        raise ValueError(
            f"Not enough rows for x_window_size={x_window_size} and horizon={horizon}. Got n={n}"
        )

    for end_idx in range(x_window_size, n - horizon + 1):
        window = df.iloc[end_idx - x_window_size:end_idx][feature_cols].values.astype(np.float32)

        entry_idx = end_idx - 1
        entry_row = df.iloc[entry_idx]
        entry_price = float(entry_row["close"])
        timestamp = entry_row["timestamp"] if "timestamp" in df.columns else entry_idx

        if side == "long":
            tp_barrier = entry_price * (1.0 + tp_pct)
            sl_barrier = entry_price * (1.0 - sl_pct)
        else:
            tp_barrier = entry_price * (1.0 - tp_pct)
            sl_barrier = entry_price * (1.0 + sl_pct)

        future_slice = df.iloc[end_idx:end_idx + horizon]

        label = None
        hit_step = None
        outcome = "unresolved"

        for step, row in enumerate(future_slice.itertuples(index=False), start=1):
            high = float(row.high)
            low = float(row.low)

            if side == "long":
                hit_tp = high >= tp_barrier
                hit_sl = low <= sl_barrier
            else:
                hit_tp = low <= tp_barrier
                hit_sl = high >= sl_barrier

            if hit_tp and hit_sl:
                hit_step = step
                outcome = "ambiguous_same_bar"
                label = None
                break

            if hit_tp:
                label = 1
                hit_step = step
                outcome = "tp_first"
                break

            if hit_sl:
                label = 0
                hit_step = step
                outcome = "sl_first"
                break

        if label is None:
            if skip_ambiguous or outcome != "ambiguous_same_bar":
                continue
            continue

        X_list.append(window)
        y_list.append(label)
        meta_rows.append(
            {
                "timestamp": timestamp,
                "entry_idx": entry_idx,
                "entry_close": entry_price,
                "tp_barrier": tp_barrier,
                "sl_barrier": sl_barrier,
                "hit_step": hit_step,
                "label": label,
                "outcome": outcome,
                "side": side,
            }
        )

    X = np.asarray(X_list, dtype=np.float32)
    y = np.asarray(y_list, dtype=np.int32)
    meta_df = pd.DataFrame(meta_rows)

    return X, y, meta_df


def _print_label_stats(name: str, y: np.ndarray):
    y = np.asarray(y).astype(int)
    pos_rate = float(y.mean()) if len(y) > 0 else float("nan")
    neg_rate = 1.0 - pos_rate if len(y) > 0 else float("nan")
    print(f"\n📊 {name} label stats")
    print(f"Samples:        {len(y)}")
    print(f"Positive rate:  {pos_rate:.4f}")
    print(f"Negative rate:  {neg_rate:.4f}")
    print(f"Pos count:      {int((y == 1).sum())}")
    print(f"Neg count:      {int((y == 0).sum())}")


def train_lstm_model(
    symbol: str = "BTCUSD",
    resolution: str = "1h",
    start_date: str = "2019-06-01",
    end_date: str | None = None,
    x_window_size: int = 100,
    epochs: int = 10,
    batch_size: int = 64,
    train_ratio: float = 0.80,

    horizon: int = 12,
    tp_pct: float = 0.02,
    sl_pct: float = 0.01,
    side: str = "long",
    skip_ambiguous: bool = True,

    model_path: str | None = None,
    scaler_path: str | None = None,
    artifacts_path: str | None = None,

    thresholds: tuple[float, ...] = (0.20, 0.25, 0.30, 0.35, 0.40, 0.50),
):
    if side not in {"long", "short"}:
        raise ValueError(f"side must be 'long' or 'short', got {side}")

    if not (0.0 < train_ratio < 1.0):
        raise ValueError(f"train_ratio must be in (0,1), got {train_ratio}")

    side_suffix = side

    if model_path is None:
        model_path = f"models/lstm/lstm_tp_horizon_stdscale_{side_suffix}.keras"
    if scaler_path is None:
        scaler_path = f"models/lstm/lstm_tp_horizon_scaler_{side_suffix}.joblib"
    if artifacts_path is None:
        artifacts_path = f"models/lstm/lstm_tp_horizon_artifacts_{side_suffix}.joblib"

    out_probs_path = f"models/lstm/lstm_tp_horizon_test_probs_{side_suffix}.csv"
    out_metrics_path = f"models/lstm/lstm_tp_horizon_metrics_{side_suffix}.csv"

    client = DeltaDataClient()
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

    feature_cols = [
        "open", "high", "low", "close", "volume",
        "log_ret_1", "body", "range", "upper_wick", "lower_wick", "clv",
        "vol_10", "vol_30",
    ]
    df = df.dropna(subset=feature_cols).reset_index(drop=True)

    X, y, meta_df = build_tp_horizon_windows(
        df=df,
        x_window_size=x_window_size,
        feature_cols=feature_cols,
        horizon=horizon,
        tp_pct=tp_pct,
        sl_pct=sl_pct,
        side=side,
        skip_ambiguous=skip_ambiguous,
    )

    if len(X) == 0:
        raise ValueError("No valid samples after TP-horizon labeling.")

    n = len(X)
    train_end = int(n * train_ratio)
    if train_end <= 0 or train_end >= n:
        raise ValueError(f"Not enough labeled samples after windowing. n={n}, train_end={train_end}")

    X_train, X_test = X[:train_end], X[train_end:]
    y_train, y_test = y[:train_end], y[train_end:]
    meta_test = meta_df.iloc[train_end:].reset_index(drop=True)

    print(f"\nTrain ratio used: {train_ratio:.2f}")
    _print_label_stats("TRAIN", y_train)
    _print_label_stats("TEST", y_test)

    n_features = X_train.shape[-1]
    scaler = StandardScaler()
    X_train_2d = X_train.reshape(-1, n_features)
    scaler.fit(X_train_2d)

    scaler_path_p = Path(scaler_path)
    scaler_path_p.parent.mkdir(parents=True, exist_ok=True)
    dump(scaler, str(scaler_path_p))
    print(f"\n✅ Saved scaler: {scaler_path_p}")

    X_train_s = scaler.transform(X_train_2d).reshape(X_train.shape).astype(np.float32)
    X_test_s = scaler.transform(X_test.reshape(-1, n_features)).reshape(X_test.shape).astype(np.float32)

    classes = np.array([0, 1])
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    class_weight = {0: float(weights[0]), 1: float(weights[1])}

    print("\n⚖️ Class weights")
    print(class_weight)

    model = Sequential([
        Input(shape=(x_window_size, n_features)),
        LSTM(100, return_sequences=False),
        Dropout(0.2),
        Dense(1, activation="sigmoid"),
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    history = model.fit(
        X_train_s,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.05,
        shuffle=False,
        callbacks=[EarlyStopping(patience=3, restore_best_weights=True)],
        verbose=1,
        class_weight=class_weight,
    )

    model_path_p = Path(model_path)
    model_path_p.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(model_path_p))
    print(f"\n✅ Saved model: {model_path_p}")

    p_test = model.predict(X_test_s, batch_size=batch_size, verbose=0).reshape(-1)

    out_df = meta_test.copy()
    out_df["y_true"] = y_test.astype(int)
    out_df["p"] = p_test.astype(float)

    Path(out_probs_path).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_probs_path, index=False)

    metrics = [eval_with_ignore_zone(y_test, p_test, threshold=t) for t in thresholds]
    metrics_df = pd.DataFrame(metrics)
    Path(out_metrics_path).parent.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(out_metrics_path, index=False)

    lstm_artifacts = {
        "model_path": str(model_path_p),
        "scaler_path": str(scaler_path_p),
        "x_window_size": int(x_window_size),
        "feature_cols": feature_cols,
        "symbol": symbol,
        "resolution": resolution,
        "start_date": start_date,
        "end_date": end_date,
        "train_ratio": float(train_ratio),
        "thresholds_eval": thresholds,
        "target_type": "tp_before_sl_within_horizon",
        "side": side,
        "horizon": int(horizon),
        "tp_pct": float(tp_pct),
        "sl_pct": float(sl_pct),
        "skip_ambiguous": bool(skip_ambiguous),
        "class_weight": class_weight,
    }

    artifacts_path_p = Path(artifacts_path)
    artifacts_path_p.parent.mkdir(parents=True, exist_ok=True)
    dump(lstm_artifacts, str(artifacts_path_p))
    print(f"✅ Saved artifacts: {artifacts_path_p}")
    print(f"✅ Saved: {out_probs_path}")
    print(f"✅ Saved: {out_metrics_path}")

    print("\n📈 Test probability summary")
    print(pd.Series(p_test).describe())

    return model, history, out_df, metrics_df, scaler