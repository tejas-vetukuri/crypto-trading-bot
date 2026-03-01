# xgb_optimise.py

import numpy as np
import pandas as pd

from xgboost import XGBClassifier
import xgboost as xgb

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    log_loss,
)

from joblib import dump

from data.delta_exchange import DeltaDataClient
from data.feature_engineering import feature_engineering_xgb


def _time_split(df: pd.DataFrame, ratio: float):
    n = len(df)
    cut = int(n * ratio)
    if cut <= 0 or cut >= n:
        raise ValueError(f"Invalid split. n={n}, cut={cut}, ratio={ratio}")
    return df.iloc[:cut].copy(), df.iloc[cut:].copy()


def _fit_with_early_stopping_if_supported(
    model: XGBClassifier,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    early_stopping_rounds: int = 50,
):
    """
    Cross-version safe fitting:
      1) Try callbacks-based early stopping (preferred).
      2) Try early_stopping_rounds kw (newer versions).
      3) Fallback: no early stopping.
    Returns: fitted model, used_early_stopping (bool)
    """
    # 1) Try callback API
    try:
        cb = [xgb.callback.EarlyStopping(rounds=early_stopping_rounds, save_best=True)]
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
            callbacks=cb,
        )
        return model, True
    except TypeError:
        pass
    except Exception:
        # Some versions raise different errors for callbacks; continue to next attempt
        pass

    # 2) Try early_stopping_rounds kw
    try:
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
            early_stopping_rounds=early_stopping_rounds,
        )
        return model, True
    except TypeError:
        pass
    except Exception:
        pass

    # 3) Fallback: plain fit (no early stopping)
    model.fit(X_train, y_train)
    return model, False


def _tune_xgb_random_search(
    X_subtrain: pd.DataFrame,
    y_subtrain: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    n_trials: int = 30,
    early_stopping_rounds: int = 50,
    random_state: int = 42,
):
    """
    Time-series-safe tuning: train on subtrain, (try) early-stop on val.
    Select best trial by validation logloss (probability quality).
    """
    rng = np.random.default_rng(random_state)

    pos = int((y_subtrain == 1).sum())
    neg = int((y_subtrain == 0).sum())
    scale_pos_weight = float(neg / max(pos, 1))

    best = None
    history = []

    for t in range(n_trials):
        params = {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "tree_method": "hist",
            "random_state": random_state,

            "max_depth": int(rng.integers(3, 7)),               # 3..6
            "min_child_weight": float(rng.integers(5, 31)),     # 5..30
            "learning_rate": float(rng.uniform(0.01, 0.08)),    # 0.01..0.08

            "gamma": float(rng.uniform(0.0, 5.0)),              # 0..5
            "reg_alpha": float(rng.uniform(0.0, 1.0)),          # 0..1
            "reg_lambda": float(rng.uniform(1.0, 20.0)),        # 1..20

            "subsample": float(rng.uniform(0.6, 0.95)),         # 0.6..0.95
            "colsample_bytree": float(rng.uniform(0.6, 0.95)),  # 0.6..0.95

            "scale_pos_weight": scale_pos_weight,

            # big; ES will stop earlier if supported
            "n_estimators": 5000,
        }

        model = XGBClassifier(**params)
        model, used_es = _fit_with_early_stopping_if_supported(
            model,
            X_subtrain,
            y_subtrain,
            X_val,
            y_val,
            early_stopping_rounds=early_stopping_rounds,
        )

        p_val = model.predict_proba(X_val)[:, 1]
        val_ll = log_loss(y_val, p_val, labels=[0, 1])
        val_auc = roc_auc_score(y_val, p_val)

        # best_iteration exists only if ES was used and supported
        best_iter = getattr(model, "best_iteration", None)
        if best_iter is None:
            # Some versions store it differently
            best_iter = getattr(model, "best_ntree_limit", None)

        record = {
            "trial": int(t),
            "val_logloss": float(val_ll),
            "val_auc": float(val_auc),
            "used_early_stopping": bool(used_es),
            "best_iteration": int(best_iter) if best_iter is not None else None,
            "params": params,
        }
        history.append(record)

        if best is None or val_ll < best["val_logloss"]:
            best = record

    return best, history


def train_xgb_model(
    symbol: str = "BTCUSD",
    resolution: str = "1h",
    start_date: str = "2019-06-01",
    end_date: str | None = None,
    train_ratio: float = 0.80,

    decision_boundary: float = 0.46,
    margin_threshold: float = 0.05,

    tune: bool = True,
    val_ratio_within_train: float = 0.20,
    n_trials: int = 30,
    early_stopping_rounds: int = 50,

    artifacts_path: str = "xgb_tuned_artifacts.joblib",
    preds_csv_path: str = "xgb_predictions.csv",
):
    # -----------------------------
    # 1) Fetch candles
    # -----------------------------
    client = DeltaDataClient()
    df = client.get_candles(
        symbol=symbol,
        resolution=resolution,
        start_date=start_date,
        end_date=end_date,
    )
    df = df.sort_values("timestamp").reset_index(drop=True)

    # -----------------------------
    # 2) Chronological train/test split
    # -----------------------------
    train_df, test_df = _time_split(df, train_ratio)

    # -----------------------------
    # 3) Feature engineering (separate to avoid leakage)
    # -----------------------------
    train_df = feature_engineering_xgb(train_df)
    test_df = feature_engineering_xgb(test_df)

    features = [
        "ema_20", "ema_50", "rsi", "atr",
        "log_ret_1", "ret_1", "ret_3", "ret_5", "ret_10",
        "body", "range", "upper_wick", "lower_wick", "body_pct", "range_pct", "clv",
        "ema_spread", "ema20_dist", "ema50_dist", "ema20_slope_3", "ema50_slope_3",
        "atr_pct", "rsi_delta", "rsi_ma_10", "rsi_dist",
        "volatility_5", "vol_10", "vol_30", "vol_ratio",
        "vol_chg_1", "vol_chg_5", "vol_z20",
    ]

    X_train_full = train_df[features]
    y_train_full = train_df["actual_trend"].astype(str)

    X_test = test_df[features]
    y_test = test_df["actual_trend"].astype(str)

    # -----------------------------
    # 4) Label encoding (keep for compatibility/artifacts)
    # -----------------------------
    le = LabelEncoder()
    le.fit(y_train_full)

    if "up" not in le.classes_ or "down" not in le.classes_:
        raise ValueError(f"Expected classes to include 'up' and 'down', got {list(le.classes_)}")

    # Use up=1, down=0 for XGB fit so p_up is always proba[:,1]
    y_train_is_up = (y_train_full.values == "up").astype(int)
    y_test_is_up = (y_test.values == "up").astype(int)

    # -----------------------------
    # 5) Chronological subtrain/val split inside train
    # -----------------------------
    if not (0.05 <= val_ratio_within_train <= 0.40):
        raise ValueError("val_ratio_within_train should be between 0.05 and 0.40.")

    n_train = len(train_df)
    subtrain_end = int(n_train * (1.0 - val_ratio_within_train))
    if subtrain_end <= 0 or subtrain_end >= n_train:
        raise ValueError(f"Invalid val split inside train. n_train={n_train}, subtrain_end={subtrain_end}")

    X_subtrain = X_train_full.iloc[:subtrain_end]
    y_subtrain = y_train_is_up[:subtrain_end]

    X_val = X_train_full.iloc[subtrain_end:]
    y_val = y_train_is_up[subtrain_end:]

    # -----------------------------
    # 6) Tune + train best model
    # -----------------------------
    tuning_summary = None
    trials = None

    if tune:
        best_trial, trials = _tune_xgb_random_search(
            X_subtrain=X_subtrain,
            y_subtrain=y_subtrain,
            X_val=X_val,
            y_val=y_val,
            n_trials=n_trials,
            early_stopping_rounds=early_stopping_rounds,
            random_state=42,
        )
        best_params = best_trial["params"].copy()

        # If early stopping worked, use best_iteration as a tighter n_estimators
        best_iteration = best_trial.get("best_iteration", None)
        if best_iteration is not None and best_iteration > 0:
            # best_iteration is typically 0-based
            best_params["n_estimators"] = int(best_iteration + 1)
        else:
            # keep a sane cap if ES not supported
            best_params["n_estimators"] = min(int(best_params.get("n_estimators", 5000)), 2000)

        model = XGBClassifier(**best_params)
        # Fit on full train; if ES supported, you can still use a val set, but we keep it simple/portable
        model.fit(X_train_full, y_train_is_up)

        tuning_summary = {
            "best_val_logloss": float(best_trial["val_logloss"]),
            "best_val_auc": float(best_trial["val_auc"]),
            "best_iteration": best_iteration,
            "used_early_stopping": bool(best_trial.get("used_early_stopping", False)),
            "n_trials": int(n_trials),
        }
    else:
        best_params = {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "n_estimators": 300,
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
            "tree_method": "hist",
        }
        model = XGBClassifier(**best_params)
        model.fit(X_train_full, y_train_is_up)

    # -----------------------------
    # 7) Predict + Decision Boundary (+ reject zone)
    # -----------------------------
    if not (0.0 < decision_boundary < 1.0):
        raise ValueError(f"decision_boundary must be in (0, 1). Got {decision_boundary}")
    if margin_threshold < 0.0:
        raise ValueError(f"margin_threshold must be >= 0. Got {margin_threshold}")

    p_up = model.predict_proba(X_test)[:, 1]
    margin = np.abs(p_up - decision_boundary)

    final_preds = np.full(len(p_up), 2, dtype=int)  # sideways
    confident = margin >= margin_threshold
    final_preds[confident] = (p_up[confident] > decision_boundary).astype(int)  # 1 up, 0 down

    probability_for_pred = np.where(final_preds == 2, decision_boundary, p_up)

    # -----------------------------
    # 8) Save predictions
    # -----------------------------
    output_df = pd.DataFrame({
        "timestamp": test_df["timestamp"].values,
        "prediction": final_preds,
        "actual_trend": y_test.values,
        "p_up": p_up,
        "decision_boundary": float(decision_boundary),
        "margin": margin,
        "used": confident.astype(int),
        "probability": probability_for_pred,
    })
    output_df.to_csv(preds_csv_path, index=False)
    print(f"✅ Predictions saved to {preds_csv_path}")

    # -----------------------------
    # 9) Evaluation
    # -----------------------------
    test_ll = log_loss(y_test_is_up, p_up, labels=[0, 1])
    test_auc = roc_auc_score(y_test_is_up, p_up)
    print(f"\n📈 Test LogLoss (all): {test_ll:.5f}")
    print(f"📈 Test ROC-AUC  (all): {test_auc:.5f}")

    mask = final_preds != 2
    filtered_preds = final_preds[mask]
    filtered_true = y_test_is_up[mask]

    if len(filtered_preds) > 0:
        print("\n📊 Classification Report (Up & Down only):")
        print(classification_report(filtered_true, filtered_preds, target_names=["down", "up"]))

        print("\n🔢 Confusion Matrix:")
        print(confusion_matrix(filtered_true, filtered_preds))
    else:
        print("\n⚠️ No predictions passed the margin threshold.")

    sideways_count = int((final_preds == 2).sum())
    sideways_pct = (sideways_count / len(final_preds)) * 100
    print(f"\n➡️ Sideways count: {sideways_count} ({sideways_pct:.2f}%)")
    print(f"➡️ Decision boundary: {decision_boundary}")
    print(f"➡️ Margin threshold: {margin_threshold}")

    print("\n📊 Label balance:")
    print(f"Train UP rate: {(y_train_is_up.mean()):.4f}")
    print(f"Test  UP rate: {(y_test_is_up.mean()):.4f}")

    if tuning_summary is not None:
        print("\n🧪 Tuning summary:")
        for k, v in tuning_summary.items():
            print(f"  - {k}: {v}")

    # -----------------------------
    # 10) Save artifacts
    # -----------------------------
    artifacts = {
        "model": model,
        "label_encoder": le,
        "features": features,
        "decision_boundary": float(decision_boundary),
        "margin_threshold": float(margin_threshold),
        "symbol": symbol,
        "resolution": resolution,
        "start_date": start_date,
        "end_date": end_date,
        "ignore_zone": f"sideways if |p_up-{decision_boundary}| < {margin_threshold}",
        "best_params": best_params,
        "tuning_summary": tuning_summary,
        "test_logloss_all": float(test_ll),
        "test_auc_all": float(test_auc),
        "val_ratio_within_train": float(val_ratio_within_train),
        "tuning_trials": trials,  # can be large; remove if you don't want it saved
    }

    dump(artifacts, artifacts_path)
    print(f"✅ Model saved to {artifacts_path}")

    return model, artifacts, output_df


if __name__ == "__main__":
    train_xgb_model(
        tune=True,
        n_trials=30,
        early_stopping_rounds=50,
        decision_boundary=0.46,
        margin_threshold=0.05,
    )