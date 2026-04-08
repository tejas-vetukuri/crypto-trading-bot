#models/xgboost/xgb.py
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

from joblib import dump, load

from data.binance import BinanceDataClient
from data.feature_engineering import feature_engineering_xgb
from data.feature_engineering_bybit import feature_engineering_bybit
from data.bybit_derivatives_features import merge_bybit_oi_lsr_features


PROJECT_ROOT = Path(__file__).resolve().parents[2]
XGB_DIR = PROJECT_ROOT / "models" / "xgboost"
SAVED_DIR = XGB_DIR / "saved"

START_DATES_BY_INTERVAL = {
    "5m": "2025-01-01",
    "15m": "2024-01-01",
    "1h": "2017-09-01",
    "4h": "2017-09-01",
}


def resolve_project_path(path_str: str | Path) -> Path:
    p = Path(path_str)
    if p.is_absolute():
        return p
    return PROJECT_ROOT / p


def get_default_start_date(resolution: str) -> str:
    if resolution not in START_DATES_BY_INTERVAL:
        raise ValueError(
            f"Unsupported resolution '{resolution}'. "
            f"Expected one of: {list(START_DATES_BY_INTERVAL.keys())}"
        )
    return START_DATES_BY_INTERVAL[resolution]


def symbol_tag(symbol: str) -> str:
    symbol = symbol.upper()
    if symbol.endswith("USDT"):
        return symbol[:-4]
    return symbol


def combo_tag(symbol: str, resolution: str) -> str:
    return f"{symbol_tag(symbol)}_{resolution}"


def build_xgb_save_paths(symbol: str, resolution: str) -> dict[str, Path]:
    tag = combo_tag(symbol, resolution)
    SAVED_DIR.mkdir(parents=True, exist_ok=True)

    return {
        "artifacts_path": SAVED_DIR / f"xgb_artifacts_{tag}.joblib",
        "preds_csv_path": SAVED_DIR / f"xgb_predictions_{tag}.csv",
    }


def train_xgb_model(
    symbol: str = "BTCUSDT",
    resolution: str = "1h",
    start_date: str | None = None,
    end_date: str | None = None,
    train_ratio: float = 0.80,

    decision_boundary: float = 0.48,
    margin_threshold: float = 0.10,

    tuned_artifacts_path: str | Path = "models/xgboost/xgb_tuned_artifacts.joblib",

    artifacts_path: str | Path | None = None,
    preds_csv_path: str | Path | None = None,

    n_estimators: int | None = None,
    learning_rate: float | None = None,
    max_depth: int | None = None,
    subsample: float | None = None,
    colsample_bytree: float | None = None,
    use_sentiment_data: bool = False,
):
    """
    XGBoost next-direction classifier with:
      - BinanceDataClient fetch
      - Optional Bybit OI / LSR merge
      - Feature engineering
      - Chronological split
      - Binary labels: down=0, up=1
      - Decision boundary + ignore zone
      - Loads tuned hyperparameters from tuned_artifacts_path by default
      - Allows manual hyperparameter overrides from the evaluation lab
      - Saves combination-specific artifacts/predictions

    Returns:
      model, artifacts, output_df
    """
    symbol = symbol.upper()

    if start_date is None:
        start_date = get_default_start_date(resolution)

    default_paths = build_xgb_save_paths(symbol, resolution)

    tuned_artifacts_path_p = resolve_project_path(tuned_artifacts_path)
    artifacts_path_p = (
        resolve_project_path(artifacts_path)
        if artifacts_path is not None
        else default_paths["artifacts_path"]
    )
    preds_csv_path_p = (
        resolve_project_path(preds_csv_path)
        if preds_csv_path is not None
        else default_paths["preds_csv_path"]
    )

    for p in [artifacts_path_p, preds_csv_path_p]:
        p.parent.mkdir(parents=True, exist_ok=True)

    print("\n================ XGB TRAIN CONFIG ================")
    print(f"Symbol:            {symbol}")
    print(f"Resolution:        {resolution}")
    print(f"Start date:        {start_date}")
    print(f"End date:          {end_date}")
    print(f"Train ratio:       {train_ratio}")
    print(f"Decision boundary: {decision_boundary}")
    print(f"Margin threshold:  {margin_threshold}")
    print(f"Use sentiment:     {use_sentiment_data}")
    print(f"Combo tag:         {combo_tag(symbol, resolution)}")
    print(f"Tuned artifacts:   {tuned_artifacts_path_p}")
    print("==================================================\n")

    # -----------------------------
    # 1) Fetch candles
    # -----------------------------
    client = BinanceDataClient(market="spot")
    df = client.get_candles(
        symbol=symbol,
        resolution=resolution,
        start_date=start_date,
        end_date=end_date,
    )

    df = df.sort_values("timestamp").reset_index(drop=True)

    if df.empty:
        raise ValueError("No candle data returned.")

    # -----------------------------
    # 2) Optional Bybit merge
    # -----------------------------
    if use_sentiment_data:
        df = merge_bybit_oi_lsr_features(
            price_df=df,
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
        )

    # -----------------------------
    # 3) Chronological split
    # -----------------------------
    n = len(df)
    train_end = int(n * train_ratio)

    if train_end <= 0 or train_end >= n:
        raise ValueError(f"Invalid split. n={n}, train_end={train_end}")

    train_df = df.iloc[:train_end].copy()
    test_df = df.iloc[train_end:].copy()

    # -----------------------------
    # 4) Feature engineering
    # -----------------------------
    if use_sentiment_data:
        train_df = feature_engineering_bybit(train_df)
        test_df = feature_engineering_bybit(test_df)

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

    if use_sentiment_data:
        features += ["oi_change", "oi_z", "lsr_z"]

    missing_train = [c for c in features + ["actual_trend"] if c not in train_df.columns]
    missing_test = [c for c in features + ["actual_trend"] if c not in test_df.columns]

    if missing_train:
        raise ValueError(f"Missing train columns after feature engineering: {missing_train}")
    if missing_test:
        raise ValueError(f"Missing test columns after feature engineering: {missing_test}")

    train_df = train_df.dropna(subset=features + ["actual_trend"]).reset_index(drop=True)
    test_df = test_df.dropna(subset=features + ["actual_trend"]).reset_index(drop=True)

    if train_df.empty or test_df.empty:
        raise ValueError("Empty train/test set after feature engineering and NA drop.")

    X_train = train_df[features]
    y_train = train_df["actual_trend"].astype(str)

    X_test = test_df[features]
    y_test = test_df["actual_trend"].astype(str)

    # -----------------------------
    # 5) Encode labels
    # -----------------------------
    y_train_bin = (y_train.values == "up").astype(int)
    y_test_bin = (y_test.values == "up").astype(int)

    le = LabelEncoder()
    le.fit(y_train)

    if "up" not in le.classes_ or "down" not in le.classes_:
        raise ValueError(f"Expected classes to include 'up' and 'down', got {list(le.classes_)}")

    # -----------------------------
    # 6) Load tuned hyperparameters
    # -----------------------------
    best_params = None
    try:
        tuned_artifacts = load(str(tuned_artifacts_path_p))
        if isinstance(tuned_artifacts, dict) and "best_params" in tuned_artifacts:
            best_params = tuned_artifacts["best_params"]
            print(f"✅ Loaded tuned best_params from {tuned_artifacts_path_p}")
        else:
            print(f"⚠️ No 'best_params' found in {tuned_artifacts_path_p}. Using defaults.")
    except Exception as e:
        print(f"⚠️ Could not load tuned artifacts from {tuned_artifacts_path_p}: {e}")
        print("⚠️ Using default XGB params.")

    # -----------------------------
    # 7) Train model
    # -----------------------------
    default_params = dict(
        objective="binary:logistic",
        eval_metric="logloss",
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        tree_method="hist",
    )

    model_params = (
        {**default_params, **best_params}
        if isinstance(best_params, dict)
        else default_params.copy()
    )

    if n_estimators is not None:
        model_params["n_estimators"] = int(n_estimators)
    if learning_rate is not None:
        model_params["learning_rate"] = float(learning_rate)
    if max_depth is not None:
        model_params["max_depth"] = int(max_depth)
    if subsample is not None:
        model_params["subsample"] = float(subsample)
    if colsample_bytree is not None:
        model_params["colsample_bytree"] = float(colsample_bytree)

    model_params["objective"] = "binary:logistic"
    model_params["eval_metric"] = "logloss"

    print("\n================ XGB PARAMS USED =================")
    for k, v in model_params.items():
        print(f"{k}: {v}")
    print("==================================================\n")

    model = XGBClassifier(**model_params)
    model.fit(X_train, y_train_bin)

    # -----------------------------
    # 8) Predict + decision boundary
    # -----------------------------
    if not (0.0 < decision_boundary < 1.0):
        raise ValueError(f"decision_boundary must be in (0, 1). Got {decision_boundary}")

    if margin_threshold < 0.0:
        raise ValueError(f"margin_threshold must be >= 0. Got {margin_threshold}")

    probs = model.predict_proba(X_test)
    p_up = probs[:, 1]

    margin = np.abs(p_up - decision_boundary)

    final_preds = np.full(len(p_up), 2, dtype=int)
    confident = margin >= margin_threshold
    final_preds[confident] = (p_up[confident] > decision_boundary).astype(int)

    probability_for_pred = np.where(final_preds == 2, decision_boundary, p_up)

    # -----------------------------
    # 9) Save predictions
    # -----------------------------
    output_df = pd.DataFrame({
        "timestamp": test_df["timestamp"].values,
        "symbol": symbol,
        "resolution": resolution,
        "prediction": final_preds,
        "actual_trend": y_test.values,
        "p_up": p_up,
        "decision_boundary": float(decision_boundary),
        "margin": margin,
        "used": confident.astype(int),
        "probability": probability_for_pred,
    })

    output_df.to_csv(str(preds_csv_path_p), index=False)
    print(f"✅ Predictions saved to {preds_csv_path_p}")

    # -----------------------------
    # 10) Used-trade evaluation
    # -----------------------------
    mask = final_preds != 2
    filtered_preds = final_preds[mask]
    filtered_true = y_test_bin[mask]

    if len(filtered_preds) > 0:
        print("\n📊 Classification Report (Up & Down only):")
        print(classification_report(filtered_true, filtered_preds, target_names=["down", "up"]))

        print("\n🔢 Confusion Matrix:")
        print(confusion_matrix(filtered_true, filtered_preds))
    else:
        print("\n⚠️ No predictions passed the margin threshold.")

    sideways_count = int((final_preds == 2).sum())
    sideways_pct = (sideways_count / len(final_preds)) * 100 if len(final_preds) else 0.0

    print(f"\n➡️ Sideways count: {sideways_count} ({sideways_pct:.2f}%)")
    print(f"➡️ Decision boundary: {decision_boundary}")
    print(f"➡️ Margin threshold: {margin_threshold}")

    print("\n📊 Label balance:")
    print(f"Train UP rate: {float(y_train_bin.mean()):.4f}")
    print(f"Test  UP rate: {float(y_test_bin.mean()):.4f}")

    # -----------------------------
    # 11) Save artifacts
    # -----------------------------
    try:
        tuned_artifacts_rel = str(tuned_artifacts_path_p.relative_to(PROJECT_ROOT))
    except ValueError:
        tuned_artifacts_rel = str(tuned_artifacts_path_p)

    try:
        preds_csv_rel = str(preds_csv_path_p.relative_to(PROJECT_ROOT))
    except ValueError:
        preds_csv_rel = str(preds_csv_path_p)

    artifacts = {
        "model": model,
        "label_encoder": le,
        "features": features,
        "decision_boundary": float(decision_boundary),
        "margin_threshold": float(margin_threshold),
        "use_sentiment_data": bool(use_sentiment_data),
        "symbol": symbol,
        "symbol_tag": symbol_tag(symbol),
        "resolution": resolution,
        "combo_tag": combo_tag(symbol, resolution),
        "start_date": start_date,
        "end_date": end_date,
        "market": "spot",
        "train_ratio": float(train_ratio),
        "ignore_zone": f"sideways if |p_up-{decision_boundary}| < {margin_threshold}",
        "best_params_used": model_params,
        "tuned_artifacts_path": tuned_artifacts_rel,
        "preds_csv_path": preds_csv_rel,
    }

    dump(artifacts, str(artifacts_path_p))
    print(f"✅ Model saved to {artifacts_path_p}")

    return model, artifacts, output_df


if __name__ == "__main__":
    a = load(str(resolve_project_path("models/xgboost/xgb_tuned_artifacts.joblib")))
    print(a.keys())
    print(a["best_params"])

    train_xgb_model()