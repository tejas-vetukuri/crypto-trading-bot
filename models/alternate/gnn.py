from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from joblib import dump, load

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.preprocessing import StandardScaler

from data.binance import BinanceDataClient
from data.feature_engineering import feature_engineering_xgb


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ALT_DIR = PROJECT_ROOT / "models" / "alternate"
SAVED_DIR = ALT_DIR / "saved"

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


def build_gnn_save_paths(symbol: str, resolution: str) -> dict[str, Path]:
    tag = combo_tag(symbol, resolution)
    SAVED_DIR.mkdir(parents=True, exist_ok=True)

    return {
        "model_path": SAVED_DIR / f"gnn_model_{tag}.pt",
        "scaler_path": SAVED_DIR / f"gnn_scaler_{tag}.joblib",
        "artifacts_path": SAVED_DIR / f"gnn_artifacts_{tag}.joblib",
        "preds_csv_path": SAVED_DIR / f"gnn_predictions_{tag}.csv",
        "metrics_path": SAVED_DIR / f"gnn_metrics_{tag}.csv",
    }


def get_gnn_feature_cols() -> list[str]:
    return [
        "ema_20", "ema_50", "rsi", "atr",
        "log_ret_1", "ret_1", "ret_3", "ret_5", "ret_10",
        "body", "range", "upper_wick", "lower_wick", "body_pct", "range_pct", "clv",
        "ema_spread", "ema20_dist", "ema50_dist", "ema20_slope_3", "ema50_slope_3",
        "atr_pct", "rsi_delta", "rsi_ma_10", "rsi_dist",
        "volatility_5", "vol_10", "vol_30", "vol_ratio",
        "vol_chg_1", "vol_chg_5", "vol_z20",
    ]


class AssetGraphDatasetBuilder:
    def __init__(
        self,
        assets: list[str],
        resolution: str,
        start_date: str,
        end_date: str | None,
        train_ratio: float,
    ):
        self.assets = [a.upper() for a in assets]
        self.resolution = resolution
        self.start_date = start_date
        self.end_date = end_date
        self.train_ratio = train_ratio
        self.feature_cols = get_gnn_feature_cols()

    def _fetch_one_asset(self, symbol: str) -> pd.DataFrame:
        client = BinanceDataClient(market="spot")
        df = client.get_candles(
            symbol=symbol,
            resolution=self.resolution,
            start_date=self.start_date,
            end_date=self.end_date,
        )
        df = df.sort_values("timestamp").reset_index(drop=True)
        df = feature_engineering_xgb(df)
        df["asset"] = symbol
        return df

    def build(self) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        dfs = []
        for asset in self.assets:
            df = self._fetch_one_asset(asset)

            required = ["timestamp", "close"] + self.feature_cols
            missing = [c for c in required if c not in df.columns]
            if missing:
                raise ValueError(f"Missing columns for {asset}: {missing}")

            df["target_up"] = (df["close"].shift(-1) > df["close"]).astype(int)
            df = df.dropna(subset=self.feature_cols).reset_index(drop=True)

            cols = ["timestamp", "asset", "target_up"] + self.feature_cols
            dfs.append(df[cols].copy())

        all_df = pd.concat(dfs, axis=0, ignore_index=True)

        pivot_frames = []
        for feat in self.feature_cols:
            tmp = all_df.pivot(index="timestamp", columns="asset", values=feat)
            tmp.columns = [f"{asset}__{feat}" for asset in tmp.columns]
            pivot_frames.append(tmp)

        wide_df = pd.concat(pivot_frames, axis=1).sort_index()

        btc_target = (
            all_df[all_df["asset"] == self.assets[0]][["timestamp", "target_up"]]
            .drop_duplicates(subset=["timestamp"])
            .set_index("timestamp")
            .sort_index()
        )

        merged = wide_df.join(btc_target, how="inner")
        merged = merged.dropna().reset_index()

        feature_matrix = []
        for asset in self.assets:
            cols = [f"{asset}__{feat}" for feat in self.feature_cols]
            feature_matrix.append(merged[cols].values)

        # shape: [num_samples, num_assets, num_features]
        X = np.stack(feature_matrix, axis=1).astype(np.float32)
        y = merged["target_up"].astype(int).values

        meta_df = merged[["timestamp"]].copy()
        meta_df["actual_trend"] = np.where(y == 1, "up", "down")

        return X, y, meta_df


class SimpleAssetGNN(nn.Module):
    """
    Graph over assets:
      - nodes = assets
      - node features = engineered per-asset features
      - fully connected message passing
      - pool asset embeddings -> predict BTC direction
    """
    def __init__(self, num_assets: int, num_features: int, hidden_dim: int = 64, dropout: float = 0.2):
        super().__init__()
        self.num_assets = num_assets
        self.hidden_dim = hidden_dim
        self.dropout = dropout

        self.node_proj = nn.Linear(num_features, hidden_dim)
        self.msg_proj = nn.Linear(hidden_dim, hidden_dim)
        self.update_proj = nn.Linear(hidden_dim * 2, hidden_dim)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

        adj = torch.ones(num_assets, num_assets, dtype=torch.float32)
        adj.fill_diagonal_(0.0)
        deg = adj.sum(dim=1, keepdim=True).clamp(min=1.0)
        self.register_buffer("adj_norm", adj / deg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [batch, num_assets, num_features]
        """
        h = F.relu(self.node_proj(x))
        h = F.dropout(h, p=self.dropout, training=self.training)

        neigh = torch.matmul(self.adj_norm.unsqueeze(0), h)
        neigh = F.relu(self.msg_proj(neigh))

        h = torch.cat([h, neigh], dim=-1)
        h = F.relu(self.update_proj(h))

        # use pooled graph embedding
        g = h.mean(dim=1)
        logits = self.classifier(g).squeeze(-1)
        return logits


def _safe_roc_auc(y_true: np.ndarray, probs: np.ndarray):
    try:
        y_true = np.asarray(y_true).astype(int)
        probs = np.asarray(probs).astype(float)
        if len(np.unique(y_true)) < 2:
            return None
        from sklearn.metrics import roc_auc_score
        return float(roc_auc_score(y_true, probs))
    except Exception:
        return None


def train_gnn_model(
    symbol: str = "BTCUSDT",
    resolution: str = "1h",
    start_date: str | None = None,
    end_date: str | None = None,
    train_ratio: float = 0.80,

    decision_boundary: float = 0.50,
    margin_threshold: float = 0.10,

    model_path: str | Path | None = None,
    scaler_path: str | Path | None = None,
    artifacts_path: str | Path | None = None,
    preds_csv_path: str | Path | None = None,
    metrics_path: str | Path | None = None,

    assets: list[str] | None = None,
    hidden_dim: int = 64,
    dropout: float = 0.20,
    learning_rate: float = 0.001,
    batch_size: int = 256,
    epochs: int = 20,
):
    symbol = symbol.upper()

    if assets is None:
        assets = ["BTCUSDT", "ETHUSDT", "BNBUSDT"]
    assets = [a.upper() for a in assets]

    if assets[0] != symbol:
        raise ValueError("First asset in assets list must match symbol, since target is built from that asset.")

    if start_date is None:
        start_date = get_default_start_date(resolution)

    default_paths = build_gnn_save_paths(symbol, resolution)

    model_path_p = resolve_project_path(model_path) if model_path is not None else default_paths["model_path"]
    scaler_path_p = resolve_project_path(scaler_path) if scaler_path is not None else default_paths["scaler_path"]
    artifacts_path_p = resolve_project_path(artifacts_path) if artifacts_path is not None else default_paths["artifacts_path"]
    preds_csv_path_p = resolve_project_path(preds_csv_path) if preds_csv_path is not None else default_paths["preds_csv_path"]
    metrics_path_p = resolve_project_path(metrics_path) if metrics_path is not None else default_paths["metrics_path"]

    for p in [model_path_p, scaler_path_p, artifacts_path_p, preds_csv_path_p, metrics_path_p]:
        p.parent.mkdir(parents=True, exist_ok=True)

    print("\n================ GNN TRAIN CONFIG ================")
    print(f"Target symbol:      {symbol}")
    print(f"Assets:             {assets}")
    print(f"Resolution:         {resolution}")
    print(f"Start date:         {start_date}")
    print(f"End date:           {end_date}")
    print(f"Train ratio:        {train_ratio}")
    print(f"Decision boundary:  {decision_boundary}")
    print(f"Margin threshold:   {margin_threshold}")
    print(f"Combo tag:          {combo_tag(symbol, resolution)}")
    print("==================================================\n")

    builder = AssetGraphDatasetBuilder(
        assets=assets,
        resolution=resolution,
        start_date=start_date,
        end_date=end_date,
        train_ratio=train_ratio,
    )

    X, y, meta_df = builder.build()

    n = len(X)
    train_end = int(n * train_ratio)
    if train_end <= 0 or train_end >= n:
        raise ValueError(f"Invalid split. n={n}, train_end={train_end}")

    X_train, X_test = X[:train_end], X[train_end:]
    y_train, y_test = y[:train_end], y[train_end:]
    meta_test = meta_df.iloc[train_end:].reset_index(drop=True)

    num_assets = X_train.shape[1]
    num_features = X_train.shape[2]

    scaler = StandardScaler()
    X_train_2d = X_train.reshape(-1, num_features)
    X_test_2d = X_test.reshape(-1, num_features)

    X_train_scaled = scaler.fit_transform(X_train_2d).reshape(X_train.shape).astype(np.float32)
    X_test_scaled = scaler.transform(X_test_2d).reshape(X_test.shape).astype(np.float32)
    dump(scaler, str(scaler_path_p))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SimpleAssetGNN(
        num_assets=num_assets,
        num_features=num_features,
        hidden_dim=int(hidden_dim),
        dropout=float(dropout),
    ).to(device)

    pos_weight_val = float((len(y_train) - y_train.sum()) / max(y_train.sum(), 1.0))
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(pos_weight_val, dtype=torch.float32, device=device)
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=float(learning_rate))

    X_train_t = torch.tensor(X_train_scaled, dtype=torch.float32, device=device)
    y_train_t = torch.tensor(y_train.astype(np.float32), dtype=torch.float32, device=device)

    model.train()
    for epoch in range(int(epochs)):
        perm = torch.randperm(X_train_t.size(0), device=device)
        epoch_loss = 0.0

        for i in range(0, X_train_t.size(0), int(batch_size)):
            idx = perm[i:i + int(batch_size)]
            xb = X_train_t[idx]
            yb = y_train_t[idx]

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            epoch_loss += float(loss.item()) * len(idx)

        epoch_loss /= max(X_train_t.size(0), 1)
        print(f"Epoch {epoch + 1}/{epochs} - loss: {epoch_loss:.6f}")

    torch.save(model.state_dict(), str(model_path_p))

    model.eval()
    with torch.no_grad():
        X_test_t = torch.tensor(X_test_scaled, dtype=torch.float32, device=device)
        logits_test = model(X_test_t)
        p_up = torch.sigmoid(logits_test).cpu().numpy()

    margin = np.abs(p_up - decision_boundary)
    final_preds = np.full(len(p_up), 2, dtype=int)
    confident = margin >= margin_threshold
    final_preds[confident] = (p_up[confident] > decision_boundary).astype(int)
    probability_for_pred = np.where(final_preds == 2, decision_boundary, p_up)

    output_df = pd.DataFrame({
        "timestamp": meta_test["timestamp"].values,
        "symbol": symbol,
        "resolution": resolution,
        "prediction": final_preds,
        "actual_trend": meta_test["actual_trend"].values,
        "y_true": y_test.astype(int),
        "p_up": p_up,
        "p": p_up,
        "decision_boundary": float(decision_boundary),
        "margin": margin,
        "used": confident.astype(int),
        "probability": probability_for_pred,
    })
    output_df.to_csv(str(preds_csv_path_p), index=False)

    metrics_df = pd.DataFrame([{
        "roc_auc": _safe_roc_auc(y_test, p_up),
        "accuracy_raw": float(((p_up >= 0.5).astype(int) == y_test).mean()),
        "coverage": float(confident.mean()),
        "ignored_rate": 1.0 - float(confident.mean()),
        "used_accuracy": float((final_preds[confident] == y_test[confident]).mean()) if confident.sum() > 0 else None,
        "used_samples": int(confident.sum()),
        "total_samples": int(len(y_test)),
    }])
    metrics_df.to_csv(str(metrics_path_p), index=False)

    artifacts = {
        "model_path": str(model_path_p),
        "scaler_path": str(scaler_path_p),
        "artifacts_path": str(artifacts_path_p),
        "preds_csv_path": str(preds_csv_path_p),
        "metrics_path": str(metrics_path_p),
        "target_symbol": symbol,
        "assets": assets,
        "resolution": resolution,
        "combo_tag": combo_tag(symbol, resolution),
        "start_date": start_date,
        "end_date": end_date,
        "train_ratio": float(train_ratio),
        "decision_boundary": float(decision_boundary),
        "margin_threshold": float(margin_threshold),
        "features": builder.feature_cols,
        "hidden_dim": int(hidden_dim),
        "dropout": float(dropout),
        "learning_rate": float(learning_rate),
        "batch_size": int(batch_size),
        "epochs": int(epochs),
    }
    dump(artifacts, str(artifacts_path_p))

    return model, artifacts, output_df


def load_gnn_artifacts(artifacts_path: str | Path) -> dict:
    return load(str(resolve_project_path(artifacts_path)))