from __future__ import annotations

import numpy as np
import pandas as pd


def _safe_zscore(series: pd.Series, window: int = 72, min_periods: int | None = None) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")

    if min_periods is None:
        min_periods = max(10, window // 3)

    rolling_mean = s.rolling(window=window, min_periods=min_periods).mean()
    rolling_std = s.rolling(window=window, min_periods=min_periods).std()

    z = (s - rolling_mean) / rolling_std.replace(0, np.nan)
    z = z.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return z


def feature_engineering_bybit(
    df: pd.DataFrame,
    oi_col: str = "by_oi",
    lsr_col: str = "by_lsr",
    z_window: int = 72,
) -> pd.DataFrame:
    """
    Builds stationary/regime-friendly Bybit derivatives features.

    Expected raw columns:
        - by_oi
        - by_lsr

    Output columns:
        - oi_change
        - oi_z
        - lsr_z
    """
    out = df.copy()

    if oi_col in out.columns:
        oi = pd.to_numeric(out[oi_col], errors="coerce")
        out[oi_col] = oi
        out["oi_change"] = oi.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
        out["oi_z"] = _safe_zscore(oi, window=z_window)
    else:
        out["oi_change"] = 0.0
        out["oi_z"] = 0.0

    if lsr_col in out.columns:
        lsr = pd.to_numeric(out[lsr_col], errors="coerce")
        out[lsr_col] = lsr
        out["lsr_z"] = _safe_zscore(lsr, window=z_window)
    else:
        out["lsr_z"] = 0.0

    return out