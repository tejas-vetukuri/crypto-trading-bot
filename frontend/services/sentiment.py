from __future__ import annotations

import requests


BINANCE_FAPI_BASE = "https://fapi.binance.com"


def _normalize_interval(interval: str) -> str:
    interval = str(interval).strip().lower()
    mapping = {
        "1m": "5m",
        "3m": "5m",
        "5m": "5m",
        "15m": "15m",
        "30m": "30m",
        "1h": "1h",
        "2h": "2h",
        "4h": "4h",
        "6h": "6h",
        "12h": "12h",
        "1d": "1d",
        "d": "1d",
    }
    return mapping.get(interval, "1h")


def _safe_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _get_json(endpoint: str, params: dict | None = None, timeout: tuple[int, int] = (2, 4)):
    url = f"{BINANCE_FAPI_BASE}{endpoint}"
    r = requests.get(url, params=params or {}, timeout=timeout)
    r.raise_for_status()
    return r.json()


def _default_snapshot() -> dict:
    return {
        "open_interest": 0.0,
        "oi_change_pct": 0.0,
        "long_ratio": 0.0,
        "short_ratio": 0.0,
        "lsr_ratio": 0.0,
        "price_change_pct": 0.0,
        "sentiment": "NEUTRAL",
        "reason": "Sentiment data unavailable",
        "timestamp_oi": None,
        "timestamp_lsr": None,
    }


def derive_sentiment_label(
    oi_change_pct: float,
    long_ratio: float,
    short_ratio: float,
    price_change_pct: float,
) -> tuple[str, str]:
    oi_up = oi_change_pct > 0
    oi_down = oi_change_pct < 0
    price_up = price_change_pct > 0
    price_down = price_change_pct < 0
    longs_dom = long_ratio > short_ratio
    shorts_dom = short_ratio > long_ratio

    if oi_up and price_up and longs_dom:
        return "BULLISH", "Price up, OI up, longs dominant"

    if oi_up and price_down and shorts_dom:
        return "BEARISH", "Price down, OI up, shorts dominant"

    if oi_up and price_up and shorts_dom:
        return "MIXED", "Price rising, but shorts still dominant"

    if oi_up and price_down and longs_dom:
        return "MIXED", "Price falling, but longs still dominant"

    if oi_down and price_up:
        return "MIXED", "Price up while OI falls"

    if oi_down and price_down:
        return "MIXED", "Price down while OI falls"

    if longs_dom and abs(price_change_pct) < 0.05:
        return "SLIGHTLY BULLISH", "Longs dominant, but price change is small"

    if shorts_dom and abs(price_change_pct) < 0.05:
        return "SLIGHTLY BEARISH", "Shorts dominant, but price change is small"

    return "NEUTRAL", "No clear directional sentiment"


def fetch_sentiment_snapshot(symbol: str, interval: str, closes_for_price_change: list[float]) -> dict:
    """
    Fast, fail-safe live snapshot.
    If Binance futures sentiment endpoints fail, returns neutral defaults
    instead of breaking or stalling the page.
    """
    try:
        symbol = symbol.upper()
        period = _normalize_interval(interval)

        price_change_pct = 0.0
        if closes_for_price_change and len(closes_for_price_change) >= 2:
            prev_close = _safe_float(closes_for_price_change[-2])
            last_close = _safe_float(closes_for_price_change[-1])
            if prev_close != 0:
                price_change_pct = ((last_close - prev_close) / prev_close) * 100.0

        # current OI
        oi_live = _get_json(
            "/fapi/v1/openInterest",
            params={"symbol": symbol},
        )

        # latest 2 OI history rows
        oi_hist = _get_json(
            "/futures/data/openInterestHist",
            params={
                "symbol": symbol,
                "period": period,
                "limit": 2,
            },
        )

        # latest 1 global LSR row
        lsr_hist = _get_json(
            "/futures/data/globalLongShortAccountRatio",
            params={
                "symbol": symbol,
                "period": period,
                "limit": 1,
            },
        )

        open_interest = _safe_float(oi_live.get("openInterest"))
        timestamp_oi = oi_live.get("time")

        oi_change_pct = 0.0
        if isinstance(oi_hist, list) and len(oi_hist) >= 2:
            prev_oi = _safe_float(oi_hist[-2].get("sumOpenInterest"))
            curr_oi_hist = _safe_float(oi_hist[-1].get("sumOpenInterest"))
            if prev_oi != 0:
                oi_change_pct = ((curr_oi_hist - prev_oi) / prev_oi) * 100.0

        long_ratio = 0.0
        short_ratio = 0.0
        lsr_ratio = 0.0
        timestamp_lsr = None

        if isinstance(lsr_hist, list) and len(lsr_hist) >= 1:
            row = lsr_hist[-1]
            long_ratio = _safe_float(row.get("longAccount"))
            short_ratio = _safe_float(row.get("shortAccount"))
            lsr_ratio = _safe_float(row.get("longShortRatio"))
            timestamp_lsr = row.get("timestamp")

        sentiment, reason = derive_sentiment_label(
            oi_change_pct=oi_change_pct,
            long_ratio=long_ratio,
            short_ratio=short_ratio,
            price_change_pct=price_change_pct,
        )

        return {
            "open_interest": open_interest,
            "oi_change_pct": oi_change_pct,
            "long_ratio": long_ratio,
            "short_ratio": short_ratio,
            "lsr_ratio": lsr_ratio,
            "price_change_pct": price_change_pct,
            "sentiment": sentiment,
            "reason": reason,
            "timestamp_oi": timestamp_oi,
            "timestamp_lsr": timestamp_lsr,
        }

    except Exception:
        snapshot = _default_snapshot()

        if closes_for_price_change and len(closes_for_price_change) >= 2:
            prev_close = _safe_float(closes_for_price_change[-2])
            last_close = _safe_float(closes_for_price_change[-1])
            if prev_close != 0:
                snapshot["price_change_pct"] = ((last_close - prev_close) / prev_close) * 100.0

        return snapshot