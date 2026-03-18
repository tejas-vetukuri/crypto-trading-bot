from __future__ import annotations

from pprint import pprint

import pandas as pd

from data.bybit_derivatives import BybitDerivativesClient


SYMBOL = "BTCUSDT"
CATEGORY = "linear"
INTERVAL_TIME = "1h"   # for open interest
PERIOD = "1h"          # for long/short ratio
START_DATE = "2017-09-01"
END_DATE = None


def print_df_summary(name: str, df: pd.DataFrame) -> None:
    print(f"\n{'=' * 80}")
    print(f"{name} DATAFRAME SUMMARY")
    print(f"{'=' * 80}")
    print("Shape:", df.shape)
    print("Columns:", list(df.columns))

    if df.empty:
        print("DataFrame is EMPTY")
        return

    print("\nHead:")
    print(df.head(10).to_string())

    print("\nTail:")
    print(df.tail(10).to_string())

    if "timestamp" in df.columns:
        print("\nTimestamp dtype:", df["timestamp"].dtype)
        print("Min timestamp:", df["timestamp"].min())
        print("Max timestamp:", df["timestamp"].max())

    non_null = df.notna().sum()
    print("\nNon-null counts:")
    print(non_null.to_string())


def raw_open_interest_call(client: BybitDerivativesClient):
    endpoint = "/v5/market/open-interest"
    params = {
        "category": CATEGORY,
        "symbol": SYMBOL,
        "intervalTime": INTERVAL_TIME,
        "limit": 5,
    }
    if START_DATE:
        params["startTime"] = client._to_millis(START_DATE)
    if END_DATE:
        params["endTime"] = client._to_millis(END_DATE)

    payload = client._get(endpoint, params)

    print(f"\n{'#' * 80}")
    print("RAW OPEN INTEREST RESPONSE")
    print(f"{'#' * 80}")
    print("Top-level keys:", list(payload.keys()))
    print("retCode:", payload.get("retCode"))
    print("retMsg:", payload.get("retMsg"))

    result = payload.get("result", {})
    print("result keys:", list(result.keys()))
    rows = result.get("list", [])
    print("row count in sample response:", len(rows))
    if rows:
        print("first row:")
        pprint(rows[0])

    return payload


def raw_lsr_call(client: BybitDerivativesClient):
    endpoint = "/v5/market/account-ratio"
    params = {
        "category": CATEGORY,
        "symbol": SYMBOL,
        "period": PERIOD,
        "limit": 5,
    }
    if START_DATE:
        params["startTime"] = client._to_millis(START_DATE)
    if END_DATE:
        params["endTime"] = client._to_millis(END_DATE)

    payload = client._get(endpoint, params)

    print(f"\n{'#' * 80}")
    print("RAW LONG/SHORT RATIO RESPONSE")
    print(f"{'#' * 80}")
    print("Top-level keys:", list(payload.keys()))
    print("retCode:", payload.get("retCode"))
    print("retMsg:", payload.get("retMsg"))

    result = payload.get("result", {})
    print("result keys:", list(result.keys()))
    rows = result.get("list", [])
    print("row count in sample response:", len(rows))
    if rows:
        print("first row:")
        pprint(rows[0])

    return payload


if __name__ == "__main__":
    client = BybitDerivativesClient()

    print("\nStarting Bybit derivatives test...")
    print(f"SYMBOL={SYMBOL}, CATEGORY={CATEGORY}, START_DATE={START_DATE}, END_DATE={END_DATE}")

    # 1) raw payload checks
    raw_open_interest_call(client)
    raw_lsr_call(client)

    # 2) normalized dataframe checks
    oi_df = client.get_open_interest_history(
        symbol=SYMBOL,
        category=CATEGORY,
        interval_time=INTERVAL_TIME,
        start_date=START_DATE,
        end_date=END_DATE,
    )
    print_df_summary("OPEN INTEREST", oi_df)

    lsr_df = client.get_long_short_ratio_history(
        symbol=SYMBOL,
        category=CATEGORY,
        period=PERIOD,
        start_date=START_DATE,
        end_date=END_DATE,
    )
    print_df_summary("LONG/SHORT RATIO", lsr_df)

    # 3) save debug CSVs
    oi_df.to_csv("bybit_open_interest_debug.csv", index=False)
    lsr_df.to_csv("bybit_long_short_ratio_debug.csv", index=False)

    print("\nSaved:")
    print("- bybit_open_interest_debug.csv")
    print("- bybit_long_short_ratio_debug.csv")