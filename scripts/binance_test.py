# test_binance_fetch.py

import pandas as pd
from data.binance import BinanceDataClient


def test_fetch(
    symbol: str = "BTCUSDT",
    resolution: str = "1h",
    start_date: str = "2015-01-01",
    end_date: str | None = None,
    market: str = "futures",   # try "spot" as well
):
    client = BinanceDataClient(market=market)

    df = client.get_candles(
        symbol=symbol,
        resolution=resolution,
        start_date=start_date,
        end_date=end_date,
    )

    print("\n================ BINANCE FETCH TEST ================")
    print(f"Market:         {market}")
    print(f"Symbol:         {symbol}")
    print(f"Resolution:     {resolution}")
    print(f"Requested from: {start_date}")
    print(f"Requested to:   {end_date}")

    print(f"\nRows fetched:   {len(df)}")

    if df.empty:
        print("\n❌ No rows returned.")
        return

    print(f"First timestamp returned: {df['timestamp'].min()}")
    print(f"Last timestamp returned:  {df['timestamp'].max()}")

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nHead:")
    print(df.head(3))

    print("\nTail:")
    print(df.tail(3))

    # Rough sanity check for 1h candles
    first_ts = pd.to_datetime(df["timestamp"].min(), utc=True)
    last_ts = pd.to_datetime(df["timestamp"].max(), utc=True)

    total_hours = (last_ts - first_ts).total_seconds() / 3600
    print(f"\nApprox hours covered: {total_hours:.0f}")

    if resolution == "1h":
        expected_rows = int(total_hours) + 1
        print(f"Approx expected 1h rows: {expected_rows}")
        if expected_rows > 0:
            print(f"Coverage ratio: {len(df) / expected_rows:.4f}")

    requested_start = pd.to_datetime(start_date, utc=True)
    diff_days = (first_ts - requested_start).days
    print(f"\nDays between requested start and actual first candle: {diff_days}")

    if first_ts > requested_start:
        print("⚠️ Binance did NOT return data starting from your requested date.")
        print("   This usually means the market/symbol did not exist that early.")
    else:
        print("✅ Binance returned data from the requested start date (or earlier rounded candle).")


if __name__ == "__main__":
    test_fetch(
        symbol="BTCUSDT",
        resolution="1h",
        start_date="2015-01-01",
        end_date=None,
        market="futures",   # first test
    )

    print("\n" + "=" * 70 + "\n")

    test_fetch(
        symbol="BTCUSDT",
        resolution="1h",
        start_date="2015-01-01",
        end_date=None,
        market="spot",      # second test
    )