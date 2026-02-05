from datetime import datetime
from data.delta_exchange import DeltaDataClient


def main():
    client = DeltaDataClient()

    print("Fetching candles...")
    df = client.get_candles(
        symbol="BTCUSD",
        resolution="5m",
        limit=50,
    )

    print("\n=== CANDLES ===")
    print(df.head())
    print(df.tail())

    print("\nRows:", len(df))
    print("Columns:", list(df.columns))

    # Basic sanity checks
    assert not df.empty, "DataFrame is empty"
    assert df.isnull().sum().sum() == 0, "NaNs found in candle data"
    assert df["timestamp"].is_monotonic_increasing, "Timestamps not sorted"

    print("\nFetching ticker...")
    ticker = client.get_ticker("BTCUSD")

    print("\n=== TICKER ===")
    print(ticker)

    assert "close" in ticker, "Ticker missing close price"

    print("\n✅ Data client test passed!")


if __name__ == "__main__":
    main()
