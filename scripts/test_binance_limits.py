import requests
import pandas as pd

BASE_URL = "https://api.binance.com/api/v3/klines"

symbols = ["BTCUSDT", "ETHUSDT"]
intervals = ["5m", "15m", "1h", "4h"]

VERY_OLD_START = "2010-01-01"


def to_millis(date_str):
    return int(pd.Timestamp(date_str, tz="UTC").timestamp() * 1000)


def get_earliest(symbol, interval):
    params = {
        "symbol": symbol,
        "interval": interval,
        "startTime": to_millis(VERY_OLD_START),
        "limit": 1,
    }

    r = requests.get(BASE_URL, params=params, timeout=20)
    r.raise_for_status()
    data = r.json()

    if not data:
        return None

    ts = pd.to_datetime(data[0][0], unit="ms", utc=True)
    return ts


def main():
    print("\n===== BINANCE SPOT EARLIEST DATA CHECK =====\n")

    for s in symbols:
        for i in intervals:
            try:
                ts = get_earliest(s, i)
                if ts is None:
                    print(f"{s} {i}: NO DATA")
                else:
                    print(f"{s} {i}: starts at {ts}")
            except Exception as e:
                print(f"{s} {i}: ERROR → {e}")


if __name__ == "__main__":
    main()